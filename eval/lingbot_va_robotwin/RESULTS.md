# First measured results — LingBot-VA on RoboTwin 2.0

Box: 8× H100 80GB, driver CUDA 13.0, torch 2.9.0+cu126.
Checkpoint: `robbyant/lingbot-va-posttrain-robotwin`.
Date: 2026-07-31 / 2026-08-01.

## 1. Correctness gate

`check_prompt_parity.py` — the server's live T5 output vs the embedding baked into the
training dataset. This is the check whose absence voided a previous 22.7-hour run.

| comparison | shape | max abs Δ | cosine (non-pad) |
|---|---|---|---|
| positive prompt: live T5 vs dataset `text_emb` | (512, 4096) | **0.000e+00** | 1.000000 |
| CFG negative `""`: live T5 vs `empty_emb.pt` | (512, 4096) | **0.000e+00** | 1.000000 |

Serving reproduces the training text conditioning exactly. Checkpoint also loads with
**zero missing tensors** (key-set diff against the model's `named_parameters`); the two
`patch_embedding.*` keys the loader warns about are a vestigial conv the architecture
replaced with `patch_embedding_mlp` (`model.py:575`, old name commented out).

## 2. Accuracy — canonical 50-task baseline

`baseline50`: all 50 official RoboTwin 2.0 tasks x 50 episodes, **stock** server, LingBot-VA
client protocol (`st_seed = 10000*(1+seed)`, `instruction_type='seen'`), `demo_clean`.
50/50 tasks, 2500/2500 episodes, **zero failures**, `aggregate.py` prints `REPORTABLE: YES`.

| metric | value |
|---|---|
| **MACRO mean over 50 tasks** (leaderboard definition) | **91.6%** |
| MICRO pooled | 91.6% (2291/2500), 95% CI [90.5, 92.7] |
| LingBot-VA published | 92.9% easy / 91.6% hard |

14 tasks at 100%. Floor: `hanging_mug` 30.0%, `open_microwave` 60.0%, `turn_switch` 62.0%,
`put_bottles_dustbin` 70.0%, `move_stapler_pad` 70.0%.

Read honestly: `demo_clean` is the *easy* setting, so the comparison is against their 92.9%, and
our 91.6% sits just below the bottom of that comparison (92.9 is outside our [90.5, 92.7]). This
is a close reproduction, not an exact one. Candidate explanations, none yet tested: 50 episodes
per task rather than 100, a different accepted-seed set, or a genuine small deficit. It is a
sound *reference* either way — every optimization is compared against this run, on these pinned
seeds, not against the paper.

Pinned accepted-seed lists are in `/home/ubuntu/iwm_seeds/baseline50` and per-chunk action
streams in `/home/ubuntu/iwm_actions/baseline50`, so a later arm can replay the identical scene
set and be compared per-episode.

## 3. Latency cost model (batch 1, idle H100)

`probe_latency.py`, 14 cycles, real episode message order. One cycle = 32 control steps =
77 transformer forwards (26 video denoise + 51 action denoise), each at batch 2 because
`guidance_scale=5` forces CFG batch duplication.

| stage | mean | share |
|---|---|---|
| `reset` (T5 encode, once per episode) | 67 ms | — |
| `compute_kv_cache` | 453 ms | 5% |
| `infer` (denoise loop) | 8429 ms | **95%** |
| **full cycle** | **8881 ms** | 277 ms/control step → **3.6 Hz** |

Per-forward: 8429/77 = **109 ms**. The model is ~5.1B params = 10.2 GB bf16; at H100 HBM
3.35 TB/s a memory-bound forward is ~3 ms. **We are ~36× off roofline** — the cost is
overhead, not arithmetic.

Latency also is *not flat*: +7.3% from the first 7 cycles to the last 7, because attention
runs over a KV population that grows 272 tokens per cycle toward a 9792-slot pool
(saturating around cycle 36). Under 8-way GPU sharing the same cycle costs ~16 s (~2 Hz).

## 4. First optimizations — 1.92×, bit-exact

Three costs, all found by reading the code, all removed via `serve_variant.py` (runtime
patches; the upstream tree is untouched). Measured back-to-back on idle GPUs, reference
and variant both under `--deterministic-seed 1234` — required, because `_infer` draws
`torch.randn` unseeded (`wan_va_server.py:449-462`), so two *stock* servers already disagree.

| variant | cycle (mean) | vs stock | control rate | max abs Δ action |
|---|---|---|---|---|
| stock | 8881 ms | 1.00× | 3.6 Hz | — |
| `--no-fsdp` | 5078 ms | **1.75×** | 6.3 Hz | **0.000e+00** |
| `+ --no-empty-cache --no-debug-dump` | 4624 ms | **1.92×** | 6.9 Hz | **0.000e+00** |

Yardstick: the reference model's own chunk-to-chunk action movement is 1.03, so a delta of
exactly zero is not a small number relative to noise — it is *no change at all*.

What each removes:

- **`--no-fsdp`** (the big one). `distributed/util.py:15-19` applies FSDP `fully_shard`
  whenever `dist.is_initialized()`, which is always: `init_distributed` runs
  unconditionally even for a single-GPU server. `fsdp.py:28-34` wraps 4 units per block
  (attn1, attn2, ffn, block) × 30 blocks + root = **121 units** with
  `reshard_after_forward=True`. At world_size=1 every all-gather is a no-op collective, but
  PyTorch still pays the flat-param copy and stream sync per unit per forward — about
  **9,300 shard/unshard round trips per cycle** to shard a model across one GPU.
- **`--no-empty-cache`**. `torch.cuda.empty_cache()` on every chunk
  (`wan_va_server.py:569`) and every KV update (`:603`), returning the caching allocator's
  pool to the driver so the next cycle re-runs `cudaMalloc`.
- **`--no-debug-dump`**. `save_async` (`utils.py:56-70`) is async only for the *disk write*;
  the `.cpu()` at :63-64 is a blocking device→host copy of the full latent and action
  tensors, three times per cycle, unconditional, with no upstream flag. Visible mostly in
  `compute_kv_cache`: 453 → 292 ms.

**These are bit-exact and therefore free.** They need no paired non-inferiority run — which
matters, since establishing non-inferiority costs roughly 10× more GPU time than measuring
a speedup, and a previous project found CUDA graph capture was *not* bit-exact and had to
pay exactly that cost.

## 5. What the numbers say about where to go next

After the free 1.92×, a forward still costs 56 ms against a ~3 ms roofline. The remaining
gap decomposes into two very different problems:

1. **Step count.** 77 forwards per chunk is the dominant term and no amount of kernel work
   touches it. Few-step distillation is the only lever with an order of magnitude in it.
   The 51-step *action* loop is the larger half and conditions on a KV cache the 26-step
   video loop already wrote, so the two are not equally compressible.
2. **Per-forward overhead.** ~53 of the remaining 56 ms is not arithmetic. CUDA graph
   capture, `torch.compile`, and an attention kernel that handles a growing KV window are
   the candidates — but note CUDA graphs were previously measured **not** bit-exact on a
   related model, so that one buys speed at the cost of an expensive accuracy certificate.

Two structural observations worth more than either:

- **CFG doubles every forward.** `guidance_scale=5 > 1` duplicates the batch on all 77
  forwards. A guidance-distilled model halves the compute with no scheduling change.
- **The paper's "asynchronous execution" is not in the server.** There are no CUDA streams
  anywhere in the repo; `wan_va_server.py:489-522` runs the video loop, then `:524-561`
  runs the action loop, strictly sequentially.

Finally, a protocol caveat that governs all of the above: **RoboTwin does not score
latency.** `take_action` blocks until the policy returns, so a 77-forward model ties a
4-forward model. Every number in §4 is real, and none of it shows up in §2. Making latency
count needs its own protocol.

## Appendix — flash-attn and the baseline environment

`baseline50` (section 2) ran with the import-only flash-attn shim and NO real flash-attn in the
venv. That matters: with real flash-attn installed, `diffusers` detects it via package metadata
and routes `autoencoder_kl_wan` down a flash-attention path instead of SDPA, which changes the
VAE's numerics. The transformer is unaffected either way (`attn_mode='torch'` → `custom_sdpa`;
`flash_attn_func` is never called).

A real `flash-attn 2.8.3` wheel finished building mid-session and was **uninstalled** so the
environment continues to match the one that produced the 91.6% baseline. Any run whose numbers
are compared against that baseline must keep it uninstalled. If flash-attn is ever wanted for
the VAE, the baseline has to be re-measured first — it is not a free environment change.

## 6. Where the time actually is (measured 2026-08-01)

`conditioning_prefill` — caching the episode-constant cross-attention K/V for all 30 layers
(`model.py:331` withholds `attn_caches` from cross-attention, so the text K/V is re-projected on
all 77 forwards) — removes **89 of 226 TFLOP per control cycle, 39% of all arithmetic**, and is
**bit-exact** (`max|delta action| = 0.000e+00` over 6 paired seeded cycles).

It bought **1.05x**.

| arm (all on top of the 1.92x bit-exact base) | cycle mean | control rate |
|---|---|---|
| base: fsdp + empty_cache + debug dumps elided | 4320.6 ms | 7.4 Hz |
| base + conditioning_prefill | 4115.4 ms | **7.8 Hz** |

That gap is the most useful measurement so far, because it decomposes the remaining cost:

| component of the 3828 ms denoise | ms | share |
|---|---|---|
| arithmetic, 226 TFLOP at ~350 TFLOPS achieved | 646 | 17% |
| weight traffic, 77 forwards x 10.2 GB at 3.35 TB/s | 234 | 6% |
| **launches, host syncs, KV gather** | **~2950** | **77%** |

The pass performed exactly as predicted *on the arithmetic it removes* (predicted ~254 ms,
measured 201 ms). Arithmetic simply is not where the time goes. Two consequences:

1. **Quantization is deprioritized.** It attacks bytes and FLOPs — 23% of the problem.
2. **Sync/gather elimination and CUDA-graph capture move to the front.** They attack the 77%.

Corroborating evidence from the same run: the per-cycle ramp as the KV pool fills grew from
**+7.3%** (stock) to **+30.2%** (optimized) over 12 cycles — 3710 -> 4829 ms. As the fixed
overheads come off, the KV-dependent term dominates, and it keeps growing until the pool
saturates around cycle 36. `model.py:452-453` re-gathers the entire valid KV pool into a fresh
tensor per layer per forward (~240 MB/layer at saturation). That is now the prime suspect and
the next pass.

## 7. ring_kv_addressing — 1.43x, bit-exact (2026-08-01)

First optimization done under the profile -> implement -> benchmark -> generalize cycle, and the
first one aimed by a profile rather than by reading code.

**Profile said:** gather/copy is 39.6% of GPU time and `aten::nonzero` fires 30 x 77 x 3 times per
`_infer`, each a data-dependent shape and therefore a host round trip. Two lines cause both
(`model.py:451-453`): `valid = mask.nonzero(...)` then `key_pool[:, valid]`.

**Implemented:** the boolean mask was only ever encoding an interval. `allocate_slots` already
takes the lowest free slots and evicts the oldest, so allocation is sequential-with-wraparound and
the live set is always a ring interval. Track it as two host ints and `valid` becomes a *slice* —
a view, not a copy. Fast path for contiguous intervals; falls back to stock when the interval
wraps, so keys stay in ascending slot order and the pass stays bit-exact.

**Benchmarked** (14 cycles, idle H100, both arms seeded, everything else held equal):

| arm | cycle | rate | max abs delta |
|---|---|---|---|
| reference | 4277.3 ms | 7.5 Hz | — |
| + ring_kv | **2982.3 ms** | **10.7 Hz** | **0.000e+00** |

**1.43x, bit-exact.** Mechanism confirmed by re-profiling rather than assumed:

| kernel, one `_infer` | before | after |
|---|---|---|
| `aten::nonzero` | 7,034 (65.3 ms) | 104 (0.0 ms) |
| `aten::index` | 6,982 (168.1 ms) | 52 (0.1 ms) |
| `aten::_index_put_impl_` | 13,852 (45.9 ms) | 52 (0.2 ms) |
| total launches | 361,565 | 250,520 |
| GPU time | 2402 ms | 1812 ms |

`aten::copy_` is unchanged at ~48k. Those are the unfused transformer-block temporaries, which is
the next pass, and the profile said so before this one was written.

**Cumulative: 8881 -> 2982 ms, 2.98x, 3.6 -> 10.7 Hz, every step bit-exact.**

Two honest caveats. The pool holds 9792 slots and grows 272 tokens/cycle, so it does not wrap
until roughly cycle 36; a 14-cycle benchmark measures the fast path throughout and therefore
*understates* the steady-state win, since the removed gather grows with occupancy. And the fast
path does not maintain the `mask`/`id`/`is_pred` arrays, so the wrapped fallback currently reads
stale bookkeeping — correct within a 36-cycle episode, wrong beyond it. Fixing that is the first
follow-up, and it is a correctness bug, not a tuning issue.

**What the correctness gate caught.** The first implementation set `pred = key_size` on a
provisional commit. But `update_cache=1` fires *twice* per cycle (video last step at
`wan_va_server.py:504`, action last step at `:544`) and stock `clear_pred_cache` drops every slot
with `is_pred` set, i.e. both blocks. Overwriting instead of accumulating leaked the video block
into the permanent cache. The gate reported `max|delta| = 1.22` against a chunk-to-chunk movement
of 1.03 — larger than the signal, so unmistakably semantic rather than numeric. Two lines to fix.
Without a bit-exactness gate this would have shipped as "1.43x with a small accuracy change".

### 7b. Wraparound correctness, and the model that was wrong

The first version fell back to stock code when the ring interval wrapped, but the fast path did not
maintain `mask`/`id`/`is_pred`, so the fallback would have read stale bookkeeping. Correct within a
36-cycle episode, wrong beyond it -- and RoboTwin episodes run 400-1700 steps, i.e. 12-53 cycles.

Investigating it falsified the model. `tests/test_ring_allocator.py` exercises the REAL
`WanAttention` allocator (constructed via `__new__`, no weights, so it crosses many wraps in
seconds) and shows allocations are always a contiguous run, but the live set goes NON-CONTIGUOUS
161 times in 600 allocations: after `clear_pred_cache` it is `live=9520, lo=0, hi=9791` -- the pool
minus a hole in the middle. Stock presents that in ASCENDING SLOT INDEX order, which a
chronological ring does not reproduce, and reordering keys changes the floating-point reduction.

Rewritten so that the live set is read as slice-or-cat in ascending order, and `mask`/`id`/`is_pred`
are maintained stock-exact BY SLICE -- predicted from the tracked interval rather than found with
`nonzero`. There is no path left on which a stale mask can be observed.

| gate | result |
|---|---|
| allocator parity vs stock, 200 cycles (~5.6 full wraps) | **800/800 checks, 0 mismatches**, identical indices in identical order |
| bit-exactness, 40 cycles (past the wrap at ~36) | `max abs delta = 0.000e+00` |
| long-horizon `put_bottles_dustbin`, 1700 steps ~= 53 cycles/episode, 3 pinned seeds | **3/3 bitwise-identical action streams**, outcomes 2/3 on both arms, McNemar p = 1.0 |

Performance improved on the corrected version:

| arm | cycle | rate | ramp over 14 cycles |
|---|---|---|---|
| reference | 3690.5 ms | 8.7 Hz | +7.9% |
| + ring_kv | **2556.2 ms** | **12.5 Hz** | **-0.3%** |

**1.44x.** The ramp collapsing from +7.9% to -0.3% is the more important number: the gather scaled
with pool occupancy, so latency used to grow through an episode and now does not. That is what
makes long-horizon tasks predictable, and it is independent evidence the mechanism is the intended
one.

**Cumulative: 8881 -> 2556 ms, 3.47x, 3.6 -> 12.5 Hz, every step bit-exact.**


---

# 8. RESTATED under the fixed benchmark protocol (2026-08-01)

Every latency number in sections 3, 4, 6 and 7 was measured with **one probe run per arm**. That
is not a valid protocol on this box, and the numbers above are superseded by this section.

## What was wrong

The first probe run after a server starts is up to **37% slower** than steady state -- cuBLAS/cuDNN
algorithm selection and allocator warm-up. `probe_latency` discarded cycle 0 but not the first
*run*, so a single run silently mixes warm-up into the mean. The same configuration measured
2556 ms and 3503 ms on consecutive days with identical flags.

Two things it was NOT, both checked before concluding:

| hypothesis | test | result |
|---|---|---|
| the harnesses disagree | in-process vs websocket, same config | 3517 vs 3503 ms, **0.4%** |
| the GPUs differ | A/A: identical config on GPU 0 and GPU 1 | 3532.6 vs 3510.6 ms, **0.6%**, same clocks |

A separate bug compounded it in the in-process profiler: `drive()` restarted its cycle counter on
every call, so the "measured" cycle ran the cycle-0 workload (4 keyframes, cold VAE encode) and
reported 4893 ms for a 3500 ms cycle.

## The protocol, now the only accepted one

`probe_latency.py --repeats N` (default 3). The first run is **discarded**; the rest are reported
with their spread, and a spread above 5% prints a refusal to quote the number. Any latency claim
must come from this path.

## Restated numbers

Four cumulative configs, one per GPU, probed sequentially, 10 cycles x 3 runs each.

| config | cycle | spread | vs stock | control rate | step |
|---|---|---|---|---|---|
| stock | 8431.5 ms | 0.0% | 1.00x | 3.8 Hz | |
| + substrate (fsdp, empty_cache, debug dumps) | 3994.0 ms | 0.4% | **2.11x** | 8.0 Hz | 2.11x |
| + conditioning prefill | 3567.5 ms | 0.1% | 2.36x | 9.0 Hz | 1.12x |
| + ring KV addressing | **2553.9 ms** | 0.7% | **3.30x** | **12.5 Hz** | 1.40x |

## What changed versus what was published

| claim | published | restated |
|---|---|---|
| stock cycle | 8881 ms | 8431.5 ms |
| substrate passes | 1.92x | **2.11x** |
| conditioning prefill | 1.05x | **1.12x** |
| ring KV | 1.44x | **1.40x** |
| **cumulative** | **3.47x** | **3.30x** |

Two went up and one went down, which is the signature of noise rather than bias -- the old
single-run numbers were not systematically flattering, they were just unreliable. The headline is
lower: **3.30x, not 3.47x.**

Unaffected: every correctness result. Bit-exactness, the 800/800 allocator parity checks, and the
3/3 identical action streams on `put_bottles_dustbin` are equality tests, immune to timing noise.

---

# 9. copy_ breakdown — measured, and the reason not to pursue it (2026-08-02)

Profiled before choosing cache reuse as the next optimization. **Recommendation: do not pursue.**

```
aten::copy_, one control cycle : 74,547 calls, 183.4 ms GPU, mean 2.46 us/call
share of the 2832 ms episode-mode cycle : 6.5%
```

| shape | n | ms | ideal (HBM) | overhead | inferred source |
|---|---|---|---|---|---|
| (2, 32, 3072) | 17264 | 41.6 | 4.05 | **90%** | action hidden state |
| (2, 240, 3072) | 8632 | 39.5 | 15.20 | 62% | video hidden state |
| (2, 240, 24, 128) | 4680 | 28.1 | 8.24 | 71% | video q/k/v, head-split |
| (2, 32, 24, 128) | 9360 | 27.1 | 2.20 | **92%** | action q/k/v, head-split |
| (1, 8, 10) | 11712 | 13.7 | 0.00 | 100% | unexplained, tiny |
| 5 more tiny families | 19936 | 26.9 | ~0.1 | ~100% | unexplained, tiny |

**80% of copy time (147.1 ms) is per-kernel overhead, not bandwidth.** Only 36.3 ms is what moving
the bytes costs. The lever is COUNT, not bytes — the opposite of how this was framed when it went
on the ranking as "46,752 copies, 79 ms GPU".

**Ceiling.** Every copy vanishing is 183.4 ms of 2832 ms = **1.069x**, unreachable by construction.
The realistic target is the four large-tensor families: 136.3 ms, **4.8% of the cycle**, and
capturing even half of it means fusing across rounding points that the block trace lists as
semantic. The tiny tail is 47.1 ms (1.7%) spread over ~29k calls of 80–1280 elements.

**Limits of this attribution, stated so it is not over-read.** `TorchDispatchMode` observed only
4,817 of the 74,547 copies — the remainder are issued below the Python dispatch key — so the
source column is inferred from shapes, not proven from call sites. The tiny families are genuinely
unexplained: 11,712 copies of `[1, 8, 10]` per cycle is ~148 per forward and no block op has that
shape. Not chased further, because 0.5% of a cycle cannot justify it.

A first pass apportioned GPU time by bytes and was wrong: 5.92 GB over 183.3 ms is 32 GB/s, ~100x
under HBM, which is precisely the signal that these copies are overhead-bound rather than
bandwidth-bound.

---

# 10. L5 operator fusion — measured at three scales, and rejected at the one that counts (2026-08-05)

The kernel layer (`instinctwm/kernels/`) had no caller until now: regions were declared, kernels
registered themselves, and nothing outside `tests/` imported either. `OperatorFusion` connects it
to a plan, and `--fuse-residual` puts it on the serving path. This section is what that bought.

**Recommendation: keep it opt-in. Do not add `--fuse-residual` to the shipped chain.**

## What was fused

`(hidden.float() + x * gate).type_as(hidden)`, the block's two gated residuals — the
self-attention one (`model.py:543-544`) and the feed-forward one (`:563-564`). Pure elementwise,
no reduction, so it is the one declared region where BITEXACT is reachable. 3 eager kernels -> 1.

## Three measurements, two of which are encouraging and irrelevant

| scale | what was measured | result |
|---|---|---|
| region, python launch | the expression alone, in a python loop | 0.73x–1.79x, size-dependent |
| region, cuda-graph replay | the same, device time only | **3.0x–7.6x** |
| one real block, graph replay | `WanTransformerBlock`, interleaved A/B | **1.033x**, `torch.equal` |
| **full control cycle** | **shipped chain + `--graph-blocks`, ABBA** | **0.994x** |

Region and block level say ship it. The cycle says otherwise, and the cycle is the number.

## The end-to-end result

8 measurements per arm, alternating ABBA, a **fresh server for every measurement** on the same
GPU, chains differing by exactly one entry.

| arm | mean | median | stdev | min | max |
|---|---|---|---|---|---|
| shipped chain | 3198.8 ms | 3198.9 ms | 17.0 | 3169.7 | 3216.4 |
| + `--fuse-residual` | 3217.0 ms | 3215.2 ms | 16.4 | 3197.5 | 3241.9 |

**0.994x — a 0.57% regression, t ≈ 2.2, borderline.** Paired per-measurement deltas are +14.0,
+40.8, +26.2, −8.3 ms: three of four against the fusion, one for it. The honest reading is *no
gain, plausibly a small loss* — not a clean regression, and nowhere near the ~3% the block-level
measurement predicts.

The fusion was verifiably running. `RESIDUAL.report()` on every reset:
`fused=34560 below_threshold=0 bypassed=0` — every residual site took the kernel, zero fallbacks.
This is not a treatment arm that failed to fire.

## Two confounds that produced wrong answers first, recorded so they are not repeated

1. **Order bias.** The first design ran base then fusion every round. The box drifts *upward*
   across a session — 3214 → 3730 → 3964 ms over three rounds — so "second" is systematically
   slower, and the treatment arm was always second. ABBA fixes it.
2. **Allocator carry-over.** `--no-empty-cache` means the caching allocator never returns
   anything and each server sits at **81 GB on an 80 GB card**. Reusing a server across
   measurements carries that state forward; the first attempt showed 46–52% spread between kept
   runs and `probe_latency` correctly refused to quote it. A fresh server per measurement brings
   within-measurement spread to 0.2–0.5%.

A third, at block level: timing stock, then hooked, then armed *in that order* reported 1.24x for
a configuration where the kernel had not been armed at all (`fused=0`). The whole difference was
the A100 boosting its clocks during the ~2,000 warm-up launches in between. Interleave, or do not
measure.

## Why the block-level win does not survive

Not established. What is established is that it does not, and the gap is large enough that the
region-level break-even sweep the installer runs is **not predictive of the cycle**. That is a gap
in this integration, not a property of this kernel: `optimizer/contract.py` says the performance
gate is `harness.cycle_ms_before/after`, and the installer gates on a region microbenchmark
instead. The region sweep is the right instrument for *which shapes*, and the wrong one for
*whether at all*.

The block-level probe also ran with a nearly-empty KV pool, where attention is cheap and the
residual is a larger share of the block. The server runs a pool that grows 272 tokens per cycle.
That direction is consistent with the gap but does not account for its size, so it is a
hypothesis, not the finding.

## What stays

The pass, the kernel, the hook and the flag all stay. `OperatorFusion` remains in
`default_passes()`, where it is self-gating: without `--graph-blocks` the break-even lands at
1,966,080 elements, above both stream shapes, so `plan.serve()` arms nothing. The measurement is
here so that "fuse the elementwise residuals" is not proposed a third time without a cycle-level
gate attached to it.

---

# 11. L4 attention — the KV extent is worth five times the attention (2026-08-06)

Layer 4 had no code before this. Two candidates were implemented and measured end to end: picking
the SDPA backend by measurement, and moving the KV extent out of the CUDA graph capture key. The
result that matters is not about attention speed.

**Recommendation: `--ring-attention` is worth a certificate and does not yet have one.
`--attention-backend` is rejected. Do not add either to the shipped chain today.**

## READ THIS FIRST: the substrate changed

Every number in sections 1-10 was measured on 8x H100 with `/home/ubuntu/.venv-lingbot`. **This
section was measured on 8x A100-SXM4-80GB**, and that venv no longer exists, so `.venv-server`
(torch 2.9.0+cu128) was used instead. Nothing here is comparable to an earlier section, and
nothing here reproduces one. These are new baselines on a different box.

The A100 baseline for the shipped default stack, `probe_latency --repeats 4`:

| run | cycle | state |
|---|---|---|
| 1 | 3165.7 ms | capturing |
| 2 | 2164.4 ms | warm |
| 3 | 2169.3 ms | warm |

The tool refuses to quote a steady-state mean (spread 46%), and it is right to: the two states are
1000 ms apart and the protocol mixes them. Warm is ~2167 ms/cycle = 67.7 ms/control step = 14.8 Hz.

## Attention is 3.6% of GPU busy, and the README's "7%" had no source

`profile_cycle.py`, one cycle, P001+P002 configuration:

| category | ms | % of GPU | launches |
|---|---|---|---|
| GEMM | 2250.0 | 39.8% | 46,605 |
| gather/copy | 1926.3 | 34.1% | 195,232 |
| elementwise/norm | 904.0 | 16.0% | 139,963 |
| other | 338.7 | 6.0% | 51,981 |
| **attention** | **201.5** | **3.6%** | 4,742 |
| memcpy | 36.4 | 0.6% | 19,579 |

4,742 launches is 30 layers x 79 forwards x 2, which confirms the attribution. Two limits: this is
the P001+P002 rung, so `gather/copy` still contains the mask-and-gather P003 removes; and it
profiles cycle 5, where the ring holds ~760 slots. Attention cost is linear in the ring extent
(microbenchmarked on this box: 110 ms/cycle at count=1000 rising to 684 ms at a full 9792 pool), so
any single-number share is a statement about one point in an episode. **The direction of the
finding survives that caveat: attention arithmetic is not where the time is.**

## Plan A: choose the SDPA backend by measurement — REJECTED

`F.scaled_dot_product_attention` dispatches by heuristic, not by measuring the shapes in front of
it. On the served shapes it picks flash, and cuDNN is faster. `AttentionBackend` measures every
backend on the site's own shapes and installs the winner. Measured by the pass itself, on the real
server, at half a pool:

```
shape (2, 240, 24, 128, 9792, bfloat16): incumbent 0.231 ms -> cudnn 0.193 ms (1.20x) [NUMERIC]
    flash          0.231 ms   max|d| vs incumbent 0.000e+00   <- what the dispatcher picks
    cudnn          0.193 ms   max|d| vs incumbent 4.883e-04
    mem_efficient  0.347 ms
    math           3.822 ms
```

End to end, sequential A/B on one GPU, warm runs only:

| | baseline | + `--attention-backend` |
|---|---|---|
| warm run 2 | 2171.8 ms | 2166.7 ms |
| warm run 3 | 2184.8 ms | 2159.1 ms |

**-15.4 ms/cycle, 0.7%** — and the within-arm spread is 13 ms, so this sits at the edge of the
protocol's resolution. Direction is consistent (the variant is faster in both warm runs and in the
capturing run) but the magnitude should not be quoted to three digits.

The cost side is not marginal:

```
probe_bitexact, 6 paired seeded cycles
  max|delta action|                   1.367
  reference chunk-to-chunk movement   1.055
  ratio                               129.6%     VERDICT: NOT bit-exact
```

A 4.883e-04 per-call difference compounds through the KV cache, because the actions it helps
produce are written back into the pool the next cycle reads. Six cycles in, the divergence exceeds
the signal. **0.7% does not buy a paired non-inferiority run**, which costs ~10x the GPU time of
measuring the speedup. The pass and its gates stay in the tree so this is not proposed again.

An earlier draft of this section claimed 1.49x for the backend swap. That came from an
under-sampled probe and does not reproduce; 1.20-1.21x is the figure that does.

## Plan B: the KV extent leaves the capture key

`graph_block_stack` keys every graph on `(start, count)` because the live KV set reaches attention
as a **slice**, and a slice puts its length in a tensor shape. Shapes are frozen at capture. The
ring advances 272 slots/cycle and `start` stays 0 for a whole episode, so `count` grows every cycle
and the key never converges. Server logs show it directly:

```
capture #1   key=(..., (0, 0))
capture #3   key=(..., (0, 240))
capture #119 key=(..., (0, 2448))
```

`kernels/ring_attention.py` is a Triton kernel taking `(q, k_pool, v_pool, extent)` where the pool
keeps its full capacity shape and `extent` is a device-resident int32[2]. Three things leak `count`
into a graph and all three had to be closed together: the read extent (a shape -> the kernel), the
write offset (a Python slice -> `index_copy_` at a device-computed index), and the key itself
(`_iwm_ring_signature` -> `None`).

The correctness argument is the masked-tile identity: a tile whose every position is masked
contributes `m <- max(m, -inf) = m`, `alpha = exp(0) = 1.0`, `l += 0`, `acc += 0` — all exact float
identities. `tests/test_ring_attention.py` gates it at zero across counts, pool capacities, and
garbage past the extent, and replays **one captured graph at two different extents** bit-identically.

`probe_episode.py`, 45 cycles, one reset, sequential arms on one GPU:

| | baseline | + `--ring-attention` |
|---|---|---|
| whole episode | 3463.0 ms | **2633.0 ms (1.32x)** |
| late episode (36+) | 3801.1 ms | **2822.4 ms (1.35x)** |
| captures / cycle | 6.0 | **0.1** |
| unique graph keys | 270 | **6** |
| evictions | 204 | **0** |
| cache hit rate | 92.457% | **99.831%** |
| late-episode spread | 0.6% | **0.2%** |

Six keys is exactly 2 token shapes x 3 `update_cache` values — the whole key space once the ring is
gone. The late-episode -978.7 ms lands within 5% of the -1032 ms predicted from 6 captures x 172 ms,
which is the evidence that the mechanism is the intended one and not a coincidence.

**A single capture costs 172.0 ms** on this box (n=120, range 162.8-220.4). The earlier estimate in
`graph_capture.py` was ~275 ms on H100.

Tier: NUMERIC. `probe_bitexact` gives `max|delta action| = 1.184` against a 1.055 reference
movement, 112%. Same compounding mechanism as Plan A — but here it buys 830 ms/cycle rather than
15, which is why this one is worth the certificate and that one is not.

## A correctness bug in P005, found by an OOM (fixed, v1.0.1)

`install` sets `_iwm_defer_commit = True` permanently, so `WanAttention.forward` stops committing
inline and the only thing that advances the ring is `_commit_all`. Both of `stack_graphed`'s
fallback returns skipped it. **From the first capture failure onwards the ring froze**: `count`
stopped growing, every later forward rewrote the same slots, and attention read a stale window. No
exception, no log line; the only symptom is a task success rate that drifts.

Capture failure is advertised as a safe degradation, and this was the one path where "safe" meant
"silently incorrect". Found because a 50-task certification run OOMed at `held=64 evicted=400` and
degraded to eager on all 8 servers. Fixed by routing both fallbacks through a `_eager()` helper
that runs the stack and then commits. The timings in sections 8-10 are unaffected — they were taken
on runs where capture never failed, and the fix adds nothing to the replay path.

## Graph eviction does not return its memory, and lowering the cap makes it worse

Evicting a captured graph does not give its private memory pool back. Over a 50-task run the
teacher servers climbed from 24 GB to the 80 GB ceiling and every one of the 8 OOMed.
`IWM_MAX_GRAPHS` was added to bound it and **the experiment falsified the idea**: at a cap of 32,

```
gpu0: captures=523 replays=20881 held=32 evicted=461  fallbacks=1
```

461 evictions for 523 captures, and all 8 servers OOMed anyway. Capping the *held* set lower does
not help when the leak is in *eviction* — it only increases the eviction count. The knob stays,
defaulting to 64, but it is not the fix.

The consequences went past latency. One A100 entered `GPU requires reset` and was lost for the rest
of the session. The `--ring-attention` arm, which holds 6 graphs and evicts none, ran the same
workload at a flat 41-42 GB with zero fallbacks. **The cost of a non-converging capture key is not
830 ms/cycle; it is that the shipped default cannot finish a 50-task evaluation on this box.**

## The certification is NOT complete, and the first teacher arm was void

Margin `-0.02` was declared before the run and 50 tasks x 10 episodes was chosen because the README
records the 2/2-step student clearing its margin on a 10-task subset and failing at 50.

**The student arm is complete and valid**: 500/500 episodes, 50/50 tasks, 0.904 success. All 8
servers reported `captures=12 held=6 evicted=0 fallbacks=0` over 342,291 replays, at a flat
41-42 GB. It took 2h05m.

**The first teacher arm is void and must not be used.** It emitted 355 episodes over 49 tasks, with
**23 tasks truncated** — `place_a2b_right` 1/10, `shake_bottle` 1/10, `stamp_seal` 2/10. Every one
of those clients exited `rc=0`; the truncation is silent.

The reason it is void is not the missing count, it is the **direction** of the missing. Episodes
that completed are the ones that ran fastest, and episode duration correlates with outcome (a
success often terminates early, a failure runs to the step limit). So the teacher lost
disproportionately many failures, which is exactly what its impossible-looking 0.955 against the
student's 0.904 reflects. Certifying against it would have systematically penalised the student
while every automated check passed. The data is retained at `l4ring_teacher_VOID_truncated/`.

**No certificate exists yet.** The teacher arm was restarted without `--graph-blocks` — legitimate
because graph capture is gated at `max|delta action| = 0`, so the teacher's *actions* are identical
with or without it, and dropping it removes the entire memory-exhaustion failure chain. That re-run
was stopped before completion.

## What is in the tree

| | |
|---|---|
| `passes/attention_backend.py` | Plan A, on the generic site interface. Rejected by measurement; kept so it is not re-proposed |
| `passes/interface.py` | `SiteKind.ATTENTION_OP` — the first site that carries the pool and the extent separately |
| `kernels/ring_attention.py` | Plan B's Triton kernel, 8 gate groups |
| `runtime/ring_attention_install.py` | Plan B on the serving path, layered on frozen P003 |
| `--attention-backend`, `--ring-attention` | opt-in flags, neither in the shipped chain |

## What is owed

1. A completed teacher arm and a certificate. Nothing about `--ring-attention` may be called
   accuracy-neutral until then.
2. A PTX assertion for the attention kernel. `matches_reference_contraction=False` is currently a
   declaration, not an assertion, which is the gap `test_triton_residual.py:test_ptx` exists to close
   for the residual kernel.
3. A fix for the eviction leak, or a reason it cannot be fixed. `--ring-attention` sidesteps it by
   never evicting; the default path still has it.
4. Split-K for the video phase. The kernel wins 1.4-1.8x on the action stream and loses 6-13% on
   the video stream, because 32 query rows over 24 heads at batch 2 is only 48 programs at
   BLOCK_M=64, under half this box's 108 SMs.
