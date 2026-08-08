# Cosmos3-Edge — first end-to-end run under the InstinctWM engine

Second reference model. Everything below was produced by
[`probe_mot_stack.py`](probe_mot_stack.py) on this box, against a fresh upstream checkout.

```bash
/home/ubuntu/cosmos-framework/.venv/bin/python eval/cosmos3_edge/probe_mot_stack.py
```

## Substrate

| | |
|:--|:--|
| upstream | [`NVIDIA/cosmos-framework`](https://github.com/NVIDIA/cosmos-framework) @ `12a9a81` (2026-08-07) |
| environment | upstream's own lock, `uv sync --group=cu130-torch213` → `/home/ubuntu/cosmos-framework/.venv` |
| torch | 2.13.0+cu130, cuDNN 92000, Python 3.13.14 |
| GPU | 1× A100-SXM4-80GB |
| attention | **cuDNN, both paths — the real dispatched kernel, no shim** |
| geometry | 28 layers, hidden 2048, 16q/8kv heads, head_dim 128, intermediate 9216 (`nvidia/Cosmos3-Edge` `config.json` → `text_config`) |
| pack | `sample_lens=[567]`, `split_lens=[111, 456]`, `attn_modes=["causal","full"]`, NFE 16 |
| params | 3.876 B (7.22 GiB bf16) |
| protocol | 20 iters × 3 repeats, median reported, spread shown |

**Not claimed: accuracy.** Random weights, no checkpoint. What is claimed is op structure, shapes,
dependency derivation, capturability, eager-vs-replay equality, and allocation traffic.

The environment is deliberately separate from `.venv-server`. The `cu130-torch213` group ships
**no flash-attn** — which is what keeps `/home/ubuntu/iwm_shims/flash_attn` (the RuntimeError shim
with no package metadata) doing its job, so `diffusers` cannot detect flash-attn, switch
`autoencoder_kl_wan` to a flash path, and invalidate the 91.6% RoboTwin baseline. Verified after
the run: `.venv-server` untouched, no flash-attn, no natten.

## 1. The engine generalized, unchanged

The adapter and the engine were written against a Cosmos tree from an earlier release. Neither was
modified. On today's HEAD:

```
5320 ops, 624 external reads, 0 unnamed
capturable: True (no host mutation detected in the traced execution)
```

`0 unnamed` closes the open finding from `tests/test_cosmos3_engine.py` — `build_name_map` no
longer needs adapter-supplied naming for this model.

## 2. One Plan, both executors, bit-exact

```
graph replay vs eager oracle : differing=0  max|d|=0.000e+00  BITEXACT   (captures=1)
second pack, new values      : differing=0  captures still 1 (rebound, not recaptured)
```

## 3. Latency — GRAPH is 2.33× on the control step

| | ms / forward | enqueue | spread | × NFE 16 (control step) | vs raw eager |
|:--|--:|--:|--:|--:|--:|
| raw eager (no engine) | 66.010 | 66.000 | 1.0% | 1056.2 ms | 1.000× |
| EagerExecutor | 65.897 | 65.886 | 0.6% | 1054.4 ms | 1.002× |
| **GraphExecutor (replay)** | **28.393** | 5.684 | 0.0% | **454.3 ms** | **2.325×** |

Across five independent invocations the graph figure was 2.300× / 2.325× / 2.327× / 2.342× /
2.361×. Quote the range, not the best one: the eager arm carries a 0.6–1.6% spread and the graph
arm effectively none (28.39–28.41 ms every time), so the ratio's variance is entirely the
denominator's.

`EagerExecutor` at 1.002× is the control that matters: the engine's own dispatch costs nothing, so
the win is capture, not bookkeeping. Eager is launch-bound (enqueue 66.000 ≈ wall 66.010); replay
drops enqueue to 5.684 ms and lands GPU-bound at 28.4 ms.

## 4. L3-P8 (ForwardScratchArena): **rejected by measurement**

The declaration is intact at this commit. `get_all_seq` (`runtime.py:505-525`) still allocates
`new_zeros` and scatters twice; `attention.py:205-206` still calls it twice in one expression with
K and V live simultaneously; `set_all_seq` (`:528`) is still never called, so the `all_seq` memo at
`:513` never fires and every call takes the allocating path.

Measured:

```
per forward      :  56 calls   62.02 MiB   (2 per layer)
per control step : 896 calls    0.97 GiB   (NFE 16)
```

The call count matches the manifest's 896 exactly. The **byte figure does not**: the manifest says
~2.08 GB, the truth is 0.97 GiB. The estimate assumed full hidden width; `get_all_seq` runs on K and
V, which are GQA-narrowed to 8 heads × 128 = 1024, not 2048. The manifest over-counts by 2.1×.

Ceiling probe — two preallocated per-role buffers, rotated, which is the minimum that cannot alias
and is exactly `ForwardScratchArena`'s structural safety argument:

```
bit-exact vs allocating path: differing=0  max|d|=0.000e+00
EAGER   66.010 -> 65.346 ms/fwd   1.010x
GRAPH   28.393 -> 28.391 ms/fwd   1.000x
=> P8 ceiling on the CYCLE: 1.010x eager, 1.000x on the shipped (graph) path
```

Removing **all** 896 allocations and the whole 0.97 GiB buys 1.0% on eager and nothing at all on the
path that would actually ship. CUDA graph capture already bakes the allocations into its private
pool, so P8 is not merely small here — it is **subsumed**. Struck, and kept so it is not
re-proposed.

## 5. L3-P3 (StaticPartitionHoist): **obsolete — upstream implemented it**

`tests/test_p3_cosmos3.py` no longer runs:

```
AttributeError: module '...sequence_packing.runtime' has no attribute 'init_sequence_pack'
```

Upstream refactored `init_sequence_pack` into `SequencePackMetadata` +
`prepare_sequence_pack_metadata` + a `prepared_metadata=` parameter on
`sequence_pack_from_packed_sequence` guarded by `matches_layout()`. That is P3's memoization, built
by NVIDIA, as opt-in reuse — and `cosmos3_vfm_network.py:1017` passes it on the served path, so the
win is already taken.

What survives of the original finding: the `.tolist()` device→host sync (`runtime.py:243`) and the
assert-only product it feeds (`:245`) are still there.

## Where this leaves the layer table

| layer | on Cosmos3-Edge |
|:--|:--|
| GRAPH | **2.33× bit-exact on the control step** (2.30–2.36× over five runs). The whole measured win. |
| CACHE | Nothing to reuse — no KV pool; the SequencePack is the state. |
| MODEL / ATTENTION / KERNEL / HARDWARE | Unbuilt for this model. |
| ~~L3-P3~~ | Obsolete, implemented upstream. |
| ~~L3-P8~~ | 1.000× on the shipped path. Subsumed by GRAPH. |

Both of the passes written specifically for Cosmos3-Edge are dead, and the one that generalized
from LingBot-VA is the one that paid. That is the opposite of what the manifest predicted.
