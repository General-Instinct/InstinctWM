# LingBot-VA × RoboTwin 2.0 — evaluation pipeline

The accuracy reference for InstinctWM. Every future optimization (step distillation, CUDA
graph capture, kernel work, KV-cache changes) is judged against numbers produced here, so
this directory optimizes for *being correct and auditable*, not for being fast to run.

## What it is

[LingBot-VA](https://github.com/robbyant/lingbot-va) is an autoregressive video-action
world model: a dual-stream Mixture-of-Transformers on a Wan2.2 backbone that predicts
future video latents and action chunks in one interleaved sequence. It reports **92.9%
easy / 91.6% hard** on RoboTwin 2.0's 50 tasks, and ships a posttrained RoboTwin
checkpoint — so unlike the earlier Cosmos3-Edge work, **no SFT is needed and there is no
embodiment mismatch**: the checkpoint's 16 used action channels are exactly RoboTwin's
`take_action(a, 'ee')` layout.

Two processes, two Python environments, one websocket:

```
  RoboTwin/.venv  (sapien 3.0.0b1, torch 2.4)          .venv-lingbot  (torch 2.9, diffusers 0.36)
  ┌──────────────────────────────────┐   ws://:2905N   ┌──────────────────────────────────┐
  │ eval_polict_client_openpi.py     │ ──────────────▶ │ wan_va_server.py                 │
  │  simulator, expert-gate, scoring │ ◀────────────── │  T5 + Wan VAE + 5B MoT           │
  └──────────────────────────────────┘                 └──────────────────────────────────┘
```

The two environments are dependency-incompatible on purpose; the websocket is the seam.
That seam is also InstinctWM's abstraction boundary — swapping the model under test should
be a change of what listens on the port, nothing else.

## Wire protocol (server reads exactly 5 keys)

| message | keys | server does |
|---|---|---|
| reset | `reset=True`, `prompt` | clears transformer KV cache + VAE cache, runs T5 on the instruction once |
| kv | `compute_kv_cache=True`, `obs=[frames]`, `state=action` | VAE-encodes real frames, folds observed history into the KV cache |
| infer | `obs` | 25+1 video denoise steps, then 50+1 action denoise steps → 32 control actions |

The server is **stateful per episode**. One client per server, always.

## Running it

```bash
cd eval/lingbot_va_robotwin
source ./env.sh

IWM_FA_SHIM=1 ./servers.sh start 8      # one server per GPU; refuses to start on a busy port
./servers.sh status

# gate: serving must reproduce the training text conditioning (see below)
$IWM_SERVER_PY check_prompt_parity.py \
  --latent  /home/ubuntu/iwm_parity/.../episode_000000_0_139.pth \
  --empty-emb /home/ubuntu/iwm_parity/empty_emb.pt

./run_eval.sh myrun 100 adjust_bottle place_dual_shoes ...   # fans tasks over the 8 GPUs
$IWM_CLIENT_PY aggregate.py $IWM_RESULT_DIR/myrun --expect-episodes 100 --expect-tasks 50
```

## The traps

Each of these is a real defect found by reading both sides of the wire. They are listed
because *most of them fail silently* — the run completes and prints a plausible number.

**Prompt parity was never checked upstream, and it is the whole ballgame.** LingBot-VA
never runs T5 during training: `lerobot_latent_dataset.py:246` reads a *precomputed*
`text_emb` from each dataset `.pth`, and substitutes a precomputed `empty_emb.pt` for the
CFG drop. The server does the opposite — `wan_va_server.py:421-435` runs T5 live with
`prompt_clean()` and pads to 512. Nothing asserts these agree. This is the exact shape of
a failure that voided a 22.7-hour run on this box. `check_prompt_parity.py` closes it, and
on our checkpoint it passes **bit-exactly** (`max|Δ| = 0.000e+00`, cosine 1.000000 on both
the positive prompt and the CFG-negative branch). Re-run it whenever the checkpoint moves.
Note `encode_prompt`'s own default is `max_sequence_length=226` and only the `_reset` call
site passes 512 — one edit away from a silently different embedding.

**Port collision between the rendezvous and the websocket.** Upstream uses
`START_PORT=29056` and `MASTER_PORT=29061`; at 8 GPUs, GPU 0's torchrun rendezvous port is
GPU 5's websocket port. Seven servers come up, one dies, and the fleet looks healthy. It is
worse than a crash because `WebsocketClientPolicy._wait_for_server` retries on *any*
exception every 5s forever — a client aimed at the dead port hangs indefinitely, which is
indistinguishable from a slow task. `env.sh` keeps the ranges far apart (ws 29056+i, rdzv
29800+i) and `servers.sh` refuses to launch on a busy port and insists on 8/8.

**A crashed task raises the headline.** `calc_stat.py` scores a task with no mp4s as
`None` and then averages only non-`None` rates, dropping it from the denominator instead of
counting it. `aggregate.py` treats a zero-episode task as a hard error and cross-checks the
mp4 evidence against the independently written `res.json`.

**Results are labelled with the wrong policy.** Upstream passes `policy_name=ACT` purely to
locate `policy/ACT/deploy_policy.yml` (a config skeleton — nothing imports an ACT model),
with the side effect that output lands in `eval_result/<task>/ACT/` and is indistinguishable
from a genuine ACT baseline. We keep the skeleton but label the run `LingBotVA`.

**Duplicate tasks overwrite each other.** Success is counted from mp4 filenames
`<test_num>_<prompt>_<True|False>.mp4` and `test_num` restarts at 0 per client process.
Upstream's `launch_client_multigpus.sh` group 6 runs `place_empty_cup` and
`blocks_ranking_rgb` four times each into one `save_root`. `run_eval.sh` refuses duplicates.

**Dead client flags.** Everything after `--overrides` is swallowed by `argparse.REMAINDER`,
so the top-level `--port`/`--test_num` defaults never apply. Worse,
`--video_guidance_scale` / `--action_guidance_scale` *are* parsed by the client and sent on
the wire, but the server never reads them (`wan_va_server.py:379,513,552` use only
`job_config`). Guidance is a server-side config value; the client flag is decoration.

**`infer_mode` is an import-order accident.** `va_robotwin_cfg.py` never sets it. It is
`'server'` only because `va_franka_cfg.py:9` mutates the shared config object after copying
it, before `configs/__init__.py` imports the robotwin config. Reordering that file would
route `--config-name robotwin` into offline video generation and produce no eval at all.

**flash-attn is a hard import that is never called.** `model.py:29-32` imports
`flash_attn_func` at module scope, but both the checkpoint config and the server select
`attn_mode='torch'` (`custom_sdpa`). `IWM_FA_SHIM=1` supplies an import-only stub that
**raises if called**, turning "flash-attn is unused" from an assumption into an enforced
invariant. Drop it once a real wheel is installed — `PYTHONPATH` precedes site-packages and
would otherwise shadow the real package.

## Protocol deviations to state whenever quoting a number

- **Seeds.** The LingBot-VA client uses `st_seed = 10000*(1+seed)`
  (`eval_polict_client_openpi.py:395`); RoboTwin's canonical harness uses
  `100000*(1+seed)`. Both are disjoint from the training seeds (low hundreds), so neither is
  contaminated — but they are *different scene sets*. Numbers from here are comparable to
  the LingBot-VA paper, not to a canonical-seed RoboTwin baseline.
- **Instructions.** `instruction_type` is hardcoded `'seen'` (line 308), ignoring the config.
  A seen-instruction eval is easier than unseen.
- **Latency is not scored.** `take_action` blocks until the policy returns, so a 77-forward
  model ties a 4-forward model. Any latency claim needs its own protocol.
- **Long-horizon tasks.** The KV pool saturates at roughly 36 cycles (~1152 control steps).
  Tasks with `step_lim ≥ 1200` (`blocks_ranking_rgb` 1200, `open_microwave` 1500,
  `put_bottles_dustbin` 1700) can evict real history mid-episode. Report them separately
  until cache occupancy is instrumented.

## Files

| file | role |
|---|---|
| `env.sh` | single source of truth: repos, checkpoint, interpreters, port plan |
| `servers.sh` | fleet lifecycle with preflight port checks and 8/8 verification |
| `check_prompt_parity.py` | **the gate** — live T5 vs baked training embedding |
| `run_eval.sh` | fan tasks over GPUs, one serial worker per GPU, records provenance |
| `aggregate.py` | honest scoring; prints `REPORTABLE: NO` rather than a partial number |
| `serve_omni_arm.py` | external reference arm: stock server + vLLM-Omni's `PromptEmbedCache` |
| `run_omni_arm.sh` | builds `.venv-omni` and measures both of its arms — never one alone |
| `omni-arm-requirements.{in,txt}` | that arm's own lock; deliberately NOT merged with `server-requirements` |
| `probe_encode_prompt.py` | ceiling for any cache keyed at `encode_prompt`, implementation-independent |

## Comparing against an external stack

Every speedup here is measured against the *unoptimized upstream server*. That is a weak
baseline — it still pays world_size=1 FSDP, a per-chunk `empty_cache`, and a blocking D2H debug
dump. `serve_omni_arm.py` answers the obvious follow-up ("how does this compare to a competent
serving stack?") by holding model, checkpoint and message order fixed and swapping exactly one
component for the shipped third-party equivalent.

```bash
./run_omni_arm.sh build     # .venv-omni from its own lock, ~8.3 GB, once
./run_omni_arm.sh reset     # where this cache actually acts: MISS ~98 ms vs HIT ~4 ms
./run_omni_arm.sh cycle     # control-cycle mean: no difference, and that is the result
```

Three rules that make the comparison mean anything, all enforced by the script:

- **Both arms are measured in `.venv-omni`.** Its stack differs from `$IWM_SERVER_PY`, so
  subtracting an absolute ms here from an absolute ms there measures the torch version, not the
  optimization. Only the within-env delta is quotable.
- **`bypassed` must be 0.** `PromptEmbedCache` silently bypasses on any unhashable argument — a
  run with a climbing `bypassed` measured the *uncached* path while looking like it worked.
- **The cache acts on `reset`, not on the cycle.** `probe_latency`'s full-cycle mean excludes the
  reset, so `cycle` showing nothing is expected, not a bug.

Measured 2026-08-04 on 8x A100-80GB: the cache works correctly and saves ~95 ms per hit, once per
episode; `conditioning_prefill` saves 288 ms per *cycle*. At 10 cycles that is ~30x, and in a
real eval the instruction is re-sampled per episode so the cache misses and saves nothing. The
two are not competitors — a generic framework caches at the *request* boundary, and closed-loop
control has no request boundary, only a 77-forward loop with the constant inside it.
