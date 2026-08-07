# Cosmos3-Edge evaluation

The generalization arm. LingBot-VA answers "does the stack make *this* model fast"; Cosmos3-Edge
answers "is the stack a framework or a LingBot-VA optimizer with ambitions".

Results and protocol: [RESULTS.md](RESULTS.md).

## Setup

Cosmos runs in **upstream's own environment**, never in `.venv-server`.

```bash
git clone https://github.com/NVIDIA/cosmos-framework /home/ubuntu/cosmos-framework
cd /home/ubuntu/cosmos-framework && uv sync --group=cu130-torch213
```

`/home/ubuntu/cosmos-framework` is not a preference — `tests/test_cosmos3_engine.py:17` and
`tests/test_p3_cosmos3.py:28` hardcode it on `sys.path`.

Use `cu130-torch213`, not `cu130-train`, for three reasons:

1. It pins cuDNN to `9.20.0.48`, which equals `CUDNN_MIN_BACKEND_VERSION` in
   `cosmos_framework/model/attention/cudnn/__init__.py`, so the **cuDNN attention backend stays
   enabled** instead of falling back. That is what removes the torch-SDPA shim.
2. It ships **no flash-attn** (upstream has no torch213 build yet). The LingBot-VA baseline depends
   on flash-attn being undetectable — `/home/ubuntu/iwm_shims/flash_attn` raises on call and carries
   no package metadata precisely so `diffusers` cannot find it and switch `autoencoder_kl_wan` to a
   flash path, which would move the VAE numerics and invalidate 91.6%. A group that installs
   flash-attn into a shared interpreter would break that; this one cannot.
3. It skips the `train` extra (megatron-core, DALI, torchtitan, lerobot), none of which the engine
   tests touch.

## Running

```bash
/home/ubuntu/cosmos-framework/.venv/bin/python eval/cosmos3_edge/probe_mot_stack.py
```

| flag | |
|:--|:--|
| `--layers N` | default 28, the shipped depth |
| `--iters` / `--repeats` | default 20 × 3; median reported with spread |
| `--shim` | force the old torch-SDPA shim, to quantify what it was hiding |

Engine mechanics, separately:

```bash
/home/ubuntu/cosmos-framework/.venv/bin/python tests/test_cosmos3_engine.py
```

## Two ways this benchmark can lie to you

**Toy width.** The config in `instinctwm/adapters/cosmos3.py:build_stack` is hidden 512 / head_dim 64,
chosen because the cuDNN SDPA of the day rejected 128. Cost *ranking* at that width does not survive
to 2048, and a benchmark that reports a ratio from toy shapes is reporting the wrong model.
`probe_mot_stack.py` uses the shipped geometry for exactly this reason.

**Region scale.** A pass that eliminates 896 allocations and 0.97 GiB per control step sounds
decisive and measures 1.000× on the path that ships. This is the fused RoPE kernel's lesson again
(`d5aae5e`: bit-exact, 1.10× at region scale, 0.3% of the cycle, did not ship) arriving from a
different direction — there the region was too small to resolve, here it is large and already
subsumed by capture. Both fail the same gate. Gate on the cycle. Section 4 of
[RESULTS.md](RESULTS.md) is that story in full.
