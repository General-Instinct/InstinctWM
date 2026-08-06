#!/usr/bin/env python3
"""Does the L5 residual fusion survive contact with a real transformer block?

The kernel-level tests (`tests/test_triton_residual.py`) prove the kernel is bit-exact against
the eager *expression*. That is not the same claim as "installing it leaves a block's output
unchanged", and it is not the same claim as "a block gets faster": the block runs attention, two
norms and a 14336-wide FFN around those two residual sites, so the fusion is a small fraction of
its cost and every gain has to survive the surrounding noise.

This probe answers the three questions the plan cannot:

  1. BIT-EXACTNESS AT THE BLOCK BOUNDARY. `torch.equal` on the block's output, stock vs armed,
     on both stream shapes. Not `allclose` — the residual feeds the next layer's input and the
     error compounds over 30 layers x 79 forwards, so there is no tolerance at which "close" is
     a claim worth making.
  2. THE FFN SITE. Upstream writes `ff_output.float() * c_gate_msa` at the second site and
     `attn_output * gate_msa` at the first. The hook uses one expression for both, which is only
     legal because c_gate_msa is fp32 and the `.float()` is therefore redundant. Asserted, not
     assumed.
  3. WHAT IT IS WORTH. Per-block wall clock, stock vs armed, at both shapes — and the
     break-even sweep that decides which shapes are armed at all.

Run under an env that can import the upstream tree (`.venv-server`):
    python probe_fused_residual.py --tokens 240
    python probe_fused_residual.py --tokens 32
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

LINGBOT = os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va")
sys.path.insert(0, os.path.join(LINGBOT, "wan_va"))
sys.path.insert(0, "/home/ubuntu/iwm_shims")
IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)

import torch  # noqa: E402

# real geometry, from transformer/config.json + va_robotwin_cfg.py (same as trace_block.py)
DIM, FFN, HEADS, EPS = 3072, 14336, 24, 1e-6
TEXT_LEN = 512


def build_block(device, dtype):
    from modules.model import WanTransformerBlock
    blk = WanTransformerBlock(dim=DIM, ffn_dim=FFN, num_heads=HEADS,
                              cross_attn_norm=True, eps=EPS, attn_mode="torch")
    return blk.to(device=device, dtype=dtype).eval().requires_grad_(False)


def bench_ms(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def bench_ab(variants, rounds=5, iters=60):
    """Interleaved A/B, median over rounds. Sequential blocks of measurement do not work here.

    The first version of this probe timed stock, then hooked, then armed, in that order, and
    reported a 1.24x speedup for a configuration in which the kernel had not been armed at all
    (`fused=0`). The whole difference was the A100 boosting its clocks during the ~2,000 warm-up
    launches that ran in between. Interleaving makes the drift common-mode; the median makes a
    single bad round harmless.
    """
    import statistics
    samples = {name: [] for name in variants}
    for _ in range(rounds):
        for name, fn in variants.items():
            samples[name].append(bench_ms(fn, iters=iters, warmup=5))
    return {name: statistics.median(v) for name, v in samples.items()}, samples


def make_inputs(B, N, dev, dt, kv_slots):
    g = torch.Generator(device="cpu").manual_seed(0)
    h = torch.randn(B, N, DIM, generator=g).to(dev, dt)
    enc = torch.randn(B, TEXT_LEN, DIM, generator=g).to(dev, dt)
    temb = torch.randn(B, N, 6, DIM, generator=g).to(dev, dt)
    # head_dim/2 complex pairs, matching trace_block.py and apply_rotary_emb (model.py:436-438)
    rot = torch.randn(1, N, 1, DIM // HEADS // 2, 2, generator=g).to(dev, torch.float32)
    rot = torch.view_as_complex(rot.contiguous())
    return h, enc, temb, rot


def test_ffn_site(dev) -> bool:
    """`ff_output.float() * c_gate` vs `ff_output * c_gate`, with c_gate fp32.

    The hook drops the explicit upcast because bf16 -> fp32 promotion inside the multiply is
    exact and produces the same fp32 product. If this ever fails, the second call site is not
    the same region as the first and the hook must stop pretending it is.
    """
    print("=== 2. the FFN site's redundant .float() ===")
    ok = True
    for shape in ((2, 240, DIM), (2, 32, DIM)):
        g = torch.Generator(device="cpu").manual_seed(1)
        ff = torch.randn(*shape, generator=g).to(dev, torch.bfloat16)
        gate = torch.randn(*shape, generator=g).to(dev, torch.float32)
        h = torch.randn(*shape, generator=g).to(dev, torch.bfloat16)
        up = (h.float() + ff.float() * gate).type_as(h)      # upstream, model.py:563-564
        no_up = (h.float() + ff * gate).type_as(h)           # what the hook computes
        same = torch.equal(up, no_up)
        ok &= same
        print(f"  {'OK  ' if same else 'FAIL'} {tuple(shape)}  torch.equal={same}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=240, help="240 = video stream, 32 = action")
    ap.add_argument("--batch", type=int, default=2, help="2 = CFG duplicated")
    ap.add_argument("--kv", type=int, default=9792, help="resident KV slots")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--graph", action="store_true",
                    help="capture the block in a CUDA graph and time replay — the mode the "
                         "shipped path actually runs in (passes/graph_capture.py)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs CUDA")
        return 2
    dev, dt = torch.device("cuda"), torch.bfloat16
    B, N = args.batch, args.tokens

    from instinctwm.runtime.fused_residual import (
        RESIDUAL, _block_forward_hooked, install_gated_residual_fusion)

    if args.graph:
        # Same precondition serve_variant enforces for --graph-blocks: the stock KV path calls
        # `(~mask).nonzero()` inside `allocate_slots` (model.py:370), a data-dependent shape that
        # aborts capture with cudaErrorStreamCaptureUnsupported. Ring addressing is what removes
        # it, so graph mode here means the SAME configuration the shipped path runs.
        from instinctwm.optimizer.passes.ring_kv import RingKVAddressing
        RingKVAddressing().install(None, None)
        print("[probe] ring-kv installed (required for capture)\n")

    blk = build_block(dev, dt)
    blk.attn1.init_kv_cache("pos", args.kv, HEADS, DIM // HEADS, dev, dt, B)
    h, enc, temb, rot = make_inputs(B, N, dev, dt, args.kv)

    import modules.model as M
    stock_forward = M.WanTransformerBlock.forward

    def run():
        with torch.no_grad():
            return blk(h, enc, temb, rot, update_cache=0, cache_name="pos")

    mode = "cuda-graph replay" if args.graph else "python launch"
    print(f"=== ONE BLOCK: B={B} N={N} dim={DIM} ffn={FFN} kv_slots={args.kv}, {mode} ===\n")

    # --- 1. correctness, settled before anything is timed -----------------------------------
    ref = run().clone()
    ok = test_ffn_site(dev)

    print("\n=== 3. hook installed, unarmed — must be a byte-identical no-op ===")
    M.WanTransformerBlock.forward = _block_forward_hooked
    same_hook = torch.equal(ref, run())
    ok &= same_hook
    print(f"  {'OK  ' if same_hook else 'FAIL'} torch.equal={same_hook}")

    print("\n=== 4. kernel armed (installer sweeps for the break-even here) ===")
    applied = install_gated_residual_fusion(None, None, dim=DIM, batch=B,
                                            graph_captured=args.graph)
    out_fused = run()
    same_fused = torch.equal(ref, out_fused)
    ok &= same_fused
    delta = (out_fused.float() - ref.float()).abs().max().item()
    print(f"\n  applied: {applied}")
    print(f"  {'OK  ' if same_fused else 'FAIL'} block output torch.equal={same_fused} "
          f"max|d|={delta:.3e}")
    armed_kernel, armed_min = RESIDUAL.kernel, RESIDUAL.min_numel

    # --- 5. the A/B, interleaved ------------------------------------------------------------
    print(f"\n=== 5. per-block wall clock, {args.rounds} interleaved rounds x {args.iters} "
          f"iters, median ===")

    def variant(kernel, min_numel, forward):
        def go():
            M.WanTransformerBlock.forward = forward
            if kernel is None:
                RESIDUAL.disarm()
            else:
                RESIDUAL.arm(kernel, "measured", min_numel)
            return run()
        return go

    variants = {
        "stock": variant(None, 0, stock_forward),
        "hooked/unarmed": variant(None, 0, _block_forward_hooked),
        "hooked/armed": variant(armed_kernel, armed_min, _block_forward_hooked),
    }
    if args.graph:
        variants = {n: _graphed(f) for n, f in variants.items()}
    med, raw = bench_ab(variants, rounds=args.rounds, iters=args.iters)
    base = med["stock"]
    for name, ms in med.items():
        spread = (max(raw[name]) - min(raw[name])) / ms * 100
        print(f"  {name:16s} {ms:8.4f} ms  {base / ms:6.3f}x  (round spread {spread:.1f}%)")

    # Re-run armed once to report what it actually did at this shape.
    M.WanTransformerBlock.forward = _block_forward_hooked
    if armed_kernel is not None:
        RESIDUAL.arm(armed_kernel, "measured", armed_min)
    run()
    print(f"  {RESIDUAL.report()}")
    if RESIDUAL.n_fused == 0:
        print(f"  NOTE: N={N} is below the measured break-even, so 'hooked/armed' ran the EAGER "
              f"path. Any difference it shows against 'hooked/unarmed' is noise — printing it "
              f"is how you can tell.")

    M.WanTransformerBlock.forward = stock_forward
    RESIDUAL.disarm()
    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def _graphed(fn):
    """Capture one call of `fn` and return its replayer, so timing sees device work only."""
    fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


if __name__ == "__main__":
    raise SystemExit(main())
