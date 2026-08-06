#!/usr/bin/env python3
"""Regression tests for the Triton gated residual -- specifically for FMA contraction.

WHY THIS FILE EXISTS

The first version of this kernel was 1.22x and looked fine under a loose tolerance. It was not
bit-exact: Triton 3.5.0 contracted `h + a*g` into a single `fma.rn.f32`, skipping the fp32 rounding
of the product that eager PyTorch performs. Only 33 of 1,474,560 elements differed, by exactly one
bf16 ULP -- small enough that any `allclose` would have waved it through.

Worse, the first attempted fix (`tl.where(mask, p, p)` as an optimization barrier) was removed by
the compiler. Both paths still contracted, both showed the identical delta, and that read as "FMA
is ruled out" when in truth FMA had never been disabled. The lesson is `test_ptx`: assert on the
emitted instruction, not on a differential test against your own flag.

    python tests/test_triton_residual.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from instinctwm.kernels.triton_residual import (
    _gated_residual_kernel, gated_residual, gated_residual_eager,
)

SHAPES = [(2, 240, 3072), (2, 32, 3072)]                   # video and action streams


def _inputs(shape, seed=0, hscale=1.0, ascale=1.0):
    B, N, C = shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (
        (torch.randn(B, N, C, generator=g) * hscale).to("cuda", torch.bfloat16),
        (torch.randn(B, N, C, generator=g) * ascale).to("cuda", torch.bfloat16),
        torch.randn(B, N, C, generator=g).to("cuda", torch.float32),   # gate is fp32 in the model
    )


def _bench(f, it=200):
    for _ in range(50):
        f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        f()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000.0                 # us


def test_bit_exact():
    print("=== 1. bit-exact against eager on both stream shapes ===")
    ok = True
    for shape in SHAPES:
        h, a, gt = _inputs(shape)
        out, ref = gated_residual(h, a, gt), gated_residual_eager(h, a, gt)
        n = (out != ref).sum().item()
        d = (out.float() - ref.float()).abs().max().item()
        print(f"  {'OK  ' if n == 0 else 'FAIL'} {str(shape):>16}  differing={n}  max|d|={d:.3e}")
        ok &= n == 0
    return ok


def test_scales():
    """Sweep the exponent range so the product straddles bf16 tie points.

    Scales are inverse to each other, keeping `a*g` near `h` in magnitude -- where a retained FMA
    guard bit is most likely to flip the final round-to-nearest.
    """
    print("=== 2. bit-exact across an adversarial scale sweep ===")
    total = elems = 0
    for trial in range(12):
        h, a, gt = _inputs((2, 240, 3072), seed=1000 + trial,
                           hscale=2.0 ** (trial - 6), ascale=2.0 ** (6 - trial))
        total += (gated_residual(h, a, gt) != gated_residual_eager(h, a, gt)).sum().item()
        elems += h.numel()
    print(f"  {'OK  ' if total == 0 else 'FAIL'} {total} differing over {elems:,} elements "
          f"across 12 scales (2^-6 .. 2^5)")
    return total == 0


def test_fma_negative_control():
    """If FMA mode ever matches eager, `test_bit_exact` is passing for free and proves nothing."""
    print("=== 3. negative control: FMA mode MUST differ ===")
    h, a, gt = _inputs((2, 240, 3072), seed=1006)          # scale with the largest disagreement
    n = (gated_residual(h, a, gt, allow_fma=True) != gated_residual_eager(h, a, gt)).sum().item()
    print(f"  {'OK  ' if n > 0 else 'FAIL'} FMA mode differs on {n} elements "
          f"({'enable_fp_fusion still controls contraction' if n else 'FLAG HAS NO EFFECT'})")
    return n > 0


def test_ptx():
    """Assert on the emitted instruction. This is the test the original bug would have failed."""
    print("=== 4. PTX: no fma.rn.f32 when contraction is disabled ===")
    h, a, gt = _inputs((2, 32, 3072))
    ok = True
    for allow_fma, want, forbid in ((False, "mul.rn.f32", "fma.rn.f32"),
                                    (True, "fma.rn.f32", None)):
        ptx = _gated_residual_kernel.warmup(
            h, a, gt, torch.empty_like(h), h.numel(), BLOCK=1024, num_warps=4,
            enable_fp_fusion=allow_fma, grid=(1,)).asm["ptx"]
        good = want in ptx and (forbid is None or forbid not in ptx)
        print(f"  {'OK  ' if good else 'FAIL'} allow_fma={str(allow_fma):5s} "
              f"has {want}={want in ptx}" + (f", has {forbid}={forbid in ptx}" if forbid else ""))
        ok &= good
    return ok


def _bench_graph(f, it=200):
    """Device time only: capture one call, time the replays.

    This exists because the python-loop gate below was failing on the action stream for a reason
    that had nothing to do with the kernel. Triton's python launcher costs a flat ~41 us per
    call at every size from 98K to 2.9M elements — at 2.9M the kernel is moving 17 MB, which an
    A100 does in ~10 us — so a python-loop benchmark of a Triton kernel largely measures the
    launcher. The shipped LingBot-VA path captures the block stack, where that cost does not
    exist, so replay is the mode the gate has to judge.
    """
    for _ in range(50):
        f()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            f()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        f()
    for _ in range(20):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it * 1000.0


def test_faster_than_eager():
    """Performance gate. Bit-exactness is necessary, not sufficient.

    Gated on GRAPH REPLAY, with the python-launch numbers printed beside it. The two disagree in
    direction, not just in magnitude — on this A100 the action stream measures 0.78x in python
    mode and 3.0x under replay — and the deployment runs under replay. The python column stays
    visible because it is what decides `min_numel` for a server running WITHOUT graph capture
    (`runtime/fused_residual.py`), so a reader needs to see both to know which one applies.
    """
    print("=== 5. faster than eager (performance gate, graph replay decides) ===")
    ok = True
    for shape in SHAPES:
        h, a, gt = _inputs(shape)
        te = _bench(lambda: gated_residual_eager(h, a, gt))
        tx = _bench(lambda: gated_residual(h, a, gt))
        tf = _bench(lambda: gated_residual(h, a, gt, allow_fma=True))
        ge = _bench_graph(lambda: gated_residual_eager(h, a, gt))
        gx = _bench_graph(lambda: gated_residual(h, a, gt))
        good = ge > gx
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} {str(shape):>16}  "
              f"graph: eager {ge:6.2f} us -> {gx:5.2f} us = {ge/gx:5.2f}x   |   "
              f"python: eager {te:6.2f} us -> {tx:5.2f} us = {te/tx:4.2f}x   "
              f"(FMA mode {tf:6.2f} us: disabling contraction costs {(tx/tf-1)*100:+.1f}%)")
    return ok


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        sys.exit(0)
    results = [t() for t in (test_bit_exact, test_scales, test_fma_negative_control,
                             test_ptx, test_faster_than_eager)]
    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} groups")
    sys.exit(0 if all(results) else 1)
