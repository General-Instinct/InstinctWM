#!/usr/bin/env python3
"""Gates for the length-parameterized ring attention kernel (Layer 4, plan B).

The claim being gated is NOT "this kernel is fast". It is:

    the output depends on the live extent and on NOTHING ELSE about the pool

which is what makes a fixed-shape pool sound, and therefore what removes (start, count) from
`graph_block_stack`'s capture key. Four independent ways for that to be false, one test each:

  1. FIXED_TRIP  walking the whole pool must equal walking only the live extent, at ZERO.
                 This is the masked-tile identity (alpha == 1.0, p == 0.0) asserted rather
                 than argued.
  2. CAPACITY    the same extent in a bigger pool must give the same answer, at ZERO.
  3. GARBAGE     overwriting everything past `count` must not move a single element.
  4. DEVICE      changing the extent tensor IN PLACE, with no re-launch bookkeeping, must
                 track a freshly built call -- the property a graph replay relies on.

Then, separately and honestly: how far this kernel is from the served path, and whether it is
worth anything. Neither is a correctness gate.

    python tests/test_ring_attention.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from instinctwm.kernels.ring_attention import (
    HAVE_TRITON, ring_attention, ring_attention_eager,
)

DEV, DT = torch.device("cuda"), torch.bfloat16
# LingBot-VA's served geometry: 72 frames x 136 tokens, 24 heads, 128 dims, batch 2 under CFG.
B, H, D, CAP = 2, 24, 128, 9792
QROWS = {"video": 240, "action": 32}
results = []


def _extent(start: int, count: int) -> torch.Tensor:
    return torch.tensor([start, count], dtype=torch.int32, device=DEV)


def _pools(cap: int = CAP):
    torch.manual_seed(0)
    return (torch.randn(B, cap, H, D, device=DEV, dtype=DT),
            torch.randn(B, cap, H, D, device=DEV, dtype=DT))


def _q(rows: int):
    torch.manual_seed(1)
    return torch.randn(B, rows, H, D, device=DEV, dtype=DT)


def _delta(a, b) -> float:
    return (a.float() - b.float()).abs().max().item()


def test_fixed_trip_equals_live_trip() -> bool:
    print("\n=== 1. fixed grid over CAPACITY == grid over the live extent ===")
    print("    (the masked-tile identity: alpha == exp(0) == 1.0, p == exp(-inf) == 0.0)")
    k, v = _pools()
    ok = True
    for phase, rows in QROWS.items():
        q = _q(rows)
        for count in (128, 1000, 5000, 7003, CAP):
            e = _extent(0, count)
            live = ring_attention(q, k, v, e, fixed_trip=False)
            fixed = ring_attention(q, k, v, e, fixed_trip=True)
            d = _delta(live, fixed)
            good = d == 0.0
            ok &= good
            print(f"  {'OK  ' if good else 'FAIL'} {phase:6s} count={count:5d}  "
                  f"max|d| = {d:.3e}")
    return ok


def test_capacity_invariance() -> bool:
    print("\n=== 2. the same extent in a LARGER pool gives the same answer ===")
    ok = True
    small_k, small_v = _pools(CAP)
    q = _q(240)
    for count in (1000, 5000):
        ref = ring_attention(q, small_k, small_v, _extent(0, count))
        # a pool twice the size, identical in its first `count` rows
        big_k = torch.randn(B, CAP * 2, H, D, device=DEV, dtype=DT)
        big_v = torch.randn(B, CAP * 2, H, D, device=DEV, dtype=DT)
        big_k[:, :count] = small_k[:, :count]
        big_v[:, :count] = small_v[:, :count]
        got = ring_attention(q, big_k, big_v, _extent(0, count))
        d = _delta(got, ref)
        good = d == 0.0
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} count={count:5d} in pool {CAP} vs {CAP*2}  "
              f"max|d| = {d:.3e}")
    return ok


def test_garbage_past_count_is_inert() -> bool:
    print("\n=== 3. anything past `count` is unreachable ===")
    print("    (a stale ring holds the PREVIOUS episode's K/V there; it must not leak)")
    k, v = _pools()
    q = _q(240)
    ok = True
    for count in (1000, 5000):
        ref = ring_attention(q, k, v, _extent(0, count)).clone()
        k2, v2 = k.clone(), v.clone()
        for filler, label in ((0.0, "zeros"), (7.5, "a large constant")):
            k2[:, count:] = filler
            v2[:, count:] = filler
            d = _delta(ring_attention(q, k2, v2, _extent(0, count)), ref)
            good = d == 0.0
            ok &= good
            print(f"  {'OK  ' if good else 'FAIL'} count={count:5d} tail := {label:18s} "
                  f"max|d| = {d:.3e}")
        k2[:, count:] = torch.randn_like(k2[:, count:])
        v2[:, count:] = torch.randn_like(v2[:, count:])
        d = _delta(ring_attention(q, k2, v2, _extent(0, count)), ref)
        good = d == 0.0
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} count={count:5d} tail := {'noise':18s} "
              f"max|d| = {d:.3e}")
    return ok


def test_extent_is_read_on_device() -> bool:
    print("\n=== 4. the extent is read from DEVICE MEMORY at run time ===")
    print("    (mutate the tensor in place -- no new launch args -- and the answer must move)")
    k, v = _pools()
    q = _q(240)
    e = _extent(0, 1000)
    got_1000 = ring_attention(q, k, v, e).clone()
    e.copy_(torch.tensor([0, 5000], dtype=torch.int32, device=DEV))   # in place
    got_5000 = ring_attention(q, k, v, e).clone()
    ref_5000 = ring_attention(q, k, v, _extent(0, 5000))

    tracked = _delta(got_5000, ref_5000) == 0.0
    moved = _delta(got_1000, got_5000) > 0.0
    print(f"  {'OK  ' if tracked else 'FAIL'} in-place extent change tracks a fresh call: "
          f"max|d| = {_delta(got_5000, ref_5000):.3e}")
    print(f"  {'OK  ' if moved else 'FAIL'} and the answer actually moved: "
          f"max|d| = {_delta(got_1000, got_5000):.3e}")

    # The same, through a captured CUDA graph: one capture, two different extents.
    g = torch.cuda.CUDAGraph()
    e.copy_(torch.tensor([0, 1000], dtype=torch.int32, device=DEV))
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            ring_attention(q, k, v, e)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    with torch.cuda.graph(g):
        out = ring_attention(q, k, v, e)
    g.replay()
    torch.cuda.synchronize()
    d1 = _delta(out, got_1000)
    e.copy_(torch.tensor([0, 5000], dtype=torch.int32, device=DEV))
    g.replay()
    torch.cuda.synchronize()
    d2 = _delta(out, got_5000)
    graph_ok = d1 == 0.0 and d2 == 0.0
    print(f"  {'OK  ' if graph_ok else 'FAIL'} ONE captured graph, replayed at count=1000 "
          f"(max|d| {d1:.3e}) then count=5000 (max|d| {d2:.3e})")
    print("        ^ this is the whole point: no recapture between two different extents")
    return tracked and moved and graph_ok


def test_full_and_wrapped_pool() -> bool:
    print("\n=== 4b. a FULL pool and a WRAPPED ring ===")
    print("    (the first integration died here: illegal memory access at cycle 36,")
    print("     which is exactly 9792 / 272 -- the moment the pool fills)")
    k, v = _pools()
    q = _q(240)
    ok = True

    # count + COUNT_EXTRA must be CLAMPED to capacity, not allowed to run off the allocation.
    for extra in (0, 272):
        out = ring_attention(q, k, v, _extent(0, CAP), count_extra=extra)
        finite = bool(torch.isfinite(out.float()).all())
        ok &= finite
        print(f"  {'OK  ' if finite else 'FAIL'} full pool count={CAP} + extra={extra:3d} "
              f"-> finite output, no OOB")
    # and the clamp must make the extra a NO-OP once full: attending to the whole pool twice
    # is not attending to more.
    a = ring_attention(q, k, v, _extent(0, CAP), count_extra=0)
    b = ring_attention(q, k, v, _extent(0, CAP), count_extra=272)
    d = _delta(a, b)
    good = d == 0.0
    ok &= good
    print(f"  {'OK  ' if good else 'FAIL'} extra beyond a full pool is a no-op: max|d| = {d:.3e}")

    # WRAPPED: start != 0 means the live set is [start, CAP) ++ [0, start+count-CAP). Compare
    # against an explicit gather of exactly those slots, in the same chronological order.
    for start, count in ((9000, 1500), (5000, 9792), (9791, 2)):
        got = ring_attention(q, k, v, _extent(start, count))
        idx = (start + torch.arange(count, device=DEV)) % CAP
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k[:, idx].transpose(1, 2), v[:, idx].transpose(1, 2)).transpose(1, 2)
        d = _delta(got, ref)
        good = d <= 8e-3 and torch.isfinite(got.float()).all()
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} wrapped start={start:4d} count={count:4d} "
              f"vs explicit gather: max|d| = {d:.3e}")
    return ok


def test_matches_sdpa_within_numeric() -> bool:
    print("\n=== 5. distance from the served path (NOT a bit-exactness gate) ===")
    k, v = _pools()
    ok = True
    for phase, rows in QROWS.items():
        q = _q(rows)
        for count in (1000, 5000):
            e = _extent(0, count)
            d = _delta(ring_attention(q, k, v, e), ring_attention_eager(q, k, v, e))
            # bf16 has ~3 decimal digits; one ULP at magnitude ~1 is 7.8e-03. Anything at or
            # below that is a reduction-order difference, not a wrong kernel.
            good = d <= 8e-3
            ok &= good
            print(f"  {'OK  ' if good else 'FAIL'} {phase:6s} count={count:5d}  "
                  f"max|d| vs custom_sdpa = {d:.3e}  (<= 8e-3 = 1 bf16 ULP)")
    print("  NOTE: non-zero by construction. Installing this kernel is a NUMERIC-tier change")
    print("        and needs a paired non-inferiority certificate, not a bit-exactness gate.")
    return ok


def test_speed() -> bool:
    print("\n=== 6. cost vs the dispatcher, on the served shapes ===")

    def bench(fn, iters=50):
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        s, ev = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        ev.record()
        torch.cuda.synchronize()
        return s.elapsed_time(ev) / iters

    k, v = _pools()
    #: forwards per control cycle, per phase, x 30 layers -- so the table can be read as a cycle
    #: rather than as a kernel. LingBot-VA: 26 video + 51 action denoise forwards.
    PER_CYCLE = {"video": 26 * 30, "action": 51 * 30}
    print(f"  {'phase':8s} {'count':>6s} {'sdpa':>9s} {'triton':>9s} {'ratio':>7s} "
          f"{'cycle delta':>12s}")
    net = {}
    for phase, rows in QROWS.items():
        q = _q(rows)
        for count in (1000, 5000, CAP):
            e = _extent(0, count)
            t_ref = bench(lambda: ring_attention_eager(q, k, v, e))
            t_new = bench(lambda: ring_attention(q, k, v, e))
            d = (t_new - t_ref) * PER_CYCLE[phase]
            net[(phase, count)] = d
            print(f"  {phase:8s} {count:6d} {t_ref:8.3f}m {t_new:8.3f}m {t_ref/t_new:6.2f}x "
                  f"{d:+11.1f}m")
    for count in (1000, 5000, CAP):
        total = net[("video", count)] + net[("action", count)]
        print(f"  NET over one control cycle at count={count:5d}: {total:+8.1f} ms "
              f"({'faster' if total < 0 else 'SLOWER'})")
    print("  NOTE: not a gate, and deliberately not a pass/fail. This kernel loses on the video")
    print("        phase and wins on the action phase; the correct consumer is a per-shape")
    print("        measured selection, not one global answer. The win it is FOR is test 4.")
    return True


def test_registered_tier() -> bool:
    print("\n=== 7. the registry DERIVES the tier; the kernel does not get to claim one ===")
    from instinctwm.kernels.lingbot_regions import SELF_ATTENTION_RING
    from instinctwm.kernels.registry import REGISTRY, derive_tier
    from instinctwm.optimizer.contract import Tier

    variants = REGISTRY._by_region.get("self_attention_ring", [])
    if not variants:
        print("  FAIL no kernel registered for self_attention_ring")
        return False
    ok = True
    for kv in variants:
        tier, why = derive_tier(SELF_ATTENTION_RING, kv)
        good = tier is Tier.NUMERIC
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} {kv.name} -> {tier.name}")
        print(f"       {why}")
    print("  ^ and therefore invisible to a BITEXACT-ceiling plan, which is correct: this")
    print("    kernel must be bought with a certificate, not picked up by default.")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    if not HAVE_TRITON:
        print("SKIP: needs triton")
        return 0
    print(f"device {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    for t in (test_fixed_trip_equals_live_trip, test_capacity_invariance,
              test_garbage_past_count_is_inert, test_extent_is_read_on_device,
              test_full_and_wrapped_pool,
              test_matches_sdpa_within_numeric, test_speed, test_registered_tier):
        results.append(t())
    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} groups")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
