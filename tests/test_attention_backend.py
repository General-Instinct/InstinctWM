#!/usr/bin/env python3
"""Gates for AttentionBackend (Layer 4, plan A).

What is actually being gated:

  0. the pass names no model symbol -- same AST check the other engine passes carry
  1. it picks by MEASUREMENT on the site's own shapes, and the pick really is faster
  2. the installed rewrite computes the same attention (within one bf16 ULP) and actually
     pins the backend it claimed
  3. it reports NUMERIC, never BITEXACT, even when a candidate measures a delta of zero
  4. every way of declining works, and says which one it took

    python tests/test_attention_backend.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from instinctwm.optimizer.contract import Tier
from instinctwm.passes.attention_backend import AttentionBackend
from instinctwm.passes.interface import Site, SiteKind, run_pass

DEV, DT = torch.device("cuda"), torch.bfloat16
B, H, D, CAP = 2, 24, 128, 9792
results = []


def model_free() -> bool:
    """The pass may know about torch. It may not know about any model."""
    import ast
    path = os.path.join(os.path.dirname(__file__), "..", "instinctwm", "passes",
                        "attention_backend.py")
    tree = ast.parse(open(path).read())
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            b = getattr(n, "body", [])
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) \
                    and isinstance(b[0].value.value, str):
                n.body = b[1:]
    code = ast.unparse(tree)
    bad = [t for t in ("modules.model", "WanAttention", "WanTransformer", "attn_caches",
                       ".blocks", "lingbot", "cosmos", "custom_sdpa") if t in code]
    print(f"  {'OK  ' if not bad else 'FAIL'} pass CODE references no model symbol"
          + (f" -- found {bad}" if bad else ""))
    return not bad


class _Surface:
    """A stand-in adapter: N layers that all publish the same shape signature.

    Deliberately N > 1. The pass must MEASURE ONCE and reuse, because 30 layers measuring
    independently is 30x the cost for one answer, and worse, lets two layers install different
    kernels for the same shape -- which splits a capture key for no reason.
    """
    model_id = "toy-wam"

    def __init__(self, n_layers=6, q_rows=240, masked=False, drop=(), op=None):
        self.n_layers, self.q_rows, self.masked, self.drop = n_layers, q_rows, masked, drop
        self.installed = {}
        self.op = op or _sdpa

    def sites(self, kind):
        if kind is not SiteKind.ATTENTION_OP:
            return
        for i in range(self.n_layers):
            attrs = {"op": self.op, "heads": H, "head_dim": D, "dtype": DT, "layout": "bshd",
                     "masked": self.masked, "capacity": CAP, "q_rows": self.q_rows, "batch": B,
                     "extent_binding": "sliced"}
            for k in self.drop:
                attrs[k] = None
            yield Site(kind=kind, id=f"toy.attention[{i}]", attrs=attrs)

    def apply(self, rewrite):
        self.installed[rewrite.site_id] = rewrite.payload(self.op)


def _sdpa(q, k, v):
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)


def _bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def test_picks_and_is_faster() -> bool:
    print("\n=== 1. picks by measurement, and the pick reproduces as faster ===")
    surf = _Surface(n_layers=6, q_rows=240)
    p = AttentionBackend(min_speedup=1.10)
    res = run_pass(p, surf, DEV)
    print(f"  {res}")
    print("  " + p.report().replace("\n", "\n  "))
    print(f"  {p.stats()}")

    if not res.fired:
        # A legitimate outcome on hardware where the dispatcher is already optimal. Then the
        # only thing to assert is that it said so with a table rather than silently.
        told = bool(p.choices) and all(c.table for c in p.choices.values())
        print(f"  {'OK  ' if told else 'FAIL'} declined WITH a measured table "
              f"(dispatcher already best on this device)")
        return told

    one_measurement = len(p.choices) == 1
    all_six = len(res.applied) == 6
    print(f"  {'OK  ' if one_measurement else 'FAIL'} measured ONCE for 6 identical layers "
          f"({len(p.choices)} signature(s))")
    print(f"  {'OK  ' if all_six else 'FAIL'} installed on all 6 layers")

    # Reproduce the claim independently of the pass's own timing loop.
    q = torch.randn(B, 240, H, D, device=DEV, dtype=DT)
    k = torch.randn(B, CAP // 2, H, D, device=DEV, dtype=DT)
    v = torch.randn(B, CAP // 2, H, D, device=DEV, dtype=DT)
    picked = surf.installed["toy.attention[0]"]
    t_ref, t_new = _bench(lambda: _sdpa(q, k, v)), _bench(lambda: picked(q, k, v))
    faster = t_new < t_ref
    print(f"  {'OK  ' if faster else 'FAIL'} independent re-measure: dispatcher {t_ref:.3f} ms "
          f"-> picked {t_new:.3f} ms ({t_ref/t_new:.2f}x)")
    return one_measurement and all_six and faster


def test_output_is_within_one_ulp() -> bool:
    print("\n=== 2. the rewrite still computes attention ===")
    surf = _Surface(n_layers=2, q_rows=240)
    p = AttentionBackend(min_speedup=1.10)
    res = run_pass(p, surf, DEV)
    if not res.fired:
        print("  SKIP (nothing installed on this device)")
        return True
    q = torch.randn(B, 240, H, D, device=DEV, dtype=DT)
    k = torch.randn(B, 1000, H, D, device=DEV, dtype=DT)
    v = torch.randn(B, 1000, H, D, device=DEV, dtype=DT)
    got = surf.installed["toy.attention[0]"](q, k, v)
    ref = _sdpa(q, k, v)
    d = (got.float() - ref.float()).abs().max().item()
    ok = d <= 8e-3 and got.shape == ref.shape
    print(f"  {'OK  ' if ok else 'FAIL'} max|d| vs dispatcher = {d:.3e} (<= 8e-3 = 1 bf16 ULP), "
          f"shape {tuple(got.shape)}")
    print(f"       non-zero is EXPECTED: a different backend is a different reduction order")
    return ok


def test_tier_is_never_bitexact() -> bool:
    print("\n=== 3. NUMERIC, always -- a measured zero does not buy BITEXACT ===")
    surf = _Surface(n_layers=1, q_rows=240)
    p = AttentionBackend(min_speedup=1.10)
    run_pass(p, surf, DEV)
    if not p.choices:
        print("  FAIL no measurement was taken")
        return False
    tiers = {c.tier for c in p.choices.values() if c.winner}
    zero_deltas = [m for c in p.choices.values() for m in c.table if m.ok and m.delta == 0.0]
    ok = Tier.BITEXACT not in tiers
    print(f"  {'OK  ' if ok else 'FAIL'} chosen tier(s) {[t.name for t in tiers] or ['none']} "
          f"-- BITEXACT never claimed")
    print(f"       ({len(zero_deltas)} candidate(s) measured max|d| = 0 and were still not "
          f"promoted)")
    # And the ceiling must actually bind.
    strict = AttentionBackend(min_speedup=1.10, tier_ceiling=Tier.BITEXACT)
    res2 = run_pass(strict, _Surface(n_layers=1, q_rows=240), DEV)
    blocked = not res2.fired
    print(f"  {'OK  ' if blocked else 'FAIL'} under a BITEXACT ceiling the pass installs nothing")
    if strict.declines:
        print(f"       reason: {strict.declines[0].reason[:100]}")
    return ok and blocked


def test_declines() -> bool:
    print("\n=== 4. every decline path, each with its own reason ===")
    cases = [
        ("masked call", _Surface(n_layers=1, masked=True), "masked"),
        ("no observed q extent", _Surface(n_layers=1, drop=("q_rows",)), "query extent"),
        ("undeclared head_dim", _Surface(n_layers=1, drop=("head_dim",)), "head_dim"),
        ("no attention callable", _Surface(n_layers=1, drop=("op",)), "no attention callable"),
    ]
    ok = True
    for label, surf, needle in cases:
        p = AttentionBackend()
        res = run_pass(p, surf, DEV)
        reason = p.declines[0].reason if p.declines else (res.skipped_reason or "")
        good = (not res.fired) and needle in reason
        ok &= good
        print(f"  {'OK  ' if good else 'FAIL'} {label:22s} -> {reason[:78]}")

    # A threshold that nothing can clear: the pass must keep the dispatcher AND show its work.
    p = AttentionBackend(min_speedup=100.0)
    res = run_pass(p, _Surface(n_layers=1), DEV)
    reason = p.declines[0].reason if p.declines else ""
    good = (not res.fired) and "threshold" in reason
    ok &= good
    print(f"  {'OK  ' if good else 'FAIL'} unreachable threshold  -> {reason[:78]}")
    return ok


class _FakeAttn:
    """Enough of an attention module for the surface to publish a site from.

    Not a mock of `WanAttention` -- it is the SHAPE OF THE CONTRACT the surface relies on:
    an `attn_op` instance attribute, `heads`, and an `attn_caches` dict holding the pool and the
    ring interval. If upstream moves any of those, this fails here rather than in a server.
    """

    def __init__(self, cache_name="pos", count=4000):
        self.heads, self.inner_dim = H, H * D
        self.attn_op = _sdpa
        k = torch.zeros(B, CAP, H, D, device=DEV, dtype=DT)
        self.attn_caches = {cache_name: {"k": k, "v": k,
                                         "_ring": {"start": 0, "count": count}}}


def test_surface_publishes_usable_sites() -> bool:
    print("\n=== 5. the LingBot surface publishes sites this pass can actually use ===")
    from instinctwm.adapter.lingbot import LingBotSurface

    blocks = [type("Blk", (), {"attn1": _FakeAttn()})() for _ in range(4)]
    model = type("M", (), {"blocks": blocks})()
    surf = LingBotSurface(model)

    sites = list(surf.sites(SiteKind.ATTENTION_OP))
    got_all = len(sites) == 4
    print(f"  {'OK  ' if got_all else 'FAIL'} published {len(sites)} attention site(s) for "
          f"4 layers")

    # Before any forward runs, the pass MUST decline -- it has no query extent to measure on.
    p_cold = AttentionBackend()
    cold = run_pass(p_cold, surf, DEV)
    cold_ok = not cold.fired and any("query extent" in d.reason for d in p_cold.declines)
    print(f"  {'OK  ' if cold_ok else 'FAIL'} declines before a forward has been observed")

    # Run one forward through the recording wrapper; now the shape is known.
    q = torch.randn(B, 240, H, D, device=DEV, dtype=DT)
    kv = torch.randn(B, 1000, H, D, device=DEV, dtype=DT)
    for b in blocks:
        b.attn1.attn_op(q, kv, kv)
    rows = {s.attrs["q_rows"] for s in surf.sites(SiteKind.ATTENTION_OP)}
    observed = rows == {240}
    print(f"  {'OK  ' if observed else 'FAIL'} recorded the query extent from a real call: {rows}")

    p = AttentionBackend(min_speedup=1.10)
    res = run_pass(p, surf, DEV)
    print(f"  {res}")
    if not res.fired:
        print("  SKIP install check (dispatcher already best on this device)")
        return got_all and cold_ok and observed

    # The rewrite must land UNDER the recording wrapper, not replace it: a second pass has to
    # still see the model's own shapes.
    a0 = blocks[0].attn1
    swapped = a0._iwm_attn_impl is not a0._iwm_attn_base
    outer_intact = a0.attn_op.__name__ == "recording"
    out = a0.attn_op(q, kv, kv)
    d = (out.float() - _sdpa(q, kv, kv).float()).abs().max().item()
    computes = d <= 8e-3
    print(f"  {'OK  ' if swapped else 'FAIL'} inner slot replaced")
    print(f"  {'OK  ' if outer_intact else 'FAIL'} recording wrapper still outermost")
    print(f"  {'OK  ' if computes else 'FAIL'} calling through the surface still computes "
          f"attention (max|d| {d:.3e})")
    return got_all and cold_ok and observed and swapped and outer_intact and computes


def main() -> int:
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    print(f"device {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    print("=== 0. model-free? ===")
    results.append(model_free())
    for t in (test_picks_and_is_faster, test_output_is_within_one_ulp,
              test_tier_is_never_bitexact, test_declines,
              test_surface_publishes_usable_sites):
        results.append(t())
    print(f"\n{'PASS' if all(results) else 'FAIL'}: {sum(results)}/{len(results)} groups")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
