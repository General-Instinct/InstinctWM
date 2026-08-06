"""AttentionBackend -- choose the attention implementation by measurement, not by reputation.

Sixth engine pass, and the first in the attention layer.

WHAT IT DOES

`torch.nn.functional.scaled_dot_product_attention` is a DISPATCHER. Which kernel it runs is chosen
by a heuristic over dtype, head dim, mask presence and alignment -- not by measuring the shapes in
front of it. On LingBot-VA's served shapes that heuristic picks the flash backend, and on this box
it is not the fastest one available:

    video q=240, KV=5000, B=2, H=24, D=128, bf16, A100-SXM4-80GB, torch 2.9.0+cu128
        flash          0.290 ms     <- what the dispatcher picks
        cudnn          0.195 ms     <- 1.49x faster
        mem_efficient  0.445 ms
        math           3.935 ms

So the pass measures every backend the device will admit, on the site's own shapes, and installs
the winner. Nothing here is specific to a model, and nothing is specific to LingBot's geometry --
the shapes come from the site.

WHY THIS IS NOT BIT-EXACT, AND WHY THE PASS SAYS SO ITSELF

A different backend is a different reduction order. Measured on the same shapes, cuDNN differs from
flash by max|delta| = 4.883e-04 on 730,892 of 1,474,560 elements -- one bf16 ULP at that magnitude,
on half the tensor. That is a NUMERIC-tier change and it needs a paired non-inferiority run before
it can ship, exactly like a step-reduction recipe.

The pass therefore MEASURES the delta against the incumbent and reports it, rather than declaring a
tier. `audit_tier` is the shared rule and it is deliberately asymmetric: a measured 0 does NOT
promote a candidate to BITEXACT, because exactness on one input is not a proof. Only a structural
argument earns BITEXACT, and "a different kernel happened to agree here" is not one.

WHY IT MUST BE ABLE TO DECLINE

The dispatcher's heuristic is right more often than it is wrong, and a backend that wins on the
video phase can lose on the action phase -- q=32 attention is bandwidth-bound at 71% of this box's
HBM roofline, where there is nothing left for a kernel to win. So the pass declines below a
threshold speedup and records the table it measured, because "we looked and the default was
already best" and "we did not look" must not be the same output.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from instinctwm.optimizer.contract import Tier
from instinctwm.passes.interface import Rewrite, RewriteKind, Site, SiteKind


@dataclass
class Decline:
    site_id: str
    reason: str

    def __str__(self) -> str:
        return f"{self.site_id}: {self.reason}"


@dataclass
class Measurement:
    """One candidate on one shape signature. `delta` is against the INCUMBENT, not against math."""
    name: str
    ms: float
    delta: float
    ok: bool = True
    error: str = ""

    def __str__(self) -> str:
        if not self.ok:
            return f"{self.name:16s}     unavailable ({self.error})"
        return f"{self.name:16s} {self.ms:8.3f} ms   max|d| vs incumbent {self.delta:.3e}"


@dataclass
class Choice:
    """What was chosen for one shape signature, and the whole table behind it."""
    signature: tuple
    incumbent_ms: float
    winner: str | None
    winner_ms: float
    tier: Tier
    reason: str
    table: list[Measurement] = field(default_factory=list)

    def report(self) -> str:
        head = (f"shape {self.signature}: incumbent {self.incumbent_ms:.3f} ms -> "
                f"{self.winner or 'KEEP INCUMBENT'}")
        if self.winner:
            head += (f" {self.winner_ms:.3f} ms ({self.incumbent_ms / self.winner_ms:.2f}x) "
                     f"[{self.tier.name}]")
        return "\n".join([head, f"    {self.reason}"]
                         + [f"    {m}" for m in self.table])


def _backends() -> dict:
    """Candidate implementations. A framework fact about torch, not a fact about any model.

    Imported lazily and filtered by availability so this module still imports on a torch build
    without `torch.nn.attention` -- the pass then simply has nothing to offer and declines.
    """
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:                                    # pragma: no cover
        return {}
    names = {"flash": "FLASH_ATTENTION", "mem_efficient": "EFFICIENT_ATTENTION",
             "cudnn": "CUDNN_ATTENTION", "math": "MATH"}
    out = {}
    for label, attr in names.items():
        be = getattr(SDPBackend, attr, None)
        if be is not None:
            out[label] = (be, sdpa_kernel)
    return out


class AttentionBackend:
    name = "attention_backend"

    #: `math` is admitted as a candidate but never wins in practice (it materialises the full
    #: score matrix). It stays in the table because a row that says 3.935 ms is evidence the
    #: measurement is discriminating, and its absence would look like a filtered result.
    def __init__(self, min_speedup: float = 1.10, tier_ceiling: Tier = Tier.NUMERIC,
                 iters: int = 20, verbose: bool = False):
        self.min_speedup = min_speedup
        self.tier_ceiling = tier_ceiling
        self.iters = iters
        self.verbose = verbose
        self.declines: list[Decline] = []
        self.choices: dict[tuple, Choice] = {}
        self.rewritten: list[str] = []

    def sites_required(self):
        return (SiteKind.ATTENTION_OP,)

    # ---- decide ---------------------------------------------------------------------------
    def plan_rewrites(self, sites, device) -> list[Rewrite]:
        self.declines.clear()
        self.choices.clear()
        out: list[Rewrite] = []
        for site in sites.get(SiteKind.ATTENTION_OP, []):
            why = self._why_not(site)
            if why:
                self._decline(site.id, why)
                continue
            sig = self._signature(site)
            if sig not in self.choices:
                # 30 layers share one signature. Measuring each would cost 30x for one answer,
                # and worse, would let two layers disagree and install different kernels for the
                # same shape -- which is a graph key that splits for no reason.
                self.choices[sig] = self._measure(sig)
            choice = self.choices[sig]
            if choice.winner is None:
                self._decline(site.id, choice.reason)
                continue
            self.rewritten.append(site.id)
            out.append(Rewrite(
                site_id=site.id, kind=RewriteKind.WRAP,
                payload=self._rewrite(choice.winner),
                note=(f"{choice.winner} at {choice.winner_ms:.3f} ms vs incumbent "
                      f"{choice.incumbent_ms:.3f} ms ({choice.incumbent_ms/choice.winner_ms:.2f}x), "
                      f"tier {choice.tier.name}")))
        return out

    def _decline(self, site_id: str, why: str) -> None:
        self.declines.append(Decline(site_id, why))
        if self.verbose:
            print(f"[attention_backend] DECLINE {site_id}: {why}", flush=True)

    def _why_not(self, site: Site) -> str | None:
        a = site.attrs
        if a.get("op") is None:
            return "site publishes no attention callable"
        if a.get("masked"):
            return ("site declares a masked attention call; backends differ in which masks they "
                    "accept, so a swap could change which kernel runs for reasons unrelated to speed")
        for f in ("heads", "head_dim", "dtype", "capacity"):
            if a.get(f) is None:
                return f"site does not declare {f!r}, so the measurement cannot be built"
        if a.get("q_rows") is None:
            return ("site has not observed a query extent yet; run one forward before planning, "
                    "so the measurement uses the shape the model actually calls with")
        if a.get("layout") != "bshd":
            return f"unsupported layout {a.get('layout')!r}; this pass builds BSHD probes"
        if not torch.cuda.is_available():
            return "no CUDA device; backend selection is a GPU question"
        return None

    @staticmethod
    def _signature(site: Site) -> tuple:
        a = site.attrs
        # The live extent is deliberately NOT in the signature: it changes every cycle, and the
        # backend choice must not. The probe uses a mid-episode extent instead (see _measure).
        return (a["batch"] or 1, a["q_rows"], a["heads"], a["head_dim"],
                a["capacity"], str(a["dtype"]))

    # ---- measure --------------------------------------------------------------------------
    def _measure(self, sig: tuple) -> Choice:
        batch, q_rows, heads, head_dim, capacity, dtype_s = sig
        dtype = getattr(torch, dtype_s.replace("torch.", ""))
        dev = torch.device("cuda")
        # HALF the pool. The live extent grows through an episode, so any single probe extent is
        # a choice; the midpoint is the one that is wrong by the least for the longest.
        kv = max(1, capacity // 2)
        q = torch.randn(batch, q_rows, heads, head_dim, device=dev, dtype=dtype)
        k = torch.randn(batch, kv, heads, head_dim, device=dev, dtype=dtype)
        v = torch.randn(batch, kv, heads, head_dim, device=dev, dtype=dtype)

        import torch.nn.functional as F

        def call(backend=None):
            def go():
                if backend is None:
                    return F.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
                be, ctx = backend
                with ctx(be):
                    return F.scaled_dot_product_attention(
                        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
            return go

        incumbent_fn = call(None)
        incumbent_ms = self._bench(incumbent_fn)
        incumbent_out = incumbent_fn().float()

        table: list[Measurement] = []
        for label, backend in _backends().items():
            fn = call(backend)
            try:
                ms = self._bench(fn)
                d = (fn().float() - incumbent_out).abs().max().item()
            except Exception as ex:                        # a backend that refuses these shapes
                table.append(Measurement(label, float("inf"), float("nan"), False,
                                         f"{type(ex).__name__}: {str(ex)[:60]}"))
                continue
            table.append(Measurement(label, ms, d))

        usable = [m for m in table if m.ok]
        usable.sort(key=lambda m: m.ms)
        if not usable:
            return Choice(sig, incumbent_ms, None, 0.0, Tier.BITEXACT,
                          "no backend could run these shapes", table)
        best = usable[0]
        speedup = incumbent_ms / best.ms
        # A candidate that IS the incumbent (delta exactly 0 and same time) is not a rewrite.
        if speedup < self.min_speedup:
            return Choice(sig, incumbent_ms, None, 0.0, Tier.BITEXACT,
                          (f"best candidate {best.name!r} at {best.ms:.3f} ms is only "
                           f"{speedup:.2f}x over the dispatcher's own choice (threshold "
                           f"{self.min_speedup:.2f}x); keeping it"), table)
        tier = self._tier_for(best.delta)
        if tier > self.tier_ceiling:
            return Choice(sig, incumbent_ms, None, 0.0, tier,
                          (f"{best.name!r} would be {speedup:.2f}x but its tier {tier.name} "
                           f"exceeds the ceiling {self.tier_ceiling.name}"), table)
        return Choice(sig, incumbent_ms, best.name, best.ms, tier,
                      (f"measured on the site's own shapes, {kv} of {capacity} KV slots live; "
                       f"tier {tier.name} because max|delta| vs the incumbent is {best.delta:.3e}"),
                      table)

    @staticmethod
    def _tier_for(delta: float) -> Tier:
        """Always NUMERIC. The argument, not the arithmetic, is what is missing.

        A measured delta of 0 does NOT buy BITEXACT here. Same rule as
        `kernels.registry.audit_tier`: exactness on one input is not a proof of exactness, and a
        backend swap has no structural argument behind it -- only a sample. Every other BITEXACT
        pass in this repo has one (identity all-gathers, a memoized pure function, an interval
        that was always an interval). This has none, so it does not get the tier, and `delta` is
        reported alongside rather than used to promote.
        """
        return Tier.NUMERIC

    # ---- what to install ------------------------------------------------------------------
    @staticmethod
    def _rewrite(backend_label: str):
        backend = _backends()[backend_label]

        def wrap(orig):
            be, ctx = backend

            def pinned(q, k, v):
                with ctx(be):
                    return orig(q, k, v)
            return pinned

        return wrap

    def _bench(self, fn) -> float:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(self.iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / self.iters

    # ---- report ---------------------------------------------------------------------------
    def report(self) -> str:
        out = [c.report() for c in self.choices.values()]
        for d in self.declines:
            out.append(f"DECLINE {d}")
        return "\n".join(out) if out else "no attention sites examined"

    def stats(self) -> str:
        tiers = {c.tier.name for c in self.choices.values() if c.winner}
        return (f"rewritten={len(self.rewritten)} declined={len(self.declines)} "
                f"signatures={len(self.choices)} tier={'/'.join(sorted(tiers)) or 'n/a'}")
