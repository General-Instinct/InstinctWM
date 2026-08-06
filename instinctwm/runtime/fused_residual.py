"""The gated-residual call site — one hook, three block bodies, one kernel.

WHY THIS MODULE EXISTS
----------------------
`(hidden.float() + x * gate).type_as(hidden)` appears **six** times per transformer block in
the code this framework actually runs: twice in upstream's `WanTransformerBlock.forward`
(`model.py:543-544` and `:563-564`), and twice more in each of the two InstinctWM rewrites of
that body — `passes/hoist_invariant_casts.py` and `adapter/lingbot.py:_install_modulate_shim`.
Three copies of one numerics-critical expression, and whichever pass installs last wins.

A fourth copy inside a fusion installer would mean the kernel is applied on exactly one of the
three paths, silently, depending on flag order. So the expression is hoisted into `RESIDUAL`
here and all three bodies call it. With no kernel armed, `RESIDUAL` *is* the eager expression,
so this is a no-op refactor until `install_gated_residual_fusion` arms it.

WHY THE DISPATCH IS BY SIZE, AND MEASURED
------------------------------------------
The Triton kernel is not a win at every shape, and *which* shapes depends on something outside
the kernel entirely. Measured on this A100-80GB, torch 2.9 / Triton 3.4, region only:

    tokens   numel      eager    triton   speedup      eager    triton   speedup
                        --- python launch ---          --- cuda graph replay ---
        32   196,608   30.65 us  42.25 us   0.73x     16.49 us   5.05 us    3.27x
       240 1,474,560   46.56 us  40.70 us   1.14x     42.11 us   6.49 us    6.49x
       480 2,949,120   72.50 us  41.42 us   1.75x     67.48 us   9.59 us    7.03x

The Triton kernel costs a flat ~41 us in Python mode at every size from 98K to 2.9M elements.
That is not the kernel: at 2.9M elements it is moving 17 MB, which an A100 does in ~10 us. It is
**Triton's Python launcher**, and it is the entire reason the kernel looks like a regression at
LingBot-VA's real shapes. Under graph replay the launcher is gone and the same kernel wins by
3.3x to 7x everywhere, including the action stream that eager-mode measurement rejects.

So the break-even is a property of the DEPLOYMENT MODE, not just of the box, and measuring it in
the wrong mode gets the answer backwards for 51 of the 77 denoise forwards. `graph_captured` is
not a hint; it selects which of the two sweeps decides. Both are measured and both are printed,
because the number that was not used is the one a reader will want to check.

`min_numel` is not a tuning constant. It is a measurement, retaken on every install.

WHAT THE SWEEP BELOW DOES NOT MEASURE
--------------------------------------
It is a REGION benchmark, and the region is not the deliverable. Measured end to end on the
shipped chain with `--graph-blocks`, 8 measurements per arm, ABBA-ordered, fresh server each:

    shipped chain              3198.8 ms   (stdev 17.0)
    + --fuse-residual          3217.0 ms   (stdev 16.4)      0.994x

A 0.57% REGRESSION, against 3.0x-7.6x on the region and 1.033x on a real block. The kernel was
demonstrably running -- `fused=34560 below_threshold=0 bypassed=0` -- so this is not a treatment
arm that failed to fire. See `eval/lingbot_va_robotwin/RESULTS.md` section 10.

So the gate this file implements answers "which shapes", not "whether at all". The second
question belongs to `harness.cycle_ms_before/after` per `optimizer/contract.py`, and until an
installer runs that, arming here is a decision about shapes taken inside a decision nobody made.
`--fuse-residual` stays opt-in for exactly that reason.
"""

from __future__ import annotations

from statistics import median

import torch

#: element counts swept to locate the break-even. B=2 (CFG duplicates the batch) and C=3072
#: (transformer/config.json), so the axis that moves is the token count: 16 = one action frame,
#: 240 = the video stream's two frames. The sweep brackets both.
_SWEEP_TOKENS = (16, 32, 48, 64, 96, 128, 192, 240, 320, 480)

#: A region measurement must beat eager by this much before the shape is armed. Not caution for
#: its own sake — it is calibration against the block, and it is the number this whole file is
#: least happy about.
#:
#: Measured in python-launch mode at 240 tokens, the region sweep reports between 0.94x and 1.14x
#: across repeats, and a block-level A/B that armed on the optimistic end measured **0.985x** —
#: a regression, reproducibly (`probe_fused_residual.py --tokens 240`). The region benchmark runs
#: the eager chain back-to-back on hot operands, which is not the situation inside a block, so it
#: overstates by roughly 10 points near the crossing. 1.15 is where the two agree here.
#:
#: It is a floor on a MEASUREMENT, not a fudge on a result: everything above it (graph mode, at
#: 3.0x-7.2x) clears it by a wide margin and nothing near the crossing gets armed on a coin flip.
_MARGIN = 1.15

#: Each timing is a median of this many independent measurements. One sample put the same shape
#: on both sides of the margin on consecutive runs, which would make the installed configuration
#: a function of when it was installed.
_REPS = 3


class GatedResidual:
    """A replaceable implementation of one expression, with the eager form as its identity.

    The counters are not decoration. vLLM-Omni's `PromptEmbedCache` silently falls back on any
    unhashable argument, and a run with a climbing bypass count measured the *uncached* path
    while looking like it worked (`eval/lingbot_va_robotwin/README.md`). The same failure is
    available here — a non-contiguous operand would route every call back to eager and report a
    fusion that never fired — so the bypass is counted and printable rather than silent.

    Under CUDA graph capture the counters advance during capture and not during replay, so read
    them as "what the capture saw", never as a call count for the cycle.
    """

    __slots__ = ("kernel", "kernel_name", "min_numel", "n_fused", "n_small", "n_bypassed")

    def __init__(self):
        self.disarm()

    def disarm(self) -> None:
        self.kernel = None
        self.kernel_name = "eager"
        self.min_numel = 0
        self.n_fused = self.n_small = self.n_bypassed = 0

    def arm(self, kernel, name: str, min_numel: int) -> None:
        self.kernel = kernel
        self.kernel_name = name
        self.min_numel = min_numel
        self.n_fused = self.n_small = self.n_bypassed = 0

    def __call__(self, hidden, x, gate):
        k = self.kernel
        if k is None:
            return (hidden.float() + x * gate).type_as(hidden)
        if hidden.numel() < self.min_numel:
            self.n_small += 1
            return (hidden.float() + x * gate).type_as(hidden)
        if not (hidden.is_contiguous() and x.is_contiguous() and gate.shape == hidden.shape):
            self.n_bypassed += 1
            return (hidden.float() + x * gate).type_as(hidden)
        self.n_fused += 1
        return k(hidden, x, gate)

    def report(self) -> str:
        return (f"gated_residual[{self.kernel_name}] min_numel={self.min_numel} "
                f"fused={self.n_fused} below_threshold={self.n_small} bypassed={self.n_bypassed}")


#: The single hook. Imported by every block body that contains the expression.
RESIDUAL = GatedResidual()


# --- the stock-shaped block body, routed through the hook -----------------------------------

def _block_forward_hooked(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                          update_cache=0, cache_name="pos"):
    """`WanTransformerBlock.forward` (model.py:515-566) with the two residuals hooked.

    Byte-for-byte the upstream body apart from the two `RESIDUAL(...)` calls. It is only
    installed when NEITHER InstinctWM block rewrite is present; both of those already call the
    hook, and re-patching over one of them would silently drop its rewrite.
    """
    from einops import rearrange

    temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
    shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
        rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
    shift_msa = shift_msa.squeeze(1)
    scale_msa = scale_msa.squeeze(1)
    gate_msa = gate_msa.squeeze(1)
    c_shift_msa = c_shift_msa.squeeze(1)
    c_scale_msa = c_scale_msa.squeeze(1)
    c_gate_msa = c_gate_msa.squeeze(1)

    norm_hidden_states = (self.norm1(hidden_states.float()) *
                          (1. + scale_msa) + shift_msa).type_as(hidden_states)
    attn_output = self.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states,
                             rotary_emb, update_cache=update_cache, cache_name=cache_name)
    hidden_states = RESIDUAL(hidden_states, attn_output, gate_msa)

    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, encoder_hidden_states,
                             None, update_cache=0, cache_name=cache_name)
    hidden_states = hidden_states + attn_output

    norm_hidden_states = (self.norm3(hidden_states.float()) *
                          (1. + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    # Upstream writes `ff_output.float() * c_gate_msa`. The `.float()` is redundant -- c_gate_msa
    # is already fp32 (it comes from `temb.float()`), so bf16 x fp32 promotes to fp32 either way
    # and the product is bit-identical. `probe_fused_residual.py:test_ffn_site` asserts that.
    hidden_states = RESIDUAL(hidden_states, ff_output, c_gate_msa)
    return hidden_states


_block_forward_hooked._iwm_calls_residual = True


# --- benchmark + install ---------------------------------------------------------------------

def _bench_us(fn, iters: int = 200, warmup: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def _bench_graph_us(fn, iters: int = 200, warmup: int = 20) -> float:
    """Device time for one call, with the Python launcher amortised away.

    Capture-then-replay rather than a Python loop. This is not a synthetic best case: the shipped
    LingBot-VA path captures the block stack (`passes/graph_capture.py:GraphBlockStack`), so
    replay IS the deployment, and a Python-loop measurement of a Triton kernel measures Triton's
    launcher instead of the fusion.
    """
    for _ in range(warmup):
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
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1000.0


def _eager(hidden, x, gate):
    return (hidden.float() + x * gate).type_as(hidden)


def measure_break_even(kernel, dim: int = 3072, batch: int = 2, tokens=_SWEEP_TOKENS,
                       device="cuda", graph_captured: bool = False
                       ) -> tuple[int, list[tuple]]:
    """Smallest element count at which `kernel` beats eager, plus the whole sweep.

    Both modes are always measured; `graph_captured` only chooses which column decides
    `min_numel`. Returns `(min_numel, rows)` with rows of
    `(tokens, numel, eager_py, kern_py, eager_graph, kern_graph, delta)`.

    `min_numel` is the smallest shape that wins AND whose every larger neighbour also wins — a
    single lucky point in the middle of a losing range is an artifact, not a break-even.

    Bit-exactness is checked at every swept shape as a side effect. This is the only place that
    sees the real dtypes on the deployment box, so the gate belongs here rather than in a test
    that may never run there.
    """
    rows = []
    for n in tokens:
        h = torch.randn(batch, n, dim, device=device, dtype=torch.bfloat16)
        x = torch.randn(batch, n, dim, device=device, dtype=torch.bfloat16)
        # gate is FULL [B,N,C] and fp32 in the real model: it is a chunk of `temb.float()`
        # (model.py:524). A broadcast or bf16 gate collapses the region to one dtype and makes
        # the whole measurement meaningless.
        g = torch.randn(batch, n, dim, device=device, dtype=torch.float32)
        delta = (kernel(h, x, g).float() - _eager(h, x, g).float()).abs().max().item()

        def med(bench, fn):
            return median(sorted(bench(fn) for _ in range(_REPS)))

        rows.append((n, h.numel(),
                     med(_bench_us, lambda: _eager(h, x, g)),
                     med(_bench_us, lambda: kernel(h, x, g)),
                     med(_bench_graph_us, lambda: _eager(h, x, g)),
                     med(_bench_graph_us, lambda: kernel(h, x, g)),
                     delta))

    bad = [(r[0], r[6]) for r in rows if r[6] != 0.0]
    if bad:
        raise RuntimeError(
            f"gated_residual_fusion: kernel is NOT bit-exact against eager at {bad}. "
            f"Refusing to install. A residual that is 'close enough' compounds over 30 layers "
            f"x 77 forwards and there is no tolerance at which that claim stays true.")

    ei, ki = (4, 5) if graph_captured else (2, 3)
    wins = [i for i, r in enumerate(rows) if r[ei] / r[ki] >= _MARGIN]
    if not wins or (len(rows) - 1) not in wins:
        return -1, rows
    i = len(rows) - 1
    while i - 1 >= 0 and (i - 1) in wins:
        i -= 1
    return rows[i][1], rows


def install_gated_residual_fusion(server_module=None, server_cls=None, *,
                                  dim: int = 3072, batch: int = 2, verbose: bool = True,
                                  graph_captured: bool = False) -> list[str]:
    """Route the block's two residuals through `RESIDUAL` and arm it with a measured kernel.

    Raises rather than degrading. A pass that silently no-ops is the failure mode this whole
    repo is built against: `plan.explain()` would keep claiming a fusion that never ran.
    """
    import instinctwm.kernels.triton_residual  # noqa: F401  (registers the Triton variant)
    from instinctwm.kernels.lingbot_regions import POST_ATTENTION
    from instinctwm.kernels.registry import REGISTRY
    from instinctwm.optimizer.contract import DeviceProfile, Tier

    if not torch.cuda.is_available():
        raise RuntimeError("operator_fusion: no CUDA device; the kernel cannot be measured.")

    # 1. WHICH kernel, from the registry rather than by import. This is the half the pass
    #    deliberately did not decide: the registry re-derives the tier from the region's
    #    structure and the kernel's declared properties, and a kernel that comes back weaker
    #    than the BITEXACT the plan claimed is refused rather than installed and re-labelled.
    device = DeviceProfile.probe()
    cands = REGISTRY.candidates(POST_ATTENTION, device, Tier.BITEXACT)
    if not cands:
        weaker = REGISTRY.candidates(POST_ATTENTION, device, Tier.BEHAVIORAL)
        raise RuntimeError(
            f"operator_fusion: no BITEXACT kernel for {POST_ATTENTION.name!r} on {device.name}. "
            f"Registered at any tier: {[k.name for k, _ in weaker]}. The plan claimed BITEXACT "
            f"for this region, so installing a weaker kernel would make explain() a lie.")

    # 2. WHICH of them, by measurement on the real shapes. Every BITEXACT candidate is swept:
    #    they are interchangeable on numerics by construction, so the only question left is
    #    speed, and it is not answerable from the source.
    swept = []
    for k, res in cands:
        min_numel, rows = measure_break_even(k.impl, dim=dim, batch=batch,
                                             graph_captured=graph_captured)
        swept.append((k, res, min_numel, rows))
        if verbose:
            print(f"[operator_fusion] candidate {k.name} [{res.tier.name}]", flush=True)
            print(_format_sweep(rows, graph_captured), flush=True)

    # best = lowest break-even; ties broken by speedup at the largest swept shape
    _ei, _ki = (4, 5) if graph_captured else (2, 3)
    usable = [s for s in swept if s[2] >= 0]
    if not usable:
        raise RuntimeError(
            "operator_fusion: no candidate beats eager at ANY swept shape on this device in "
            f"{'graph-replay' if graph_captured else 'python-launch'} mode. Refusing to install "
            "— legal and slower is still rejected.\n"
            + "\n".join(f"{k.name}:\n{_format_sweep(rows, graph_captured)}"
                        for k, _, _, rows in swept))
    k, res, min_numel, rows = min(
        usable, key=lambda s: (s[2], -(s[3][-1][_ei] / s[3][-1][_ki])))

    # 3. the call site. Only patch the body if nothing already routes through the hook.
    import modules.model as M

    Blk = M.WanTransformerBlock
    if getattr(Blk.forward, "_iwm_calls_residual", False):
        site = "hook already present (another InstinctWM block rewrite installed it)"
    else:
        Blk.forward = _block_forward_hooked
        site = "WanTransformerBlock.forward replaced with the hooked body"

    RESIDUAL.arm(k.impl, k.name, min_numel)

    if verbose:
        print("[operator_fusion] " + site, flush=True)
        print(f"[operator_fusion] selected {k.name} [{res.tier.name}]: {res.reason}", flush=True)
        print(f"[operator_fusion] break-even at numel >= {min_numel} "
              f"({'graph-replay' if graph_captured else 'python-launch'} mode); "
              f"smaller shapes stay eager", flush=True)
    return [f"operator_fusion({POST_ATTENTION.name}, {k.name}, min_numel={min_numel}, "
            f"mode={'graph' if graph_captured else 'python'})"]


def _format_sweep(rows, graph_captured: bool) -> str:
    star = "  <- deciding" if graph_captured else ""
    out = [f"  {'tokens':>6} {'numel':>9} | {'eagerPY':>8} {'kernPY':>8} {'x':>6}"
           f" | {'eagerG':>8} {'kernG':>8} {'x':>6}{star}"]
    for n, numel, ep, kp, eg, kg, d in rows:
        out.append(f"  {n:6d} {numel:9d} | {ep:8.2f} {kp:8.2f} {ep / kp:6.2f}"
                   f" | {eg:8.2f} {kg:8.2f} {eg / kg:6.2f}")
    out.append(f"  (PY = python launch, G = cuda-graph replay; "
               f"{'G' if graph_captured else 'PY'} decides the break-even here, "
               f"and must clear {_MARGIN:.2f}x to count)")
    return "\n".join(out)
