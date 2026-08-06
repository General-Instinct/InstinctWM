#!/usr/bin/env python3
"""Launch the LingBot-VA server with InstinctWM optimization variants toggled on.

This is an A/B harness, not a fork. It imports the upstream server unmodified and patches
named behaviours at runtime, so `git status` in lingbot-va stays clean and every variant is
one flag away from the stock baseline. Each variant exists because a specific cost was
measured or read out of the code.

Variants (all default OFF; stock behaviour is the baseline):

  --no-fsdp
      Skip `shard_model`. `distributed/util.py:15-19` applies FSDP `fully_shard` whenever
      `dist.is_initialized()`, which is ALWAYS -- `init_distributed` is called
      unconditionally, even for a single-GPU server. `fsdp.py:28-34` wraps 4 units per
      block (attn1, attn2, ffn, block) across 30 blocks plus the root = 121 units, with
      `reshard_after_forward=True`. At world_size=1 the all-gather is a no-op collective
      but PyTorch still pays the flat-param copy and stream sync on every unit on every
      forward: ~9,300 shard/unshard round trips per 77-forward cycle. Expected to be
      numerically identical at world_size=1 (MixedPrecisionPolicy param_dtype=bf16 on an
      already-bf16 model), but that is a claim to VERIFY, not assume.

  --no-empty-cache
      Neuter `torch.cuda.empty_cache()`, which the server calls on every chunk
      (`wan_va_server.py:569`) and every KV update (`:603`). It releases the caching
      allocator back to the driver, forcing cudaMalloc again next cycle.

  --no-debug-dump
      Neuter `save_async`. It is async only for the disk write: `utils.py:63-64` does a
      BLOCKING `.cpu()` of the full latent/action tensors on the critical path, three times
      per cycle, unconditionally, with no upstream flag to disable it.

Usage mirrors the upstream launcher:
    python serve_variant.py --config-name robotwin --port 29061 --save_root /tmp/x --no-fsdp
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Repo root, derived from this file rather than written down. The tree has moved once already
# (/home/ubuntu/InstinctWM -> /home/ubuntu/Code/InstinctWM), and the stale absolute path turned
# --conditioning-prefill into an ImportError while every other variant kept working -- i.e. it
# broke exactly one arm of an A/B comparison. IWM_ROOT still wins when it is set.
IWM_ROOT = os.environ.get("IWM_ROOT") or str(Path(__file__).resolve().parents[2])
if IWM_ROOT not in sys.path:
    sys.path.insert(0, IWM_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="robotwin")
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--save_root", default=None)
    ap.add_argument("--no-fsdp", action="store_true")
    ap.add_argument("--no-empty-cache", action="store_true")
    ap.add_argument("--no-debug-dump", action="store_true")
    ap.add_argument(
        "--conditioning-prefill", action="store_true",
        help="Cache the episode-constant cross-attention K/V for all 30 layers. model.py:331 "
             "withholds attn_caches from cross-attention, so the text K/V is re-projected on "
             "all 77 forwards; that is ~39%% of a control cycle's arithmetic, and 67.6%% of an "
             "action forward's layer FLOPs (32-token query vs 512-token text).")
    ap.add_argument("--hoist-casts", action="store_true",
        help="Cast loop-invariant constants once per episode instead of once per forward: "
             "FP32LayerNorm's weight/bias (4,740 casts of a constant per cycle) and the block's "
             "scale_shift_table, which lets the modulation add promote bf16->fp32 for free and "
             "deletes a 35.4 MB materialization per block.")
    ap.add_argument("--ring-kv", action="store_true",
        help="Address the KV pool by ring interval instead of boolean mask. Removes the "
             "per-layer-per-forward mask.nonzero() host sync and the full-pool advanced-index "
             "gather (model.py:451-453); valid becomes a slice. Falls back to stock when the "
             "interval wraps, so key order stays ascending and the pass stays bit-exact.")
    ap.add_argument("--graph-blocks", action="store_true",
        help="[NOT SHIPPABLE -- 2.17x but NOT bit-exact, max|d action| 1.398 = 136%% of real "
             "movement. The captured region mutates host-side ring bookkeeping that replay never "
             "re-executes. See graph_capture.py. Kept for measurement only.] "
             "Run the 30-block transformer stack from a captured CUDA graph (E3). Requires "
             "--ring-kv: the stock mask.nonzero() is a data-dependent shape and capture of a "
             "stock block fails with cudaErrorStreamCaptureInvalidated. Measured per-op cost is "
             "6.2 us of which 83.6%% is cudaLaunchKernel; replay is ~1.17 us.")
    ap.add_argument("--generic-only", default=None,
        help="comma-separated subset of generic passes to APPLY: pools,hoist,promote,stepidx. "
             "Sites are still enumerated for all of them, so the adapter's rewritable-site shims "
             "are installed either way -- which is what lets shim cost be separated from rewrite "
             "cost.")
    ap.add_argument("--generic-dry-run", action="store_true",
        help="Enumerate every site (installing all shims) and apply NO rewrites. Isolates the "
             "eager tax of exposing a rewritable surface from the effect of rewriting it.")
    ap.add_argument("--degrade-nfe", default=None,
        help="VIDEO,ACTION denoise steps, e.g. '2,2'. Stands in for a distilled student until a "
             "real one exists: it is the same descriptor delta a step-reduction recipe produces "
             "(phases[].nfe), applied to the teacher weights. Used to prove the certification "
             "workflow can REJECT a genuine regression, not just pass synthetic unit tests.")
    ap.add_argument("--legacy-passes", action="store_true",
        help="ORACLE/FALLBACK. Use the backend-specific P004/P006 install() functions instead of "
             "the pass framework. The generic path is the DEFAULT as of 2026-08-02: it measured "
             "1213.6 ms vs legacy 1207.8 ms (within the protocol's resolution, legacy itself "
             "spans 1207.8-1211.3) and is bit-exact across episodes. Kept so the two can be "
             "diffed when a regression is suspected. Passing --hoist-casts or --stable-pools "
             "also selects the legacy path.")
    ap.add_argument("--stable-pools", action="store_true",
        help="E1: reset clears logical KV state in place instead of reallocating the pools, so "
             "captured graphs stay valid across episodes. Only has an effect with --graph-blocks, "
             "which still verifies every pool pointer survived before keeping its graphs.")
    ap.add_argument("--no-keep-graphs", action="store_true",
        help="Drop captured graphs at every reset even when --stable-pools certifies pointer "
             "stability. Preservation is ON by default and gated by that certificate; this is the "
             "escape hatch if a new pass introduces episode-scoped device state the certificate "
             "does not yet cover.")
    ap.add_argument(
        "--fuse-residual", action="store_true",
        help="Fuse the block's two gated residuals into one Triton kernel "
             "(model.py:543-544 and 563-564). 3 eager kernels -> 1. Bit-exact "
             "(enable_fp_fusion=False, asserted on PTX) and 3.0-7.6x on the region under graph "
             "replay -- but MEASURED 0.994x END TO END on the shipped chain (RESULTS.md #10). "
             "Kept opt-in for that reason: the win is real at region and block level and does "
             "not survive into the control cycle.")
    ap.add_argument(
        "--ring-attention", action="store_true",
        help="Make the KV extent a device-resident VALUE instead of a tensor shape, so "
             "(start, count) leaves the CUDA graph capture key. Requires --ring-kv. Measured on "
             "this box: capture costs 172.0 ms, the key never converges (6/cycle, 20 distinct "
             "`count` values in a 10-cycle run), and a capturing run costs ~1020 ms/cycle more "
             "than a warm one. NUMERIC tier: the Triton kernel has its own reduction order.")
    ap.add_argument(
        "--attention-backend", action="store_true",
        help="Pick the SDPA backend by MEASUREMENT on the served shapes instead of letting the "
             "dispatcher's heuristic choose. On this box the dispatcher picks flash and cudnn is "
             "faster on the video shape (0.285 -> 0.235 ms, 1.21x). NUMERIC tier: a different "
             "backend is a different reduction order (max|d| 9.766e-04), so this arm needs a "
             "paired non-inferiority certificate before it can ship.")
    ap.add_argument(
        "--deterministic-seed", type=int, default=None,
        help="Seed torch before each chunk's noise draw. REQUIRED to compare two variants: "
             "_infer draws torch.randn for the initial video latents and action tokens "
             "(wan_va_server.py:449-462) with no seeding, so two stock servers already "
             "disagree and any A/B on output values is meaningless without this.")
    args = ap.parse_args()

    # Every variant below calls the SAME installer that `plan.serve()` calls. They used to be
    # separate inline patches here, which meant an A/B could measure a patch that production
    # never applied.
    from instinctwm.runtime.lingbot_install import (
        import_lingbot_server,
        install_allocator_churn_elision,
        install_conditioning_prefill,
        install_debug_dump_elision,
        install_deterministic_seed,
        install_fsdp_elision,
    )

    S = import_lingbot_server()

    applied = []

    if getattr(args, "degrade_nfe", None):
        v, ac = (int(x) for x in args.degrade_nfe.split(","))
        _orig_run = S.run

        def run_degraded(a_):
            # VA_CONFIGS is the registry wan_va_server resolves --config-name against, so
            # mutating the entry is the same descriptor delta a step-reduction recipe emits.
            for cfg in S.VA_CONFIGS.values():
                if hasattr(cfg, "num_inference_steps"):
                    cfg.num_inference_steps = v
                    cfg.action_num_inference_steps = ac
            return _orig_run(a_)

        S.run = run_degraded
        applied.append(f"degrade-nfe={v},{ac}")

    if args.no_fsdp:
        applied += install_fsdp_elision(S)

    if args.no_empty_cache:
        applied += install_allocator_churn_elision(S)

    if args.no_debug_dump:
        applied += install_debug_dump_elision(S)

    if getattr(args, "conditioning_prefill", False):
        applied += install_conditioning_prefill(S, S.VA_Server)

    if getattr(args, "fuse_residual", False):
        # Installed BEFORE the block-rewriting passes below. Both of those route their residuals
        # through the same hook, so order does not change what runs -- but arming the hook first
        # means the sweep prints before the server's own startup noise.
        from instinctwm.runtime.fused_residual import RESIDUAL
        from instinctwm.runtime.lingbot_install import install_operator_fusion
        # --graph-blocks decides which sweep gates the install, so it is read here rather than
        # defaulted. Under capture the Triton launcher cost is amortised away and the kernel
        # wins at both stream shapes; without capture it only wins on the video stream.
        applied += install_operator_fusion(
            S, S.VA_Server, graph_captured=getattr(args, "graph_blocks", False))

        # Print what the hook actually did, once per reset. Without this, "the fusion is
        # installed" and "the fusion ran" are indistinguishable from outside the process -- and
        # they come apart quietly, because a shape below the break-even or a non-contiguous
        # operand routes back to eager by design. An A/B whose treatment arm never fired looks
        # exactly like an A/B with no effect.
        _orig_reset_fr = S.VA_Server._reset

        def _reset_report(self, prompt=None):
            out = _orig_reset_fr(self, prompt=prompt)
            print(f"[operator_fusion] {RESIDUAL.report()}", flush=True)
            return out

        S.VA_Server._reset = _reset_report

    if getattr(args, "ring_kv", False):
        from instinctwm.optimizer.passes.ring_kv import RingKVAddressing
        RingKVAddressing().install(S, S.VA_Server)
        applied.append("ring-kv")

    if getattr(args, "hoist_casts", False):
        from instinctwm.optimizer.passes.hoist_invariant_casts import HoistInvariantCasts
        HoistInvariantCasts().install(S, S.VA_Server)
        applied.append("hoist-casts")

    if getattr(args, "ring_attention", False):
        # Strictly after --ring-kv: this overrides three of P003's seams (forward, _iwm_commit,
        # _iwm_ring_signature) and cannot be installed before they exist.
        if not getattr(args, "ring_kv", False):
            print("REFUSING: --ring-attention requires --ring-kv. It rides on P003's ring "
                  "bookkeeping; without it there is no interval to make device-resident.",
                  flush=True)
            return 2
        from instinctwm.runtime.ring_attention_install import install_ring_attention
        applied += install_ring_attention(S, S.VA_Server)

    _surface = []            # one-element cell: `global` cannot rebind a local of main()
    _hoist_g = _pools_g = _promote_g = _graph_pass = None
    _use_legacy = (getattr(args, "legacy_passes", False)
                   or getattr(args, "hoist_casts", False)
                   or getattr(args, "stable_pools", False))
    if not _use_legacy:
        from instinctwm.adapter.lingbot import LingBotSurface
        from instinctwm.passes.hoist_invariant import HoistInvariant
        from instinctwm.passes.interface import run_pass
        from instinctwm.passes.explicit_step_index import ExplicitStepIndex
        from instinctwm.passes.promote_small_operand import PromoteSmallOperand
        from instinctwm.passes.stable_pools import StablePools

        _hoist_g, _pools_g = HoistInvariant(), StablePools()
        _promote_g = PromoteSmallOperand()
        _stepidx_g = ExplicitStepIndex()

        # The surface needs a live model, which only exists after the server builds it. Install on
        # the first reset, then let the passes run against the sites the adapter publishes.
        _orig_reset_gp = S.VA_Server._reset

        def _reset_generic(self, prompt=None):
            out = _orig_reset_gp(self, prompt=prompt)
            if not _surface and hasattr(self, "transformer"):
                _surface.append(LingBotSurface(self.transformer, server=self))
                # ORDER MATTERS and is a real dependency: hoist caches the fp32 constant that
                # the promotion rewrite then reuses instead of re-casting per call.
                named = (("pools", _pools_g), ("hoist", _hoist_g),
                         ("promote", _promote_g), ("stepidx", _stepidx_g))
                sel = (None if not getattr(args, "generic_only", None)
                       else {x.strip() for x in args.generic_only.split(",")})
                from instinctwm.passes.interface import SiteKind
                if getattr(args, "generic_dry_run", False):
                    # touch every site kind so all shims install, then apply nothing
                    for k in SiteKind:
                        list(_surface[0].sites(k))
                    print("[generic_passes] DRY RUN: all shims installed, 0 rewrites applied",
                          flush=True)
                else:
                    for nm, p_ in named:
                        if sel is not None and nm not in sel:
                            # still enumerate, so this pass's shim is installed and only its
                            # REWRITE is withheld
                            for k in p_.sites_required():
                                list(_surface[0].sites(k))
                            print(f"[generic_passes] {nm}: sites enumerated, rewrite withheld",
                                  flush=True)
                            continue
                        print(f"[generic_passes] {run_pass(p_, _surface[0], None)}", flush=True)
                        for d in getattr(p_, "declines", [])[:3]:
                            print(f"[generic_passes]   decline {d}", flush=True)
                # Seed the certificate from the buffers that exist RIGHT NOW, not from the first
                # wrapped allocation. The passes install at the end of reset 1, so this episode's
                # buffers pre-date stabilization; anything captured against them becomes stale at
                # reset 2. Seeding here makes the check notice that once -- graphs drop, recapture
                # against stable storage, and survive every reset after. Recording pointers at the
                # first wrapped call instead reported "stable" while the graphs were already stale,
                # which cost max|delta action| = 1.527.
                # Baseline is NOT set here: at this point the wrappers are installed but no
                # wrapped allocation has happened yet, so the buffers in play still pre-date
                # stabilization. `pending` stays non-empty until they are replaced, and
                # `pointers_stable` refuses until someone re-certifies.
            return out

        S.VA_Server._reset = _reset_generic
        applied.append("generic-passes(default)")

    _pools_pass = None
    if getattr(args, "stable_pools", False):
        from instinctwm.optimizer.passes.stable_pools import StableStatePools
        _pools_pass = StableStatePools()
        _pools_pass.install(S, S.VA_Server)
        applied.append("stable-pools")

    if getattr(args, "graph_blocks", False):
        if not getattr(args, "ring_kv", False):
            print("REFUSING: --graph-blocks requires --ring-kv. The stock KV path calls "
                  "mask.nonzero() per layer per forward, which is a data-dependent shape; "
                  "capture fails with cudaErrorStreamCaptureInvalidated.", flush=True)
            return 2
        from instinctwm.optimizer.passes.graph_capture import GraphBlockStack
        # IWM_MAX_GRAPHS exists because the default cap of 64 is not survivable on every run:
        # a 50-task certification OOMed at held=64 evicted=400 on an 80 GB A100, since evicted
        # graphs do not give their private memory pools back. Lower it to bound the pool. It is
        # a knob, not a new default -- 64 is the measured working set and going under it trades
        # memory for recapture (FIFO at 24 once measured 214 captures instead of 60).
        _graph_pass = GraphBlockStack(          # noqa: F841 -- rebinds the cell declared above
            max_graphs=int(os.environ.get("IWM_MAX_GRAPHS", "64")))
        _graph_pass.install(S, S.VA_Server)
        applied.append("graph-blocks")

        if _pools_g is not None:
            def _generic_stability():
                if not _surface:
                    return False, "surface not built yet"
                tracked = dict(_surface[0].pools())
                tracked.update(_hoist_g.hoisted_values())   # the 60 formerly-anonymous buffers
                ok, why = _pools_g.pointers_stable(tracked)
                if not ok and not _pools_g.pointers:
                    _pools_g.set_baseline(tracked)          # first certification
                    return False, "baseline established; graphs dropped once"
                if not ok and _pools_g.pending:
                    _pools_g.set_baseline(tracked)          # storage moved; re-certify from here
                    return False, why
                return ok, why

            _graph_pass.stability_check = _generic_stability

            # ROOT CAUSE, found by diffing derived dependency signatures across a reset
            # (deps.py) rather than by inspection:
            #
            #   legacy   reads=873 unnamed=0   buffers MOVED across reset: 3 (test inputs only)
            #   generic  reads=873 unnamed=60  buffers MOVED across reset: 210
            #                                    kv        147
            #                                    cross_kv   60
            #
            # Two defects, both now fixed:
            #  1. StablePools recorded its baseline INSIDE the allocation path, so a reset that
            #     reallocated refreshed the reference before anything compared against it --
            #     fail-open. `set_baseline` is now the only door, and `pending` makes the check
            #     refuse until storage is re-certified.
            #  2. The generic hoist kept its caches in closures, invisible to build_name_map, so
            #     60 of the region's reads were ANONYMOUS and no check could cover them.
            #     `hoisted_values()` exposes them and they are tracked here.
            #
            # Both are the same failure class as the two that preceded them: a certificate that
            # covers less than the region reads. The tracer is what makes that measurable.

        if _pools_pass is not None:
            # Bind the pools on first use, then let the graph pass consult them at every reset.
            _orig_reset_bind = S.VA_Server._reset

            def _reset_bind(self, prompt=None, _p=_pools_pass, _o=_orig_reset_bind):
                out = _o(self, prompt=prompt)
                if hasattr(self, "transformer"):
                    _p.bind(self.transformer)
                return out

            S.VA_Server._reset = _reset_bind
            _graph_pass.bind_hook = _pools_pass.bind
            if not getattr(args, "no_keep_graphs", False):
                _graph_pass.stability_check = lambda: _pools_pass.pointers_stable()

        # Report capture/replay counts at the end of each chunk so the recapture rate is visible
        # rather than inferred -- if the key churns, the win evaporates and we need to know.
        _orig_infer_g = S.VA_Server._infer

        def _infer_reporting(self, obs, frame_st_id=0):
            out = _orig_infer_g(self, obs, frame_st_id=frame_st_id)
            extra = ""
            if _pools_pass:
                extra = f" | {_pools_pass.stats()}"
            elif _pools_g:
                extra = (f" | generic {_pools_g.stats()} | {_hoist_g.stats()}"
                         f" | {_promote_g.stats()} | {_stepidx_g.stats()}")
            print(f"[graph_block_stack] {_graph_pass.stats()}{extra}", flush=True)
            return out

        S.VA_Server._infer = _infer_reporting

    if getattr(args, "attention_backend", False):
        # Armed after the FIRST _infer, not after the first _reset, and that is forced by the
        # pass rather than chosen here: it measures on the site's own query extent and REFUSES
        # to guess one, so with no forward yet observed it declines with "site has not observed
        # a query extent". The first cycle therefore runs stock and is not part of the arm.
        from instinctwm.adapter.lingbot import LingBotSurface as _LBS
        from instinctwm.passes.attention_backend import AttentionBackend
        from instinctwm.passes.interface import SiteKind as _SiteKind, run_pass as _run_pass

        _attn_g = AttentionBackend(min_speedup=1.05, verbose=True)
        _armed: list = []

        # ENUMERATE at reset, RUN after the first infer. Splitting the two is not tidiness, it
        # is required: enumerating is what installs the adapter's recording wrapper, and the
        # wrapper is what supplies the query extent the pass refuses to guess. Doing both after
        # the first infer -- the first version of this -- installed the wrapper too late to have
        # observed that infer, so the pass declined on all 30 layers with "no observed query
        # extent" and the arm silently served the baseline. Same shape as the `--generic-only`
        # path, which also enumerates sites it will not rewrite.
        _orig_reset_a = S.VA_Server._reset

        def _reset_attn(self, prompt=None):
            out = _orig_reset_a(self, prompt=prompt)
            if not _surface and hasattr(self, "transformer"):
                _surface.append(_LBS(self.transformer, server=self))
            if _surface:
                list(_surface[0].sites(_SiteKind.ATTENTION_OP))
            return out

        S.VA_Server._reset = _reset_attn
        _orig_infer_a = S.VA_Server._infer

        def _infer_attn(self, obs, frame_st_id=0):
            out = _orig_infer_a(self, obs, frame_st_id=frame_st_id)
            if not _armed:
                _armed.append(True)
                if not _surface:
                    _surface.append(_LBS(self.transformer, server=self))
                print(f"[attention_backend] {_run_pass(_attn_g, _surface[0], None)}", flush=True)
                for line in _attn_g.report().splitlines():
                    print(f"[attention_backend] {line}", flush=True)
                if _graph_pass is not None:
                    # Every graph captured during that first cycle baked in the OLD attention
                    # kernel. Replaying them would keep running it while `explain()` claimed the
                    # swap had happened -- the exact silent-staleness failure this file's other
                    # passes are gated against.
                    _graph_pass.drop_graphs("attention_backend swapped the attention op")
            return out

        S.VA_Server._infer = _infer_attn
        applied.append("attention-backend")

    if args.deterministic_seed is not None:
        applied += install_deterministic_seed(S, args.deterministic_seed)

    print("=" * 72, flush=True)
    print(f"InstinctWM serve_variant: {applied if applied else ['STOCK BASELINE']}", flush=True)
    print(f"  ckpt   : {os.environ.get('LINGBOT_CKPT')}", flush=True)
    print(f"  config : {args.config_name}   port: {args.port}", flush=True)
    print("=" * 72, flush=True)

    class _A:
        pass

    a = _A()
    a.config_name = args.config_name
    a.port = args.port
    a.save_root = args.save_root

    S.init_logger()
    S.run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
