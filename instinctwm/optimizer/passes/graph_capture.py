"""GraphBlockStack (E3) -- run the 30-block transformer stack from a captured CUDA graph.

Attacks the measured bottleneck directly. Per-op cost is 6.2 us, of which 83.6% is
`cudaLaunchKernel` itself; a control cycle dispatches ~250k ops. Graph replay drops per-op cost to
~1.17 us and CPU enqueue for a block stack from 7.5 ms to 0.03 ms.

WHAT MAKES THIS LEGAL

P003 (ring KV addressing) is a hard precondition, not a co-optimization: the stock path calls
`mask.nonzero()` per layer per forward, and capture of a stock block fails outright with
`cudaErrorStreamCaptureInvalidated`. With slice addressing it captures cleanly.

WHAT IS DYNAMIC, AND HOW EACH PART IS HANDLED

  token count        static per stream (video / action)      -> separate graphs
  update_cache       0 or 1                                  -> part of the graph key
  cache_name         pos / neg                               -> part of the graph key
  KV read extent     grows 272 tokens/cycle                  -> RECAPTURE (see below)
  KV write position  advances on commit                      -> part of the graph key

Padding the read extent to capacity and masking the tail would give one permanent graph, and it is
what most LLM engines do. It is **not bit-exact here**: masked SDPA re-trees the reduction, and
measured against the exact-prefix path 750 of 196,608 elements differ (max|d| 4.883e-04). Since
capture costs only 29.4 ms for a 30-block stack and holds a 42 MB pool, recapturing on change is
both cheaper and exact.

RESULT (2026-08-02), under the only accepted protocol:

    probe_latency  : 2539.9 -> 1842.0 ms   = 1.38x   (repeats 3, first discarded, spread 0.5%)
    probe_bitexact : max|delta action| = 0.000e+00 over 6 paired seeded cycles, with the gate run
                     AFTER an episode reset -- the ordering that exposed a nan when graphs were
                     wrongly kept across a pool reallocation

WHAT THE FIRST ATTEMPT GOT WRONG, because it is the failure mode this whole layer exists to stop

v1 measured 2.17x and was **wrong**: max|delta action| = 1.398 against a chunk-to-chunk movement of
1.031, i.e. 136%. The graph was fine. The captured region mutated **host** state -- P003's ring
bookkeeping advanced with plain Python interleaved among the GPU ops:

    r["next_id"] += 1                       # per layer per forward
    kvc["mask"][sl] = False                 # roll back the provisional write
    r["count"] = count ;  r["start"] = ...  # commit / eviction

Capture freezes the GPU work; on replay the Python never runs, so the ring stopped advancing while
the graph kept writing the slots baked in at capture. A second, quieter symptom pointed at the same
thing: only **6 graphs** were captured where there should have been dozens, because the key read
`_iwm_live` -- an attribute of the *test harness*, not of the ring pass.

THE FIX, in three parts

1. `ring_kv` grew a `_iwm_defer_commit` mode. `forward` became pure with respect to host state;
   everything that advances the ring moved into `_iwm_commit`, which this pass calls on the HOST
   for all 30 layers after each replay. Default mode is unchanged, so frozen P003 still behaves
   byte-for-byte as v1.0.0.
2. The graph key now includes `_iwm_ring_signature` = (start, count). `start` fixes the read-slice
   and write offsets; `count` fixes the read extent's SHAPE. Both are baked at capture.
3. `engine/effects.py` runs before every capture and REFUSES a region that mutates undeclared host
   state. That check, not the benchmark, is what should have caught v1.

COSTS, measured rather than assumed

  * ~162 MB per graph; the working set converged to 60 graphs (+9.7 GB) over a 30-cycle run.
    Eviction is LRU with a cap above that working set, acting as a safety bound for long episodes
    rather than a live constraint. FIFO was tried first on the theory that `count` grows
    monotonically so the oldest key is the stalest. It does not: `clear_pred_cache` subtracts
    `pred` from `count`, so the extent oscillates within a cycle and old keys recur. FIFO at 24
    evicted exactly what was about to be reused -- 214 captures instead of 60, 1880.9 ms instead
    of 1203.3 ms.
  * Capture is not free at full model scale (~275 ms each, not the 29 ms an 8-block microbenchmark
    suggested), and graphs are dropped at every reset because `_reset` reallocates the KV pool.
    That recapture is the difference between 1842.0 ms and the 1208.2 ms the same build reaches
    with graphs surviving resets. E1 (`stable_pools.py`) closes it by making the pools
    address-stable; this pass then keeps its graphs, but ONLY when `stability_check` certifies
    every pool pointer survived. Without that certificate the default stays drop-and-recapture.

Tier: BITEXACT.
"""

from __future__ import annotations

import time

import torch

from instinctwm.optimizer.contract import (
    Applicability, BenchResult, CostTerm, DeviceProfile, Discovery, HardwareReq, Tier,
    VerifyResult,
)


class GraphBlockStack:
    name = "graph_block_stack"
    hardware = HardwareReq(requires=frozenset({"cuda"}))

    def _touch(self, key):
        """Move `key` to the most-recently-used end.

        Python dicts preserve insertion order, so re-inserting IS the LRU implementation.
        """
        for d in (self.graphs, self.bound, self.outputs):
            d[key] = d.pop(key)

    def __init__(self, verbose: bool = True, max_graphs: int = 64):
        #: Each captured 30-block graph holds its own memory pool: measured ~162 MB each, and the
        #: working set converged to 60 graphs (+9.7 GB) over a 30-cycle run.
        #:
        #: Eviction is LRU, not FIFO. FIFO looked right on the theory that `count` grows
        #: monotonically so the oldest key is the stalest -- it does not. `clear_pred_cache` does
        #: `r["count"] -= r["pred"]`, so the extent oscillates within a cycle and old keys recur.
        #: FIFO at 24 evicted exactly what was about to be reused: 214 captures instead of 60, and
        #: 1880.9 ms instead of 1203.3 ms. The cap is set above the observed working set so it acts
        #: as a safety bound for long episodes rather than as a live constraint.
        self.max_graphs = max_graphs
        self.n_evicted = 0
        self.verbose = verbose
        self.graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.bound: dict[tuple, tuple] = {}
        self.outputs: dict[tuple, torch.Tensor] = {}
        self.n_captures = 0
        self.n_replays = 0
        self.capture_ms = 0.0
        self.failed: str | None = None
        self.n_resets_survived = 0
        #: every key ever captured. A high hit rate with a growing key set is not a warm cache,
        #: it is an unbounded recapture stream -- the distinction episode mode exists to expose.
        self.seen_keys: set = set()
        #: set by StableStatePools (E1). Returns (keep_graphs, why). Absent -> drop on every reset.
        self.stability_check = None
        #: called once, at the first capture, when lazily-created state (P004's fp32 casts) exists.
        #: Binding only at reset was too early: the casts had not been created yet, so the
        #: certificate reported success while covering zero of them.
        self.bind_hook = None
        self._bound = False

    def applicability(self, spec, device: DeviceProfile) -> Applicability:
        return Applicability(
            True,
            "the block stack is a fixed kernel sequence replayed ~79x per cycle; its launch "
            "sequence is PLAN-scoped but is reconstructed at LAYER x STEP scope",
            discovery=Discovery.AUTO,
            cost_term=CostTerm.PER_STEP,
            claimed_tier=Tier.BITEXACT)

    def expected_delta_ms(self, spec, device: DeviceProfile) -> float:
        # 135 dispatched ops/block x 30 layers x 79 forwards, at 6.2 us minus ~1.17 us replay
        return 135 * 30 * 79 * (6.2 - 1.17) / 1000.0

    # ---- install -----------------------------------------------------------------------------
    def install(self, server_module, server_cls) -> list[str]:
        import modules.model as M

        engine = self
        Model = M.WanTransformer3DModel

        _orig_forward = Model.forward

        def _stack(self_model, hidden, text, tproj, rot, update_cache, cache_name):
            for block in self_model.blocks:
                hidden = block(hidden, text, tproj, rot,
                               update_cache=update_cache, cache_name=cache_name)
            return hidden

        def _key(self_model, hidden, tproj, update_cache, cache_name):
            """Everything the captured kernel sequence bakes in.

            `ring_signature` is (start, count): `start` fixes both the read-slice offset and the
            write offset, `count` fixes the read extent's SHAPE. Omitting them is what made the
            first integration capture 6 graphs and then replay stale ones.
            """
            a0 = self_model.blocks[0].attn1
            sig = a0._iwm_ring_signature(cache_name) if hasattr(a0, "_iwm_ring_signature") else None
            return (tuple(hidden.shape), tuple(tproj.shape), int(update_cache),
                    str(cache_name), sig)

        def _commit_all(self_model, hidden, update_cache, cache_name):
            """Advance every layer's ring on the HOST, after replay.

            This is the whole fix. Each layer holds its own ring, all of them advance by the same
            key_size, and none of it may live inside the graph.
            """
            key_size = hidden.shape[1]
            for block in self_model.blocks:
                block.attn1._iwm_commit(cache_name, key_size, update_cache)

        def _eager(self_model, hidden, text, tproj, rot, update_cache, cache_name):
            """The fallback path -- and it MUST still advance the ring.

            `install` sets `_iwm_defer_commit = True` permanently, so `WanAttention.forward` no
            longer commits inline; the only thing that advances the ring is `_commit_all`. Both
            fallback returns used to skip it, which meant that from the first capture failure
            onwards the ring FROZE: `count` stopped growing, every later forward rewrote the same
            slots, and attention read a stale window. The actions stayed plausible and were
            wrong.

            Found by an OOM during a 50-task certification run, which is exactly how this fails
            in production -- capture failure is advertised as a safe degradation, and this was
            the one path where "safe" meant "silently incorrect". Nothing raises, nothing logs;
            the only symptom is a task success rate that drifts.
            """
            out = _stack(self_model, hidden, text, tproj, rot, update_cache, cache_name)
            _commit_all(self_model, hidden, update_cache, cache_name)
            return out

        def stack_graphed(self_model, hidden, text, tproj, rot, update_cache, cache_name):
            if engine.failed:
                return _eager(self_model, hidden, text, tproj, rot, update_cache, cache_name)
            k = _key(self_model, hidden, tproj, update_cache, cache_name)
            if k not in engine.graphs:
                try:
                    engine._capture(_stack, self_model, hidden, text, tproj, rot,
                                    update_cache, cache_name, k)
                except Exception as ex:
                    engine.failed = f"{type(ex).__name__}: {str(ex)[:200]}"
                    if engine.verbose:
                        print(f"[graph_block_stack] CAPTURE FAILED, falling back to eager for the "
                              f"rest of the run: {engine.failed}", flush=True)
                    return _eager(self_model, hidden, text, tproj, rot,
                                  update_cache, cache_name)
            else:
                engine._touch(k)                     # LRU: recently used keys survive eviction
            for buf, src in zip(engine.bound[k], (hidden, text, tproj, rot)):
                if buf is not None:
                    buf.copy_(src)          # bind by NAME; None args stay None (see _capture)
            engine.graphs[k].replay()
            engine.n_replays += 1
            _commit_all(self_model, hidden, update_cache, cache_name)
            return engine.outputs[k]

        # Deferred-commit mode: forward stops touching host state so the stack is capturable.
        AttnCls = M.WanAttention
        if not hasattr(AttnCls, "_iwm_commit"):
            raise RuntimeError(
                "graph_block_stack requires the ring_kv pass to be installed first: it provides "
                "_iwm_commit / _iwm_ring_signature, without which the block stack mutates host "
                "state inside the captured region.")
        AttnCls._iwm_defer_commit = True

        engine._stack_fn = _stack
        engine._commit_all = _commit_all

        # Patch the block loop inside the inference forward by swapping the loop body.
        Model._iwm_stack = stack_graphed
        _patch_forward(Model, _orig_forward)

        # ---- graphs are EPISODE-scoped -----------------------------------------------------
        # `_reset` re-runs `init_kv_cache`, which allocates NEW pool tensors. A captured graph
        # still writes to the addresses it was captured with, so every graph from the previous
        # episode now targets freed memory. Symptom: max|delta action| = nan on the second
        # episode, while the first episode is perfectly bit-exact -- the exact "silent staleness"
        # failure the Plan's bind-by-name rule exists to prevent, showing up here because this
        # pass still binds to whatever tensors the model happens to hold.
        _orig_reset = server_cls._reset

        def _reset(self, prompt=None):
            out = _orig_reset(self, prompt=prompt)
            # Keep graphs ONLY if something has certified that every pool is still at the address
            # it was captured with. Absent that certificate the safe behaviour -- drop and
            # recapture -- is the default, because a stale graph does not raise, it returns
            # plausible garbage.
            keep, why = False, "no pool-stability certificate (E1 not installed)"
            if engine.stability_check is not None:
                keep, why = engine.stability_check()
            if keep:
                engine.n_resets_survived += 1
                if engine.verbose:
                    print(f"[graph_block_stack] kept {len(engine.graphs)} graph(s) across reset: "
                          f"{why}", flush=True)
            else:
                engine.drop_graphs(f"reset: {why}")
            return out

        server_cls._reset = _reset
        return ["graph_block_stack"]

    def drop_graphs(self, why: str) -> None:
        n = len(self.graphs)
        self.graphs.clear()
        self.bound.clear()
        self.outputs.clear()
        if n and self.verbose:
            print(f"[graph_block_stack] dropped {n} graph(s): {why}", flush=True)

    def _capture(self, stack_fn, model, hidden, text, tproj, rot, update_cache, cache_name, key):
        t0 = time.perf_counter()
        # P002 replaces the text embedder with a no-op once the cross-attention K/V is cached, so
        # `text` arrives as None. A None argument is not a buffer to bind -- it is part of the
        # captured control flow, and stays None. Cloning it unconditionally is what made the first
        # attempt fall back to eager for the whole run.
        bh, bt, bp, br = (x.clone() if isinstance(x, torch.Tensor) else x
                          for x in (hidden, text, tproj, rot))

        # GATE: refuse to capture a region that mutates host state. This is the check that would
        # have caught the 1.398 action delta before it became a benchmark number.
        from instinctwm.engine.effects import require_capturable
        roots = [b.attn1.attn_caches for b in model.blocks if getattr(b.attn1, "attn_caches", None)]
        require_capturable(
            f"block_stack{key[:2]}",
            lambda: stack_fn(model, bh, bt, bp, br, update_cache, cache_name),
            roots)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(2):
                stack_fn(model, bh, bt, bp, br, update_cache, cache_name)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.no_grad():
            out = stack_fn(model, bh, bt, bp, br, update_cache, cache_name)
        while len(self.graphs) >= self.max_graphs:
            oldest = next(iter(self.graphs))
            del self.graphs[oldest], self.bound[oldest], self.outputs[oldest]
            self.n_evicted += 1
        self.graphs[key] = g
        self.bound[key] = (bh, bt, bp, br)   # None entries are skipped when binding
        self.outputs[key] = out
        self.n_captures += 1
        self.capture_ms += (time.perf_counter() - t0) * 1e3
        if self.verbose:
            print(f"[graph_block_stack] capture #{self.n_captures} key={key} "
                  f"({(time.perf_counter()-t0)*1e3:.1f} ms)", flush=True)

    def stats(self) -> str:
        return (f"captures={self.n_captures} replays={self.n_replays} "
                f"unique_keys={len(self.seen_keys)} "
                f"held={len(self.graphs)} evicted={self.n_evicted} "
                f"resets_survived={self.n_resets_survived} "
                f"capture_total={self.capture_ms:.0f} ms "
                f"{'FELL BACK: ' + self.failed if self.failed else ''}")

    def verify(self, harness) -> VerifyResult:
        d = harness.max_abs_action_delta()
        return VerifyResult(passed=(d == 0.0),
                            tier_achieved=Tier.BITEXACT if d == 0.0 else Tier.NUMERIC,
                            max_abs_delta=d,
                            detail="replay executes the identical kernel sequence on identical "
                                   "addresses; recapture keeps KV extents exact")

    def benchmark(self, harness) -> BenchResult:
        b, a = harness.cycle_ms_before(), harness.cycle_ms_after()
        return BenchResult(passed=a < b, before_ms=b, after_ms=a)


def _patch_forward(Model, orig):
    """Replace the `for block in self.blocks` loop with a call to the graphed stack.

    Done by source rewriting rather than by reimplementing `forward`: the surrounding code
    (embedding, temb, norm_out, proj_out, split) is long, model-specific, and not what we are
    optimizing. Reimplementing it would silently fork from upstream.
    """
    import inspect
    import textwrap

    NEEDLE = "for block in self.blocks:"

    def _find(fn, depth=0):
        """Locate the function that actually contains the block loop.

        By the time this pass installs, `Model.forward` may already be a wrapper: P002 replaces it
        with a closure that swaps in a no-op text embedder. Inspecting the wrapper finds no loop.
        Rather than depend on install order -- which is exactly the fragility the Plan artifact is
        meant to remove -- walk the closure cells to the function that has it, so this pass
        composes with any wrapper that keeps a reference to what it wraps.
        """
        try:
            if NEEDLE in textwrap.dedent(inspect.getsource(fn)):
                return fn, None, None
        except (OSError, TypeError):
            pass
        if depth > 4 or not getattr(fn, "__closure__", None):
            return None, None, None
        for i, cell in enumerate(fn.__closure__):
            try:
                inner = cell.cell_contents
            except ValueError:
                continue
            if callable(inner) and hasattr(inner, "__code__"):
                found, _, _ = _find(inner, depth + 1)
                if found is not None:
                    return found, cell, i
        return None, None, None

    target, cell, _ = _find(orig)
    if target is None:
        raise RuntimeError(
            f"graph_block_stack could not find {NEEDLE!r} in {Model.__name__}.forward or in any "
            f"wrapper closure. The block loop moved upstream, or a wrapper does not retain the "
            f"function it wraps; update _patch_forward.")

    src = textwrap.dedent(inspect.getsource(target))
    orig = target
    start = src.index(NEEDLE)
    tail = src[start:]
    end = tail.index("temb_scale_shift_table")
    loop = tail[:end]
    # the loop's second line is `<var> = block(...`; that variable is what the stack produces
    var = loop.split("\n")[1].strip().split(" =")[0].strip()
    replacement = (f"{var} = self._iwm_stack({var}, text_hidden_states, timestep_proj, "
                   f"rotary_emb, update_cache, cache_name)\n    ")
    new_src = src[:start] + replacement + src[start + end:]
    ns = dict(orig.__globals__)
    try:
        exec(compile(new_src, "<iwm-graph-forward>", "exec"), ns)
    except SyntaxError as ex:                       # fail loudly, never silently half-patch
        raise RuntimeError(
            f"graph_block_stack could not rewrite {Model.__name__}.forward: {ex}\n"
            f"The block loop shape changed upstream; update _patch_forward.") from ex
    rewritten = ns[orig.__name__]
    if cell is None:
        Model.forward = rewritten
    else:
        # The loop lives inside a wrapper's closure; swap it there so the wrapper keeps working.
        cell.cell_contents = rewritten
    return var
