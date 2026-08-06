"""Backend Adapter helpers for LingBot-VA.

Only the parts the generic engine needs from the adapter side. The optimization passes are still
model-specific `install()` functions elsewhere; that is a known gap, not this file's job.
"""

from __future__ import annotations


def state_roots(model) -> dict:
    """Named roots the engine walks to name device buffers.

    The engine used to hard-code this schema. Moving it here is what lets the same tracer name
    Cosmos3's state without knowing what a `SequencePack` is.
    """
    # Specific roots FIRST: `build_name_map` keeps the first name it sees, and a generic walk from
    # the model reaches the same tensors by an unhelpful path
    # (`model.blocks[0]._modules.attn1.attn_caches.pos.k`). Order is the whole mechanism.
    roots: dict = {}
    for i, blk in enumerate(getattr(model, "blocks", []) or []):
        for attr in ("attn1", "attn2"):
            a = getattr(blk, attr, None)
            if a is None:
                continue
            if getattr(a, "attn_caches", None):
                roots[f"kv[{i}].{attr}"] = a.attn_caches
            kv = getattr(a, "_iwm_cross_kv", None)
            if kv is not None:
                roots[f"cross_kv[{i}].{attr}"] = kv
        # P004's hoisted fp32 parameter views, which live on the modules themselves
        for name, mod in blk.named_modules():
            for cache in ("_iwm_w32", "_iwm_b32", "_iwm_sst32"):
                t = getattr(mod, cache, None)
                if t is not None:
                    roots[f"hoisted[{i}].{name or 'block'}.{cache}"] = t
    roots["model"] = model        # catch-all, deliberately last
    return roots


# =================================================================================================
# AdapterSurface: LingBot-VA published as sites.
# =================================================================================================

class LingBotSurface:
    """Site publisher for LingBot-VA.

    Everything model-specific about capture lives HERE now: which callable is the unit, what its
    arguments are, what makes its structural signature change (the ring interval), and which host
    state must be advanced outside the captured region. The pass reads none of it by name.
    """

    model_id = "lingbot-va"

    def __init__(self, model, cache_name="pos", server=None):
        self.model = model
        self._server_arg = server
        self.cache_name = cache_name
        self._wrapped = {}
        self._cast_sites: dict = {}
        self._alloc_sites: dict = {}
        self._cross_sites: dict = {}
        self._mod_sites: dict = {}
        self._sched_sites: dict = {}
        self._attn_sites: dict = {}
        self._server = server

    # -- WHERE ------------------------------------------------------------------------------
    def sites(self, kind):
        from instinctwm.passes.interface import Scope, Site, SiteKind

        blocks = getattr(self.model, "blocks", []) or []
        if kind is SiteKind.CAPTURE_UNIT:
            a0 = blocks[0].attn1 if blocks else None
            deferred = bool(getattr(type(a0), "_iwm_defer_commit", False)) if a0 else False
            yield Site(
                kind=kind, id="lingbot.block_stack",
                attrs={
                    # Capture is only sound once the ring bookkeeping has been deferred out of the
                    # region. The ADAPTER knows that; the pass just reads the flag.
                    "capturable": deferred,
                    "effect_roots": tuple(b.attn1.attn_caches for b in blocks
                                          if getattr(b.attn1, "attn_caches", None)),
                    "arity": 4,
                    # the ring interval is baked into every captured graph
                    "extent_fn": self._extent,
                    "note": "argument 0 is hidden states; 1 encoder; 2 timestep_proj; 3 rope",
                })
        elif kind is SiteKind.EXECUTION_REGION:
            for i, _b in enumerate(blocks):
                yield Site(kind=kind, id=f"lingbot.block[{i}]", attrs={"index": i})
        elif kind is SiteKind.STATE_ADDRESSING:
            # The solver's step index. Rediscovered per step by searching the schedule for the
            # timestep value, at the cost of 3 host round trips -- while the caller is in
            # `for i, t in enumerate(timesteps)` and already holds i.
            for name, sched in self._schedulers().items():
                self._install_step_shim(type(sched))
                sid = f"lingbot.step_index[{name}]"
                self._sched_sites[sid] = sched
                yield Site(kind=kind, id=sid,
                           attrs={"addressing": "index_by_search",
                                  "monotonic_calls": True,     # one step() per timestep, in order
                                  "syncs_per_call": 3,
                                  "scope": Scope.PLAN, "evaluated_at": Scope.STEP,
                                  "reset_hook": self._reset_hook(sched),
                                  "note": "timestep.cpu() + argmin(table) + .item() comparison"})
            for i, b in enumerate(blocks):
                r = (getattr(b.attn1, "attn_caches", None) or {}).get(self.cache_name, {})
                if isinstance(r, dict) and "_ring" in r:
                    yield Site(kind=kind, id=f"lingbot.kv_ring[{i}]",
                               attrs={"addressing": "ring_interval",
                                      "scope": Scope.CYCLE, "evaluated_at": Scope.LAYER})
        elif kind is SiteKind.ATTENTION_OP:
            # `attn_op` is an INSTANCE attribute assigned in WanAttention.__init__
            # (model.py:302-303 picks custom_sdpa for attn_mode='torch'), so it is directly
            # replaceable per layer without touching the class.
            self._install_attn_shim()
            for i, b in enumerate(blocks):
                a = b.attn1
                op = getattr(a, "attn_op", None)
                if op is None:
                    continue
                sid = f"lingbot.attention[{i}]"
                self._attn_sites[sid] = a
                pool = (getattr(a, "attn_caches", None) or {}).get(self.cache_name) or {}
                k = pool.get("k")
                ring = pool.get("_ring")
                yield Site(
                    kind=kind, id=sid,
                    attrs={
                        "op": op,
                        "heads": a.heads,
                        "head_dim": (k.shape[3] if k is not None
                                     else a.inner_dim // a.heads),
                        "dtype": k.dtype if k is not None else None,
                        "layout": "bshd",
                        # custom_sdpa passes NO mask and NO is_causal (model.py:37-40). A pass
                        # that would only be legal for an unmasked call needs to know that, and
                        # must not have to read the model to find out.
                        "masked": False,
                        "capacity": (k.shape[1] if k is not None else None),
                        # Observed at trace time. The two phases have different query extents
                        # (240 video / 32 action rows at batch 2) and a pass measuring on the
                        # wrong one measures the wrong thing.
                        "q_rows": getattr(a, "_iwm_last_q_rows", None),
                        "batch": getattr(a, "_iwm_last_batch", None),
                        # HOW the live set reaches the op today. "sliced" means its extent is in
                        # a tensor shape, which is what lands in a capture key.
                        "extent_binding": "sliced",
                        "extent": (lambda _r=ring: (_r["start"], _r["count"])) if ring else None,
                        "pool": (lambda _p=pool: (_p.get("k"), _p.get("v"))) if k is not None
                                else None,
                        "note": "self-attention over the ring KV pool; model.py:444-455",
                    })

        elif kind is SiteKind.ALLOCATION:
            self._install_alloc_shim()
            for i, b in enumerate(blocks):
                a = b.attn1
                shape = getattr(a, "_iwm_pool_shape", None)
                if shape is None:
                    continue
                sid = f"lingbot.kv_pool[{i}].{self.cache_name}"
                self._alloc_sites[sid] = a
                yield Site(
                    kind=kind, id=sid,
                    attrs={
                        # storage is MODEL-scoped: shape depends only on config, never on the
                        # episode. Contents are EPISODE-scoped. The model conflates them and
                        # reallocates the pool on every reset, which invalidates any captured graph.
                        "physical_lifetime": Scope.MODEL,
                        "logical_reset": Scope.EPISODE,
                        "evaluated_at": Scope.EPISODE,
                        "extent": shape,
                        "ownership": f"attn1[{i}]",
                        "allocate": self._pool_allocator(a),
                        # WHAT "logically clear" MEANS is the adapter's to declare: mask/id/is_pred
                        # are reset because clear_pred_cache and the stock path read them; k/v are
                        # deliberately NOT zeroed (3.5 GB of writes) because the ring live set is
                        # empty after a reset so they are unreachable.
                        "clear": self._pool_clear,
                    })

            # P002's cross-attention K/V. Same physical lifetime, DIFFERENT reset semantics: the
            # text changes every episode, so the contents are recomputed -- but they must land in
            # the same storage or every captured graph reading them goes stale.
            if hasattr(type(self.model), "populate_cross_cache"):
                self._install_cross_shim()
                for i, b in enumerate(blocks):
                    kv = getattr(b.attn2, "_iwm_cross_kv", None)
                    if kv is None:
                        continue
                    sid = f"lingbot.cross_kv[{i}]"
                    self._cross_sites[sid] = b.attn2
                    yield Site(
                        kind=kind, id=sid,
                        attrs={"physical_lifetime": Scope.MODEL,
                               "logical_reset": Scope.EPISODE,
                               "evaluated_at": Scope.EPISODE,
                               "extent": tuple(tuple(t.shape) for t in kv),
                               "ownership": f"attn2[{i}]",
                               "allocate": self._cross_allocator(b.attn2),
                               "copy_into": self._cross_copy})

        elif kind is SiteKind.DTYPE_PROMOTION:
            import torch
            self._install_modulate_shim()
            for i, b in enumerate(blocks):
                tbl = getattr(b, "scale_shift_table", None)
                if tbl is None:
                    continue
                sid = f"lingbot.modulation[{i}]"
                self._mod_sites[sid] = b
                n_tok = getattr(b, "_iwm_last_temb_elems", None)
                yield Site(
                    kind=kind, id=sid,
                    attrs={"narrow": torch.bfloat16, "wide": torch.float32,
                           "constant_elems": tbl.numel(),
                           # the activation is [B, N, 6, C]; measured at trace time, else assume
                           # the video stream's 240 tokens x batch 2
                           "activation_elems": n_tok or (2 * 240 * tbl.numel()),
                           "constant": self._mod_constant(b),
                           "note": "scale_shift_table + temb.float()"})

        elif kind is SiteKind.INVARIANT_CONDITIONING:
            LN = self._install_producer_shim()
            for i, b in enumerate(blocks):
                if getattr(b, "scale_shift_table", None) is not None:
                    # consumed by the modulation combine via `_iwm_cast_scale_shift_table`
                    yield Site(kind=kind, id=f"lingbot.cast[{i}].block.scale_shift_table",
                               attrs={"scope": Scope.MODEL, "evaluated_at": Scope.LAYER,
                                      "pure": True, "dtype": "fp32",
                                      "produce": self._remember(
                                          f"lingbot.cast[{i}].block.scale_shift_table",
                                          b, "scale_shift_table")})
                for name, mod in b.named_modules():
                    # ONLY modules whose consumer actually reads the producer. Publishing a site
                    # the adapter cannot route means the pass installs a cache nothing calls and
                    # then reports a rewrite that did nothing: an inflated count, and worse, a
                    # correctness claim about a code path that never ran. An earlier version
                    # published every weight/bias in the block and claimed 81 rewrites where only
                    # the layer-norm ones were live.
                    if not isinstance(mod, LN):
                        continue
                    for src in ("weight", "bias"):
                        t = getattr(mod, src, None)
                        if t is None or not hasattr(t, "float"):
                            continue
                        # A PARAMETER cast to fp32. The parameter does not change once the model is
                        # loaded, so the cast is MODEL-scoped; the model performs it inside every
                        # layer of every forward, i.e. at LAYER scope. 4,740 casts of a constant
                        # per control cycle.
                        yield Site(
                            kind=kind, id=f"lingbot.cast[{i}].{name or 'block'}.{src}",
                            attrs={"scope": Scope.MODEL, "evaluated_at": Scope.LAYER,
                                   "pure": True, "dtype": "fp32",
                                   "produce": self._remember(
                                       f"lingbot.cast[{i}].{name or 'block'}.{src}", mod, src)})

    # -- allocation plumbing ----------------------------------------------------------------

    @staticmethod
    def _pool_allocator(attn):
        def allocate(**kw):
            return attn._iwm_raw_alloc(**kw)
        return allocate

    @staticmethod
    def _pool_clear(pool):
        pool["mask"].fill_(False)
        pool["id"].fill_(-1)
        pool["is_pred"].fill_(False)
        r = pool.get("_ring")
        if r is not None:
            r.update(start=0, count=0, pred=0, next_id=0)

    def _install_alloc_shim(self):
        """Route pool creation through a wrappable allocator, and record the static extent.

        `init_kv_cache` both allocates and publishes. Splitting the allocation out is what gives a
        generic pass something to wrap; the adapter keeps the publishing.
        """
        import modules.model as M
        Attn = M.WanAttention
        cache_name = self.cache_name
        if not getattr(Attn, "_iwm_alloc_shim", False):
            _orig = Attn.init_kv_cache

            def raw_alloc(self, **kw):
                _orig(self, kw["cache_name"], kw["total"], kw["heads"], kw["head_dim"],
                      kw["device"], kw["dtype"], kw["batch"])
                return self.attn_caches[kw["cache_name"]]

            def init_kv_cache(self, name, total, heads, head_dim, device, dtype, batch):
                self._iwm_pool_shape = (batch, total, heads, head_dim)
                self._iwm_alloc_kw = dict(cache_name=name, total=total, heads=heads,
                                          head_dim=head_dim, device=device, dtype=dtype,
                                          batch=batch)
                alloc = getattr(self, "_iwm_pool_alloc", None) or self._iwm_raw_alloc
                self.attn_caches[name] = alloc(**self._iwm_alloc_kw)

            Attn._iwm_raw_alloc = raw_alloc
            Attn.init_kv_cache = init_kv_cache
            Attn._iwm_alloc_shim = True
        # record the extent for pools already created before the shim went in
        for b in getattr(self.model, "blocks", []) or []:
            a = b.attn1
            c = (getattr(a, "attn_caches", None) or {}).get(cache_name)
            if c and getattr(a, "_iwm_pool_shape", None) is None and c.get("k") is not None:
                a._iwm_pool_shape = tuple(c["k"].shape)
                a._iwm_alloc_kw = dict(cache_name=cache_name, total=c["k"].shape[1],
                                       heads=c["k"].shape[2], head_dim=c["k"].shape[3],
                                       device=c["k"].device, dtype=c["k"].dtype,
                                       batch=c["k"].shape[0])

    def reallocate(self):
        """Do what `_reset` does: ask every pool to be created again."""
        for b in getattr(self.model, "blocks", []) or []:
            a = b.attn1
            if getattr(a, "_iwm_alloc_kw", None):
                kw = a._iwm_alloc_kw
                a.init_kv_cache(kw["cache_name"], kw["total"], kw["heads"], kw["head_dim"],
                                kw["device"], kw["dtype"], kw["batch"])

    def pools(self) -> dict:
        out = {sid: a.attn_caches[self.cache_name] for sid, a in self._alloc_sites.items()}
        out.update({sid: getattr(a2, "_iwm_cross_kv", None)
                    for sid, a2 in self._cross_sites.items()})
        return {k: v for k, v in out.items() if v is not None}

    # -- solver step-index plumbing ----------------------------------------------------------

    def _schedulers(self) -> dict:
        srv = self._server
        out = {}
        if srv is None:
            return out
        for attr in ("scheduler", "action_scheduler"):
            sc = getattr(srv, attr, None)
            if sc is not None:
                out[attr] = sc
        return out

    @staticmethod
    def _reset_hook(sched):
        def register(fn):
            hooks = getattr(sched, "_iwm_reset_hooks", None)
            if hooks is None:
                hooks = sched._iwm_reset_hooks = []
            hooks.append(fn)
        return register

    @staticmethod
    def _install_step_shim(SchedCls):
        """Route the index lookup through a replaceable hook, and rewind it per schedule.

        The default hook reproduces the stock search exactly, so installing the shim alone changes
        nothing; only a pass replacing it changes behaviour.
        """
        if getattr(SchedCls, "_iwm_step_shim", False):
            return
        import torch
        _orig_step = SchedCls.step
        _orig_set = SchedCls.set_timesteps

        def default_resolve(self_s, timestep):
            t = timestep.cpu() if isinstance(timestep, torch.Tensor) else timestep
            return int(torch.argmin((self_s.timesteps - t).abs()))

        def step(self_s, model_output, timestep, sample, to_final=False, **kw):
            resolve = getattr(self_s, "_iwm_resolve_index", None)
            idx = resolve() if resolve is not None else default_resolve(self_s, timestep)
            sigma = self_s.sigmas[idx]                       # python int index: no host sync
            if to_final or idx + 1 >= len(self_s.timesteps):
                sigma_ = 1 if (self_s.inverse_timesteps or self_s.reverse_sigmas) else 0
            else:
                sigma_ = self_s.sigmas[idx + 1]
            return sample + model_output * (sigma_ - sigma)

        def set_timesteps(self_s, *a, **kw):
            out = _orig_set(self_s, *a, **kw)
            for fn in getattr(self_s, "_iwm_reset_hooks", ()):   # new schedule, rewind the counter
                fn()
            return out

        SchedCls.step = step
        SchedCls.set_timesteps = set_timesteps
        SchedCls._iwm_step_shim = True

    # -- attention plumbing ------------------------------------------------------------------

    def _install_attn_shim(self):
        """Interpose one recording wrapper, and give a pass a slot to replace UNDER it.

        Three layers, on purpose:

            attn_op            the recording wrapper -- installed once, never replaced
            _iwm_attn_impl     what actually computes -- this is what a rewrite swaps
            _iwm_attn_base     the model's own op, kept so a rewrite can fall back to it

        Wrapping `attn_op` directly would work exactly once: the second pass to run would wrap the
        first pass's wrapper and the recorded shapes would stop being the model's. Recording is the
        adapter's business and stays outermost; the pass owns only the inner slot.

        Recording is one-shot per layer. Two phases with different query extents both get seen
        because the FIRST call of each shape wins per attribute, and a pass that needs both asks
        for the site twice -- once per phase -- rather than the adapter guessing.
        """
        for b in getattr(self.model, "blocks", []) or []:
            a = b.attn1
            if getattr(a, "_iwm_attn_shim", False):
                continue
            base = a.attn_op
            a._iwm_attn_base = base
            a._iwm_attn_impl = base

            def recording(q, k, v, _a=a):
                if _a._iwm_last_q_rows is None:
                    _a._iwm_last_q_rows = int(q.shape[1])
                    _a._iwm_last_batch = int(q.shape[0])
                return _a._iwm_attn_impl(q, k, v)

            a._iwm_last_q_rows = None
            a._iwm_last_batch = None
            a.attn_op = recording
            a._iwm_attn_shim = True

    # -- modulation combine plumbing ---------------------------------------------------------

    @staticmethod
    def _mod_constant(block):
        """The wide constant, preferring a hoisted cache if HoistInvariant already made one.

        This is where the two passes compose: hoist caches `scale_shift_table.float()` at MODEL
        scope, and this returns that cache so the promotion rewrite does not redo the cast.
        """
        def constant():
            cached = getattr(block, "_iwm_cast_scale_shift_table", None)
            t = cached() if cached is not None else block.scale_shift_table.float()
            return t[None]
        return constant

    def _install_modulate_shim(self):
        """Route the block's modulation combine through a replaceable hook."""
        import modules.model as M
        Blk = M.WanTransformerBlock
        if getattr(Blk, "_iwm_modulate_shim", False):
            return
        from einops import rearrange

        # Shared with the stock body and with hoist_invariant_casts; see runtime/fused_residual.py.
        # Unarmed it is the eager expression, so this shim's numerics are unchanged.
        from instinctwm.runtime.fused_residual import RESIDUAL

        def default_combine(self_b, temb):
            self_b._iwm_last_temb_elems = temb.numel()
            return self_b.scale_shift_table[None] + temb.float()

        _orig = Blk.forward
        if not getattr(_orig, "_iwm_calls_residual", False):
            # The un-rewritten fallback below delegates to whatever was installed before this
            # shim. If that is the STOCK upstream body it contains the residual expression
            # inline, so a kernel armed on the hook would reach the rewritten path and not the
            # fallback — the fusion would apply or not depending on whether an unrelated pass
            # happened to fire. Substitute the hooked copy of the stock body, which is
            # byte-for-byte identical apart from the hook. Another InstinctWM body already
            # carries the marker and is left alone.
            from instinctwm.runtime.fused_residual import _block_forward_hooked
            _orig = _block_forward_hooked

        def forward(self_b, hidden_states, encoder_hidden_states, temb, rotary_emb,
                    update_cache=0, cache_name="pos"):
            combine = getattr(self_b, "_iwm_modulate", None)
            if combine is None:
                return _orig(self_b, hidden_states, encoder_hidden_states, temb, rotary_emb,
                             update_cache, cache_name)
            self_b._iwm_last_temb_elems = temb.numel()
            t = combine(temb)
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
                rearrange(t, "b l n c -> b n l c").chunk(6, dim=1)
            shift_msa, scale_msa, gate_msa = (x.squeeze(1) for x in
                                              (shift_msa, scale_msa, gate_msa))
            c_shift_msa, c_scale_msa, c_gate_msa = (x.squeeze(1) for x in
                                                    (c_shift_msa, c_scale_msa, c_gate_msa))
            norm_hidden_states = (self_b.norm1(hidden_states.float()) * (1. + scale_msa)
                                  + shift_msa).type_as(hidden_states)
            attn_output = self_b.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states,
                                       rotary_emb, update_cache=update_cache,
                                       cache_name=cache_name)
            hidden_states = RESIDUAL(hidden_states, attn_output, gate_msa)
            norm_hidden_states = self_b.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self_b.attn2(norm_hidden_states, encoder_hidden_states,
                                       encoder_hidden_states, None, update_cache=0,
                                       cache_name=cache_name)
            hidden_states = hidden_states + attn_output
            norm_hidden_states = (self_b.norm3(hidden_states.float()) * (1. + c_scale_msa)
                                  + c_shift_msa).type_as(hidden_states)
            ff_output = self_b.ffn(norm_hidden_states)
            return RESIDUAL(hidden_states, ff_output, c_gate_msa)

        forward._iwm_calls_residual = True
        Blk.forward = forward
        Blk._iwm_default_combine = default_combine
        Blk._iwm_modulate_shim = True

    # -- cross-attention K/V plumbing -------------------------------------------------------

    @staticmethod
    def _cross_allocator(attn2):
        def allocate(**kw):
            return attn2._iwm_cross_fresh
        return allocate

    @staticmethod
    def _cross_copy(dst, src):
        for d, s_ in zip(dst, src):
            d.copy_(s_)

    def _install_cross_shim(self):
        """Route cross-cache publication through the per-layer allocator the pass wraps."""
        Model = type(self.model)
        if getattr(Model, "_iwm_cross_shim", False):
            return
        _orig = Model.populate_cross_cache

        def populate_cross_cache(self_m, text_emb):
            _orig(self_m, text_emb)
            for b in self_m.blocks:
                fresh = getattr(b.attn2, "_iwm_cross_kv", None)
                if fresh is None:
                    continue
                b.attn2._iwm_cross_fresh = fresh
                alloc = getattr(b.attn2, "_iwm_cross_alloc", None)
                if alloc is not None:
                    b.attn2._iwm_cross_kv = alloc()      # stable storage, fresh values

        Model.populate_cross_cache = populate_cross_cache
        Model._iwm_cross_shim = True

    # -- invariant-conditioning plumbing ----------------------------------------------------
    # The adapter owns the mechanism; the pass owns the policy. `produce` computes the value,
    # `install` puts whatever the pass hands back where the consumer will call it.

    def _remember(self, site_id, mod, src):
        """Record producer+installer for this site so `apply` can find them by id alone."""
        produce, install = self._producer(mod, src), self._installer(mod, src)
        self._cast_sites[site_id] = (produce, install)
        return produce

    @staticmethod
    def _producer(mod, src):
        def produce():
            return getattr(mod, src).float()
        return produce

    @staticmethod
    def _installer(mod, src):
        def install(cached):
            setattr(mod, f"_iwm_cast_{src}", cached)
        return install

    def _install_producer_shim(self):
        """Make FP32LayerNorm read its fp32 params through a producer, so a pass can wrap it.

        Without this the cast is buried inside `F.layer_norm(x.float(), ..., self.weight.float())`
        and there is nothing for a generic pass to hold on to. Exposing it is the adapter's job:
        the model must present a rewritable surface before anything can rewrite it.
        """
        import modules.model as M
        LN = M.FP32LayerNorm
        if getattr(LN, "_iwm_producer_shim", False):
            return LN
        import torch.nn.functional as F
        _orig = LN.forward

        def forward(self, inputs):
            wc = getattr(self, "_iwm_cast_weight", None)
            bc = getattr(self, "_iwm_cast_bias", None)
            if wc is None and bc is None:
                return _orig(self, inputs)
            w = wc() if wc is not None else (None if self.weight is None else self.weight.float())
            b = bc() if bc is not None else (None if self.bias is None else self.bias.float())
            return F.layer_norm(inputs.float(), self.normalized_shape, w, b,
                                self.eps).to(inputs.dtype)

        LN.forward = forward
        LN._iwm_producer_shim = True
        return LN

    def _extent(self) -> int:
        blocks = getattr(self.model, "blocks", []) or []
        if not blocks:
            return 0
        sig = getattr(blocks[0].attn1, "_iwm_ring_signature", None)
        s = sig(self.cache_name) if sig else None
        return 0 if not s else s[0] * 100003 + s[1]     # fold (start, count) into one int

    # -- APPLY ------------------------------------------------------------------------------
    def apply(self, rewrite):
        from instinctwm.passes.interface import RewriteKind

        if rewrite.site_id == "lingbot.block_stack" and rewrite.kind is RewriteKind.WRAP:
            self._wrapped["block_stack"] = rewrite.payload(self._raw_stack)
            return
        if rewrite.site_id.startswith("lingbot.attention[") and rewrite.kind is RewriteKind.WRAP:
            a = self._attn_sites[rewrite.site_id]
            # Replace the INNER slot. The recording wrapper stays outermost, so a second pass
            # still sees the model's shapes rather than this pass's wrapper.
            a._iwm_attn_impl = rewrite.payload(a._iwm_attn_impl)
            return
        if rewrite.site_id.startswith("lingbot.kv_pool[") and rewrite.kind is RewriteKind.WRAP:
            a = self._alloc_sites[rewrite.site_id]
            a._iwm_pool_alloc = rewrite.payload(self._pool_allocator(a))
            return
        if rewrite.site_id.startswith("lingbot.modulation[") and rewrite.kind is RewriteKind.WRAP:
            blk = self._mod_sites[rewrite.site_id]
            base = (lambda temb, _b=blk: _b.scale_shift_table[None] + temb.float())
            blk._iwm_modulate = rewrite.payload(base)
            return
        if rewrite.site_id.startswith("lingbot.cross_kv[") and rewrite.kind is RewriteKind.WRAP:
            a2 = self._cross_sites[rewrite.site_id]
            a2._iwm_cross_alloc = rewrite.payload(self._cross_allocator(a2))
            return
        if rewrite.site_id.startswith("lingbot.step_index[") and rewrite.kind is RewriteKind.WRAP:
            sched = self._sched_sites[rewrite.site_id]
            sched._iwm_resolve_index = rewrite.payload(None)
            return
        if rewrite.site_id.startswith("lingbot.cast[") and rewrite.kind is RewriteKind.WRAP:
            spec = self._cast_sites.get(rewrite.site_id)
            if spec is None:
                raise KeyError(f"unknown cast site {rewrite.site_id}")
            produce, install = spec
            install(rewrite.payload(produce))
            return
        raise NotImplementedError(f"lingbot surface cannot apply {rewrite}")

    def _raw_stack(self, hidden, encoder, tproj, rot):
        x = hidden
        for b in self.model.blocks:
            x = b(x, encoder, tproj, rot, update_cache=0, cache_name=self.cache_name)
        return x

    def stack(self, hidden, encoder, tproj, rot):
        fn = self._wrapped.get("block_stack", self._raw_stack)
        return fn(hidden, encoder, tproj, rot)
