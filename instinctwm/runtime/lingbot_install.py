"""Runtime installation of optimizer passes onto the stock LingBot-VA server.

Today the "runtime" is the upstream server plus runtime patches. That is deliberate and
temporary: it keeps every pass verifiable against the existing bit-exactness gate
(`probe_bitexact.py`) before anything is rewritten, and it keeps the vendored upstream tree
clean so `git diff` stays reviewable. When the multi-stream KV core exists these installers get
replaced by real backend methods; the *passes* do not change, which is the point of the layering.

Each installer asserts its structural preconditions at install time and raises if one is false.
A pass that silently no-ops, or worse silently mis-caches, is much more expensive than one that
refuses to load.
"""

from __future__ import annotations

import os
import sys
from typing import Callable, Sequence

import torch


# --- substrate passes -------------------------------------------------------------------
# These three used to live as inline patches in eval/lingbot_va_robotwin/serve_variant.py.
# They are here so the A/B harness and `plan.serve()` apply the SAME code: a measured
# speedup that came from a different patch than the one production installs is not a
# measurement of anything.


def install_fsdp_elision(server_module, va_server_cls=None) -> list[str]:
    """Do not shard across one GPU. See `passes/substrate.py:FSDPElision` for the cost.

    `wan_va_server` binds `_configure_model` at import time, so the BOUND name is what has to
    be replaced — patching `distributed.util` would be too late.
    """

    def _configure_model_nofsdp(model, shard_fn, param_dtype, device, eval_mode=True):
        if eval_mode:
            model.eval().requires_grad_(False)
        model.to(param_dtype)
        model.to(device)
        return model

    if not hasattr(server_module, "_configure_model"):
        raise RuntimeError(
            "fsdp_elision: wan_va_server has no _configure_model to replace. The upstream "
            "server changed shape; refusing to report an optimization that was not applied."
        )
    server_module._configure_model = _configure_model_nofsdp
    return ["fsdp_elision"]


def install_allocator_churn_elision(server_module, va_server_cls=None) -> list[str]:
    """Stop handing the caching allocator back to the driver between control steps.

    Patched on `torch.cuda` rather than on the server module because the server calls through
    to `torch.cuda.empty_cache` from two different sites (`wan_va_server.py:569`, `:603`).
    """
    torch.cuda.empty_cache = lambda *a, **k: None
    return ["allocator_churn_elision"]


def install_debug_dump_elision(server_module, va_server_cls=None) -> list[str]:
    """Take the blocking device->host telemetry copy off the critical path."""
    if not hasattr(server_module, "save_async"):
        raise RuntimeError(
            "debug_dump_elision: wan_va_server has no save_async to neuter. The upstream "
            "server changed shape; refusing to report an optimization that was not applied."
        )
    server_module.save_async = lambda obj, path: None
    return ["debug_dump_elision"]


def install_conditioning_prefill(server_module, va_server_cls) -> list[str]:
    """Cache the episode-constant cross-attention K/V for all layers.

    Patches three things:
      1. `WanAttention.forward` — take a fast path when `cross_kv` is populated.
      2. `WanTransformer3DModel` — gain populate/clear/query methods.
      3. `VA_Server._reset` — release, then repopulate once the prompt embeds exist.
    """
    import modules.model as M

    Attn = M.WanAttention if hasattr(M, "WanAttention") else None
    if Attn is None:
        # Find the attention class structurally rather than by name.
        for obj in vars(M).values():
            if isinstance(obj, type) and hasattr(obj, "attn_caches") is False \
               and getattr(obj, "__module__", "") == M.__name__ \
               and "Attention" in obj.__name__:
                Attn = obj
                break
    if Attn is None:
        raise RuntimeError("could not locate the attention class in modules.model")

    Model = M.WanTransformer3DModel

    # ---- 1. attention fast path ----------------------------------------------------------
    _orig_forward = Attn.forward

    def forward(self, q, k, v, rotary_emb, update_cache=0, cache_name="pos"):
        cross_kv = getattr(self, "_iwm_cross_kv", None)
        if cross_kv is None:
            return _orig_forward(self, q, k, v, rotary_emb, update_cache, cache_name)

        # Preconditions that make the cache correct. Both hold for cross-attention
        # (model.py:552-553 pass rotary_emb=None, update_cache=0). Assert rather than assume:
        # if a future edit routes a rotary or a cache write through here, fail loudly.
        if rotary_emb is not None or update_cache != 0:
            raise RuntimeError(
                "conditioning_prefill: cached cross-attention received "
                f"rotary_emb={rotary_emb is not None}, update_cache={update_cache}. "
                "The cache is only valid when neither is used; refusing to serve a wrong value."
            )

        key, value = cross_kv
        query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
        hidden_states = self.attn_op(query, key, value)
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        return self.to_out[1](self.to_out[0](hidden_states))

    Attn.forward = forward

    @torch.no_grad()
    def _project_cross_kv(self, encoder_hidden_states):
        # Byte-for-byte the k/v half of the stock forward (model.py:426-431).
        key = self.norm_k(self.to_k(encoder_hidden_states)).unflatten(2, (self.heads, -1))
        value = self.to_v(encoder_hidden_states).unflatten(2, (self.heads, -1))
        return key, value

    Attn._iwm_project_cross_kv = _project_cross_kv

    # ---- 2. model-level populate / clear -------------------------------------------------
    @torch.no_grad()
    def populate_cross_cache(self, text_emb):
        t = self.condition_embedder.text_embedder(text_emb)
        # Build every layer BEFORE publishing any, so a failure mid-way cannot leave the model
        # half-cached (the same transactional discipline vLLM-Omni uses in
        # manager.populate_cross_attention).
        built = [b.attn2._iwm_project_cross_kv(t) for b in self.blocks]
        if len(built) != len(self.blocks):
            raise RuntimeError("conditioning_prefill: incomplete projection")
        for b, kv in zip(self.blocks, built):
            b.attn2._iwm_cross_kv = kv
        del t

    def clear_cross_cache(self):
        for b in self.blocks:
            b.attn2._iwm_cross_kv = None

    def cross_cache_populated(self):
        return getattr(self.blocks[0].attn2, "_iwm_cross_kv", None) is not None

    Model.populate_cross_cache = populate_cross_cache
    Model.clear_cross_cache = clear_cross_cache
    Model.cross_cache_populated = cross_cache_populated

    # ---- 3. skip the text_embedder on the warm path --------------------------------------
    _orig_model_forward = Model.forward

    def model_forward(self, input_dict, *a, **kw):
        if self.cross_cache_populated():
            # attn2 never dereferences encoder_hidden_states on the cached path, but the stock
            # forward still projects text_emb -> text_hidden_states (model.py:843). Swap in a
            # zero-cost stand-in so that projection is skipped too.
            emb = self.condition_embedder.text_embedder
            self.condition_embedder.text_embedder = _NoopTextEmbedder()
            try:
                return _orig_model_forward(self, input_dict, *a, **kw)
            finally:
                self.condition_embedder.text_embedder = emb
        return _orig_model_forward(self, input_dict, *a, **kw)

    Model.forward = model_forward

    # ---- 4. lifecycle: release on reset, repopulate once prompts exist -------------------
    _orig_reset = va_server_cls._reset

    def _reset(self, prompt=None):
        if hasattr(self, "transformer"):
            self.transformer.clear_cross_cache()
        _orig_reset(self, prompt=prompt)
        # _reset is the ONLY writer of prompt_embeds (wan_va_server.py:424,:426) and also
        # recomputes use_cfg (:379), which is the only thing that can change the batch dim.
        if getattr(self, "prompt_embeds", None) is not None:
            text_emb = self._iwm_text_emb()
            self.transformer.populate_cross_cache(text_emb)

    def _iwm_text_emb(self):
        """Reproduce exactly what _repeat_input_for_cfg puts in input_dict['text_emb']."""
        pos = self.prompt_embeds.to(self.dtype)
        if self.use_cfg:
            return torch.cat([pos, self.negative_prompt_embeds.to(self.dtype)], dim=0)
        return pos

    va_server_cls._reset = _reset
    va_server_cls._iwm_text_emb = _iwm_text_emb

    return ["conditioning_prefill"]


class _NoopTextEmbedder(torch.nn.Module):
    """Stands in for the text embedder while the cross-attention cache is warm.

    Returns None: attn2 ignores its k/v operands on the cached path, and nothing else in the
    forward reads `text_hidden_states`. If that ever stops being true this returns None into an
    op and fails loudly, which is the behaviour we want.
    """

    def forward(self, x):
        return None


# --- plan installation ------------------------------------------------------------------

def install_operator_fusion(server_module, va_server_cls=None, *,
                            graph_captured: bool = False) -> list[str]:
    """Route the block's two gated residuals through a measured Triton kernel.

    Thin on purpose: the work is in `runtime/fused_residual.py`, which both this and the A/B
    harness call, so a measured speedup cannot come from a different patch than production
    installs. It raises if the kernel is not bit-exact or does not beat eager anywhere.

    `graph_captured` decides WHICH measurement gates the install, and it is not a detail: a
    Triton kernel pays ~41 us of Python launcher per call that graph replay erases entirely, so
    the same kernel measures 0.73x in one mode and 3.27x in the other at the action stream's
    shape. Pass it from whether the block stack is being captured, never from a default.
    """
    from instinctwm.runtime.fused_residual import install_gated_residual_fusion

    return install_gated_residual_fusion(server_module, va_server_cls,
                                         graph_captured=graph_captured)


#: pass name -> installer. Every entry here changes the running server.
INSTALLERS: dict[str, Callable[..., list[str]]] = {
    "fsdp_elision": install_fsdp_elision,
    "allocator_churn_elision": install_allocator_churn_elision,
    "debug_dump_elision": install_debug_dump_elision,
    "conditioning_prefill": install_conditioning_prefill,
    "operator_fusion": install_operator_fusion,
}

#: pass name -> why this backend needs no runtime action for it. Separate from INSTALLERS on
#: purpose: "we did nothing, here is why" and "we did something" are different claims, and a
#: pass that quietly fell into neither bucket is the failure this split exists to prevent.
NO_RUNTIME_ACTION: dict[str, str] = {
    "obs_decode_elision": (
        "the serving path never invokes the decoder — `_infer` returns latents the RoboTwin "
        "client drops (wan_va_server.py:623-624) — so there is no execution to elide. The "
        "residency win this pass describes is decided by the checkpoint loader, not by a "
        "runtime patch, and is not available through this installer."
    ),
}


def install_plan(server_module, va_server_cls, plan) -> list[str]:
    """Apply every applied pass in `plan` to the imported upstream server.

    Raises on any applied pass this backend cannot install. That is the point: a server whose
    `plan.explain()` claims a pass fired, while the pass was silently skipped, invalidates
    every number measured against it. The failure mode this framework sells against is
    precisely a plausible wrong number.

    The whole plan is checked BEFORE anything is patched, for the same reason
    `populate_cross_cache` builds every layer before publishing any: a partial install leaves
    the server module mutated in a state no plan describes, and the patches are monkeypatches
    on an imported module, so there is nothing to roll back to.
    """
    unsupported = [
        r.name for r in plan.applied
        if r.name not in INSTALLERS and r.name not in NO_RUNTIME_ACTION
    ]
    if unsupported:
        raise NotImplementedError(
            f"the lingbot-va backend has no installer for {unsupported}. Either implement one "
            f"in instinctwm/runtime/lingbot_install.py, or drop the pass from the plan with "
            f"plan.without({', '.join(repr(n) for n in unsupported)}) — but do not serve a "
            f"plan whose explain() output claims work that was never applied."
        )

    applied: list[str] = []
    for result in plan.applied:
        installer = INSTALLERS.get(result.name)
        if installer is not None:
            applied.extend(installer(server_module, va_server_cls))
        else:
            applied.append(f"{result.name} (no runtime action: {NO_RUNTIME_ACTION[result.name]})")
    return applied


def install_deterministic_seed(server_module, seed: int) -> list[str]:
    """Seed the noise draw so two servers can be compared at all.

    `_infer` draws `torch.randn` for the initial video latents and action tokens
    (`wan_va_server.py:449-462`) with no seeding, so two *stock* servers already disagree and
    any A/B on output values measures the noise draw rather than the variant.

    Seeded as a function of `frame_st_id`, not a constant: a constant would start every chunk
    in an episode from the SAME noise, which is not the stock distribution and would itself be
    a behaviour change.
    """
    _orig_infer = server_module.VA_Server._infer

    def _seeded_infer(self, obs, frame_st_id=0):
        torch.manual_seed(seed + frame_st_id)
        torch.cuda.manual_seed_all(seed + frame_st_id)
        return _orig_infer(self, obs, frame_st_id=frame_st_id)

    server_module.VA_Server._infer = _seeded_infer
    return [f"deterministic_seed={seed}"]


def resolve_lingbot_root(explicit: str | None = None) -> str:
    """Locate the upstream lingbot-va tree, or say exactly what is missing.

    Order: explicit argument, `LINGBOT_ROOT`, then the historical default. Raises rather than
    letting `import wan_va_server` fail with a bare ModuleNotFoundError several frames later.
    """
    root = explicit or os.environ.get("LINGBOT_ROOT") or "/home/ubuntu/lingbot-va"
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"lingbot-va checkout not found at {root!r}. Set LINGBOT_ROOT to the upstream "
            f"tree (it provides wan_va/wan_va_server.py). InstinctWM patches that server at "
            f"runtime rather than vendoring it, so the checkout is required to serve."
        )
    return root


def import_lingbot_server(lingbot_root: str | None = None, extra_paths: Sequence[str] = ()):
    """Put the upstream tree on `sys.path` and import its server module."""
    root = resolve_lingbot_root(lingbot_root)
    for entry in (os.path.join(root, "wan_va"), root, *extra_paths):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    import wan_va_server  # noqa: E402  (importable only after the sys.path insert above)

    return wan_va_server
