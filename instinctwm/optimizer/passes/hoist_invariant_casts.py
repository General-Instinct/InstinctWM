"""HoistInvariantCasts (L1-P4) — cast constants once per episode, not once per forward.

Two instances of one pattern, both found by dispatch-level tracing of a single block
(`docs/BLOCK_EXECUTION_PLAN.md`) and invisible in the profiler because they aggregate into
`aten::_to_copy`.

1. PARAMETER UPCASTS. `FP32LayerNorm.forward` does

       F.layer_norm(inputs.float(), shape, self.weight.float(), self.bias.float(), eps)

   `weight` and `bias` are *parameters*. They are cast on every call: 2 casts x 30 layers x 79
   forwards = **4,740 casts of a constant per control cycle**.

2. THE MODULATION UPCAST, which is the same bug wearing a disguise:

       self.scale_shift_table[None] + temb.float()          # upcasts the BIG operand

   `scale_shift_table` is `[1, 6, 3072]` (18 KB) and constant; `temb` is `[B, N, 6, 3072]`
   (35.4 MB) and changes every call. The expression upcasts the one that changes. Casting the
   constant instead gives the identical fp32 result -- addition promotes bf16 to fp32 exactly,
   because bf16 is a strict subset of fp32 -- and deletes a 35.4 MB materialization per block.

Why this is one pass and not two: both are `cast(constant)` inside a loop, and the fix is to
evaluate it at episode scope. The pass keys on that property, so it will fire on any model whose
adapter declares a parameter-derived cast inside a phase.

A REJECTED alternative, recorded because it is the obvious one. "Compute the six modulation chunks
lazily" would replace 2 ops / 70.8 MB with 12 ops / 70.8 MB -- **+10 dispatches per block, or
+23,700 per cycle, about +158 ms of enqueue for zero bytes saved.** On a launch-bound workload the
intuitive fix is a regression; the profitable one is smaller and duller.

Tier: BITEXACT. No arithmetic changes, no rounding point moves, no reduction is re-treed. The
values cast are identical; only *when* they are cast differs.
"""

from __future__ import annotations

import torch

from instinctwm.optimizer.contract import (
    Applicability, BenchResult, CostTerm, DeviceProfile, Discovery, HardwareReq, Tier,
    VerifyResult,
)


class HoistInvariantCasts:
    name = "hoist_invariant_casts"
    hardware = HardwareReq()

    def applicability(self, spec, device: DeviceProfile) -> Applicability:
        return Applicability(
            True,
            "parameter-derived casts execute inside the denoise loop; their operands are "
            "constant for the episode",
            discovery=Discovery.AUTO,      # detectable: a cast whose input is an nn.Parameter
            cost_term=CostTerm.PER_STEP,
            claimed_tier=Tier.BITEXACT)

    def expected_delta_ms(self, spec, device: DeviceProfile) -> float:
        layers, fwd = 30, spec.total_forwards() if hasattr(spec, "total_forwards") else 79
        casts_removed = 3 * layers * fwd          # 2 norm params + 1 temb upcast per block
        return casts_removed * device.launch_overhead_us / 1000.0

    # ---- install -----------------------------------------------------------------------------
    def install(self, server_module, server_cls) -> list[str]:
        import modules.model as M

        # The two gated residuals below go through the shared hook rather than being written
        # out again. With no kernel armed `RESIDUAL` *is* the eager expression, so this changes
        # nothing here; it is what lets `operator_fusion` reach this body instead of silently
        # applying to only whichever block rewrite happened to install last.
        from instinctwm.runtime.fused_residual import RESIDUAL

        applied = []

        # --- 1. FP32LayerNorm: cache the fp32 parameter views -----------------------------------
        LN = M.FP32LayerNorm
        _orig_ln = LN.forward

        def ln_forward(self, inputs: torch.Tensor) -> torch.Tensor:
            w = getattr(self, "_iwm_w32", None)
            if w is None and self.weight is not None:
                w = self.weight.float()
                self._iwm_w32 = w
            b = getattr(self, "_iwm_b32", None)
            if b is None and self.bias is not None:
                b = self.bias.float()
                self._iwm_b32 = b
            return torch.nn.functional.layer_norm(
                inputs.float(), self.normalized_shape, w, b, self.eps).to(inputs.dtype)

        LN.forward = ln_forward
        applied.append("fp32_layernorm_params")

        # --- 2. the block's modulation table: upcast the CONSTANT, not the activation -----------
        Blk = M.WanTransformerBlock
        _orig_blk = Blk.forward

        def blk_forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                        update_cache=0, cache_name="pos"):
            sst = getattr(self, "_iwm_sst32", None)
            if sst is None:
                sst = self.scale_shift_table.float()
                self._iwm_sst32 = sst
            # Stash the fp32 table where the patched arithmetic below can reach it, then defer to
            # the original body with `temb` already promoted for free by the add.
            return _blk_body(self, sst, hidden_states, encoder_hidden_states, temb, rotary_emb,
                             update_cache, cache_name)

        def _blk_body(self, sst32, hidden_states, encoder_hidden_states, temb, rotary_emb,
                      update_cache, cache_name):
            from einops import rearrange

            # WAS: self.scale_shift_table[None] + temb.float()   -- upcasts 35.4 MB of activation
            # NOW: fp32 constant + bf16 activation, promoted inside the add. Same fp32 result.
            t = sst32[None] + temb
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
                rearrange(t, "b l n c -> b n l c").chunk(6, dim=1)
            shift_msa = shift_msa.squeeze(1)
            scale_msa = scale_msa.squeeze(1)
            gate_msa = gate_msa.squeeze(1)
            c_shift_msa = c_shift_msa.squeeze(1)
            c_scale_msa = c_scale_msa.squeeze(1)
            c_gate_msa = c_gate_msa.squeeze(1)

            norm_hidden_states = (self.norm1(hidden_states.float()) * (1. + scale_msa)
                                  + shift_msa).type_as(hidden_states)
            attn_output = self.attn1(norm_hidden_states, norm_hidden_states, norm_hidden_states,
                                     rotary_emb, update_cache=update_cache, cache_name=cache_name)
            hidden_states = RESIDUAL(hidden_states, attn_output, gate_msa)

            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states,
                                     encoder_hidden_states, None, update_cache=0,
                                     cache_name=cache_name)
            hidden_states = hidden_states + attn_output

            norm_hidden_states = (self.norm3(hidden_states.float()) * (1. + c_scale_msa)
                                  + c_shift_msa).type_as(hidden_states)
            ff_output = self.ffn(norm_hidden_states)
            hidden_states = RESIDUAL(hidden_states, ff_output, c_gate_msa)
            return hidden_states

        blk_forward._iwm_calls_residual = True
        Blk.forward = blk_forward
        applied.append("modulation_constant_upcast")

        # These caches are refreshed on reset, but REFRESHED IN PLACE -- never deleted.
        #
        # They used to be dropped with delattr, on the reasoning that a reset may reload or move
        # weights. That is still handled (the values are recomputed), but delattr also reallocates
        # them at NEW ADDRESSES, and they are read inside the region P005 captures. So every reset
        # silently invalidated 90 buffers that no stability check covered, which is the third and
        # last cause of the reset-isolation failure. It was found by `engine/deps.py`, not by
        # inspection: 90 of the region's read buffers had no name, and 89 of them moved.
        #
        # `copy_` keeps the storage, so the graphs stay valid and a genuine weight change still
        # propagates.
        _orig_reset = server_cls._reset
        _SRC = {"_iwm_w32": "weight", "_iwm_b32": "bias", "_iwm_sst32": "scale_shift_table"}

        def _reset(self, prompt=None):
            if hasattr(self, "transformer"):
                for m in self.transformer.modules():
                    for attr, src in _SRC.items():
                        cached = getattr(m, attr, None)
                        p = getattr(m, src, None)
                        if cached is None:
                            continue
                        if p is None or p.shape != cached.shape:
                            delattr(m, attr)       # genuinely stale: let it rebuild
                        else:
                            cached.copy_(p.float())            # same storage, fresh values
            return _orig_reset(self, prompt=prompt)

        server_cls._reset = _reset
        return applied

    def verify(self, harness) -> VerifyResult:
        d = harness.max_abs_action_delta()
        return VerifyResult(passed=(d == 0.0),
                            tier_achieved=Tier.BITEXACT if d == 0.0 else Tier.NUMERIC,
                            max_abs_delta=d,
                            detail="identical values, cast at episode scope instead of per call; "
                                   "bf16->fp32 promotion inside the add is exact")

    def benchmark(self, harness) -> BenchResult:
        b, a = harness.cycle_ms_before(), harness.cycle_ms_after()
        return BenchResult(passed=a < b, before_ms=b, after_ms=a)
