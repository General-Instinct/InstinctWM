"""LingBot-VA's fusible regions — declarations only, read from source.

`modules/model.py:524-544`. Note the shape of the eager code, because it is what decides the tier:

    norm_hidden_states = (self.norm1(hidden_states.float())      # fp32 compute
                          * (1. + scale_msa) + shift_msa
                         ).type_as(hidden_states)                # <- MATERIALISES in bf16
    ...
    hidden_states = (hidden_states.float()
                     + attn_output * gate_msa
                    ).type_as(hidden_states)                     # <- MATERIALISES in bf16

Each `.type_as(hidden_states)` is a rounding to bf16 that a fused kernel must reproduce to be
bit-exact. Inside the parentheses the arithmetic is already fp32, so the intermediate `norm * scale`
is NOT materialised and carries no rounding of its own — which is why `norm1` is marked
`materializes_as=None` while the modulation output is `bf16`.

The `.float()` / `.type_as()` pair is also why these two regions account for so many launches: each
one is a separate elementwise kernel over [2, N, 3072], and the profile attributes 16,932 copies of
[2,32,3072] and 8,632 of [2,240,3072] per `_infer` to exactly this pattern.
"""

from __future__ import annotations

from instinctwm.kernels.regions import FusibleRegion, FusionDescriptor, OpKind, OpSpec

_LAYERS = 30

PRE_ATTENTION = FusibleRegion(
    name="pre_attention_modulated_norm",
    ops=(
        OpSpec("upcast", OpKind.ELEMENTWISE, materializes_as=None, computes_in="fp32"),
        # LayerNorm is a REDUCTION. Its presence is what keeps a naive fused kernel out of the
        # BITEXACT tier unless it also preserves the reduction tree.
        OpSpec("norm1", OpKind.REDUCTION, materializes_as=None, computes_in="fp32"),
        OpSpec("scale", OpKind.ELEMENTWISE, materializes_as=None, computes_in="fp32"),
        OpSpec("shift", OpKind.ELEMENTWISE, materializes_as="bf16", computes_in="fp32"),
    ),
    boundary_in=("hidden_states", "scale_msa", "shift_msa"),
    boundary_out=("norm_hidden_states",),
    phases=("kv_refresh", "video", "action"),
    # Two sites per block, same as the residual: the pre-self-attention norm (model.py:533-535,
    # norm1/scale_msa/shift_msa) and the pre-FFN norm (model.py:557-559, norm3/c_scale_msa/
    # c_shift_msa). `norm2` is NOT one of them -- it is an unmodulated cross-attention norm.
    occurrences_per_forward=2 * _LAYERS,
    note="model.py:533-535 and 557-559",
)

POST_ATTENTION = FusibleRegion(
    name="post_attention_gated_residual",
    ops=(
        OpSpec("upcast", OpKind.ELEMENTWISE, materializes_as=None, computes_in="fp32"),
        # RE-CORRECTED to materializes_as=None, which is what it originally said.
        #
        # The intervening "correction" claimed `attn_out * gate` is bf16 x bf16 and lands in
        # bf16 before the fp32 add. That is not what the model does: `gate_msa` is a chunk of
        # `self.scale_shift_table[None] + temb.float()` (model.py:524), so it is **fp32**, the
        # product promotes to fp32, and nothing is materialised between the multiply and the add.
        #
        # The evidence that prompted the wrong correction was real -- a kernel deriving BITEXACT
        # measured max|d| = 6.25e-02 -- but the cause was FP CONTRACTION, not a missing rounding:
        # the backend fused the multiply-add into one FMA and skipped the *fp32* rounding of the
        # product. `matches_reference_contraction` was added later and catches exactly that, so
        # the three torch.compile variants derive NUMERIC for the right reason now. Leaving the
        # region wrong as well meant two errors cancelling: the region over-declared a rounding
        # point, and the Triton kernel over-declared reproducing it.
        OpSpec("gate", OpKind.ELEMENTWISE, materializes_as=None, computes_in="fp32"),
        OpSpec("residual_add", OpKind.ELEMENTWISE, materializes_as="bf16", computes_in="fp32"),
    ),
    boundary_in=("hidden_states", "attn_output", "gate_msa"),
    boundary_out=("hidden_states",),
    phases=("kv_refresh", "video", "action"),
    # TWO sites per block, not one: the self-attention residual (model.py:543-544) and the
    # feed-forward residual (model.py:563-564). They differ only in an explicit `.float()` on
    # the second operand, which is a no-op against an fp32 gate, so ONE kernel serves both and
    # the occurrence count is 2 per layer. Counting one site per layer halved every launch
    # estimate this region feeds.
    occurrences_per_forward=2 * _LAYERS,
    note="model.py:543-544 and 563-564 -- pure elementwise, NO reduction. This is the region "
         "where a rounding-preserving kernel can legitimately reach BITEXACT.",
)


#: The self-attention call itself. Declared as a region so the registry can weigh a replacement
#: against it -- NOT because it is "fusible" in the elementwise sense.
#:
#: `gather` is the op that matters and it is why this region exists. After P003 the live set is a
#: SLICE rather than a masked gather, which is cheap (measured 1.00x vs a contiguous copy on this
#: box), but a slice still encodes its length as a shape, and that shape is what
#: `graph_block_stack` has to put in its capture key. A kernel that takes the extent as a value
#: removes it. `attention` is marked ATTENTION, so `derive_tier` treats any replacement as having
#: its own reduction order and returns NUMERIC unless the kernel argues otherwise -- which is the
#: correct default for anything that re-implements softmax.
SELF_ATTENTION_RING = FusibleRegion(
    name="self_attention_ring",
    ops=(
        OpSpec("gather", OpKind.RESHAPE, materializes_as=None, computes_in="bf16"),
        OpSpec("attention", OpKind.ATTENTION, materializes_as="bf16", computes_in="fp32"),
    ),
    boundary_in=("query", "key_pool", "value_pool", "extent"),
    boundary_out=("attn_output",),
    phases=("kv_refresh", "video", "action"),
    occurrences_per_forward=_LAYERS,
    note="model.py:451-455 (stock: mask.nonzero + advanced index) / ring_kv.py (slice). The "
         "extent reaches the op as a shape, which is what puts (start, count) in the capture key.",
)


def lingbot_fusion_descriptor() -> FusionDescriptor:
    # SELF_ATTENTION_RING is DELIBERATELY NOT LISTED. `FusionDescriptor` means "regions offered
    # for operator fusion", and `OperatorFusion` takes every region in it whose admissible tier
    # clears the ceiling. That region admits NUMERIC, so listing it made the fusion pass select
    # it -- and the fusion installer cannot install it: it is a whole-op replacement that needs
    # the device-resident extent `runtime/ring_attention_install.py` sets up, not a fused
    # elementwise chain. A region containing an ATTENTION op is not an operator fusion.
    #
    # It stays a module-level declaration because `KernelRegistry` takes a region object
    # directly, which is all the tier derivation and kernel selection need.
    return FusionDescriptor(
        model_id="lingbot-va-posttrain-robotwin",
        regions=(PRE_ATTENTION, POST_ATTENTION),
        # measured from the post-ring-KV profile: elementwise/norm 160,225 launches and
        # gather/copy 163,596 launches per cycle, dominated by these two regions
        launches_per_region={
            "pre_attention_modulated_norm": 4,     # upcast, norm, scale+shift, type_as
            "post_attention_gated_residual": 3,    # upcast, gate-mul, add+type_as
            # slice + transpose x3 + sdpa + transpose. Stock was 9 (nonzero, index x2, ...);
            # P003 removed the gather, and what is left is mostly view bookkeeping.
            "self_attention_ring": 6,
        },
    )
