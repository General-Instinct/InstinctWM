"""Reference kernels for the declared regions.

Two variants of the SAME fusion, differing only in whether they reproduce the eager path's
intermediate rounding. They exist as a pair on purpose: it is the cheapest way to demonstrate that
the framework derives the tier from structure rather than from a claim, and to measure what
bit-exactness actually costs.

These are `torch.compile`d rather than hand-written Triton. That is deliberate for the first
proof — the framework's job is to decide *which* kernel is legal and profitable, and a Triton
kernel plugs into the identical `@register_kernel` seam. Writing Triton before the seam is proven
would be writing a LingBot-specific kernel, which is the thing we are avoiding.
"""

from __future__ import annotations

import torch

from instinctwm.kernels.registry import register_kernel
from instinctwm.optimizer.contract import HardwareReq


# ---------------------------------------------------------------------------------------------
# post_attention_gated_residual:  hidden = (hidden.float() + attn_out * gate).type_as(hidden)
# Pure elementwise, no reduction -- the region where BITEXACT is reachable.
# ---------------------------------------------------------------------------------------------

def _post_attn_eager(hidden, attn_out, gate):
    return (hidden.float() + attn_out * gate).type_as(hidden)


@torch.compile(dynamic=False, fullgraph=True)
def _post_attn_fused_impl(hidden, attn_out, gate):
    return (hidden.float() + attn_out * gate).type_as(hidden)


@register_kernel(
    region="post_attention_gated_residual",
    hardware=HardwareReq(requires=frozenset({"triton"})),
    # CORRECTED from True. Identical SOURCE does not imply identical rounding: inductor is free to
    # keep `attn_out * gate` in an fp32 register instead of materialising it in bf16, which skips
    # a rounding the eager path performs. Measured max|d| = 6.25e-02 against eager.
    preserves_intermediate_rounding=False,
    preserves_reduction_order=True,
    compute_dtype="fp32",
    note="fp32 accumulate, but lets inductor elide the bf16 rounding of attn_out*gate")
def post_attn_gated_residual_fused(hidden, attn_out, gate):
    return _post_attn_fused_impl(hidden, attn_out, gate)


@torch.compile(dynamic=False, fullgraph=True)
def _post_attn_exact_impl(hidden, attn_out, gate):
    # Written to re-materialise the product in the storage dtype, on the belief that the eager
    # chain rounds there. It does not -- `gate` is fp32, so eager keeps the product in fp32 (see
    # the RE-CORRECTED note in `lingbot_regions.py`). So this variant adds a bf16 rounding eager
    # never performs and is *further* from the reference than the plain fused one, which its
    # measured max|d| = 7.8e-03 shows. Kept as the worked example of a kernel that is wrong in
    # the direction everyone assumes is safe: reproducing a rounding that was not there.
    prod = (attn_out * gate).to(hidden.dtype)
    return (hidden.float() + prod.float()).type_as(hidden)


@register_kernel(
    region="post_attention_gated_residual",
    hardware=HardwareReq(requires=frozenset({"triton"})),
    preserves_intermediate_rounding=True,
    preserves_reduction_order=True,
    compute_dtype="fp32",
    note="rounds the product to bf16, which the fp32-gated reference does NOT do; kept as the "
         "counter-example to 'more rounding is safer'")
def post_attn_gated_residual_exact(hidden, attn_out, gate):
    return _post_attn_exact_impl(hidden, attn_out, gate)


@torch.compile(dynamic=False, fullgraph=True)
def _post_attn_bf16_impl(hidden, attn_out, gate):
    # Same algebra, but the multiply and add run in bf16. Faster; not the same answer.
    return hidden + (attn_out * gate).to(hidden.dtype)


@register_kernel(
    region="post_attention_gated_residual",
    hardware=HardwareReq(requires=frozenset({"triton"})),
    preserves_intermediate_rounding=False,    # skips the fp32 accumulate
    preserves_reduction_order=True,
    compute_dtype="bf16",
    note="bf16 arithmetic; skips the eager path's fp32 accumulate, so it rounds differently")
def post_attn_gated_residual_bf16(hidden, attn_out, gate):
    return _post_attn_bf16_impl(hidden, attn_out, gate)


# ---------------------------------------------------------------------------------------------
# pre_attention_modulated_norm: (norm1(h.float()) * (1+scale) + shift).type_as(h)
# Contains a LayerNorm, i.e. a reduction. A kernel must declare whether it preserves that
# reduction's order, and torch.compile's generated norm does not guarantee it does.
# ---------------------------------------------------------------------------------------------

def _pre_attn_eager(hidden, norm, scale, shift):
    return (norm(hidden.float()) * (1.0 + scale) + shift).type_as(hidden)


@torch.compile(dynamic=False, fullgraph=True)
def _pre_attn_fused_impl(hidden, w, b, eps, scale, shift):
    h = hidden.float()
    normed = torch.nn.functional.layer_norm(h, (h.shape[-1],), w, b, eps)
    return (normed * (1.0 + scale) + shift).type_as(hidden)


@register_kernel(
    region="pre_attention_modulated_norm",
    hardware=HardwareReq(requires=frozenset({"triton"})),
    preserves_intermediate_rounding=True,
    preserves_reduction_order=False,          # HONEST: the fused norm may retree the reduction
    compute_dtype="fp32",
    note="fp32 layer_norm + modulation; does not guarantee the eager reduction tree, so the "
         "framework derives NUMERIC for it rather than BITEXACT")
def pre_attn_modulated_norm_fused(hidden, w, b, eps, scale, shift):
    return _pre_attn_fused_impl(hidden, w, b, eps, scale, shift)
