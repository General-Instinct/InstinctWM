"""Length-parameterized attention over a ring KV pool: the extent becomes a VALUE, not a shape.

THE PROBLEM THIS EXISTS TO SOLVE, AND IT IS NOT "ATTENTION IS SLOW"

`graph_block_stack` keys every captured graph on `(start, count)` because the live KV extent is
delivered to attention as a SLICE, and a slice puts its length in a tensor shape. Shapes are frozen
at capture. The ring advances 152 slots per control cycle and `start` stays 0 for a whole episode,
so `count` grows on every single cycle and the key never converges: 6.0 captures/cycle, 92.5% hit
rate, 204 evictions per episode. Measured in-process with no resets, ~85% of a cycle was capture.

So the dominant cost of attention on this model is not attention. It is that attention's extent
lives in the type system.

    sliced      attn(q, pool[:, s:s+n])      extent is a SHAPE -> in the capture key
    this file   attn(q, pool, extent_dev)    extent is a VALUE -> not in the capture key

`pool` is passed at its FULL capacity, so its shape is constant for the model's lifetime. `extent`
is an int32[2] DEVICE tensor holding (start, count); a captured graph replays the same kernel with
the same pointer and reads whatever is in it at replay time. One graph, every cycle.

WHY THIS IS EXACT ACROSS EXTENTS -- the property the whole design rests on

In an online-softmax kernel a tile whose every position is masked contributes:

    m <- max(m, -inf) = m        alpha = exp(m - m) = exp(0) = 1.0
    l <- l * 1.0 + 0 = l         acc <- acc * 1.0 + dot(0, v) = acc

Every one of those is an exact float identity -- multiplying by 1.0 and adding 0.0 change nothing,
in any rounding mode. So iterating the KV loop to the pool's capacity and masking the tail produces
BIT-IDENTICAL output to iterating only to `count`. `FIXED_TRIP` exposes both and
`tests/test_ring_attention.py` gates them against each other at zero, because that identity is the
reason the fixed-shape pool is sound and it deserves an assertion rather than a paragraph.

That is also precisely what `F.scaled_dot_product_attention` CANNOT give you. Padding its input and
handing it a mask is not the same operation: measured on these shapes it changes 730,892 of
1,474,560 elements (max|d| 4.883e-04) and costs 1.60x MORE, because a mask pushes the dispatcher off
the flash backend. The tail content is irrelevant to that -- zeros and stale KV give the identical
delta -- so it is reduction re-treeing, not leakage. Owning the kernel is what buys the property.

WHAT THIS IS NOT

NOT bit-exact against the served path. This kernel's reduction order is its own, and against
`custom_sdpa` it differs by ~4.3e-04 on these shapes. Installing it is a NUMERIC-tier change that
needs a paired non-inferiority certificate, the same as any step-reduction recipe. The kernel is
registered here so `KernelRegistry.select` can weigh it; it is deliberately NOT on a default path.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except ImportError:                                        # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _ring_attn_kernel(
        Q, K, V, Out, EXTENT,
        sm_scale,
        stride_qb, stride_qm, stride_qh, stride_qd,
        stride_kb, stride_kn, stride_kh, stride_kd,
        stride_ob, stride_om, stride_oh, stride_od,
        M, CAP,
        H: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        D: tl.constexpr,
        FIXED_TRIP: tl.constexpr,
        COUNT_EXTRA: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H

        start = tl.load(EXTENT + 0)
        # COUNT_EXTRA is the provisional block this forward just wrote. It is a CONSTEXPR, not a
        # device value, because the number of tokens a forward commits is fixed by the phase and
        # is already part of the graph's shape key -- so adding it here costs nothing and keeps
        # the caller from having to materialise a second extent tensor inside the captured region.
        #
        # The CLAMP is load-bearing and its absence was a crash, not a slow path: once the pool
        # is full `count` equals CAP, `count + COUNT_EXTRA` runs off the end of the allocation,
        # and every masked-in lane reads unmapped memory. Measured: died at cycle 36, which is
        # exactly 9792 / 272. P003 has the same clamp written as `if count >= total: use the
        # whole pool`.
        count = tl.minimum(tl.load(EXTENT + 1) + COUNT_EXTRA, CAP)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        m_live = offs_m < M

        q_ptrs = (Q + b * stride_qb + offs_m[:, None] * stride_qm
                  + h * stride_qh + offs_d[None, :] * stride_qd)
        q = tl.load(q_ptrs, mask=m_live[:, None], other=0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, D], tl.float32)

        # FIXED_TRIP walks the whole pool and lets masked tiles fall through as the identity
        # above; the default walks only the live extent. They are bit-identical by construction
        # and the fixed form exists to make that testable, not because it is needed for capture:
        # a graph freezes the LAUNCH, and a data-dependent trip count inside the kernel is not
        # part of it.
        n_end = CAP if FIXED_TRIP else count

        for start_n in range(0, n_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            live = offs_n < count
            # WRAP. Once the pool is full the live set is [start, CAP) ++ [0, start+count-CAP),
            # so the position must come back round. Walking it chronologically is not ascending
            # SLOT order, which is what P003 reproduces to stay bit-exact against stock -- this
            # kernel is NUMERIC already, and softmax is permutation-invariant over keys, so
            # chronological is semantically right and only reassociates the sum.
            pos = (start + offs_n) % CAP
            k_ptrs = (K + b * stride_kb + pos[:, None] * stride_kn
                      + h * stride_kh + offs_d[None, :] * stride_kd)
            k = tl.load(k_ptrs, mask=live[:, None], other=0.0)
            v_ptrs = (V + b * stride_kb + pos[:, None] * stride_kn
                      + h * stride_kh + offs_d[None, :] * stride_kd)
            v = tl.load(v_ptrs, mask=live[:, None], other=0.0)

            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(live[None, :], qk, float("-inf"))

            # A fully masked tile leaves m_i untouched, so alpha is exactly 1.0 and p is exactly
            # 0.0. m_i is -inf only before the first tile, and tile 0 always holds a live column
            # (count >= 1 is asserted by the wrapper), so -inf minus -inf never arises.
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])

            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = (Out + b * stride_ob + offs_m[:, None] * stride_om
                  + h * stride_oh + offs_d[None, :] * stride_od)
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=m_live[:, None])

    def _config(M: int) -> tuple[int, int, int, int]:
        """(BLOCK_M, BLOCK_N, num_warps, num_stages), from a sweep on the served shapes.

        A fixed table rather than `@triton.autotune`, for a reason specific to this kernel:
        autotune benchmarks on its first call with a given key, and a first call that launches
        the same kernel a hundred times cannot happen inside `torch.cuda.graph`. The whole point
        of this kernel is to be capturable, so its launch parameters must be decided before
        capture, not during it.

        Measured, A100-SXM4-80GB / torch 2.13.0 / triton, B=2 H=24 D=128 bf16, 50 timed
        iterations after 10 warmup, against the SDPA dispatcher on identical inputs:

            q=240 count=5000   sdpa 0.335   this 0.363   0.92x   BM=128 BN=128 w=8 s=3
            q=240 count=9792   sdpa 0.528   this 0.556   0.95x   BM=128 BN=128 w=8 s=3
            q=32  count=5000   sdpa 0.156   this 0.102   1.53x   BM=16  BN=128 w=4 s=3
            q=32  count=9792   sdpa 0.223   this 0.152   1.47x   BM=16  BN=128 w=4 s=3

        So it LOSES by 5-8% on the video phase and WINS by ~1.5x on the action phase, and the
        reason is occupancy rather than arithmetic: 32 query rows over 24 heads at batch 2 is
        only 48 programs at BLOCK_M=64, under half of this box's 108 SMs. BLOCK_M=16 doubles the
        grid. Per control cycle at count=5000 that nets to about -61 ms (26 video forwards
        +22 ms, 51 action forwards -83 ms), before counting anything the capture key saves.

        The right consumer of that table is `KernelRegistry.select`, which measures per shape and
        would keep SDPA for video while taking this for action. A single global answer is the
        wrong shape for this question.

        METHODOLOGY NOTE, because the first version of this table was WRONG. It came from a sweep
        of 72 configs at 20 iterations each, reporting the minimum. The minimum over 72 noisy
        measurements is biased low -- it selects for whichever config got the luckiest run -- and
        it reported BM=64 at 0.347 ms (0.97x) for a config that reproduces at 0.425 ms (0.79x).
        Same trap as `RESULTS.md` section 8, one level down: there it was one probe run per arm,
        here it is one run per config with a min-reducer. Re-measure the WINNER at high iteration
        count before believing a sweep.
        """
        if M <= 32:
            return 16, 128, 4, 3        # measured
        if M <= 128:
            return 32, 128, 4, 3        # INTERPOLATED between the two measured rows, not measured
        return 128, 128, 8, 3           # measured at M=240

    def ring_attention(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
                       extent: torch.Tensor, *, count_extra: int = 0,
                       fixed_trip: bool = False,
                       block_m: int | None = None, block_n: int | None = None,
                       num_warps: int | None = None, num_stages: int | None = None,
                       allow_fma: bool = False) -> torch.Tensor:
        """Attention of `q` against `k_pool[:, start:start+count]`, BSHD in and out.

        `extent` is an int32 CUDA tensor of shape (2,) holding `(start, count)`. It is read on the
        device inside the kernel, which is the whole point: it may change between replays of a
        captured graph without invalidating the graph.

        `k_pool` / `v_pool` keep their full capacity shape. Nothing is sliced, so nothing about
        this call's shape depends on how full the pool is.
        """
        assert q.dim() == 4 and k_pool.dim() == 4 and v_pool.dim() == 4
        B, M, H, D = q.shape
        assert k_pool.shape[0] == B and k_pool.shape[2] == H and k_pool.shape[3] == D
        assert k_pool.shape == v_pool.shape, "K and V pools must share a layout"
        assert k_pool.stride() == v_pool.stride(), "K and V pools must share strides"
        assert extent.dtype == torch.int32 and extent.numel() == 2, \
            "extent must be an int32[2] device tensor holding (start, count)"
        assert D in (16, 32, 64, 128, 256), f"unsupported head dim {D}"

        d_bm, d_bn, d_w, d_s = _config(M)
        block_m = d_bm if block_m is None else block_m
        block_n = d_bn if block_n is None else block_n
        num_warps = d_w if num_warps is None else num_warps
        num_stages = d_s if num_stages is None else num_stages

        CAP = k_pool.shape[1]
        out = torch.empty_like(q)
        grid = (triton.cdiv(M, block_m), B * H)
        _ring_attn_kernel[grid](
            q, k_pool, v_pool, out, extent,
            D ** -0.5,
            *q.stride(), *k_pool.stride(), *out.stride(),
            M, CAP,
            H=H, BLOCK_M=block_m, BLOCK_N=block_n, D=D, FIXED_TRIP=fixed_trip,
            COUNT_EXTRA=count_extra,
            num_warps=num_warps, num_stages=num_stages,
            enable_fp_fusion=allow_fma,
        )
        return out

    # -- registration ---------------------------------------------------------------------------
    # Every `preserves_*` flag is False, and that is not modesty -- it is the derivation working.
    # This kernel's reduction order is its own, so `derive_tier` returns NUMERIC for a region that
    # contains an ATTENTION op, and `KernelRegistry.candidates` will refuse it under the default
    # BITEXACT ceiling. Installing it requires a caller that has explicitly raised the ceiling and
    # bought a certificate, which is exactly the gate this kernel should have to pass.
    from instinctwm.kernels.registry import HardwareReq, register_kernel

    @register_kernel(
        region="self_attention_ring",
        hardware=HardwareReq(requires=frozenset({"cuda", "triton"})),
        preserves_intermediate_rounding=False,
        preserves_reduction_order=False,
        matches_reference_contraction=False,
        compute_dtype="fp32",
        note="length-parameterized: the live extent is a device-resident VALUE, so the capture "
             "key no longer contains (start, count). Bit-identical across extents by the "
             "masked-tile identity; NOT bit-exact against custom_sdpa (~4.3e-04 on served shapes)")
    def self_attention_ring_triton(q, k_pool, v_pool, extent):
        return ring_attention(q, k_pool, v_pool, extent)


else:                                                       # pragma: no cover
    def ring_attention(*a, **k):
        raise RuntimeError("triton not available")


def ring_attention_eager(q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
                         extent: torch.Tensor) -> torch.Tensor:
    """The reference: what the model does today. Slices, then calls the dispatcher.

    Every audit compares against this. It is also the thing whose SHAPE is the problem -- the
    slice below is where `count` enters the type system and, from there, the capture key.
    """
    import torch.nn.functional as F
    start, count = (int(x) for x in extent.tolist())
    k = k_pool[:, start:start + count]
    v = v_pool[:, start:start + count]
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
