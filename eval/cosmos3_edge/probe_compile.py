#!/usr/bin/env python3
"""torch.compile on the Cosmos3-Edge MoT stack, and whether attention should be excluded from it.

`profile_stack.py` says the graph-replay forward is 55% GEMM and 39% non-GEMM glue spread over
2919 small kernels. torch.compile is the tool aimed at exactly that glue, and it is also what
upstream ships (`Cosmos3-Edge.yaml`: `compile.enabled: true`, `compiled_region: language`), so
measuring it is measuring parity, not invention.

Two questions, one run:

  1. What does compile buy on top of graph capture, and at what tier?
  2. Compile decomposes Cosmos's cuDNN fused SDPA. Is holding attention out of the compiled
     region worth it? Two ways to do that are measured:
        dynamo.disable  -- opaque AND a graph break; splits the layer's fusion region
        custom op       -- opaque, NO graph break; inductor keeps fusing around a black box

Run:  /home/ubuntu/cosmos-framework/.venv/bin/python eval/cosmos3_edge/probe_compile.py
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/cosmos-framework")
sys.path.insert(0, "/home/ubuntu/Code/InstinctWM")

import torch
from torch.profiler import ProfilerActivity, profile

import probe_mot_stack as P
from profile_stack import bucket_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=P.EDGE["num_hidden_layers"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)

    cfg, layers = P.build_stack(args.layers, dev, dt)
    pack = P.build_pack(dev, cfg.hidden_size, dt)

    from cosmos_framework.data.generator.sequence_packing.runtime import get_all_seq, zeros_like
    from cosmos_framework.model.generator.mot.attention import SplitInfo
    import cosmos_framework.model.generator.mot.attention as mot_attn

    hd = P.EDGE["head_dim"]
    cos, sin = zeros_like(pack, (-1, hd)), zeros_like(pack, (-1, hd))
    for p in (cos, sin):
        p["causal_seq"] = torch.randn_like(p["causal_seq"])
        p["full_only_seq"] = torch.randn_like(p["full_only_seq"])
    pos = (cos, sin)
    mask = SplitInfo(split_lens=list(P.SPLIT_LENS), attn_modes=list(P.ATTN_MODES),
                     sample_lens=list(P.SAMPLE_LENS), actual_len=sum(P.SAMPLE_LENS))

    ORIG_ATTN = mot_attn.attention
    torch._dynamo.config.cache_size_limit = 512

    @torch.library.custom_op("iwm::cosmos_attn", mutates_args=())
    def _cosmos_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     is_causal: bool, causal_type: int) -> torch.Tensor:
        from cosmos_framework.model.attention.masks import CausalType
        kw = {"is_causal": is_causal}
        if is_causal:
            kw["causal_type"] = CausalType(causal_type) if causal_type >= 0 else None
        return ORIG_ATTN(q, k, v, **kw)

    @_cosmos_attn.register_fake
    def _(q, k, v, is_causal, causal_type):
        return q.new_empty(q.shape)

    def custom_op_attention(q, k, v, *, is_causal=False, causal_type=None, **kw):
        if kw:                              # varlen path; not the served dense shape
            return ORIG_ATTN(q, k, v, is_causal=is_causal, causal_type=causal_type, **kw)
        ct = causal_type.value if causal_type is not None and hasattr(causal_type, "value") else -1
        return torch.ops.iwm.cosmos_attn(q, k, v, bool(is_causal), int(ct))

    from instinctwm.adapters.cosmos3 import build_plan
    from instinctwm.executors.executor import GraphExecutor

    def stack(inp):
        x = inp
        for l in layers:
            x = l(x, mask, pos)[0]
        return x

    def buckets(runner):
        for _ in range(5):
            runner()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for _ in range(10):
                runner()
            torch.cuda.synchronize()
        bb, bn = collections.Counter(), collections.Counter()
        for e in pr.key_averages():
            if e.self_device_time_total > 0:
                bb[bucket_of(e.key)] += e.self_device_time_total
                bn[bucket_of(e.key)] += e.count
        return bb, bn

    print("=" * 108)
    print("torch.compile on Cosmos3-Edge -- what it buys, what it costs, and whether to "
          "exclude attention")
    print("=" * 108)
    print(f"  {args.layers} layers, hidden {P.EDGE['hidden_size']}, pack "
          f"{P.SPLIT_LENS[0]} und + {P.SPLIT_LENS[1]} gen, NFE {P.NFE}, torch {torch.__version__}")
    print(f"  protocol: {args.iters} iters x {args.repeats} repeats, median, spread shown")

    with torch.no_grad():
        eager_ref = get_all_seq(stack(pack)).clone()
        e_ms, _, e_sp = P.bench(lambda: stack(pack), args.iters, args.repeats)
    g0 = GraphExecutor(build_plan(layers, mask, pos), dev)
    g0.prepare()
    g_ms, _, g_sp = P.bench(lambda: g0.run("mot_stack/default", pack=pack),
                            args.iters, args.repeats)
    gb, gn = buckets(lambda: g0.run("mot_stack/default", pack=pack))

    print(f"\n  {'eager':32s}{e_ms:9.3f} ms/fwd  spread {e_sp:.1f}%")
    print(f"  {'graph replay (shipped)':32s}{g_ms:9.3f} ms/fwd  spread {g_sp:.1f}%   "
          f"BITEXACT baseline, attn {gb['attention']/1e4:.2f} ms / {gn['attention']/10:.0f} kernels")

    print(f"\n  {'variant':32s}{'ms/fwd':>9s}{'spread':>8s}{'attn ms':>9s}{'attn k':>8s}"
          f"{'GEMM ms':>9s}{'kernels':>9s}{'vs graph':>10s}{'max|d| vs eager':>18s}")

    rows = []
    for tag, impl in (("compile (all in)", ORIG_ATTN),
                      ("compile + dynamo.disable", None),
                      ("compile + custom op", custom_op_attention)):
        torch._dynamo.reset()
        mot_attn.attention = torch._dynamo.disable(ORIG_ATTN) if impl is None else impl
        comp = [torch.compile(l, dynamic=False) for l in layers]

        def cstack(inp):
            x = inp
            for l in comp:
                x = l(x, mask, pos)[0]
            return x

        t0 = time.perf_counter()
        with torch.no_grad():
            cstack(pack)
        ct = time.perf_counter() - t0

        gc_ = GraphExecutor(build_plan(comp, mask, pos, model_id=f"c3-{len(rows)}"), dev)
        gc_.prepare()
        out = get_all_seq(gc_.run("mot_stack/default", pack=pack)).clone()
        ms, _, sp = P.bench(lambda: gc_.run("mot_stack/default", pack=pack),
                            args.iters, args.repeats)
        bb, bn = buckets(lambda: gc_.run("mot_stack/default", pack=pack))
        mot_attn.attention = ORIG_ATTN

        md = (out.float() - eager_ref.float()).abs().max().item()
        rows.append((tag, ms, md))
        print(f"  {tag:32s}{ms:9.3f}{sp:7.1f}%{bb['attention']/1e4:9.2f}{bn['attention']/10:8.0f}"
              f"{bb['GEMM']/1e4:9.2f}{sum(bn.values())/10:9.0f}{g_ms/ms:9.3f}x{md:18.3e}"
              f"   (compile {ct:.1f}s)")

    best = min(rows, key=lambda r: r[1])
    print(f"\n  control step (x{P.NFE}): eager {e_ms*P.NFE:.1f} ms -> graph {g_ms*P.NFE:.1f} ms "
          f"-> {best[0]} {best[1]*P.NFE:.1f} ms")
    print(f"  end to end vs eager: {e_ms/best[1]:.3f}x, at max|d| = {best[2]:.3e} (NUMERIC, "
          f"not BITEXACT)")
    print("\n" + "=" * 108)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
