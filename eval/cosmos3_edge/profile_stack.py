#!/usr/bin/env python3
"""Where the 28.4 ms goes.

`probe_mot_stack.py` establishes that graph replay lands the Cosmos3-Edge MoT stack GPU-bound at
~28.4 ms per forward. Launch overhead is spent; anything further has to come off the GPU. This
answers what the GPU is actually doing, and puts two floors underneath it so a candidate can be
ranked against what is physically available rather than against intuition.

Run:  /home/ubuntu/cosmos-framework/.venv/bin/python eval/cosmos3_edge/profile_stack.py
"""
from __future__ import annotations

import argparse
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/cosmos-framework")
sys.path.insert(0, "/home/ubuntu/Code/InstinctWM")

import torch
from torch.profiler import ProfilerActivity, profile

import probe_mot_stack as P

# A100-SXM4-80GB datasheet. Used only to state a floor, never to claim one was reached.
HBM_GB_S = 2039.0
PEAK_BF16_TFLOPS = 312.0

# Kernel-name buckets. Order matters: first match wins.
#
# Inductor kernels MUST be matched first, by prefix. Its names embed the fused epilogue ops --
# `triton_tem_fused_add_mul_silu_t_16` is a matmul TEMPLATE with a silu epilogue -- so a generic
# `mul|add|copy` rule files matmuls under elementwise, and a rule that only knows cuBLAS/CUTLASS
# names drops them into `other` entirely. The second failure reads as "GEMM got 3.6x faster" when
# all that happened is the kernels stopped being counted. It cost a wrong conclusion once already.
#
#   triton_tem_*  matmul/conv template (+ fused epilogue)  -> GEMM
#   triton_poi_*  pointwise                                -> elementwise
#   triton_red_*  reduction                                -> reduce
BUCKETS = (
    ("GEMM",        r"triton_tem|triton_mm|triton_bmm"),
    ("elementwise", r"triton_poi"),
    ("reduce",      r"triton_red"),
    ("GEMM",        r"gemm|cutlass|s16816|nvjet|ampere_bf16|sm80_xmma|cublas|matmul|addmm"),
    ("attention",   r"attention|fmha|flash|cudnn|mha|softmax"),
    ("norm/RMS",    r"norm|rms|layer_norm"),
    ("RoPE",        r"rope|rotary"),
    ("scatter/idx", r"scatter|index_|gather|take|embedding"),
    ("copy/cast",   r"copy|cast|convert|contiguous|memcpy"),
    ("elementwise", r"elementwise|mul|add|silu|gelu|relu|activation|unrolled|vectorized"),
    ("reduce",      r"reduce|sum|mean"),
)


def bucket_of(name: str) -> str:
    low = name.lower()
    for label, pat in BUCKETS:
        if re.search(pat, low):
            return label
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=P.EDGE["num_hidden_layers"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)

    und, gen = P.SPLIT_LENS
    tot = sum(P.SAMPLE_LENS)
    sha = os.popen("git -C /home/ubuntu/cosmos-framework rev-parse --short HEAD").read().strip()

    print("=" * 92)
    print("Cosmos3-Edge MoT stack -- where the GPU time goes")
    print("=" * 92)
    print(f"  cosmos-framework @ {sha}   torch {torch.__version__}   "
          f"{torch.cuda.get_device_name(0)}")
    print(f"  {args.layers} layers, hidden {P.EDGE['hidden_size']}, pack {und} und + {gen} gen "
          f"= {tot} tokens, NFE {P.NFE}")

    cfg, layers = P.build_stack(args.layers, dev, dt)
    pack = P.build_pack(dev, cfg.hidden_size, dt)

    from cosmos_framework.data.generator.sequence_packing.runtime import zeros_like
    from cosmos_framework.model.generator.mot.attention import SplitInfo

    hd = P.EDGE["head_dim"]
    cos, sin = zeros_like(pack, (-1, hd)), zeros_like(pack, (-1, hd))
    for p in (cos, sin):
        p["causal_seq"] = torch.randn_like(p["causal_seq"])
        p["full_only_seq"] = torch.randn_like(p["full_only_seq"])
    pos = (cos, sin)
    mask = SplitInfo(split_lens=list(P.SPLIT_LENS), attn_modes=list(P.ATTN_MODES),
                     sample_lens=list(P.SAMPLE_LENS), actual_len=tot)

    def stack(inp):
        x = inp
        for l in layers:
            x = l(x, mask, pos)[0]
        return x

    from instinctwm.adapters.cosmos3 import build_plan
    from instinctwm.executors.executor import GraphExecutor

    graph = GraphExecutor(build_plan(layers, mask, pos), dev)
    graph.prepare()
    g_ms, _, _ = P.bench(lambda: graph.run("mot_stack/default", pack=pack), 20, 3)

    # ---- floors ------------------------------------------------------------------------
    nparam = sum(p.numel() for l in layers for p in l.parameters())
    wbytes = nparam * 2
    # MoT routes und tokens through the und tower and gen tokens through the gen tower, so each
    # token multiplies against roughly half the parameters, but EVERY weight is still read once.
    flops = 2 * (nparam / 2) * tot
    mem_floor_ms = wbytes / (HBM_GB_S * 1e9) * 1e3
    cmp_floor_ms = flops / (PEAK_BF16_TFLOPS * 1e12) * 1e3

    print(f"\n--- floors (datasheet, not achieved) ----------------------------------------")
    print(f"  weights {wbytes / 2**30:.2f} GiB, read once per forward")
    print(f"  memory floor : {mem_floor_ms:6.2f} ms   ({wbytes / 2**30:.2f} GiB @ {HBM_GB_S:.0f} GB/s)")
    print(f"  compute floor: {cmp_floor_ms:6.2f} ms   ({flops / 1e12:.2f} TFLOP @ "
          f"{PEAK_BF16_TFLOPS:.0f} TFLOPS bf16)")
    print(f"  measured     : {g_ms:6.2f} ms   graph replay")
    print(f"  => {g_ms / max(mem_floor_ms, cmp_floor_ms):.1f}x above the binding floor "
          f"({'compute' if cmp_floor_ms > mem_floor_ms else 'memory'}); "
          f"MFU {flops / 1e12 / (g_ms / 1e3) / PEAK_BF16_TFLOPS * 100:.1f}%, "
          f"HBM {wbytes / 2**30 * 1.074 / (g_ms / 1e3) / HBM_GB_S * 100:.1f}%")

    # ---- kernel breakdown on the SHIPPED path ------------------------------------------
    for _ in range(5):
        graph.run("mot_stack/default", pack=pack)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(args.iters):
            graph.run("mot_stack/default", pack=pack)
        torch.cuda.synchronize()

    evs = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    total_us = sum(e.self_device_time_total for e in evs)
    per_fwd_ms = total_us / args.iters / 1e3

    print(f"\n--- GPU kernel time, graph replay ({args.iters} forwards) -------------------")
    print(f"  summed kernel self-time {per_fwd_ms:.2f} ms/forward vs {g_ms:.2f} ms wall "
          f"({per_fwd_ms / g_ms * 100:.1f}% -- the remainder is gaps between kernels)")

    by_bucket = collections.Counter()
    by_bucket_n = collections.Counter()
    for e in evs:
        b = bucket_of(e.key)
        by_bucket[b] += e.self_device_time_total
        by_bucket_n[b] += e.count

    print(f"\n  {'bucket':14s}{'ms/fwd':>9s}{'% GPU':>8s}{'kernels/fwd':>13s}"
          f"{'x NFE 16':>11s}")
    for b, us in by_bucket.most_common():
        ms = us / args.iters / 1e3
        print(f"  {b:14s}{ms:9.2f}{us / total_us * 100:7.1f}%"
              f"{by_bucket_n[b] / args.iters:13.0f}{ms * P.NFE:10.1f} ms")

    print(f"\n  top {args.top} individual kernels")
    print(f"  {'ms/fwd':>8s}{'% GPU':>8s}{'calls/fwd':>11s}  kernel")
    for e in sorted(evs, key=lambda x: -x.self_device_time_total)[:args.top]:
        ms = e.self_device_time_total / args.iters / 1e3
        nm = e.key if len(e.key) <= 78 else e.key[:75] + "..."
        print(f"  {ms:8.3f}{e.self_device_time_total / total_us * 100:7.1f}%"
              f"{e.count / args.iters:11.1f}  {nm}")

    n_kernels = sum(by_bucket_n.values()) / args.iters
    print(f"\n  {n_kernels:.0f} kernels per forward over {args.layers} layers "
          f"= {n_kernels / args.layers:.1f} per layer; "
          f"{n_kernels * P.NFE:.0f} per control step")
    print(f"  mean kernel duration {total_us / sum(by_bucket_n.values()):.1f} us")

    print("\n" + "=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
