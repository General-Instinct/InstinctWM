#!/usr/bin/env python3
"""Cosmos3-Edge MoT stack under the InstinctWM engine, at the REAL served width.

Replaces the toy config in instinctwm/adapters/cosmos3.py (hidden 512 / head_dim 64) with the
shipped Cosmos3-Edge text-tower geometry read off nvidia/Cosmos3-Edge config.json:

    hidden 2048, 28 layers, 16 q heads, 8 kv heads, head_dim 128, intermediate 9216

and drops the torch-SDPA shim: the cuDNN backend resolves at these shapes on this box, so the
attention kernel is the one Cosmos actually dispatches to.

STILL NOT a served-accuracy claim: random weights, no checkpoint. What IS claimed is op
structure, shapes, dependency derivation, capturability, eager-vs-replay equality, and the
allocation traffic a pass would target.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, "/home/ubuntu/cosmos-framework")
sys.path.insert(0, "/home/ubuntu/Code/InstinctWM")

import torch

# Shipped Cosmos3-Edge text tower (nvidia/Cosmos3-Edge config.json -> text_config).
EDGE = dict(hidden_size=2048, num_hidden_layers=28, num_attention_heads=16,
            num_key_value_heads=8, head_dim=128, intermediate_size=9216, rms_norm_eps=1e-5)

# Served pack geometry, as declared in instinctwm/runtime/state/manifests.py:cosmos3_edge_manifest.
SAMPLE_LENS = [567]
SPLIT_LENS = [111, 456]          # und (text) prefix, gen (video+action) body
ATTN_MODES = ["causal", "full"]
NFE = 16                         # forwards per control step, the served rung


def build_stack(n_layers, device, dtype=torch.bfloat16):
    from cosmos_framework.model.generator.mot.unified_mot import LayerTypes, MoTDecoderLayer
    from cosmos_framework.model.generator.reasoner.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLTextConfig,
    )
    cfg = Qwen3VLTextConfig(
        hidden_size=EDGE["hidden_size"], num_attention_heads=EDGE["num_attention_heads"],
        num_key_value_heads=EDGE["num_key_value_heads"], num_hidden_layers=n_layers,
        intermediate_size=EDGE["intermediate_size"], head_dim=EDGE["head_dim"],
        rms_norm_eps=EDGE["rms_norm_eps"], attention_bias=False,
    )
    lt = LayerTypes("qwen3_vl_dense")
    layers = [MoTDecoderLayer(cfg, layer_idx=i, layer_types=lt,
                              qk_norm_for_text=True, qk_norm_for_diffusion=True)
              .to(device, dtype).eval() for i in range(n_layers)]
    for l in layers:
        l.requires_grad_(False)
    return cfg, layers


def build_pack(device, hidden, dtype, seed=0):
    from cosmos_framework.data.generator.sequence_packing.runtime import (
        sequence_pack_from_packed_sequence,
    )
    g = torch.Generator(device="cpu").manual_seed(seed)
    und, gen = SPLIT_LENS
    seq = torch.randn(sum(SAMPLE_LENS), hidden, generator=g).to(device, dtype)
    return sequence_pack_from_packed_sequence(
        packed_sequence=seq, attn_modes=ATTN_MODES, split_lens=SPLIT_LENS,
        sample_lens=list(SAMPLE_LENS),
        packed_und_token_indexes=torch.tensor(list(range(und)), device=device),
        packed_gen_token_indexes=torch.tensor(list(range(und, und + gen)), device=device))


def bench(fn, iters, repeats, warmup=5):
    """Returns (median wall ms, median enqueue ms, spread %) over `repeats` independent runs."""
    walls, enqs = [], []
    for _ in range(repeats):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        enqs.append((time.perf_counter() - t0) / iters * 1e3)
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - t0) / iters * 1e3)
    spread = (max(walls) - min(walls)) / statistics.median(walls) * 100
    return statistics.median(walls), statistics.median(enqs), spread


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=EDGE["num_hidden_layers"])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--shim", action="store_true", help="force the torch-SDPA shim instead of cuDNN")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    dev, dt = torch.device("cuda"), torch.bfloat16
    torch.manual_seed(0)

    sha = os.popen("git -C /home/ubuntu/cosmos-framework rev-parse --short HEAD").read().strip()
    print("=" * 92)
    print("Cosmos3-Edge MoT stack -- InstinctWM engine, real served width")
    print("=" * 92)
    print(f"  torch {torch.__version__}  cudnn {torch.backends.cudnn.version()}  "
          f"{torch.cuda.get_device_name(0)}")
    print(f"  cosmos-framework @ {sha}   (uv group cu130-torch213, no flash-attn)")
    print(f"  config: {args.layers} layers, hidden {EDGE['hidden_size']}, "
          f"{EDGE['num_attention_heads']}q/{EDGE['num_key_value_heads']}kv heads, "
          f"head_dim {EDGE['head_dim']}, intermediate {EDGE['intermediate_size']}")
    print(f"  pack:   sample_lens={SAMPLE_LENS} split_lens={SPLIT_LENS} "
          f"modes={ATTN_MODES}, NFE={NFE}")
    print(f"  protocol: {args.iters} iters x {args.repeats} repeats, median reported, spread shown")

    # ---- attention backend -------------------------------------------------------------
    if args.shim:
        from instinctwm.adapters.cosmos3 import use_torch_sdpa
        backend = use_torch_sdpa()
    else:
        from cosmos_framework.model.attention.backends import choose_backend
        from cosmos_framework.model.attention.masks import CausalType
        und, gen, tot = SPLIT_LENS[0], SPLIT_LENS[1], sum(SAMPLE_LENS)
        nq, nkv, hd = (EDGE["num_attention_heads"], EDGE["num_key_value_heads"], EDGE["head_dim"])
        sz = torch.Size
        bc = choose_backend(sz((1, und, nq, hd)), sz((1, und, nkv, hd)), sz((1, und, nkv, hd)),
                            dt, dev, requires_grad=False, is_causal=True,
                            causal_type=CausalType.TopLeft, is_varlen=False, raise_error=False)
        bf = choose_backend(sz((1, gen, nq, hd)), sz((1, tot, nkv, hd)), sz((1, tot, nkv, hd)),
                            dt, dev, requires_grad=False, is_causal=False, causal_type=None,
                            is_varlen=False, raise_error=False)
        backend = f"{bc} (causal) / {bf} (full)  -- REAL kernel, no shim"
    print(f"  attention backend: {backend}")

    cfg, layers = build_stack(args.layers, dev, dt)
    nparam = sum(p.numel() for l in layers for p in l.parameters())
    print(f"  params: {nparam / 1e9:.3f} B  ({nparam * 2 / 2**30:.2f} GiB bf16)")

    pack = build_pack(dev, cfg.hidden_size, dt)
    from cosmos_framework.data.generator.sequence_packing.runtime import get_all_seq, zeros_like
    from cosmos_framework.model.generator.mot.attention import SplitInfo

    hd = EDGE["head_dim"]
    cos, sin = zeros_like(pack, (-1, hd)), zeros_like(pack, (-1, hd))
    for p in (cos, sin):
        p["causal_seq"] = torch.randn_like(p["causal_seq"])
        p["full_only_seq"] = torch.randn_like(p["full_only_seq"])
    pos = (cos, sin)
    mask = SplitInfo(split_lens=list(SPLIT_LENS), attn_modes=list(ATTN_MODES),
                     sample_lens=list(SAMPLE_LENS), actual_len=sum(SAMPLE_LENS))

    def stack(inp):
        x = inp
        for l in layers:
            x = l(x, mask, pos)[0]
        return x

    # ---- 1. dependency derivation ------------------------------------------------------
    from instinctwm.adapters.cosmos3 import build_plan, state_roots
    from instinctwm.planners.deps import derive_signature

    print("\n--- 1. dependency signature (engine, unchanged) ------------------------------")
    with torch.no_grad():
        sig = derive_signature(lambda: stack(pack), roots=[pack, cos, sin],
                               name_roots=state_roots(layers, pack, pos))
    cap, why = sig.capturable()
    print(f"  {sig.n_ops} ops, {len(sig.reads)} external reads, {sig.unnamed_reads} unnamed")
    print(f"  capturable: {cap} ({why})")

    # ---- 2. one Plan, both executors ---------------------------------------------------
    from instinctwm.executors.executor import EagerExecutor, GraphExecutor

    print("\n--- 2. one Plan under EagerExecutor and GraphExecutor ------------------------")
    plan = build_plan(layers, mask, pos)
    eager, graph = EagerExecutor(plan, dev), GraphExecutor(plan, dev)
    graph.prepare()

    ref = get_all_seq(eager.run("mot_stack/default", pack=pack)).clone()
    got = get_all_seq(graph.run("mot_stack/default", pack=pack))
    nd = (got != ref).sum().item()
    md = (got.float() - ref.float()).abs().max().item()
    print(f"  graph replay vs eager oracle : differing={nd}  max|d|={md:.3e}  "
          f"{'BITEXACT' if nd == 0 else 'MISMATCH'}   (captures={graph.n_captures})")
    pack2 = build_pack(dev, cfg.hidden_size, dt, seed=7)
    ref2 = get_all_seq(eager.run("mot_stack/default", pack=pack2)).clone()
    got2 = get_all_seq(graph.run("mot_stack/default", pack=pack2))
    nd2 = (got2 != ref2).sum().item()
    print(f"  second pack, new values      : differing={nd2}  "
          f"captures still {graph.n_captures} (rebound, not recaptured)")

    # ---- 3. latency --------------------------------------------------------------------
    print("\n--- 3. latency ---------------------------------------------------------------")
    with torch.no_grad():
        r_ms, r_enq, r_sp = bench(lambda: stack(pack), args.iters, args.repeats)
    e_ms, e_enq, e_sp = bench(lambda: eager.run("mot_stack/default", pack=pack),
                              args.iters, args.repeats)
    g_ms, g_enq, g_sp = bench(lambda: graph.run("mot_stack/default", pack=pack),
                              args.iters, args.repeats)
    print(f"  {'':24s}{'fwd (ms)':>10s}{'enqueue':>10s}{'spread':>9s}"
          f"{'control step x16':>19s}{'vs raw eager':>14s}")
    for nm, ms, eq, sp in (("raw eager (no engine)", r_ms, r_enq, r_sp),
                           ("EagerExecutor", e_ms, e_enq, e_sp),
                           ("GraphExecutor (replay)", g_ms, g_enq, g_sp)):
        print(f"  {nm:24s}{ms:10.3f}{eq:10.3f}{sp:8.1f}%{ms * NFE:17.1f} ms{r_ms / ms:13.3f}x")

    # ---- 4. P8 target: get_all_seq alloc traffic, and its measured ceiling --------------
    print("\n--- 4. L3-P8 target: get_all_seq allocate-and-scatter ------------------------")
    import cosmos_framework.model.generator.mot.attention as mot_attn
    _orig = mot_attn.get_all_seq

    stats = {"calls": 0, "bytes": 0}

    def counting(p):
        out = _orig(p)
        stats["calls"] += 1
        stats["bytes"] += out.numel() * out.element_size()
        return out

    mot_attn.get_all_seq = counting
    with torch.no_grad():
        stack(pack)
    mot_attn.get_all_seq = _orig
    calls, byts = stats["calls"], stats["bytes"]
    print(f"  per forward      : {calls:5d} calls  {byts / 2**20:8.2f} MiB "
          f"({calls // args.layers} per layer -- attention.py:205-206, K and V live together)")
    print(f"  per control step : {calls * NFE:5d} calls  {byts * NFE / 2**30:8.2f} GiB (NFE {NFE})")
    print(f"  'all_seq' memo in pack: {'all_seq' in pack}  "
          f"(runtime.py:513; set_all_seq is never called, so every call allocates)")

    # Measured ceiling: two preallocated per-role buffers, rotated. Two slots is the minimum
    # that cannot alias, which is exactly ForwardScratchArena's structural safety argument.
    slot = [torch.empty(sum(SAMPLE_LENS), EDGE["num_key_value_heads"], EDGE["head_dim"],
                        device=dev, dtype=dt) for _ in range(2)]
    turn = {"i": 0}

    def pooled(p):
        if "all_seq" in p:
            return p["all_seq"]
        buf = slot[turn["i"]]
        turn["i"] ^= 1
        buf.zero_()
        ci, fi = p["_causal_indices"], p["_full_indices"]
        if p["causal_seq"].shape[0] > 0:
            buf[ci] = p["causal_seq"][: ci.shape[0]]
        if p["full_only_seq"].shape[0] > 0:
            buf[fi] = p["full_only_seq"][: fi.shape[0]]
        return buf

    with torch.no_grad():
        base = get_all_seq(stack(pack)).clone()
    mot_attn.get_all_seq = pooled
    with torch.no_grad():
        pooled_out = get_all_seq(stack(pack)).clone()
        p_ms, p_enq, p_sp = bench(lambda: stack(pack), args.iters, args.repeats)
    mot_attn.get_all_seq = _orig

    pnd = (pooled_out != base).sum().item()
    print(f"\n  ceiling probe -- 2-slot preallocated per-role scratch (no aliasing possible):")
    print(f"    bit-exact vs allocating path: differing={pnd} "
          f"max|d|={(pooled_out.float() - base.float()).abs().max().item():.3e}")
    print(f"    EAGER  {r_ms:8.3f} -> {p_ms:8.3f} ms/fwd   {r_ms / p_ms:.3f}x  (spread {p_sp:.1f}%)")

    # The shipped path is graph replay, where the allocations are baked into the graph's private
    # pool. If P8 buys nothing THERE, it is subsumed rather than merely small.
    mot_attn.get_all_seq = pooled
    gp_plan = build_plan(layers, mask, pos, model_id="cosmos3-edge-pooled")
    gp = GraphExecutor(gp_plan, dev)
    gp.prepare()
    gp_out = get_all_seq(gp.run("mot_stack/default", pack=pack)).clone()
    gp_ms, gp_enq, gp_sp = bench(lambda: gp.run("mot_stack/default", pack=pack),
                                 args.iters, args.repeats)
    mot_attn.get_all_seq = _orig
    gnd = (gp_out != base).sum().item()
    print(f"    GRAPH  {g_ms:8.3f} -> {gp_ms:8.3f} ms/fwd   {g_ms / gp_ms:.3f}x  "
          f"(spread {gp_sp:.1f}%, differing={gnd})")
    print(f"    control step: eager {r_ms * NFE:.1f} -> {p_ms * NFE:.1f} ms | "
          f"graph {g_ms * NFE:.1f} -> {gp_ms * NFE:.1f} ms")
    print(f"    => P8 ceiling on the CYCLE: {r_ms / p_ms:.3f}x eager, {g_ms / gp_ms:.3f}x on the "
          f"shipped (graph) path")

    print("\n" + "=" * 92)
    print("NOT a served-accuracy claim: random weights, no checkpoint. Claimed: op structure,")
    print("shapes, dependency derivation, capturability, replay equality, allocation traffic.")
    return 0 if nd == 0 and nd2 == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
