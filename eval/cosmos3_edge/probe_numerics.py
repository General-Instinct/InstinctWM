#!/usr/bin/env python3
"""Can torch.compile's numeric deviation be pushed back to BITEXACT, and is it actually error?

`probe_compile.py` measures compile at 1.48x over graph replay, at max|d| = 2.812e-01 against the
eager oracle -- NUMERIC, not BITEXACT. This asks whether that tier cost is recoverable, four ways,
and then whether "not bit-exact" even means "less accurate".

  A  depth sweep      is it a per-layer math change, or accumulation?
  B  inductor knobs   emulate_precision_casts / split_reductions
  C  sub-module       compile norms / MLP / attention separately -- is there a bit-exact subset
     bisect           that still pays?
  D  fp32 oracle      both bf16 arms against a widened evaluation of the same weights. "Differs
                      from eager" is agreement with one rounding of one kernel schedule; it is
                      not, by itself, error.

Run:  /home/ubuntu/cosmos-framework/.venv/bin/python eval/cosmos3_edge/probe_numerics.py
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ubuntu/cosmos-framework")
sys.path.insert(0, "/home/ubuntu/Code/InstinctWM")

import torch
import probe_mot_stack as P

from cosmos_framework.data.generator.sequence_packing.runtime import get_all_seq, zeros_like
from cosmos_framework.model.generator.mot.attention import SplitInfo

NORMS = ("input_layernorm", "input_layernorm_moe_gen",
         "post_attention_layernorm", "post_attention_layernorm_moe_gen")
MLPS = ("mlp", "mlp_moe_gen")
ATTN = ("self_attn",)


def make(nl, dev, dt=torch.bfloat16):
    torch.manual_seed(0)
    cfg, layers = P.build_stack(nl, dev, dt)
    pack = P.build_pack(dev, cfg.hidden_size, dt)
    hd = P.EDGE["head_dim"]
    cos, sin = zeros_like(pack, (-1, hd)), zeros_like(pack, (-1, hd))
    for p in (cos, sin):
        p["causal_seq"] = torch.randn_like(p["causal_seq"])
        p["full_only_seq"] = torch.randn_like(p["full_only_seq"])
    mask = SplitInfo(split_lens=list(P.SPLIT_LENS), attn_modes=list(P.ATTN_MODES),
                     sample_lens=list(P.SAMPLE_LENS), actual_len=sum(P.SAMPLE_LENS))
    return layers, pack, mask, (cos, sin)


def run(ls, pk, mask, ps):
    x = pk
    for l in ls:
        x = l(x, mask, ps)[0]
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=P.EDGE["num_hidden_layers"])
    ap.add_argument("--skip-depth", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("SKIP: needs CUDA")
        return 0
    dev = torch.device("cuda")
    torch._dynamo.config.cache_size_limit = 2048

    print("=" * 100)
    print("torch.compile numerics on Cosmos3-Edge -- is the tier cost recoverable?")
    print("=" * 100)

    # ---- A. depth ----------------------------------------------------------------------
    if not args.skip_depth:
        print("\nA. deviation vs depth (compile default). bf16 carries 8 mantissa bits.")
        print(f"  {'layers':>7s}{'differing':>11s}{'max|d|':>12s}{'max|ref|':>12s}"
              f"{'rel':>11s}{'~bf16 ulps':>12s}")
        for nl in (1, 2, 4, 8, args.layers):
            layers, pack, mask, pos = make(nl, dev)
            torch._dynamo.reset()
            comp = [torch.compile(l, dynamic=False) for l in layers]
            with torch.no_grad():
                ref = get_all_seq(run(layers, pack, mask, pos)).clone()
                got = get_all_seq(run(comp, pack, mask, pos)).clone()
            sc = ref.float().abs().max().item()
            md = (got.float() - ref.float()).abs().max().item()
            print(f"  {nl:7d}{(got != ref).sum().item():11d}{md:12.3e}{sc:12.3e}"
                  f"{md / sc:11.2e}{md / sc * 256:12.1f}")
            del layers, comp
            torch.cuda.empty_cache()

    # ---- shared full-depth setup -------------------------------------------------------
    layers, pack, mask, pos = make(args.layers, dev)
    from instinctwm.adapters.cosmos3 import build_plan
    from instinctwm.executors.executor import GraphExecutor

    with torch.no_grad():
        ref = get_all_seq(run(layers, pack, mask, pos)).clone()
        e_ms, _, _ = P.bench(lambda: run(layers, pack, mask, pos), 20, 3)
    g0 = GraphExecutor(build_plan(layers, mask, pos), dev)
    g0.prepare()
    g_ms, _, _ = P.bench(lambda: g0.run("mot_stack/default", pack=pack), 20, 3)
    scale = ref.float().abs().max().item()
    print(f"\n  eager {e_ms:.3f} ms/fwd | graph replay {g_ms:.3f} ms/fwd  BITEXACT baseline"
          f" | max|ref| {scale:.3f}")

    def measure(tag, model_id):
        # `layers` is mutated in place by the bisect: sub-modules are swapped for compiled ones,
        # so the same list object is the variant.
        gc_ = GraphExecutor(build_plan(layers, mask, pos, model_id=model_id), dev)
        gc_.prepare()
        got = get_all_seq(gc_.run("mot_stack/default", pack=pack)).clone()
        ms, _, _ = P.bench(lambda: gc_.run("mot_stack/default", pack=pack), 20, 3)
        n = (got != ref).sum().item()
        md = (got.float() - ref.float()).abs().max().item()
        print(f"  {tag:28s}{ms:9.3f}{g_ms / ms:9.3f}x{n:11d}{md:12.3e}{md / scale:11.2e}"
              f"{'   BITEXACT' if n == 0 else ''}")

    # ---- B. knobs ----------------------------------------------------------------------
    import torch._inductor.config as ind
    print("\nB. inductor knobs")
    print(f"  {'knob':28s}{'ms/fwd':>9s}{'vs graph':>10s}{'differing':>11s}{'max|d|':>12s}"
          f"{'rel':>11s}")
    saved = {k: getattr(ind, k, None) for k in ("emulate_precision_casts", "split_reductions")}
    for tag, kw in (("default", {}),
                    ("emulate_precision_casts", {"emulate_precision_casts": True}),
                    ("no split_reductions", {"split_reductions": False}),
                    ("emulate + no split", {"emulate_precision_casts": True,
                                            "split_reductions": False})):
        for k, v in saved.items():
            if v is not None:
                setattr(ind, k, v)
        for k, v in kw.items():
            setattr(ind, k, v)
        torch._dynamo.reset()
        comp = [torch.compile(l, dynamic=False) for l in layers]
        gc_ = GraphExecutor(build_plan(comp, mask, pos, model_id=f"kn-{abs(hash(tag)) % 9999}"), dev)
        gc_.prepare()
        got = get_all_seq(gc_.run("mot_stack/default", pack=pack)).clone()
        ms, _, _ = P.bench(lambda: gc_.run("mot_stack/default", pack=pack), 20, 3)
        n = (got != ref).sum().item()
        md = (got.float() - ref.float()).abs().max().item()
        print(f"  {tag:28s}{ms:9.3f}{g_ms / ms:9.3f}x{n:11d}{md:12.3e}{md / scale:11.2e}"
              f"{'   BITEXACT' if n == 0 else ''}")
    for k, v in saved.items():
        if v is not None:
            setattr(ind, k, v)

    # ---- C. sub-module bisect ----------------------------------------------------------
    print("\nC. sub-module bisect -- is there a bit-exact subset that still pays?")
    pristine = [{n: getattr(l, n) for n in NORMS + MLPS + ATTN} for l in layers]

    def restore():
        for l, sv in zip(layers, pristine):
            for n, m in sv.items():
                setattr(l, n, m)

    print(f"  {'compiled sub-modules':28s}{'ms/fwd':>9s}{'vs graph':>10s}{'differing':>11s}"
          f"{'max|d|':>12s}{'rel':>11s}")
    for tag, names in (("norms only", NORMS), ("MLP only", MLPS), ("self_attn only", ATTN),
                       ("norms + MLP", NORMS + MLPS),
                       ("everything (sub-modules)", NORMS + MLPS + ATTN)):
        restore()
        for l in layers:
            for n in names:
                setattr(l, n, torch.compile(getattr(l, n), dynamic=False))
        measure(tag, f"bs-{abs(hash(tag)) % 9999}")
        torch._dynamo.reset()
    restore()

    # ---- D. fp32 oracle ----------------------------------------------------------------
    if args.skip_oracle:
        return 0
    print("\nD. both bf16 arms vs an fp32 evaluation of the same weights")
    with torch.no_grad():
        eager_bf16 = get_all_seq(run(layers, pack, mask, pos)).float().clone()
    comp = [torch.compile(l, dynamic=False) for l in layers]
    with torch.no_grad():
        comp_bf16 = get_all_seq(run(comp, pack, mask, pos)).float().clone()
    del comp
    torch._dynamo.reset()
    torch.cuda.empty_cache()

    from instinctwm.adapters.cosmos3 import use_torch_sdpa
    print(f"  CAVEAT: the fp32 oracle runs {use_torch_sdpa()} because Cosmos's cuDNN backend is")
    print("  bf16/fp16. Both bf16 arms meet the SAME oracle, so the comparison BETWEEN them is")
    print("  sound; the absolute distances carry the oracle's own attention difference.")
    f32 = copy.deepcopy(layers)
    for l in f32:
        l.float()
    up = lambda d: {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
                    for k, v in d.items()}
    with torch.no_grad():
        oracle = get_all_seq(run(f32, up(pack), mask, tuple(up(p) for p in pos))).float().clone()
    osc = oracle.abs().max().item()
    orms = oracle.pow(2).mean().sqrt().item()
    print(f"\n  {'arm':20s}{'max|d| vs fp32':>16s}{'rel':>11s}{'RMS':>12s}{'rel RMS':>11s}")
    for tag, arm in (("eager bf16", eager_bf16), ("compiled bf16", comp_bf16)):
        d = (arm - oracle).abs()
        rms = d.pow(2).mean().sqrt().item()
        print(f"  {tag:20s}{d.max().item():16.4e}{d.max().item() / osc:11.2e}"
              f"{rms:12.4e}{rms / orms:11.2e}")
    de, dc = (eager_bf16 - oracle).abs(), (comp_bf16 - oracle).abs()
    closer, tie, n = (dc < de).sum().item(), (dc == de).sum().item(), oracle.numel()
    ratio = dc.pow(2).mean().sqrt().item() / de.pow(2).mean().sqrt().item()
    print(f"\n  compiled closer to fp32 on {closer / n * 100:.1f}% of elements, eager closer on "
          f"{(n - closer - tie) / n * 100:.1f}%, tied {tie / n * 100:.1f}%")
    print(f"  RMS error ratio compiled/eager = {ratio:.4f}  "
          f"({'compiled is MORE accurate' if ratio < 1 else 'compiled is LESS accurate'})")
    print("\n" + "=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
