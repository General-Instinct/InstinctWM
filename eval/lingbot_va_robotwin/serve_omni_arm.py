#!/usr/bin/env python
"""External reference arm: the stock server plus vLLM-Omni's `PromptEmbedCache`.

WHY THIS EXISTS

Every speedup in RESULTS.md is measured against the *unoptimized upstream server* -- a server
still paying world_size=1 FSDP, a per-chunk `empty_cache`, and a blocking D2H debug dump.
Beating that 3.3x is real, but it invites the obvious question: how does this compare to a
COMPETENT serving stack? This arm answers it by holding the model, checkpoint and message order
fixed and swapping exactly one component for the shipped third-party equivalent.

`serve_variant.py` is the A/B server for OUR passes; this is the same shape for THEIRS.

RUNS IN A DIFFERENT ENVIRONMENT ON PURPOSE
------------------------------------------
`$IWM_OMNI_PY` (.venv-omni: torch 2.11 / diffusers 0.38 / transformers 5.x / vllm-omni 0.26),
not `$IWM_SERVER_PY` (torch 2.9 / diffusers 0.36). Those two are dependency-incompatible --
vllm-omni >= 0.22 pins `diffusers==0.38.0` and vllm pins `torch==2.11.0` -- and merging them
would change the served numerics, which per RESULTS.md's appendix invalidates the accuracy
baseline. The arms talk over the websocket instead, exactly as the RoboTwin client (torch 2.4)
already coexists with the server (torch 2.9). See env.sh: "Never try to merge these."

BECAUSE the stack differs, absolute ms from this arm are NOT comparable to arms measured under
`$IWM_SERVER_PY`. Only the WITHIN-ENV delta is:

    D_base = this script without  --prompt-embed-cache
    D      = this script with     --prompt-embed-cache
    delta  = D_base - D           <- this is what may be compared to a pass's delta

`run_omni_arm.sh` runs both arms here so that constraint cannot be violated by accident.

THE TRAP: INSTANCE, NOT CLASS
-----------------------------
`PromptEmbedCache` builds its key from the *bound* arguments of `encode_prompt` and BYPASSES the
cache for any argument it cannot hash safely (tensors, custom objects) -- deliberately, to
guarantee correctness. Installed on the CLASS, `self` becomes argument 0 and is exactly such an
object: every call would bypass, the cache would never engage, and the arm would silently
measure the uncached path while reporting success. Installed on the INSTANCE, `encode_prompt` is
already bound and `self` never enters the key.

So the install is deferred into `VA_Server.__init__`, and `_reset` prints `stats()` after every
episode. A run whose `bypassed` counter climbs is a BROKEN EXPERIMENT, not a slow cache -- which
is why it is printed rather than assumed.

Usage (mirrors serve_variant.py):
    $IWM_OMNI_PY -m torch.distributed.run --nproc_per_node 1 --master_port 29800 \
        serve_omni_arm.py --port 29056 [--prompt-embed-cache]
"""
from __future__ import annotations

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="robotwin")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--save_root", default=os.environ.get("IWM_VIS_DIR", "/home/ubuntu/iwm_vis"))
    ap.add_argument("--prompt-embed-cache", action="store_true",
                    help="install vllm_omni.diffusion.cache.PromptEmbedCache on the server "
                         "instance. This is the ONLY variable between the two arms.")
    ap.add_argument("--deterministic-seed", type=int, default=None)
    args = ap.parse_args()

    from instinctwm.runtime.lingbot_install import (
        import_lingbot_server, install_allocator_churn_elision, install_debug_dump_elision,
        install_fsdp_elision,
    )

    S = import_lingbot_server()

    # The substrate three, so the arm sits at the same place in the chain as `serve_variant.py
    # --no-fsdp --no-empty-cache --no-debug-dump`. Both arms here get them, so they cancel in
    # the delta; they are applied only so the measurement is not dominated by known waste.
    applied: list[str] = []
    applied += install_fsdp_elision(S)
    applied += install_allocator_churn_elision(S)
    applied += install_debug_dump_elision(S)

    if args.deterministic_seed is not None:
        from instinctwm.runtime.lingbot_install import install_deterministic_seed
        applied += install_deterministic_seed(S, args.deterministic_seed)

    if args.prompt_embed_cache:
        from vllm_omni.diffusion.cache import install_prompt_embed_cache

        _orig_init = S.VA_Server.__init__

        def _init(self, *a, **kw):
            _orig_init(self, *a, **kw)
            cache = install_prompt_embed_cache(self, model_tag="lingbot-va-posttrain-robotwin")
            if cache is None:
                raise RuntimeError(
                    "install_prompt_embed_cache returned None -- VA_Server has no encode_prompt, "
                    "or signature introspection failed. Refusing to run an arm that would "
                    "measure nothing while looking like it worked.")
            self._iwm_pec = cache
            print(f"[omni-arm] PromptEmbedCache installed: {cache}", flush=True)

        S.VA_Server.__init__ = _init

        _orig_reset = S.VA_Server._reset

        def _reset(self, prompt=None):
            out = _orig_reset(self, prompt=prompt)
            c = getattr(self, "_iwm_pec", None)
            if c is not None:
                print(f"[omni-arm] prompt_embed_cache stats: {c.stats()}", flush=True)
            return out

        S.VA_Server._reset = _reset
        applied.append("vllm_omni.PromptEmbedCache")

    import diffusers
    import torch
    import transformers

    print("=" * 72, flush=True)
    print(f"InstinctWM serve_omni_arm: {applied}", flush=True)
    print(f"  stack : torch {torch.__version__} | diffusers {diffusers.__version__} "
          f"| transformers {transformers.__version__}", flush=True)
    print(f"  ckpt  : {os.environ.get('LINGBOT_CKPT')}", flush=True)
    print(f"  config: {args.config_name}   port: {args.port}", flush=True)
    print("=" * 72, flush=True)

    class _A:
        pass

    a = _A()
    a.config_name = args.config_name
    a.port = args.port
    a.save_root = args.save_root

    S.init_logger()
    S.run(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
