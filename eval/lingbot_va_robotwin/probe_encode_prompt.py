#!/usr/bin/env python
"""Time `VA_Server.encode_prompt` in isolation — the ceiling for any cache keyed there.

WHY THIS EXISTS

`encode_prompt` is called from exactly one place, `_reset` (`wan_va_server.py:426-436`), i.e.
ONCE PER EPISODE. So any cache keyed at that level — vLLM-Omni's `PromptEmbedCache`, or any
future equivalent — is bounded above by one `encode_prompt`, no matter how it is implemented or
how often it hits. Measuring that bound is stronger than measuring one implementation: the
result holds for all of them.

Use it to sanity-check a measured cache win. `run_omni_arm.sh reset` measures the real thing
end to end (~95 ms saved per hit); this says what the most any such cache could ever save is,
without needing the cache installed at all.

Built like `check_prompt_parity.py`: `VA_Server.__new__` so the 10 GB transformer is never
loaded, then attach only what `encode_prompt` touches. That keeps this on the real server code
path — same `prompt_clean`, same 512 padding, same dtype, both CFG branches — rather than
timing a reimplementation.

NOTE the reset round-trip reported by `probe_latency --repeats 1` is a DIFFERENT quantity: it
also clears the transformer KV cache and the VAE cache, which no prompt cache can skip. The two
have been observed to disagree (95 ms round-trip vs 113 ms isolated, direction unexplained —
the server runs in ASYNC mode and may return before T5 completes). Quote the LARGER as the
ceiling: an upper bound argued against yourself is the only kind worth publishing.

Run (after `source ./env.sh`):
    $IWM_SERVER_PY probe_encode_prompt.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

PROMPTS = [
    "Use the left arm to lift the plastic drink bottle head-up",
    "Position the red-capped bottle head-up and lift it",
    "Find the narrow-necked bottle on the table and raise it using the left arm.",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.environ.get("LINGBOT_CKPT"))
    ap.add_argument("--lingbot-root", default=os.environ.get("LINGBOT_ROOT", "/home/ubuntu/lingbot-va"))
    ap.add_argument("--iters", type=int, default=20, help="timed calls per prompt")
    ap.add_argument("--warmup", type=int, default=3,
                    help="discarded calls: cuBLAS algorithm selection and allocator warm-up "
                         "make the first calls up to 37%% slow (RESULTS.md section 8)")
    args = ap.parse_args()
    if not args.ckpt:
        print("LINGBOT_CKPT is unset; source ./env.sh first", file=sys.stderr)
        return 2

    sys.path.insert(0, os.path.join(args.lingbot_root, "wan_va"))
    from configs import VA_CONFIGS
    from modules.utils import load_text_encoder, load_tokenizer
    from wan_va_server import VA_Server

    cfg = VA_CONFIGS["robotwin"]
    cfg.wan22_pretrained_model_name_or_path = args.ckpt

    srv = VA_Server.__new__(VA_Server)
    srv.job_config = cfg
    srv.dtype = cfg.param_dtype
    srv.device = torch.device("cuda:0")
    srv.tokenizer = load_tokenizer(os.path.join(args.ckpt, "tokenizer"))
    srv.text_encoder = load_text_encoder(
        os.path.join(args.ckpt, "text_encoder"), torch_dtype=cfg.param_dtype,
        torch_device=srv.device)

    def call(p):
        return srv.encode_prompt(
            prompt=p, negative_prompt=None,
            do_classifier_free_guidance=cfg.guidance_scale > 1,
            num_videos_per_prompt=1, prompt_embeds=None, negative_prompt_embeds=None,
            max_sequence_length=512, device=srv.device, dtype=srv.dtype)

    for _ in range(args.warmup):
        call(PROMPTS[0])
    torch.cuda.synchronize()

    print(f"{'prompt':66s} {'mean ms':>9s} {'min':>8s} {'max':>8s}")
    every = []
    for p in PROMPTS:
        ts = []
        for _ in range(args.iters):
            torch.cuda.synchronize()
            t = time.perf_counter()
            call(p)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t) * 1000)
        every += ts
        print(f"{p[:64]:66s} {sum(ts)/len(ts):9.2f} {min(ts):8.2f} {max(ts):8.2f}")

    mean = sum(every) / len(every)
    print(f"\nencode_prompt (T5 + pad 512, both CFG branches): {mean:.2f} ms over {len(every)} calls")
    print("This is the CEILING for any cache keyed at encode_prompt, once per episode.")
    print("Compare against a per-CYCLE saving before concluding anything: at 10 cycles/episode a")
    print("288 ms/cycle pass saves 2884 ms, i.e. ~30x this number.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
