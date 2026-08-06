"""LingBot-VA backend adapter — declarations, plus the thin glue that serves them.

Every field in `lingbot_va_spec()` is a fact read out of the upstream lingbot-va tree (set
`LINGBOT_ROOT`; the historical default is `/home/ubuntu/lingbot-va`), cited to `file:line`.
Nothing there is an optimization; the optimizer derives those. If you find yourself wanting to
add a `use_fast_path=True` field, it belongs in a pass instead.

`LingBotVA` at the bottom is the `BackendAdapter` implementation: declarations, plus install
and serve. It holds no optimization logic either — it maps pass names onto installers and
refuses to serve a plan it could not fully apply.
"""

from __future__ import annotations

from typing import Sequence

from instinctwm.adapter.base import (
    AdapterSpec, CommitMode, GuidanceMode, GuidanceRule, KVLifetime, KVStreamSpec,
    PhaseSpec, PurityKey,
)
from instinctwm.kernels.lingbot_regions import lingbot_fusion_descriptor

# --- geometry, from wan_va/configs/va_robotwin_cfg.py --------------------------------------
_HEIGHT, _WIDTH = 256, 320
_FRAME_CHUNK = 2            # frame_chunk_size
_ACTION_PER_FRAME = 16      # action_per_frame
_PATCH = (1, 2, 2)          # patch_size
# env_type 'robotwin_tshape': cam_high full res, the two wrist cams at half res, composited into
# a T. latent grid = ((256//16)*3)//2 x (320//16) = 24 x 20, patched by (2,2) -> 12 x 10 = 120.
_VIDEO_TOKENS_PER_FRAME = ((_HEIGHT // 16) * 3 // 2) * (_WIDTH // 16) // (_PATCH[1] * _PATCH[2])
_ACTION_TOKENS_PER_FRAME = _ACTION_PER_FRAME


def lingbot_va_spec() -> AdapterSpec:
    return AdapterSpec(
        model_id="lingbot-va-posttrain-robotwin",
        param_bytes=10_179_017_396,  # transformer safetensors, bf16

        # Two CO-EQUAL committed streams. Both are written by the same `update_cache` path
        # (model.py:444-447) and both are persisted with update_cache=2 in _compute_kv_cache
        # (wan_va_server.py:593-601), so both are attended in every later control step. The
        # action stream is NOT scratch — this is the fact that vLLM-Omni's single
        # `tokens_per_frame` + `max_scratch_tokens_per_branch` spec cannot express.
        streams=(
            KVStreamSpec(
                name="video",
                tokens_per_frame=_VIDEO_TOKENS_PER_FRAME,     # 120
                lifetime=KVLifetime.EPISODE,
                commit_mode=CommitMode.CONFIRMED,
                window_frames=72,                              # attn_window
                supports_provisional=True,                     # update_cache=1 writes is_pred
            ),
            KVStreamSpec(
                name="action",
                tokens_per_frame=_ACTION_TOKENS_PER_FRAME,     # 16
                lifetime=KVLifetime.EPISODE,
                commit_mode=CommitMode.CONFIRMED,
                window_frames=72,
                supports_provisional=True,
            ),
        ),

        # Three phases per control step. The video loop must fully complete before the action
        # loop starts: the 26th video forward writes provisional K/V that all 51 action forwards
        # read (wan_va_server.py:504-508 then :542-548). That is a hard barrier, declared here
        # via depends_on so the scheduler plans around it instead of discovering it.
        phases=(
            PhaseSpec(
                name="kv_refresh", nfe=2,
                reads=frozenset({"video", "action"}),
                writes=frozenset({"video", "action"}),
                commit_steps=frozenset({0, 1}),     # BOTH forwards run update_cache=2
            ),
            PhaseSpec(
                name="video", nfe=26,               # num_inference_steps 25, +1 padded timestep
                reads=frozenset({"video", "action"}),
                writes=frozenset({"video"}),
                commit_steps=frozenset({25}),       # last_step -> update_cache=1
                truncatable=True,                   # video_exec_step already exists, set to -1
                min_nfe=1,
                depends_on=("kv_refresh",),
            ),
            PhaseSpec(
                name="action", nfe=51,              # action_num_inference_steps 50, +1 padded
                reads=frozenset({"video", "action"}),
                writes=frozenset({"action"}),
                commit_steps=frozenset({50}),
                truncatable=True,
                min_nfe=1,
                depends_on=("video",),
            ),
        ),

        # guidance_scale=5 on video, action_guidance_scale=1 on action. The action stream's
        # combine takes the else branch and keeps [:1] (wan_va_server.py:552-555), i.e. its
        # negative branch is computed and discarded. Declared as a FACT; CFGBranchElision is
        # what turns it into an optimization.
        guidance={
            "video": GuidanceRule(mode=GuidanceMode.CFG, scale=5.0, batchable=True),
            "action": GuidanceRule(mode=GuidanceMode.POSITIVE_ONLY, scale=1.0, batchable=True),
        },

        # Episode-constant conditioning. The instruction is encoded once in _reset
        # (wan_va_server.py:421-435) and never changes within an episode, yet cross-attention
        # gets attn_caches=None (model.py:331) so its k/v projections over the 512-token text
        # embedding are recomputed in all 30 layers on all 77 forwards.
        purity=(
            PurityKey(artifact="text_kv", fields=("prompt",), scope=KVLifetime.EPISODE),
            PurityKey(artifact="negative_text_kv", fields=("negative_prompt",),
                      scope=KVLifetime.EPISODE),
        ),

        # The predicted video is never consumed by the RoboTwin client — it asks only for
        # actions. `_infer` returns latents that the caller drops (wan_va_server.py:623-624).
        obs_decode_modules=("vae.decoder",),

        # The block's two fusible op sequences, read out of `modules/model.py:515-566`. Facts
        # about what the eager path materialises, not a request to fuse anything: `OperatorFusion`
        # reads them, and the LayerNorm inside `pre_attention_modulated_norm` is what keeps that
        # region out of the BITEXACT tier for every kernel registered so far.
        fusion=lingbot_fusion_descriptor(),

        notes={
            "attn_mode": "torch (custom_sdpa); forced by the server and by transformer/config.json",
            "kv_pool": "9792 slots, grows 272 tokens/cycle, saturates ~cycle 36, 6.72 GiB",
            "measured_stock_cycle_ms": "8881 on idle H100 = 32 actions = 3.6 Hz",
            "known_sync": "model.py:451 mask.nonzero() per layer per forward",
            "known_gather": "model.py:452-453 key_pool[:, valid] re-gathers the whole pool",
        },
    )


class LingBotVA:
    """`BackendAdapter` for LingBot-VA posttrained on RoboTwin 2.0.

    The upstream server is patched at runtime rather than forked. That keeps `git status` in
    the lingbot-va checkout clean, keeps every variant one flag away from stock, and keeps the
    bit-exactness gate (`eval/lingbot_va_robotwin/probe_bitexact.py`) meaningful.
    """

    model_id = "lingbot-va-posttrain-robotwin"

    def __init__(self, lingbot_root: str | None = None):
        #: None means "resolve from LINGBOT_ROOT at serve time", so constructing an adapter
        #: never touches the filesystem — `spec()` must work on a box with no checkpoint.
        self.lingbot_root = lingbot_root

    def spec(self) -> AdapterSpec:
        return lingbot_va_spec()

    def install(self, server_module: object, plan) -> Sequence[str]:
        # Imported here, not at module scope: the runtime layer needs torch, and reading a
        # model's declarations must not.
        from instinctwm.runtime.lingbot_install import install_plan

        return install_plan(server_module, server_module.VA_Server, plan)

    def serve(
        self,
        plan,
        port: int,
        config_name: str = "robotwin",
        save_root: str | None = None,
        deterministic_seed: int | None = None,
    ):
        """Import the upstream server, install `plan`, and run it on `port`.

        Installation happens BEFORE `run()` because `run()` is what builds the model, and
        `fsdp_elision` replaces the bound `_configure_model` the build calls through.
        """
        from instinctwm.runtime.lingbot_install import (
            import_lingbot_server,
            install_deterministic_seed,
        )

        server_module = import_lingbot_server(self.lingbot_root)
        applied = list(self.install(server_module, plan))
        if deterministic_seed is not None:
            applied += install_deterministic_seed(server_module, deterministic_seed)

        print("=" * 72, flush=True)
        print(f"InstinctWM serving {self.model_id} on :{port}", flush=True)
        print(f"  plan tier : {plan.tier().name}", flush=True)
        print(f"  applied   : {applied if applied else ['STOCK BASELINE']}", flush=True)
        print("=" * 72, flush=True)

        args = _ServerArgs(config_name=config_name, port=port, save_root=save_root)
        server_module.init_logger()
        return server_module.run(args)


class _ServerArgs:
    """The three attributes upstream's `run()` reads off its argparse namespace."""

    def __init__(self, config_name: str, port: int | None, save_root: str | None):
        self.config_name = config_name
        self.port = port
        self.save_root = save_root
