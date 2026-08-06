#!/usr/bin/env bash
# Single source of truth for the LingBot-VA x RoboTwin 2.0 evaluation pipeline.
# Source this; do not execute it.
#
# Port scheme (deliberately non-overlapping -- see NOTE below):
#   websocket serving port : IWM_WS_PORT_BASE  + gpu_index   -> 29056..29063
#   torch.distributed rdzv : IWM_RDZV_PORT_BASE + gpu_index  -> 29800..29807
#
# NOTE: the upstream launch scripts use START_PORT=29056 and MASTER_PORT=29061,
# i.e. the rendezvous port of GPU 0 collides with the websocket port of GPU 5
# once you fan out to 8 GPUs. That collision does not fail loudly at the fleet
# level -- 7 servers come up, one dies, and a client pointed at the dead port
# blocks forever in WebsocketClientPolicy._wait_for_server (it retries on *any*
# exception, every 5s, silently). Keep the two ranges far apart.

set -u

# ---- repos ------------------------------------------------------------------
# Derived from this file's own location rather than written down, because the tree has moved
# once (/home/ubuntu/InstinctWM -> /home/ubuntu/Code/InstinctWM) and a stale IWM_ROOT breaks
# only the arms that import instinctwm -- a broken A/B rather than a broken run.
# BASH_SOURCE is the right variable here: this file is sourced, never executed.
export IWM_ROOT=${IWM_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export ROBOTWIN_ROOT=${ROBOTWIN_ROOT:-/home/ubuntu/RoboTwin}
export LINGBOT_ROOT=${LINGBOT_ROOT:-/home/ubuntu/lingbot-va}

# ---- model ------------------------------------------------------------------
export LINGBOT_CKPT=${LINGBOT_CKPT:-/home/ubuntu/ckpt_lingbot/lingbot-va-posttrain-robotwin}

# ---- interpreters (two envs, on purpose) ------------------------------------
# The server needs torch 2.9 / diffusers 0.36; the client needs sapien 3.0.0b1 on
# torch 2.4. They are dependency-incompatible, which is exactly why upstream put a
# websocket between them. Never try to merge these.
#
# A lock for the server env now exists -- server-requirements.in next to this file,
# compiled to server-requirements.txt. It is deliberately not part of pyproject.toml:
# uv builds one universal lockfile, and sharing it would pin the development envs to
# the server's torch. That header explains it in full. Build it with:
#                 ./scripts/task.sh test-lingbot   (which syncs it first)
#          or:    uv venv .venv-server --python 3.10 \
#                   && VIRTUAL_ENV=.venv-server uv pip sync \
#                        eval/lingbot_va_robotwin/server-requirements.txt
#
# BUT THE DEFAULT BELOW IS STILL THE HAND-ROLLED VENV, ON PURPOSE.
# /home/ubuntu/.venv-lingbot is the environment every frozen number was measured in --
# P001's 2.11x through the 3.38x episode-mode chain -- including the deliberate removal
# of flash-attn. A venv rebuilt from the lock is equivalent only if the lock reproduces
# it exactly, and that is an empirical claim we have not tested. Flipping the default
# first and checking later would silently re-measure the whole chain against a different
# substrate.
#
# To migrate: build .venv-server, run probe_bitexact (max|delta action| = 0) and one
# probe_episode arm against .venv-lingbot, and only then change this line. The lock is
# the right destination -- a hand-rolled venv cannot be re-created from the repo -- but
# the parity check comes before the switch, not after.
export IWM_SERVER_PY=${IWM_SERVER_PY:-/home/ubuntu/.venv-lingbot/bin/python}
export IWM_CLIENT_PY=${IWM_CLIENT_PY:-${ROBOTWIN_ROOT}/.venv/bin/python}

# On a box that HAS the hand-rolled venv the line above wins and nothing below fires, so
# the frozen numbers keep their substrate. On a box that does not have it -- a fresh clone,
# or this one -- fall back to the lock-built env rather than dying on a missing interpreter.
#
# This is NOT the migration the block above defers. The migration is about which env is
# AUTHORITATIVE for a published number, and it still needs the probe_bitexact /
# probe_episode parity run against .venv-lingbot before the default may change. This is
# only about being runnable where .venv-lingbot does not exist and cannot be rebuilt.
# It announces itself on every source precisely so a number measured here is never
# mistaken for a reproduction -- silence is what commit abe41f5 was guarding against, not
# the fallback itself.
if [ ! -x "$IWM_SERVER_PY" ] && [ -x "${IWM_ROOT}/.venv-server/bin/python" ]; then
  export IWM_SERVER_PY="${IWM_ROOT}/.venv-server/bin/python"
  echo "NOTE: /home/ubuntu/.venv-lingbot is absent; IWM_SERVER_PY falls back to" >&2
  echo "      \$IWM_ROOT/.venv-server (built from server-requirements.txt)." >&2
  echo "      This is a DIFFERENT substrate from the one RESULTS.md was measured on." >&2
  echo "      Numbers from here are a new baseline, not a reproduction." >&2
fi

if [ ! -x "$IWM_SERVER_PY" ]; then
  echo "WARNING: IWM_SERVER_PY does not exist: $IWM_SERVER_PY" >&2
  echo "         run './scripts/task.sh test-lingbot' from $IWM_ROOT, or see the" >&2
  echo "         build command in $(basename "${BASH_SOURCE[0]}")" >&2
fi

# ---- third interpreter: the vLLM-Omni comparison arm ------------------------
# A THIRD env, for the same reason there are already two: it is dependency-incompatible with
# the other two and the websocket is the seam. vllm-omni pulls diffusers 0.38 / torch 2.11,
# the server pins diffusers 0.36 / torch 2.9. Built from its OWN lock next to this file
# (omni-arm-requirements.in -> .txt), never merged into server-requirements.
#
# It exists ONLY to run serve_omni_arm.py as an external reference arm. Because its stack
# differs, absolute ms from it are NOT comparable to arms measured under IWM_SERVER_PY --
# only the within-env delta is. run_omni_arm.sh enforces that by measuring both of its arms
# here. Build it with:  ./run_omni_arm.sh build
export IWM_OMNI_PY=${IWM_OMNI_PY:-${IWM_ROOT}/.venv-omni/bin/python}

# ---- flash-attn import shim -------------------------------------------------
# wan_va/modules/model.py imports flash_attn unconditionally at module scope even
# though the RoboTwin path runs attn_mode='torch'. Set IWM_FA_SHIM=1 to make the
# import-only shim visible. It raises if ever CALLED, so it cannot change numerics.
# Leave unset once a real flash-attn wheel is installed -- PYTHONPATH precedes
# site-packages and would otherwise shadow the real package.
export IWM_FA_SHIM_DIR=${IWM_FA_SHIM_DIR:-/home/ubuntu/iwm_shims}

# ---- ports ------------------------------------------------------------------
export IWM_WS_PORT_BASE=${IWM_WS_PORT_BASE:-29056}
export IWM_RDZV_PORT_BASE=${IWM_RDZV_PORT_BASE:-29800}
export IWM_NUM_GPUS=${IWM_NUM_GPUS:-8}

# ---- run artifacts ----------------------------------------------------------
export IWM_LOG_DIR=${IWM_LOG_DIR:-/home/ubuntu/iwm_logs}
export IWM_RESULT_DIR=${IWM_RESULT_DIR:-/home/ubuntu/iwm_results}
# The server dumps latents/actions/obs tensors here on EVERY chunk via save_async.
# On a full 50-task run this is the largest artifact by far -- keep it on the big disk.
export IWM_VIS_DIR=${IWM_VIS_DIR:-/home/ubuntu/iwm_vis}

mkdir -p "$IWM_LOG_DIR" "$IWM_RESULT_DIR" "$IWM_VIS_DIR"

iwm_ws_port()   { echo $(( IWM_WS_PORT_BASE   + $1 )); }
iwm_rdzv_port() { echo $(( IWM_RDZV_PORT_BASE + $1 )); }

# True if something is already listening on $1.
iwm_port_busy() {
  local p=$1
  if command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :${p}" 2>/dev/null | grep -q LISTEN
  else
    (exec 3<>"/dev/tcp/127.0.0.1/${p}") 2>/dev/null && { exec 3<&- 3>&-; return 0; } || return 1
  fi
}
