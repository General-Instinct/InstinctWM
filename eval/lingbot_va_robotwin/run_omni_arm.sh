#!/usr/bin/env bash
# Build and measure the vLLM-Omni comparison arm (see serve_omni_arm.py for why it exists).
#
#   ./run_omni_arm.sh build     # build .venv-omni from omni-arm-requirements.txt (~8.3 GB, once)
#   ./run_omni_arm.sh cycle     # control-cycle mean: D_base vs D
#   ./run_omni_arm.sh reset     # reset: MISS vs HIT -- where this cache actually acts
#   ./run_omni_arm.sh all       # build + cycle + reset
#   ./run_omni_arm.sh stop      # stop a leftover arm
#
# Design rules, each of which exists because of a specific way this measurement can go wrong:
#
#  1. BOTH ARMS ARE MEASURED HERE, in .venv-omni. The stack differs from $IWM_SERVER_PY, so an
#     absolute ms from this arm minus an absolute ms from a serve_variant arm measures the torch
#     version, not the optimization. Only D_base - D is meaningful, so this script always
#     produces both.
#
#  2. ONE MEASUREMENT CLIENT for both arms ($IWM_SERVER_PY). The client cost is then identical
#     on both sides and cancels in the delta.
#
#  3. THE CACHE ACTS ON reset, NOT ON THE CYCLE. `probe_latency`'s full-cycle mean excludes the
#     reset, so `cycle` is EXPECTED to show no difference -- that is a result, not a bug, and it
#     is why `reset` exists as a separate task. Only `--repeats 1` prints the reset line
#     (probe_latency.py:130-131 is after the --repeats>1 early return).
#
#  4. VERIFY `bypassed: 0`. PromptEmbedCache silently bypasses on any unhashable argument. A run
#     with a climbing `bypassed` counter measured the uncached path and must be thrown away.
#
#  5. KILL BY PROCESS GROUP, never `pkill -f`. servers.sh documents why: a pattern match hits the
#     caller's own command line and kills the operator's shell. It has happened.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

PORT=${PORT:-$(iwm_ws_port 0)}
RDZV=${RDZV:-$(iwm_rdzv_port 0)}
GPU=${GPU:-0}
CYCLES=${CYCLES:-10}
REPEATS=${REPEATS:-3}
LOCK="$IWM_ROOT/eval/lingbot_va_robotwin/omni-arm-requirements.txt"

die() { echo "ERROR: $*" >&2; exit 1; }

stop_arm() {
  local pid pg
  pid=$(ss -ltnp "sport = :$PORT" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1)
  [ -n "$pid" ] || return 0
  pg=$(ps -o pgid= -p "$pid" | tr -d ' ')
  kill -TERM -- "-$pg" 2>/dev/null
  for _ in $(seq 30); do iwm_port_busy "$PORT" || return 0; sleep 2; done
  echo "WARN: port $PORT still busy" >&2
}

start_arm() {                      # start_arm <logfile> [extra serve_omni_arm.py flags...]
  local log=$1; shift
  iwm_port_busy "$PORT" && die "port $PORT already in use -- run './run_omni_arm.sh stop' first"
  ( cd "$LINGBOT_ROOT" && \
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH="$IWM_FA_SHIM_DIR" LINGBOT_CKPT="$LINGBOT_CKPT" \
    setsid nohup "$IWM_OMNI_PY" -m torch.distributed.run \
      --nproc_per_node 1 --master_port "$RDZV" \
      "$IWM_ROOT/eval/lingbot_va_robotwin/serve_omni_arm.py" \
      --port "$PORT" --save_root "$IWM_VIS_DIR" "$@" > "$log" 2>&1 & )
  echo "  starting (first load of the 23 GB checkpoint takes 1-3 min)..."
  for _ in $(seq 120); do
    iwm_port_busy "$PORT" && { grep -m1 "serve_omni_arm:" "$log"; return 0; }
    grep -qE "Traceback|ChildFailedError" "$log" 2>/dev/null && { tail -25 "$log"; die "arm failed to start"; }
    sleep 5
  done
  die "timed out waiting for the arm; see $log"
}

probe_cycle() {
  "$IWM_SERVER_PY" probe_latency.py --port "$PORT" --cycles "$CYCLES" --repeats "$REPEATS" 2>&1 | tail -2
}

probe_reset() {                    # probe_reset <prompt> -> "NN.N ms"
  "$IWM_SERVER_PY" probe_latency.py --port "$PORT" --cycles 2 --repeats 1 --prompt "$1" 2>&1 \
    | grep -E "^reset" | grep -oE "[0-9.]+ ms" | head -1
}

show_stats() {                     # rule 4
  echo "-- cache stats (bypassed MUST be 0) --"
  grep "prompt_embed_cache stats" "$1" | tail -3
}

do_build() {
  command -v uv >/dev/null || die "uv not on PATH: curl -LsSf https://astral.sh/uv/install.sh | sh"
  [ -f "$LOCK" ] || die "missing $LOCK -- regenerate with: uv pip compile omni-arm-requirements.in --python-version 3.10 -o omni-arm-requirements.txt"
  if [ -x "$IWM_OMNI_PY" ]; then echo "  .venv-omni already present, skipping"; return 0; fi
  ( cd "$IWM_ROOT" && \
    uv venv --python 3.10 .venv-omni && \
    VIRTUAL_ENV="$IWM_ROOT/.venv-omni" uv pip sync "$LOCK" && \
    VIRTUAL_ENV="$IWM_ROOT/.venv-omni" uv pip install --no-deps -e . ) || die "build failed"
  "$IWM_OMNI_PY" -c "
from vllm_omni.diffusion.cache import install_prompt_embed_cache
import torch, diffusers, transformers
print('  OK: torch', torch.__version__, '| diffusers', diffusers.__version__,
      '| transformers', transformers.__version__)" || die "built env cannot import the cache"
}

do_cycle() {
  echo "===== control-cycle mean: D_base (no cache) ====="
  stop_arm; start_arm "$IWM_LOG_DIR/omni_Dbase.log"
  probe_cycle
  echo
  echo "===== control-cycle mean: D (PromptEmbedCache) ====="
  stop_arm; start_arm "$IWM_LOG_DIR/omni_D.log" --prompt-embed-cache
  probe_cycle
  show_stats "$IWM_LOG_DIR/omni_D.log"
  echo "  (no difference here is EXPECTED -- see rule 3 in this file's header)"
  stop_arm
}

do_reset() {
  echo "===== reset: MISS (fresh prompt) vs HIT (repeated prompt) ====="
  stop_arm; start_arm "$IWM_LOG_DIR/omni_reset.log" --prompt-embed-cache
  echo "  (round 1's HIT column is that prompt's first sighting, i.e. a miss -- expected)"
  for i in 1 2 3 4 5; do
    m=$(probe_reset "unique instruction number $i lift the bottle")
    h=$(probe_reset "a fixed repeated instruction for cache hits")
    echo "  round $i:  MISS=$m   HIT=$h"
  done
  show_stats "$IWM_LOG_DIR/omni_reset.log"
  stop_arm
}

[ -x "$IWM_SERVER_PY" ] || die "no measurement client: $IWM_SERVER_PY (run ../../scripts/task.sh test-lingbot)"
[ -d "$LINGBOT_CKPT" ] || die "no checkpoint: $LINGBOT_CKPT"
[ -d "$IWM_FA_SHIM_DIR/flash_attn" ] || die "no flash-attn shim: $IWM_FA_SHIM_DIR/flash_attn"

case "${1:-all}" in
  build) do_build ;;
  cycle) do_cycle ;;
  reset) do_reset ;;
  all)   do_build; do_cycle; echo; do_reset ;;
  stop)  stop_arm; echo "stopped" ;;
  -h|--help|help) sed -n '2,8p' "${BASH_SOURCE[0]}" ;;
  *) echo "usage: $0 {build|cycle|reset|all|stop}" >&2; exit 2 ;;
esac
