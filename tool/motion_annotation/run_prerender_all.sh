#!/usr/bin/env bash
# One-click launcher for the pairwise labeler, pre-render pipeline and public tunnel.
#
# Required environment:
#   TASK_FILE    pairs.jsonl produced by tool/rm_pipeline
#   HF_LOGS_DIR  directory annotation parquet shards are written to
#
# Optional: SCENE_XML, CACHE_ROOT, GRADIO_FRPC, PYTHON.

set -euo pipefail

if (( $# != 0 )); then
  echo "Usage: bash tool/motion_annotation/run_prerender_all.sh" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python}"
GRADIO_FRPC="${GRADIO_FRPC:-$HOME/.cache/huggingface/gradio/frpc/frpc_linux_amd64_v0.3}"
TASK_FILE="${TASK_FILE:?set TASK_FILE to a pairs.jsonl produced by the rm_pipeline}"
HF_LOGS_DIR="${HF_LOGS_DIR:?set HF_LOGS_DIR to the directory annotations are written to}"
SCENE_XML="${SCENE_XML:-$REPO_ROOT/storage/assets/unitree_g1_5010/scene_mjx_track_papergray.xml}"
CACHE_ROOT="${CACHE_ROOT:-$SCRIPT_DIR/.video_cache}"
GPU_IDS="0,1,2,3,4,5,6,7"
PORT=7860
PUBLIC_TUNNEL_DIR="$SCRIPT_DIR/.public_tunnel"
PUBLIC_TUNNEL_LOG="$PUBLIC_TUNNEL_DIR/gradio.log"
PUBLIC_URL_FILE="$PUBLIC_TUNNEL_DIR/url"

if ! command -v "$PYTHON" >/dev/null; then
  echo "Missing Python interpreter: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$TASK_FILE" ]]; then
  echo "Missing task file: $TASK_FILE" >&2
  exit 1
fi
if [[ ! -f "$SCENE_XML" ]]; then
  echo "Missing scene XML: $SCENE_XML" >&2
  exit 1
fi
if ! command -v ffmpeg >/dev/null; then
  echo "Missing required command: ffmpeg" >&2
  exit 1
fi
if [[ ! -x "$GRADIO_FRPC" ]]; then
  echo "Missing executable Gradio tunnel client: $GRADIO_FRPC" >&2
  exit 1
fi
if ! command -v flock >/dev/null; then
  echo "Missing required command: flock" >&2
  exit 1
fi
if ! command -v setsid >/dev/null; then
  echo "Missing required command: setsid" >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:$SCRIPT_DIR"
export MUJOCO_GL=egl
cd "$SCRIPT_DIR"

mkdir -p "$PUBLIC_TUNNEL_DIR"
exec 9>"$SCRIPT_DIR/.run_prerender_all.lock"
if ! flock -n 9; then
  echo "run_prerender_all.sh is already running" >&2
  exit 1
fi

echo "[launcher] task-file:   $TASK_FILE"
echo "[launcher] scene-xml:   $SCENE_XML"
echo "[launcher] cache-root:  $CACHE_ROOT"
echo "[launcher] hf-logs-dir: $HF_LOGS_DIR"
echo "[launcher] gpu-ids:     $GPU_IDS"
echo "[launcher] labeler:     http://0.0.0.0:$PORT"

prerender_pid=""
server_pid=""
tunnel_pid=""
cleanup() {
  for pid in "$prerender_pid" "$server_pid" "$tunnel_pid"; do
    if [[ -n "$pid" ]]; then
      kill -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "$prerender_pid" "$server_pid" "$tunnel_pid"; do
    if [[ -n "$pid" ]]; then
      wait "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

setsid "$PYTHON" -u -m web_labeler.prerender_all \
  --task-file "$TASK_FILE" \
  --scene-xml "$SCENE_XML" \
  --cache-root "$CACHE_ROOT" \
  --gpu-ids "$GPU_IDS" \
  --combine-workers 8 &
prerender_pid=$!

setsid "$PYTHON" -u -m web_labeler.app \
  --task-file "$TASK_FILE" \
  --hf-logs-dir "$HF_LOGS_DIR" \
  --cache-dir "$CACHE_ROOT" \
  --scene-xml "$SCENE_XML" \
  --annotations-per-pair 1 \
  --lease-seconds 120 \
  --port "$PORT" \
  --external-prerender &
server_pid=$!

rm -f "$PUBLIC_URL_FILE"
: >"$PUBLIC_TUNNEL_LOG"
setsid "$PYTHON" -u -m web_labeler.public_tunnel >"$PUBLIC_TUNNEL_LOG" 2>&1 &
tunnel_pid=$!

public_url=""
for _ in {1..30}; do
  if ! kill -0 "$tunnel_pid" 2>/dev/null; then
    wait "$tunnel_pid" || true
    cat "$PUBLIC_TUNNEL_LOG" >&2
    exit 1
  fi
  if [[ -s "$PUBLIC_URL_FILE" ]]; then
    read -r public_url <"$PUBLIC_URL_FILE"
    break
  fi
  sleep 1
done
if [[ -z "$public_url" ]]; then
  cat "$PUBLIC_TUNNEL_LOG" >&2
  echo "Timed out waiting for Gradio public URL" >&2
  exit 1
fi
echo "[launcher] public:      $public_url"

set +e
wait -n "$prerender_pid" "$server_pid" "$tunnel_pid"
first_status=$?
set -e

if ! kill -0 "$server_pid" 2>/dev/null; then
  exit "$first_status"
fi
if ! kill -0 "$tunnel_pid" 2>/dev/null; then
  cat "$PUBLIC_TUNNEL_LOG" >&2
  (( first_status != 0 )) && exit "$first_status"
  exit 1
fi
if (( first_status != 0 )); then
  exit "$first_status"
fi

echo "[launcher] prerender complete; labeler and public tunnel remain active"
set +e
wait -n "$server_pid" "$tunnel_pid"
service_status=$?
set -e
if ! kill -0 "$tunnel_pid" 2>/dev/null; then
  cat "$PUBLIC_TUNNEL_LOG" >&2
  (( service_status != 0 )) && exit "$service_status"
  exit 1
fi
exit "$service_status"
