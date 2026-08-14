#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )) || [[ ! "$1" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "Usage: bash tool/rm_pipeline/prerender.sh YYYYMMDD_HHMMSS" >&2
  exit 2
fi

run_id="$1"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${PYTHON:-python}"
pair_root=$repo/storage/preference_pair/$run_id
scene=$repo/storage/assets/unitree_g1_5010/scene_mjx_track_papergray.xml
cache_root=$repo/tool/motion_annotation/.video_cache/$run_id

command -v "$python" >/dev/null
command -v ffmpeg >/dev/null
test -f "$pair_root/PAIR_COMPLETE"
test -f "$pair_root/pairs.jsonl"
test -f "$scene"
test ! -e "$pair_root/PRERENDER_COMPLETE"
test ! -e "$pair_root/prerender.pid"

printf '%s\n' "$$" >"$pair_root/prerender.pid"
exec >>"$pair_root/prerender.log" 2>&1

export PYTHONPATH="$repo:$repo/tool/motion_annotation"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

child_pid=""
stop_child() {
  if [[ -n "$child_pid" ]]; then
    kill -- "-$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
}
trap stop_child EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[prerender] started pid=$$"
cd "$repo/tool/motion_annotation"
setsid "$python" -u -m web_labeler.prerender_all \
  --task-file "$pair_root/pairs.jsonl" \
  --scene-xml "$scene" \
  --cache-root "$cache_root" \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --combine-workers 8 &
child_pid=$!
wait "$child_pid"
child_pid=""
touch "$pair_root/PRERENDER_COMPLETE"
echo "[prerender] complete"
