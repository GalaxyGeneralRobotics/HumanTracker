#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )) || [[ ! "$1" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "Usage: bash tool/rm_pipeline/pairs.sh YYYYMMDD_HHMMSS" >&2
  exit 2
fi

run_id="$1"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${PYTHON:-python}"
dataset="${HUMANTRACKER_DATASET:?set HUMANTRACKER_DATASET to the motion dataset root}"
rollout_root=$repo/storage/dataset/tracker_rollouts/$run_id
clip_root=$rollout_root/clips
pair_root=$repo/storage/preference_pair/$run_id

command -v "$python" >/dev/null
test -d "$dataset"
test -f "$rollout_root/ROLLOUT_COMPLETE"
test -f "$rollout_root/rollouts.jsonl"
test ! -e "$clip_root"
test ! -e "$pair_root"

printf '%s\n' "$$" >"$rollout_root/pipeline.pid"
printf '%s\n' "$$" >"$rollout_root/postprocess.pid"
exec >>"$rollout_root/pipeline.log" 2>&1

cd "$repo"
export PYTHONPATH="$repo"
export PYTHONUNBUFFERED=1
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

echo "[postprocess] clipping started pid=$$"
setsid "$python" -m tool.rm_pipeline clip-rollouts \
  --rollout-root "$rollout_root" \
  --manifest "$rollout_root/rollouts.jsonl" \
  --clips-root "$clip_root" \
  --run-id "$run_id" \
  --clip-seconds 5 \
  --keep-tail &
child_pid=$!
wait "$child_pid"
child_pid=""

"$python" -m tool.rm_pipeline.build_motion_pairs "$clip_root/clips.jsonl" "$pair_root" \
  --train-json "$dataset/train.json" \
  --dataset-root "$dataset"
"$python" -m tool.rm_pipeline validate-pairs --pairs "$pair_root/pairs.jsonl"
touch "$pair_root/PAIR_COMPLETE"

echo "[postprocess] pair construction complete"
exec bash "$repo/tool/rm_pipeline/prerender.sh" "$run_id"
