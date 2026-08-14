#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: bash tool/rm_pipeline/rollout.sh <run_id>" >&2
  exit 2
fi

run_id="$1"
if [[ ! "$run_id" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
  echo "run_id must use YYYYMMDD_HHMMSS" >&2
  exit 2
fi

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python="${PYTHON:-python}"
dataset="${HUMANTRACKER_DATASET:?set HUMANTRACKER_DATASET to the motion dataset root}"
manifest=$dataset/train.json
checkpoint="${HUMANTRACKER_RM_CHECKPOINT:-$repo/storage/checkpoints/reward_model/best.pt}"
scene=$repo/storage/assets/unitree_g1_5010/scene_mjx_track_papergray.xml
robot=$repo/storage/assets/unitree_g1_5010/g1_mjx_track_papergray.xml
hgpt_policy=$repo/thirdparty/Humanoid-GPT/storage/ckpts/pns_wo_priv216.onnx
rollout_root=$repo/storage/dataset/tracker_rollouts/$run_id
clip_root=$rollout_root/clips
pair_root=$repo/storage/preference_pair/$run_id

command -v "$python" >/dev/null
for path in "$manifest" "$checkpoint" "$scene" "$robot" "$hgpt_policy"; do
  test -e "$path"
done
test ! -e "$rollout_root"
test ! -e "$pair_root"

mkdir -p "$rollout_root/logs" "$rollout_root/results" "$rollout_root/pids"
exec >"$rollout_root/pipeline.log" 2>&1
printf '%s\n' "$$" >"$rollout_root/pipeline.pid"
cp "$0" "$rollout_root/rollout.sh"

cd "$repo"
export PYTHONPATH="$repo"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export G1_VERSION=5010
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

trackers=(sonic twist2 hgpt gmt)
gpus=(4 5 6 7)
pids=()

cleanup() {
  set +e
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid" 2>/dev/null
  done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "[pipeline] run_id=$run_id"
echo "[pipeline] rollout_root=$rollout_root"
echo "[pipeline] pair_root=$pair_root"

for index in "${!trackers[@]}"; do
  tracker=${trackers[$index]}
  gpu=${gpus[$index]}
  extra=()
  if [[ "$tracker" == hgpt ]]; then
    extra=(--policy "$hgpt_policy" --robot_xml "$robot")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$python" -m humantracker.eval.eval_parallel_tracker \
    --tracker "$tracker" \
    --termination_metric trunk \
    --mocap_path "$dataset" \
    --test_json "$manifest" \
    --skip_flipped \
    --workers 8 \
    --rm_checkpoint "$checkpoint" \
    --xml_path "$scene" \
    --video_interval 0 \
    --rollout_dir "$rollout_root" \
    --rollout_run_id "$run_id" \
    --rollout_tracker "$tracker" \
    --output_json "$rollout_root/results/$tracker.json" \
    "${extra[@]}" >"$rollout_root/logs/$tracker.log" 2>&1 &
  pids+=("$!")
  printf '%s\n' "${pids[$index]}" >"$rollout_root/pids/$tracker.pid"
  echo "[pipeline] tracker=$tracker gpu=$gpu pid=${pids[$index]}"
done

for index in "${!pids[@]}"; do
  tracker=${trackers[$index]}
  if wait "${pids[$index]}"; then
    echo "[pipeline] tracker=$tracker complete"
  else
    status=$?
    echo "[pipeline] tracker=$tracker failed status=$status" >&2
    exit "$status"
  fi
done
pids=()

"$python" -m tool.rm_pipeline.verify_rollout_dataset "$manifest" "$rollout_root"
touch "$rollout_root/ROLLOUT_COMPLETE"
exec bash "$repo/tool/rm_pipeline/pairs.sh" "$run_id"
