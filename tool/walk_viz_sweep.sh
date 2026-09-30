#!/usr/bin/env bash
# Side-by-side rollout videos for the walk visualization subset.
# Usage: tool/walk_viz_sweep.sh [output_dir]
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${1:-outputs/eval_runs/walk_viz}
MANIFEST=outputs/eval_runs/walk_viz_clips.json
mkdir -p "$OUT"

if [ ! -f "$MANIFEST" ]; then
  echo "missing $MANIFEST — run: python tool/build_walk_manifest.py" >&2
  exit 1
fi

HGPT264=storage/checkpoints/trackers/hgpt/pns_wo_priv264.onnx
HGPT216=storage/checkpoints/trackers/hgpt/pns_wo_priv216.onnx
COMMON=(
  --mocap_path storage/dataset/HumanTracker/motions
  --test_json "$MANIFEST"
  --rm_checkpoint storage/checkpoints/reward_model/best.pt
  --termination_metric whole_body
  --device cpu --rm_device cpu
  --multiprocessing_start_method spawn
  --video_interval 1 --video_width 1280 --video_height 720
  --workers 1
)

export PYTHONPATH=src MUJOCO_GL=cgl G1_VERSION=5010 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY_BASE=${PYTHON:-python}
PY_HGPT=$PY_BASE

run_vid () {
  local tracker=$1 profile=$2 scale=$3 vel=$4 onnx=$5
  local tag
  if [ "$scale" = "0" ]; then tag="${tracker}_clean"; else tag="${tracker}_${profile}_s${scale}_${vel}"; fi
  local vdir="$OUT/videos_$tag"
  local json="$OUT/${tag}.json"
  [ -f "$json" ] && { echo "[skip] $json"; return; }
  echo "=== viz $tag"
  local profile_flag=$profile
  [ "$scale" = "0" ] && profile_flag=none
  "$PY_HGPT" -m humantracker.eval.eval_parallel_tracker --tracker hgpt \
    --policy "$onnx" \
    --ref_noise "$profile_flag" --ref_noise_seed 0 \
    --ref_noise_scale "$scale" --ref_noise_vel "$vel" \
    "${COMMON[@]}" --video_dir "$vdir" --output_json "$json"
}

# Own-live only — this is the robot-input condition.
run_vid hgpt264 gmr_stream 1.0 filtered "$HGPT264"
run_vid hgpt216 gmr_stream 1.0 filtered "$HGPT216"

echo WALK_VIZ_SWEEP_DONE
