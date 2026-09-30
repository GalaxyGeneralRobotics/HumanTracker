#!/usr/bin/env bash
# Expanded Daily/walk eval with GT-free feel metrics.
# Manifest: outputs/eval_runs/walk_expand_clips.json
#
# Usage: tool/walk_expand_sweep.sh [output_dir] [tracker ...]
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${1:-outputs/eval_runs/noise_sweep_expand}
shift || true
WANT=("$@")
MANIFEST=outputs/eval_runs/walk_expand_clips.json
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
)

export PYTHONPATH=src MUJOCO_GL=cgl G1_VERSION=5010 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY_BASE=${PYTHON:-python}
PY_HGPT=$PY_BASE
python_for () {
  case $1 in
    hgpt216|hgpt264) echo "$PY_HGPT" ;;
    *) echo "$PY_BASE" ;;
  esac
}
for interp in "$PY_BASE" "$PY_HGPT"; do
  command -v "$interp" >/dev/null || { echo "missing interpreter: $interp" >&2; exit 1; }
done

tag () {
  if [ "$3" = "0" ]; then echo "$1_clean"; else echo "$1_$2_s$3_$4"; fi
}

run () {
  local out; out="$OUT/$(tag "$@").json"
  [ -f "$out" ] && { echo "[skip] $out"; return; }
  local profile=$2 scale=$3
  [ "$scale" = "0" ] && profile=none
  local noise=(--ref_noise "$profile" --ref_noise_seed 0
               --ref_noise_scale "$scale" --ref_noise_vel "$4")
  echo "=== $(tag "$@")"
  local py; py=$(python_for "$1")
  case $1 in
    hgpt264)
      "$py" -m humantracker.eval.eval_parallel_tracker --tracker hgpt \
        --policy "$HGPT264" \
        "${noise[@]}" "${COMMON[@]}" --workers 4 --output_json "$out" ;;
    hgpt216)
      "$py" -m humantracker.eval.eval_parallel_tracker --tracker hgpt \
        --policy "$HGPT216" \
        "${noise[@]}" "${COMMON[@]}" --workers 4 --output_json "$out" ;;
    *) echo "unknown tracker $1" >&2; return 1 ;;
  esac
}

want () {
  local t=$1
  if [ ${#WANT[@]} -eq 0 ]; then return 0; fi
  local x
  for x in "${WANT[@]}"; do
    [ "$x" = "$t" ] && return 0
  done
  return 1
}

# Own-live first (robot-input ranking), then clean, then the other scales.
if want hgpt264; then
  run hgpt264 gmr_stream 1.0 filtered
  run hgpt264 none 0 clean
  run hgpt264 gmr_stream 0.5 filtered
  run hgpt264 gmr_stream 2.0 filtered
fi
if want hgpt216; then
  run hgpt216 gmr_stream 1.0 filtered
  run hgpt216 none 0 clean
  run hgpt216 gmr_stream 0.5 filtered
  run hgpt216 gmr_stream 2.0 filtered
fi

echo WALK_EXPAND_SWEEP_DONE
