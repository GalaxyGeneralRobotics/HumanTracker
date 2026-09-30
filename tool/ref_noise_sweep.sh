#!/usr/bin/env bash
# Reference-noise robustness sweep across trackers.
#
# The trackers do not read the same reference fields, so the sweep is split:
#
#   vel=clean     pose and timing corrupted, reference velocity left alone.
#                 The setting comparable across trackers whatever reference
#                 fields they read.
#   vel=filtered  velocity from a causally low-passed noisy pose, i.e. what a
#                 real online stack reports.
#   vel=raw_diff  velocity differentiated straight off the noisy pose. Kept to
#                 show how much of a tracker's apparent fragility is that
#                 amplification rather than reference error.
#
# Usage: tool/ref_noise_sweep.sh [output_dir]
set -euo pipefail

OUT=${1:-outputs/eval_runs/noise_sweep_v2}
MANIFEST=outputs/eval_runs/ref_noise_sweep_clips.json
SCALES=${SCALES:-"0 0.5 1.0 2.0"}
mkdir -p "$OUT"

SONIC_DIR=storage/checkpoints/trackers/sonic
HGPT_ONNX=storage/checkpoints/trackers/hgpt/pns_wo_priv264.onnx

COMMON=(
  --mocap_path storage/dataset/HumanTracker/motions
  --test_json "$MANIFEST"
  --rm_checkpoint storage/checkpoints/reward_model/best.pt
  --termination_metric whole_body
  --device cpu --rm_device cpu
  --multiprocessing_start_method spawn
)

export PYTHONPATH=src MUJOCO_GL=cgl G1_VERSION=5010 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# All backends use the active HumanTracker environment.
PY_BASE=${PYTHON:-python}
PY_HGPT=$PY_BASE
python_for () { [ "$1" = hgpt ] && echo "$PY_HGPT" || echo "$PY_BASE"; }
for interp in "$PY_BASE" "$PY_HGPT"; do
  command -v "$interp" >/dev/null || { echo "missing interpreter: $interp" >&2; exit 1; }
done

# scale=0 is an exact no-op in every stage, so one baseline covers all velocity
# modes and both profiles; only the tracker identity matters.
tag () {  # tracker profile scale velmode
  if [ "$3" = "0" ]; then echo "$1_clean"; else echo "$1_$2_s$3_$4"; fi
}

run () {  # tracker profile scale velmode
  local out; out="$OUT/$(tag "$@").json"
  [ -f "$out" ] && { echo "[skip] $out"; return; }
  local profile=$2 scale=$3
  [ "$scale" = "0" ] && profile=none
  local noise=(--ref_noise "$profile" --ref_noise_seed 0
               --ref_noise_scale "$scale" --ref_noise_vel "$4")
  echo "=== $(tag "$@")"
  local py; py=$(python_for "$1")
  case $1 in
    sonic)
      "$py" -m humantracker.eval.eval_parallel_tracker --tracker sonic \
        --encoder "$SONIC_DIR/model_encoder.onnx" \
        --decoder "$SONIC_DIR/model_decoder.onnx" \
        "${noise[@]}" "${COMMON[@]}" --workers 6 --output_json "$out" ;;
    hgpt)
      "$py" -m humantracker.eval.eval_parallel_tracker --tracker hgpt \
        --policy "$HGPT_ONNX" \
        "${noise[@]}" "${COMMON[@]}" --workers 4 --output_json "$out" ;;
    *) echo "unknown tracker $1" >&2; return 1 ;;
  esac
}

for t in sonic hgpt; do
  for s in $SCALES; do
    run "$t" online_tracking "$s" clean
  done
  run "$t" online_tracking 1.0 filtered
  run "$t" online_tracking 1.0 raw_diff
  run "$t" vla_chunk 1.0 clean
done

echo REF_NOISE_SWEEP_DONE
