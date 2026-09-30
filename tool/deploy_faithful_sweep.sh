#!/usr/bin/env bash
# Deploy-faithful reference distributions on Humanoid-GPT: the ScaleBridge
# human-proportion cloud (xsens_human) and the GMR live stream (gmr_stream). Same 100-clip manifest as the
# earlier noise sweeps. Metrics / HumanScore stay on the clean GT.
#
# Usage: tool/deploy_faithful_sweep.sh [output_dir]
set -euo pipefail

OUT=${1:-outputs/eval_runs/noise_sweep_deploy}
MANIFEST=outputs/eval_runs/ref_noise_sweep_clips.json
CLEAN=outputs/eval_runs/noise_sweep_v2
mkdir -p "$OUT"

# Reuse the bit-identical clean baselines. scale=0 is a no-op in every profile.
if [ -f "$CLEAN/hgpt_clean.json" ] && [ ! -f "$OUT/hgpt264_clean.json" ]; then
  ln -s "../noise_sweep_v2/hgpt_clean.json" "$OUT/hgpt264_clean.json"
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

tag () {  # tracker profile scale vel
  if [ "$3" = "0" ]; then echo "$1_clean"; else echo "$1_$2_s$3_$4"; fi
}

run () {  # tracker profile scale vel
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

# Clean 216 (264 already linked). scale=0, profile ignored.
run hgpt216 none 0 clean

for t in hgpt264 hgpt216; do
  for s in 0.5 1.0 2.0; do
    run "$t" xsens_human "$s" clean
  done
done

for t in hgpt264 hgpt216; do
  for s in 0.5 1.0 2.0; do
    run "$t" gmr_stream "$s" filtered
  done
done

# Extra column: HGPT differentiated off the raw GMR qpos (LiveRefConverter
# default is filtered; this isolates the velocity blow-up).
for t in hgpt264 hgpt216; do
  run "$t" gmr_stream 1.0 raw_diff
done

echo DEPLOY_FAITHFUL_SWEEP_DONE
