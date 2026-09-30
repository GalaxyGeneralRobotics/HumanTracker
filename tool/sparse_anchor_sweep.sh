#!/usr/bin/env bash
# After-FK mid-link sweep on Humanoid-GPT: the five endpoints stay on the
# delayed skeleton while the other nine links get an independent residual.
#
# Reuses the clean baselines from noise_sweep_v2 (bit-identical at scale=0).
# Usage: tool/sparse_anchor_sweep.sh [output_dir]
set -euo pipefail

OUT=${1:-outputs/eval_runs/noise_sweep_anchor}
MANIFEST=outputs/eval_runs/ref_noise_sweep_clips.json
CLEAN=outputs/eval_runs/noise_sweep_v2
mkdir -p "$OUT"

# Same clean files -- scale=0 is a no-op in every profile.
for t in hgpt; do
  src="$CLEAN/${t}_clean.json"
  dst="$OUT/${t}_clean.json"
  if [ -f "$src" ] && [ ! -f "$dst" ]; then
    ln -s "../noise_sweep_v2/${t}_clean.json" "$dst"
  fi
done

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
PY_BASE=${PYTHON:-python}
PY_HGPT=$PY_BASE
python_for () { [ "$1" = hgpt ] && echo "$PY_HGPT" || echo "$PY_BASE"; }

tag () { echo "$1_$2_s$3_clean"; }

run () {  # tracker profile scale
  local out; out="$OUT/$(tag "$@").json"
  [ -f "$out" ] && { echo "[skip] $out"; return; }
  echo "=== $(tag "$@")"
  local py; py=$(python_for "$1")
  local noise=(--ref_noise "$2" --ref_noise_seed 0
               --ref_noise_scale "$3" --ref_noise_vel clean)
  case $1 in
    hgpt)
      "$py" -m humantracker.eval.eval_parallel_tracker --tracker hgpt \
        --policy "$HGPT_ONNX" \
        "${noise[@]}" "${COMMON[@]}" --workers 4 --output_json "$out" ;;
    *) echo "unknown tracker $1" >&2; return 1 ;;
  esac
}

for t in hgpt; do
  for s in 0.5 1.0 2.0; do
    run "$t" sparse_anchor "$s"
  done
  run "$t" vla_fill 1.0
done

echo SPARSE_ANCHOR_SWEEP_DONE
