#!/usr/bin/env bash
# Evaluate the released sim2real policies, one process per GPU, from sim2real's
# environment. Run src/humantracker/eval/sim2real/setup.sh first.
#
# Usage: bash src/humantracker/eval/sim2real/eval.sh [tracker ...]   (default: all nine)
#
# Required environment:
#   HUMANTRACKER_DATASET        motion dataset root containing test.json
#
# Optional environment:
#   HUMANTRACKER_RM_CHECKPOINT  HumanScore checkpoint (default: storage/checkpoints/reward_model/best.pt)
#   SIM2REAL_ROOT               sim2real checkout (default: thirdparty/sim2real)
#   SIM2REAL_MOTION_VIEW        motion view, on the dataset's filesystem
#                               (default: $HUMANTRACKER_DATASET/.humantracker_eval_view)
#   GPUS                        GPUs to cycle through (default: "0 1 2 3 4 5 6 7")
#   WORKERS                     worker processes per tracker (default: 8)
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
dataset="${HUMANTRACKER_DATASET:?set HUMANTRACKER_DATASET to the motion dataset root}"
test_json=$dataset/test.json
rm_checkpoint="${HUMANTRACKER_RM_CHECKPOINT:-$repo/storage/checkpoints/reward_model/best.pt}"
sim2real="${SIM2REAL_ROOT:-$repo/thirdparty/sim2real}"
motion_view="${SIM2REAL_MOTION_VIEW:-$dataset/.humantracker_eval_view}"
read -r -a gpus <<< "${GPUS:-0 1 2 3 4 5 6 7}"
workers="${WORKERS:-8}"
python=$sim2real/.venv/bin/python
export PYTHONPATH="$repo/src" LOGURU_LEVEL=ERROR

test -f "$test_json"
test -f "$rm_checkpoint"
test -x "$python"

if [ $# -gt 0 ]; then
    trackers=("$@")
else
    read -r -a trackers <<< "$("$python" -c \
        'from humantracker.eval.backends import SIM2REAL_POLICIES; print(*SIM2REAL_POLICIES)')"
fi

# Also caches sim2real's G1 model, so the workers below can run offline.
"$python" -m humantracker.eval.sim2real.prepare_motion_view \
    --mocap_path "$dataset" --test_json "$test_json" --output "$motion_view"

run_id=$(date +%Y%m%d_%H%M%S)_sim2real_sweep
run_root=$repo/outputs/eval_runs/$run_id
test ! -e "$run_root"
mkdir -p "$run_root/logs" "$run_root/pids" "$run_root/results"
cp "$0" "$run_root/launch.sh"
cd "$repo"

for i in "${!trackers[@]}"; do
    tracker=${trackers[$i]}
    gpu=${gpus[$((i % ${#gpus[@]}))]}
    # A private FK cache per tracker: concurrent runs sharing one race on its files.
    nohup env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        HF_HUB_OFFLINE=1 \
        ANY4HDMI_QPOS_CACHE_ROOT="$sim2real/.cache/humantracker/$tracker" \
        ANY4HDMI_CACHE_BUILD_NUM_WORKERS=0 \
        PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        SIM2REAL_ORT_NUM_THREADS=1 \
        "$python" -m humantracker.eval.eval_parallel_tracker \
            --tracker "$tracker" \
            --sim2real_root "$sim2real" \
            --motion_view "$motion_view" \
            --mocap_path "$dataset" \
            --test_json "$test_json" \
            --rm_checkpoint "$rm_checkpoint" \
            --rm_device cuda:0 \
            --termination_metric whole_body \
            --workers "$workers" \
            --seed 0 \
            --output_json "$run_root/results/$tracker.json" \
        >"$run_root/logs/$tracker.log" 2>&1 </dev/null &
    printf '%s\n' "$!" >"$run_root/pids/$tracker.pid"
done

printf 'run_id=%s\nrun_root=%s\n' "$run_id" "$run_root"
for pid_file in "$run_root"/pids/*.pid; do
    printf '%s=%s\n' "$(basename "$pid_file" .pid)" "$(<"$pid_file")"
done
