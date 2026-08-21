#!/usr/bin/env bash
# Reproduce the four-tracker zero-shot comparison, one process per GPU.
#
# Required environment:
#   HUMANTRACKER_DATASET        motion dataset root containing test.json
#
# Optional environment:
#   HUMANTRACKER_RM_CHECKPOINT  HumanScore checkpoint (default: storage/checkpoints/reward_model/best.pt)
#
# Run with the environment activated; set PYTHON to override the interpreter.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python="${PYTHON:-python}"
dataset="${HUMANTRACKER_DATASET:?set HUMANTRACKER_DATASET to the motion dataset root}"
test_json=$dataset/test.json
rm_checkpoint="${HUMANTRACKER_RM_CHECKPOINT:-$repo/storage/checkpoints/reward_model/best.pt}"
workers=8
video_interval=20
video_width=2560
video_height=1440

# Tracker assets
scene_xml=$repo/storage/assets/unitree_g1_5010/scene_mjx_track_papergray.xml
robot_xml=$repo/storage/assets/unitree_g1_5010/g1_mjx_track_papergray.xml
sonic_encoder=$repo/thirdparty/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
sonic_decoder=$repo/thirdparty/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
twist2_policy=$repo/thirdparty/TWIST2/assets/ckpts/twist2_1017_25k.onnx
hgpt_policy=$repo/thirdparty/Humanoid-GPT/storage/ckpts/pns_wo_priv264.onnx
gmt_policy=$repo/thirdparty/humanoid-general-motion-tracking/assets/pretrained_checkpoints/pretrained.pt

command -v "$python" >/dev/null
test -d "$dataset"
for input in \
    "$test_json" \
    "$rm_checkpoint" \
    "$scene_xml" \
    "$robot_xml" \
    "$sonic_encoder" \
    "$sonic_decoder" \
    "$twist2_policy" \
    "$hgpt_policy" \
    "$gmt_policy"; do
    test -f "$input"
done

run_id=$(date +%Y%m%d_%H%M%S)_tracker_sweep
run_root=$repo/outputs/eval_runs/$run_id
video_root=$repo/outputs/eval_videos/$run_id

test ! -e "$run_root"
test ! -e "$video_root"
mkdir -p \
    "$run_root/logs" \
    "$run_root/pids" \
    "$run_root/results" \
    "$video_root"
cp "$0" "$run_root/launch.sh"
cd "$repo"

launch_tracker() {
    local gpu=$1
    local tracker=$2
    shift 2

    mkdir "$video_root/$tracker"
    nohup env \
        CUDA_VISIBLE_DEVICES="$gpu" \
        MUJOCO_EGL_DEVICE_ID="$gpu" \
        G1_VERSION=5010 \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        XLA_PYTHON_CLIENT_PREALLOCATE=false \
        PYTHONUNBUFFERED=1 \
        OMP_NUM_THREADS=1 \
        MKL_NUM_THREADS=1 \
        OPENBLAS_NUM_THREADS=1 \
        NUMEXPR_NUM_THREADS=1 \
        PYTHONPATH="$repo/src" \
        "$python" -m humantracker.eval.eval_parallel_tracker \
            --tracker "$tracker" \
            --mocap_path "$dataset" \
            --test_json "$test_json" \
            --rm_checkpoint "$rm_checkpoint" \
            --xml_path "$scene_xml" \
            --termination_metric whole_body \
            --device cuda \
            --rm_device cuda \
            --workers "$workers" \
            --multiprocessing_start_method spawn \
            --output_json "$run_root/results/$tracker.json" \
            --video_dir "$video_root/$tracker" \
            --video_interval "$video_interval" \
            --video_width "$video_width" \
            --video_height "$video_height" \
            "$@" \
        >"$run_root/logs/$tracker.log" 2>&1 </dev/null &
    printf '%s\n' "$!" >"$run_root/pids/$tracker.pid"
}

launch_tracker 0 sonic \
    --encoder "$sonic_encoder" \
    --decoder "$sonic_decoder"

launch_tracker 1 twist2 \
    --policy "$twist2_policy"

launch_tracker 2 hgpt \
    --policy "$hgpt_policy" \
    --robot_xml "$robot_xml"

launch_tracker 3 gmt \
    --policy "$gmt_policy"

printf 'run_id=%s\nrun_root=%s\nvideo_root=%s\n' "$run_id" "$run_root" "$video_root"
for pid_file in "$run_root"/pids/*.pid; do
    printf '%s=%s\n' "$(basename "$pid_file" .pid)" "$(<"$pid_file")"
done
