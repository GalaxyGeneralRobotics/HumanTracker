#!/usr/bin/env bash
set -euo pipefail

# Train the HumanScore reward model with the paper's reported hyperparameters
# (appendix Table 6). Single-GPU: select the device with CUDA_VISIBLE_DEVICES.
#
# Required environment:
#   DATA_DIR  directory of annotated preference-pair parquet shards
#
# Optional: RUN_NAME, CACHE_DIR, OUTPUT_DIR, PYTHON and every hyper-parameter
# below can be overridden from the environment.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_DIR="${DATA_DIR:?set DATA_DIR to the annotated preference-pair directory}"
RUN_NAME="${RUN_NAME:-rm_$(date +%Y%m%d_%H%M%S)}"
CACHE_DIR="${CACHE_DIR:-$ROOT/storage/dataset/reward_model/$RUN_NAME}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/storage/checkpoints/reward_model}"

cd "$ROOT"
export WANDB_PROJECT="${WANDB_PROJECT:-humantracker-reward-model}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$ROOT/storage/logs"
export TMPDIR="${TMPDIR:-$ROOT/storage/tmp}"
mkdir -p "$TMPDIR" "$WANDB_DIR"

"$PYTHON" src/humantracker/reward_model/train/trainer.py \
    --data_dir "$DATA_DIR" \
    --cache_dir "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    --wandb_project "$WANDB_PROJECT" \
    --batch_size "${BATCH_SIZE:-8}" \
    --max_epochs "${MAX_EPOCHS:-20}" \
    --learning_rate "${LEARNING_RATE:-1e-4}" \
    --weight_decay "${WEIGHT_DECAY:-1e-5}" \
    --tie_weight "${TIE_WEIGHT:-1.0}" \
    --temperature "${TEMPERATURE:-1.0}" \
    --d_model "${D_MODEL:-256}" \
    --nhead "${NHEAD:-8}" \
    --num_layers "${NUM_LAYERS:-4}" \
    --dim_feedforward "${DIM_FEEDFORWARD:-1024}" \
    --dropout "${DROPOUT:-0.1}" \
    --downsample_rate "${DOWNSAMPLE_RATE:-1}"
