#!/usr/bin/env bash
# Stage 1: LVIS Pre-training (85,200 iterations)
# This script pre-trains the DINO detection transformer on LVIS base classes.

# --- Path configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"

export DETECTRON2_DATASETS=${DATA_ROOT}

python tools/train_net.py \
    --config-file lami_dino_lvis/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --num-gpus 8 \
    --resume \
    train.init_checkpoint=${CLIP_CKPT_PATH} \
    model.clip_head_path=${CLIP_HEAD_PATH} \
    train.ddp.find_unused_parameters=True \
    train.eval_period=10000 \
    train.checkpointer.period=5000 \
    train.output_dir="${OUTPUT_DIR}/idow_convnext_large_12ep_test" \
    dataloader.train.total_batch_size=32
