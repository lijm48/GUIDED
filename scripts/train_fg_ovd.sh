#!/usr/bin/env bash
# Stage 2: FG-OVD Fine-tuning with Attribute Attention (2,000 iterations)
# This script fine-tunes the detection transformer on the FG-OVD training set.

# --- Path configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"

export DETECTRON2_DATASETS=${DATA_ROOT}/
export DETECTRON2_DATASETS2=${CODE_ROOT}/

CUDA_VISIBLE_DEVICES=4,5,6,7 python lami_dino_mix/train.py \
    --config-file lami_dino_mix/configs/dino_convnext_large_4scale_12ep_lvis_attr.py \
    --num-gpus 4 \
    train.init_checkpoint=${OUTPUT_DIR}/idow_convnext_large_12ep_lvis_attn/model_final.pth \
    model.clip_head_path=${CLIP_HEAD_PATH} \
    dataloader.train.total_batch_size=16 \
    train.ddp.find_unused_parameters=True \
    train.eval_period=1000 \
    train.checkpointer.period=1000 \
    train.log_period=100 \
    train.output_dir=${OUTPUT_DIR}/fg_ovd_guided  \
    dataloader.evaluator.output_dir=${OUTPUT_DIR}/fg_ovd_guided \
    train.max_iter=2000 \
    optimizer.lr=1e-5
