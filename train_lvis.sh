#!/usr/bin/env bash

# --- Path configuration ---
DATA_ROOT=/apdcephfs_cq12/share_1150325/jiaminghli/data
CODE_ROOT=/apdcephfs_cq12/share_1150325/jiaminghli/code/OVD
PRETRAIN_DIR=${CODE_ROOT}/pretrain_models
OUTPUT_DIR=${CODE_ROOT}/output

export DETECTRON2_DATASETS=${DATA_ROOT}

python tools/train_net.py \
    --config-file lami_dino_lvis/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --num-gpus 8 \
    --resume \
    train.init_checkpoint=${PRETRAIN_DIR}/clip_convnext_large_trans.pth \
    model.clip_head_path=${PRETRAIN_DIR}/clip_convnext_large_head.pth \
    train.ddp.find_unused_parameters=True \
    train.eval_period=10000 \
    train.checkpointer.period=5000 \
    train.output_dir="${OUTPUT_DIR}/idow_convnext_large_12ep_lvis_attn" \
    dataloader.train.total_batch_size=32
