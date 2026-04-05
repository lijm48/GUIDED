#!/usr/bin/env bash
# run_mix_data_train_attr.sh
# Training script for Mix-data + VLM-score + Attribute_Attention.

# --- Path configuration ---
DATA_ROOT=/apdcephfs_cq12/share_1150325/jiaminghli/data
CODE_ROOT=/apdcephfs_cq12/share_1150325/jiaminghli/code/OVD
PRETRAIN_DIR=${CODE_ROOT}/pretrain_models
OUTPUT_DIR=${CODE_ROOT}/output

export DETECTRON2_DATASETS=${DATA_ROOT}/
export DETECTRON2_DATASETS2=/apdcephfs_cq12/share_1150325/jiaminghli/OVD/

CUDA_VISIBLE_DEVICES=4,5,6,7 python lami_dino_mix/train.py \
    --config-file lami_dino_mix/configs/dino_convnext_large_4scale_12ep_lvis_attr.py \
    --num-gpus 4 \
    train.init_checkpoint=${OUTPUT_DIR}/idow_convnext_large_12ep_lvis_attn/model_final.pth \
    model.clip_head_path=${PRETRAIN_DIR}/clip_convnext_large_head.pth \
    dataloader.train.total_batch_size=16 \
    train.ddp.find_unused_parameters=True \
    train.eval_period=500 \
    train.checkpointer.period=500 \
    train.log_period=100 \
    train.output_dir=${OUTPUT_DIR}/fg_train_with_vlm_scr_final_multi_diff2_attr \
    dataloader.evaluator.output_dir=${OUTPUT_DIR}/fg_train_with_vlm_scr_final_attr \
    train.max_iter=2000 \
    optimizer.lr=1e-5
