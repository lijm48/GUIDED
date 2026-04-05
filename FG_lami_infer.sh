#!/usr/bin/env bash

# --- Path configuration ---
DATA_ROOT=/apdcephfs_cq12/share_1150325/jiaminghli/data
CODE_ROOT=/apdcephfs_cq12/share_1150325/jiaminghli/code/OVD
PRETRAIN_DIR=${CODE_ROOT}/pretrain_models
OUTPUT_DIR=${CODE_ROOT}/output

export CUDA_VISIBLE_DEVICES="4"
export DETECTRON2_DATASETS=${DATA_ROOT}

# --- Experiment configuration ---
Name=multi_diff_attr
PYTHON_SCRIPT="FG_OVD_TEST/FG_inf.py"

BASE_DATASET_DIR="${DATA_ROOT}/FG_OVD/benchmarks/with_multi_vocab_single_final"
FG_BENCH_GT_DIR="${DATA_ROOT}/FG_OVD/benchmarks"
BASE_OUT_DIR="${OUTPUT_DIR}/FG_bench/${Name}"

INIT_CHECKPOINT="${OUTPUT_DIR}/fg_train_with_vlm_scr_final_multi_diff2_attr/model_0001999.pth"
CLIP_HEAD_PATH="${PRETRAIN_DIR}/clip_convnext_large_head.pth"
COMMON_ARGS="train.init_checkpoint=${INIT_CHECKPOINT} model.clip_head_path=${CLIP_HEAD_PATH}"

mkdir -p ${BASE_OUT_DIR}

# --- Inference ---

echo "Running tests with n_hardnegatives = 5"
N_HARDNEGATIVES=5

python "${PYTHON_SCRIPT}" \
    --dataset "${BASE_DATASET_DIR}/1_attributes_with_subject_with_multi_vocab_single.json" \
    --out "${BASE_OUT_DIR}/lami_1_attributes.pkl" \
    --n_hardnegatives "${N_HARDNEGATIVES}" \
    ${COMMON_ARGS}

python "${PYTHON_SCRIPT}" \
    --dataset "${BASE_DATASET_DIR}/2_attributes_with_subject_with_multi_vocab_single.json" \
    --out "${BASE_OUT_DIR}/lami_2_attributes.pkl" \
    --n_hardnegatives "${N_HARDNEGATIVES}" \
    ${COMMON_ARGS}

python "${PYTHON_SCRIPT}" \
    --dataset "${BASE_DATASET_DIR}/3_attributes_with_subject_with_multi_vocab_single.json" \
    --out "${BASE_OUT_DIR}/lami_3_attributes.pkl" \
    --n_hardnegatives "${N_HARDNEGATIVES}" \
    ${COMMON_ARGS}

python "${PYTHON_SCRIPT}" \
    --dataset "${BASE_DATASET_DIR}/shuffle_negatives_with_subject_with_multi_vocab_single.json" \
    --out "${BASE_OUT_DIR}/shuffle_negatives.pkl" \
    --n_hardnegatives "${N_HARDNEGATIVES}" \
    ${COMMON_ARGS}

# --- Evaluation ---

python FG_OVD_TEST/eval_map.py \
    --predictions "${BASE_OUT_DIR}/lami_1_attributes.pkl" \
    --ground_truth "${FG_BENCH_GT_DIR}/1_attributes.json" \
    --out "${BASE_OUT_DIR}/lami_1_attributes.txt" \
    --evaluate_all_vocabulary

python FG_OVD_TEST/eval_map.py \
    --predictions "${BASE_OUT_DIR}/lami_2_attributes.pkl" \
    --ground_truth "${FG_BENCH_GT_DIR}/2_attributes.json" \
    --out "${BASE_OUT_DIR}/lami_2_attributes.txt" \
    --evaluate_all_vocabulary

python FG_OVD_TEST/eval_map.py \
    --predictions "${BASE_OUT_DIR}/lami_3_attributes.pkl" \
    --ground_truth "${FG_BENCH_GT_DIR}/3_attributes.json" \
    --out "${BASE_OUT_DIR}/lami_3_attributes.txt" \
    --evaluate_all_vocabulary

python FG_OVD_TEST/eval_map.py \
    --predictions "${BASE_OUT_DIR}/lami_color.pkl" \
    --ground_truth "${FG_BENCH_GT_DIR}/color.json" \
    --out "${BASE_OUT_DIR}/lami_color.txt" \
    --evaluate_all_vocabulary

python FG_OVD_TEST/eval_map.py \
    --predictions "${BASE_OUT_DIR}/material.pkl" \
    --ground_truth "${FG_BENCH_GT_DIR}/material.json" \
    --out "${BASE_OUT_DIR}/material.txt" \
    --evaluate_all_vocabulary

python FG_OVD_TEST/eval_map.py \
    --predictions "${BASE_OUT_DIR}/pattern.pkl" \
    --ground_truth "${FG_BENCH_GT_DIR}/pattern.json" \
    --out "${BASE_OUT_DIR}/pattern.txt" \
    --evaluate_all_vocabulary

python FG_OVD_TEST/eval_map.py \
    --predictions "${BASE_OUT_DIR}/shuffle_negatives.pkl" \
    --ground_truth "${FG_BENCH_GT_DIR}/shuffle_negatives.json" \
    --out "${BASE_OUT_DIR}/shuffle_negatives.txt" \
    --evaluate_all_vocabulary

python FG_OVD_TEST/eval_map.py \
    --predictions "${BASE_OUT_DIR}/transparency.pkl" \
    --ground_truth "${FG_BENCH_GT_DIR}/transparency.json" \
    --out "${BASE_OUT_DIR}/transparency.txt" \
    --evaluate_all_vocabulary
