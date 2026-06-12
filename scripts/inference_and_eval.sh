#!/usr/bin/env bash
# Inference & Evaluation on FG-OVD Benchmark
# Usage: bash scripts/inference_and_eval.sh [EXPERIMENT_NAME] [MULTI_ATTR_N_HARDNEGATIVES] [SINGLE_ATTR_N_HARDNEGATIVES]

set -euo pipefail

# --- Path configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/paths.sh"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"4"}
export DETECTRON2_DATASETS=${DATA_ROOT}

# --- Experiment configuration ---
Name=${1:-"fgovd_"}
MULTI_ATTR_N_HARDNEGATIVES=${2:-5}
SINGLE_ATTR_N_HARDNEGATIVES=${3:-2}
PYTHON_SCRIPT="FG_OVD_TEST/FG_inf.py"

BASE_DATASET_DIR="${DATA_ROOT}/FG_OVD/benchmarks/with_subject_and_atomic_phrases"
FG_BENCH_GT_DIR="${DATA_ROOT}/FG_OVD/benchmarks"
BASE_OUT_DIR="${OUTPUT_DIR}/FG_bench/${Name}"

INIT_CHECKPOINT="${OUTPUT_DIR}/fg_ovd_guided/model_final.pth"
CLIP_HEAD_PATH=${CLIP_HEAD_PATH:-"${PRETRAIN_DIR}/clip_convnext_large_head.pth"}
COMMON_ARGS=(
    "train.init_checkpoint=${INIT_CHECKPOINT}"
    "model.clip_head_path=${CLIP_HEAD_PATH}"
)

mkdir -p "${BASE_OUT_DIR}"

run_inference() {
    local dataset_file=$1
    local output_file=$2
    local n_hardnegatives=$3

    echo "Running inference: ${dataset_file} (n_hardnegatives=${n_hardnegatives})"
    python "${PYTHON_SCRIPT}" \
        --dataset "${BASE_DATASET_DIR}/${dataset_file}" \
        --out "${BASE_OUT_DIR}/${output_file}" \
        --n_hardnegatives "${n_hardnegatives}" \
        "${COMMON_ARGS[@]}"
}

run_eval() {
    local prediction_file=$1
    local ground_truth_file=$2
    local output_file=$3

    echo "Evaluating: ${prediction_file}"
    python FG_OVD_TEST/eval_map.py \
        --predictions "${BASE_OUT_DIR}/${prediction_file}" \
        --ground_truth "${FG_BENCH_GT_DIR}/${ground_truth_file}" \
        --out "${BASE_OUT_DIR}/${output_file}" \
        --evaluate_all_vocabulary
}

# --- Inference ---
# Single-attribute tracks follow the original evaluation setting: n_hardnegatives=2.
run_inference "color_with_subject_with_multi_vocab_single.json" "lami_color.pkl" "${SINGLE_ATTR_N_HARDNEGATIVES}"
run_inference "material_with_subject_with_multi_vocab_single.json" "material.pkl" "${SINGLE_ATTR_N_HARDNEGATIVES}"
run_inference "pattern_with_subject_with_multi_vocab_single.json" "pattern.pkl" "${SINGLE_ATTR_N_HARDNEGATIVES}"
run_inference "transparency_with_subject_with_multi_vocab_single.json" "transparency.pkl" "${SINGLE_ATTR_N_HARDNEGATIVES}"

# Multi-attribute and shuffle tracks follow the original evaluation setting: n_hardnegatives=5.
run_inference "1_attributes_with_subject_with_multi_vocab_single.json" "lami_1_attributes.pkl" "${MULTI_ATTR_N_HARDNEGATIVES}"
run_inference "2_attributes_with_subject_with_multi_vocab_single.json" "lami_2_attributes.pkl" "${MULTI_ATTR_N_HARDNEGATIVES}"
run_inference "3_attributes_with_subject_with_multi_vocab_single.json" "lami_3_attributes.pkl" "${MULTI_ATTR_N_HARDNEGATIVES}"
run_inference "shuffle_negatives_with_subject_with_multi_vocab_single.json" "shuffle_negatives.pkl" "${MULTI_ATTR_N_HARDNEGATIVES}"

# --- Evaluation ---
echo "Evaluating mAP metrics..."
run_eval "lami_1_attributes.pkl" "1_attributes.json" "lami_1_attributes.txt"
run_eval "lami_2_attributes.pkl" "2_attributes.json" "lami_2_attributes.txt"
run_eval "lami_3_attributes.pkl" "3_attributes.json" "lami_3_attributes.txt"
run_eval "lami_color.pkl" "color.json" "lami_color.txt"
run_eval "material.pkl" "material.json" "material.txt"
run_eval "pattern.pkl" "pattern.json" "pattern.txt"
run_eval "shuffle_negatives.pkl" "shuffle_negatives.json" "shuffle_negatives.txt"
run_eval "transparency.pkl" "transparency.json" "transparency.txt"

echo "All evaluations completed. Results saved to ${BASE_OUT_DIR}/"
