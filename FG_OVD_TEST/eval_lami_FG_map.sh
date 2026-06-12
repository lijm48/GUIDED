#!/usr/bin/env bash
# Evaluation script for FG-OVD mAP metrics
# Usage: bash FG_OVD_TEST/eval_lami_FG_map.sh [EXPERIMENT_NAME]

set -euo pipefail

# --- Path configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/scripts/paths.sh"

DATA_ROOT=${GUIDED_DATASET_PATH:-"${REPO_ROOT}/data"}
OUTPUT_DIR=${GUIDED_OUTPUT_DIR:-"${REPO_ROOT}/output"}

Name=${1:-"multi_diff_attr"}
PRED_DIR="${OUTPUT_DIR}/FG_bench/${Name}"
GT_DIR="${DATA_ROOT}/FG_OVD/benchmarks"

run_eval() {
    local prediction_file=$1
    local ground_truth_file=$2
    local output_file=$3

    echo "Evaluating: ${prediction_file}"
    python "${SCRIPT_DIR}/eval_map.py" \
        --predictions "${PRED_DIR}/${prediction_file}" \
        --ground_truth "${GT_DIR}/${ground_truth_file}" \
        --out "${PRED_DIR}/${output_file}" \
        --evaluate_all_vocabulary
}

# --- Evaluate each track ---
run_eval "lami_1_attributes.pkl" "1_attributes.json" "lami_1_attributes.txt"
run_eval "lami_2_attributes.pkl" "2_attributes.json" "lami_2_attributes.txt"
run_eval "lami_3_attributes.pkl" "3_attributes.json" "lami_3_attributes.txt"
run_eval "lami_color.pkl" "color.json" "lami_color.txt"
run_eval "material.pkl" "material.json" "material.txt"
run_eval "pattern.pkl" "pattern.json" "pattern.txt"
run_eval "shuffle_negatives.pkl" "shuffle_negatives.json" "shuffle_negatives.txt"
run_eval "transparency.pkl" "transparency.json" "transparency.txt"
