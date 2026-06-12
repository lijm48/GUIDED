#!/usr/bin/env bash
# Evaluation script for FG-OVD ranking metrics
# Usage: bash eval_lami_FG_rank.sh [EXPERIMENT_NAME]

# --- Path configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../scripts/paths.sh"

Name=${1:-"multi_diff_attr"}
PRED_DIR="${OUTPUT_DIR}/FG_bench/${Name}"
GT_DIR="${DATA_ROOT}/FG_OVD/benchmarks"

# --- Evaluate ranking for each track ---
python eval_rank.py \
    --predictions "${PRED_DIR}/lami_3_attributes.pkl" \
    --ground_truth "${GT_DIR}/3_attributes.json"

python eval_rank.py \
    --predictions "${PRED_DIR}/lami_1_attributes.pkl" \
    --ground_truth "${GT_DIR}/1_attributes.json"

python eval_rank.py \
    --predictions "${PRED_DIR}/lami_2_attributes.pkl" \
    --ground_truth "${GT_DIR}/2_attributes.json"

python eval_rank.py \
    --predictions "${PRED_DIR}/lami_color.pkl" \
    --ground_truth "${GT_DIR}/color.json"

python eval_rank.py \
    --predictions "${PRED_DIR}/material.pkl" \
    --ground_truth "${GT_DIR}/material.json"

python eval_rank.py \
    --predictions "${PRED_DIR}/pattern.pkl" \
    --ground_truth "${GT_DIR}/pattern.json"

python eval_rank.py \
    --predictions "${PRED_DIR}/shuffle_negatives.pkl" \
    --ground_truth "${GT_DIR}/shuffle_negatives.json"
