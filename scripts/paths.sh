#!/usr/bin/env bash
# Centralized path configuration for GUIDED shell scripts.
# Source this file at the top of any script that needs path variables.
#
# Usage:
#   source scripts/paths.sh
#
# To override, set environment variables before sourcing, e.g.:
#   export GUIDED_DATASET_PATH=/mnt/data
#   source scripts/paths.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CODE_ROOT=${GUIDED_CODE_ROOT:-"${DEFAULT_CODE_ROOT}"}
export PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}"

GUIDED_PYTHON=${PYTHON:-python}
if ! command -v "${GUIDED_PYTHON}" >/dev/null 2>&1; then
    GUIDED_PYTHON=python3
fi

get_guided_path() {
    local attr="$1"
    local fallback="$2"
    local value=""

    if command -v "${GUIDED_PYTHON}" >/dev/null 2>&1; then
        value="$(PYTHONPATH="${CODE_ROOT}:${PYTHONPATH:-}" "${GUIDED_PYTHON}" - "${attr}" <<'PY' 2>/dev/null || true
import sys
from guided_config import paths
print(getattr(paths, sys.argv[1]))
PY
)"
    fi

    printf '%s' "${value:-${fallback}}"
}

DATA_ROOT=${GUIDED_DATASET_PATH:-"$(get_guided_path dataset_path "./data")"}
PRETRAIN_DIR=${GUIDED_PRETRAIN_DIR:-"$(get_guided_path pretrain_dir "./pretrain_models")"}
OUTPUT_DIR=${GUIDED_OUTPUT_DIR:-"$(get_guided_path output_dir "./output")"}
CLIP_CKPT_PATH=${GUIDED_CLIP_CKPT:-"$(get_guided_path clip_ckpt "${PRETRAIN_DIR}/timm_clip_convnext_large_trans.pth")"}
CLIP_HEAD_PATH=${GUIDED_CLIP_HEAD:-"$(get_guided_path clip_head "${PRETRAIN_DIR}/clip_convnext_large_head.pth")"}
