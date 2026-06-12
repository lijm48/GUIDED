#!/usr/bin/env bash
# Download pretrained weights for GUIDED
# Usage: bash scripts/download_weights.sh [OUTPUT_DIR]

set -e

OUTPUT_DIR=${1:-"pretrain_models"}
mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "Downloading GUIDED pretrained weights"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================"

# --- OpenCLIP ConvNeXt-Large checkpoint ---
CLIP_FILE="${OUTPUT_DIR}/timm_clip_convnext_large_trans.pth"
if [ -f "${CLIP_FILE}" ]; then
    echo "[SKIP] ${CLIP_FILE} already exists"
else
    echo "[Downloading] OpenCLIP ConvNeXt-Large checkpoint..."
    wget -O "${CLIP_FILE}" \
        "https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/resolve/main/open_clip_pytorch_model.bin" \
        || { echo "ERROR: Failed to download CLIP checkpoint. Please download manually from:"; \
             echo "  https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup"; }
fi

# --- CLIP Head weights ---
HEAD_FILE="${OUTPUT_DIR}/clip_convnext_large_head.pth"
if [ -f "${HEAD_FILE}" ]; then
    echo "[SKIP] ${HEAD_FILE} already exists"
else
    echo "[TODO] CLIP head weights need to be obtained from LaMI-DETR."
    echo "  Please download from: https://github.com/AtIsElsT/LaMI-DETR"
    echo "  Place the file at: ${HEAD_FILE}"
fi


echo ""
echo "============================================"
echo "Download complete!"
echo "============================================"
