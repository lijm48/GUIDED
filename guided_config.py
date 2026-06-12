"""
Centralized path configuration for the GUIDED project.

Instead of scattering path defaults across config files and shell scripts,
all path-related settings are managed here. Users can override any path
by editing this file or by setting the corresponding environment variable.

Usage in Python:
    from guided_config import paths
    dataset_path = paths.dataset_path
    clip_ckpt = paths.clip_ckpt

Usage in shell scripts:
    source scripts/paths.sh
    echo $DATA_ROOT
    echo $PRETRAIN_DIR
"""

import os


class _PathsConfig:
    """Centralized path configuration with environment variable overrides.

    Each attribute first checks for an environment variable, then falls back
    to a hardcoded default. To customize paths, either:
    1. Set the corresponding environment variable before running, or
    2. Edit the defaults below directly.
    """

    @property
    def dataset_path(self) -> str:
        """Root directory for datasets (LVIS, FG-OVD, COCO)."""
        # return os.environ.get("GUIDED_DATASET_PATH", "./data")
        return os.environ.get("GUIDED_DATASET_PATH", "./data")
        


    @property
    def clip_ckpt(self) -> str:
        """Path to the pretrained CLIP ConvNeXt-Large checkpoint."""
        return os.environ.get(
            "GUIDED_CLIP_CKPT",
            "../pretrain_models/timm_clip_convnext_large_trans.pth",
        )

    @property
    def clip_head(self) -> str:
        """Path to the CLIP head weights."""
        return os.environ.get(
            "GUIDED_CLIP_HEAD",
            "../pretrain_models/clip_convnext_large_head.pth",
        )

    @property
    def fgclip_mlp(self) -> str:
        """Path to the FG-CLIP MLP weights used by the ConvNeXt text encoder."""
        return os.environ.get(
            "GUIDED_FGCLIP_MLP",
            "clip_models/FG_CLIP/ckpt_convnext_norm_emb/fg-ovd_convnext-mlp_norm_emb.pth",
        )

    @property
    def fgclip_mlp_vit(self) -> str:
        """Path to the optional FG-CLIP MLP weights for the ViT-B/16 text encoder."""
        return os.environ.get(

            "GUIDED_FGCLIP_MLP_VIT",
            "clip_models/FG_CLIP/checkpoints/fg-ovd_linear-mlp_ve-freezed.pt",
        )

    @property
    def pretrain_dir(self) -> str:
        """Directory for pretrained model weights."""
        return os.environ.get("GUIDED_PRETRAIN_DIR", "../pretrain_models")

    @property
    def output_dir(self) -> str:
        """Directory for training outputs and checkpoints."""
        return os.environ.get("GUIDED_OUTPUT_DIR", "../output")

    @property
    def code_root(self) -> str:
        """Root directory of the GUIDED codebase."""
        return os.environ.get("GUIDED_CODE_ROOT", ".")

    @property
    def coco_path(self) -> str:
        """Path to COCO dataset root."""
        return os.environ.get("GUIDED_COCO_PATH", f"{self.dataset_path}/coco")

    @property
    def fgovd_benchmark(self) -> str:
        """Path to FG-OVD benchmark directory."""
        return os.environ.get(
            "GUIDED_FGOVD_BENCHMARK", f"{self.dataset_path}/FG_OVD/benchmarks"
        )

    @property
    def openai_api_key(self) -> str:
        """OpenAI API key for GPT-based annotation generation."""
        return os.environ.get("GUIDED_OPENAI_API_KEY", "")

    @property
    def openai_api_base(self) -> str:
        """OpenAI API base URL."""
        return os.environ.get(
            "GUIDED_OPENAI_API_BASE", "https://api.gptsapi.net/v1"
        )


paths = _PathsConfig()
