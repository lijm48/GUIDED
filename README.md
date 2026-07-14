# GUIDED: Granular Understanding via Identification, Detection, and Discrimination for Fine-Grained Open-Vocabulary Object Detection

[![arXiv](https://img.shields.io/badge/arXiv-2603.27014-b31b1b.svg)](https://arxiv.org/abs/2603.27014)

Official implementation of **GUIDED**, a decomposition framework for Fine-Grained Open-Vocabulary Object Detection (FG-OVD).

![GUIDED Framework](https://img.shields.io/badge/Framework-Figure-blue)


## 📋 Table of Contents

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Pretrained Weights](#pretrained-weights)
- [Training](#training)
- [Inference & Evaluation](#inference--evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## 🛠 Installation

The installation follows [LaMI-DETR](https://github.com/eternaldolphin/LaMI-DETR). The code has been tested with `python=3.9`, `torch=1.13.0`, `cuda=11.7`.

### Step 1: Create conda environment and install PyTorch

```bash
conda create -n guided python=3.9
conda activate guided
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install numpy==1.23.5
export CUDA_HOME=/usr/local/cuda-11.7
```

### Step 2: Install GUIDED and dependencies

```bash
# Clone the repository
git clone https://github.com/lijm48/GUIDED.gitv

cd GUIDED

# Install the modified detrex package from the repository root
pip install -e .

# Install the modified detectron2 package
pip install -e detectron2

# Install other dependencies
pip install -r requirements.txt
```

### Path Configuration

All path settings are centralized in **`guided_config.py`**. If your directory layout matches the defaults below, no configuration is needed:

```
./data/                 # datasets (LVIS, FG-OVD, COCO)
./pretrain_models/      # pretrained weights
./output/               # training outputs
```

To customize paths, edit the defaults in `guided_config.py`:

```python
# guided_config.py
class _PathsConfig:
    @property
    def dataset_path(self) -> str:   # default: "./data"
        return os.environ.get("GUIDED_DATASET_PATH", "./data")

    @property
    def clip_ckpt(self) -> str:      # default: "./pretrain_models/timm_clip_convnext_large_trans.pth"
        return os.environ.get("GUIDED_CLIP_CKPT",
            "pretrain_models/timm_clip_convnext_large_trans.pth")
    # ... see guided_config.py for all options
```



## 📦 Data Preparation

### LVIS

Download the LVIS dataset following the [official instructions](https://www.lvisdataset.org/). The expected directory structure:

```
data/
├── lvis/
│   ├── lvis_v1_val.json
│   ├── lvis_v1_train.json
│   ├── lvis_v1_all_classes.json
│   ├── lvis_v1_seen_classes.json
│   └── lvis_v1_train_norare_cat_info.json
├── coco/
│   ├── train2017/
│   └── val2017/
```

### FG-OVD

Download the FG-OVD benchmark from the [official repository](https://github.com/therosFG/FG-OVD). Then download our generated subject-and-atomic-phrase benchmark and training files from [this link](https://drive.google.com/drive/folders/16QzgT1lIpRmwTTnFsBA3D8ZloXLpG9s2?usp=sharing).

Place the generated `with_subject_and_atomic_phrases/` benchmark directory under `FG_OVD/benchmarks/`, and place the generated training directory under `FG_OVD/training_sets/with_subject_and_atomic_phrases/`.

These generated files extend each FG-OVD category with an explicit subject and decomposed atomic phrases. Stage 2 fine-tuning uses the 1-attribute FG-OVD training split together with LVIS training data.

Expected structure after preparation:

```
${GUIDED_DATASET_PATH}/
├── FG_OVD/
│   ├── benchmarks/
│   │   ├── 1_attributes.json
│   │   ├── 2_attributes.json
│   │   ├── 3_attributes.json
│   │   ├── color.json
│   │   ├── material.json
│   │   ├── pattern.json
│   │   ├── shuffle_negatives.json
│   │   ├── transparency.json
│   │   └── with_subject_and_atomic_phrases/  # generated benchmark files
│   └── training_sets/
│       └── with_subject_and_atomic_phrases/
│           └── 1_attributes_with_subject_with_multi_vocab_single.json  # used for Stage 2 fine-tuning
```



## 🔑 Pretrained Weights

Download the following pretrained weights:

| File | Destination | Description | Download |
|------|-------------|-------------|----------|
| `timm_clip_convnext_large_trans.pth` | `pretrain_models/` | OpenCLIP ConvNeXt-Large checkpoint | [Link](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup) |
| `clip_convnext_large_head.pth` | `pretrain_models/` | CLIP head weights | See LaMI-DETR |
| `fg-ovd_convnext-mlp_norm_emb.pth` | `clip_models/FG_CLIP/ckpt_convnext_norm_emb/` | [FG-CLIP](https://github.com/lorebianchi98/FG-CLIP)  ConvNeXt text MLP weights used during FG-OVD inference | [this link](https://github.com/lorebianchi98/FG-CLIP) |

`fg-ovd_convnext-mlp_norm_emb.pth` is our reproduced FG-CLIP result with the training code under `clip_models/FG_CLIP/`.



## 🚀 Training

GUIDED follows a two-stage training pipeline:

### Stage 1: LVIS Pre-training

Pre-train the detection transformer on LVIS base classes for 85,200 iterations:

```bash
bash scripts/train_lvis.sh
```

Or manually:

```bash
python tools/train_net.py \
    --config-file lami_dino_lvis/configs/dino_convnext_large_4scale_12ep_lvis.py \
    --num-gpus 8 \
    train.output_dir=output/idow_convnext_large_12ep_lvis_CGOD
```

### Stage 2: FG-OVD Fine-tuning

Fine-tune on the FG-OVD training set with Attribute Attention for 2,000 iterations:

```bash
bash scripts/train_fg_ovd.sh
```

Or manually:

```bash
python lami_dino_mix/train.py \
    --config-file lami_dino_mix/configs/dino_convnext_large_4scale_12ep_lvis_attr.py \
    --num-gpus 4 \
    train.init_checkpoint=output/idow_convnext_large_12ep_lvis_CGOD/model_final.pth \
    model.clip_head_path=pretrain_models/clip_convnext_large_head.pth \
    dataloader.train.total_batch_size=16 \
    train.max_iter=2000 \
    optimizer.lr=1e-5 \
    train.output_dir=output/fg_ovd_guided
```

## 📊 Inference & Evaluation

### Inference on FG-OVD Benchmark

```bash
# Default: experiment name `multi_diff_attr`, 5 hard negatives for 1/2/3-attribute
# and shuffle tracks, and 2 hard negatives for color/material/pattern/transparency tracks.
bash scripts/inference_and_eval.sh

# Optional overrides:
# bash scripts/inference_and_eval.sh [EXPERIMENT_NAME] [MULTI_ATTR_N_HARDNEGATIVES] [SINGLE_ATTR_N_HARDNEGATIVES]
bash scripts/inference_and_eval.sh my_experiment 5 2
```

This runs inference on all FG-OVD tracks (1/2/3 attributes, color, material, pattern, transparency, shuffle_negatives) using the generated files under `FG_OVD/benchmarks/with_subject_and_atomic_phrases/`, and then evaluates mAP against the original FG-OVD ground-truth files.

### Evaluation Only

```bash
# mAP evaluation for saved predictions
bash FG_OVD_TEST/eval_lami_FG_map.sh [EXPERIMENT_NAME]
```


## 📁 Project Structure

```
GUIDED/
├── lami_dino_lvis/              # Stage 1: LVIS pre-training config & model
│   ├── configs/
│   └── modeling/
├── lami_dino_mix/               # Stage 2: FG-OVD fine-tuning (with Attribute Attention)
│   ├── configs/
│   │   ├── dino_convnext_large_4scale_12ep_lvis_attr.py
│   │   ├── dino_convnext_large_4scale_12ep_lvis.py
│   │   └── models/
│   │       ├── dino_convnextl.py
│   │       └── dino_convnextl_attr.py    # DINOAttr model definition
│   ├── modeling/
│   │   ├── dino.py               # DINO detector with FG modules
│   │   ├── dino_attr.py          # DINOAttr with Attribute Attention
│   │   └── dino_transformer_attr.py
│   └── train.py
├── FG_OVD_TEST/                  # Inference & evaluation
│   ├── FG_inf.py                 # Main inference script
│   ├── eval_map.py               # mAP evaluation
│   ├── eval_rank.py              # Ranking evaluation
│   ├── eval_lami_FG_map.sh
│   └── eval_lami_FG_rank.sh
├── clip_models/                  # CLIP model wrappers
│   ├── FG_clip_model.py          # FG-CLIP with MLP projection head
│   ├── enc_text.py               # CLIP text encoder utilities
│   ├── FG_CLIP/                  # FG-CLIP submodule
│   └── OpenClip/                 # OpenCLIP submodule
├── configs/                      # Shared config utilities
├── tools/                        # Training entry point (train_net.py)
├── utils/                        # Common utilities
├── scripts/                      # Run scripts
│   ├── train_lvis.sh
│   ├── train_fg_ovd.sh
│   ├── inference_and_eval.sh
├── detectron2/                   # Modified Detectron2
├── detrex/                       # Modified Detrex
├── dataset/                      # Metadata (npy, pt files, downloaded separately)
└── setup.py
```

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{li2026guided,
  title={Guided: Granular understanding via identification, detection, and discrimination for fine-grained open-vocabulary object detection},
  author={Li, Jiaming and Liang, Zhijia and Chen, Weikai and Ma, Lin and Li, Guanbin},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={41320--41339},
  year={2026}
}
```

## 🙏 Acknowledgements

This codebase is built upon [LaMI-DETR](https://github.com/eternaldolphin/LaMI-DETR), [Detectron2](https://github.com/facebookresearch/detectron2), and [Detrex](https://github.com/IDEA-Research/detrex). We also use [OpenCLIP](https://github.com/mlfoundations/open_clip) and the [FG-OVD](https://github.com/therosFG/FG-OVD) benchmark. We thank the authors for their great work.
