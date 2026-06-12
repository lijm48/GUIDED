# FG-CLIP MLP Training for GUIDED

This note documents how the GUIDED checkpoint `fg-ovd_convnext-mlp_norm_emb.pth` is produced from the code under `clip_models/FG_CLIP/`.

The code is adapted from [lorebianchi98/FG-CLIP](https://github.com/lorebianchi98/FG-CLIP). The checkpoint is a ConvNeXt-based FG-CLIP text-side MLP projection head used by GUIDED during FG-OVD inference.

## Output checkpoint

GUIDED uses the following checkpoint by default:

```text
clip_models/FG_CLIP/ckpt_convnext_norm_emb/fg-ovd_convnext-mlp_norm_emb.pth
```

It is loaded through `paths.fgclip_mlp` in `guided_config.py`, and then passed to `FGConvnextClip` as the text MLP weights.

## Required pre-extracted features

The training script does not read raw images directly. It trains on pre-extracted ConvNeXt CLIP features:

```text
clip_models/FG_CLIP/
├── features/convnext_large_d_320/
│   ├── train.json
│   └── val.json
└── fg-ovd_feature_extraction/
    ├── training_sets/
    │   ├── 1_attributes.pt
    │   └── shuffle_negatives.pt
    └── val_sets/
        └── 1_attributes.pt
```

The relevant loader is `PACCO2CLIPDataset` in `src/dataset.py`. For each FG-OVD annotation, `image` is the crop visual feature, and `annotation` is the stack of the positive category feature plus hard-negative category features.

## Model configuration

The checkpoint is trained with:

```text
configs/model/convnext-mlp_norm_emb.yaml
```

Key settings:

```yaml
mlp_dims: []
no_act: True
keep_embeds: [True, False]
initial_weights: 'ckpt_convnext_norm_emb/triplet_convnext-mlp_norm_emb.pth'
cosine: True
embedding_dim: 768
```

This means:

- visual embeddings are kept unchanged;
- text embeddings are projected by the MLP branch;
- embeddings are L2-normalized before cosine scoring;
- the FG-OVD stage starts from the COCO warm-up checkpoint `triplet_convnext-mlp_norm_emb.pth`.

## Stage 1: COCO warm-up

First train the ConvNeXt MLP projection head on COCO features:

```bash
cd clip_models/FG_CLIP
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train.py \
    --train_config configs/train/triplet.yaml \
    --model_config configs/model/convnext-mlp_norm_emb.yaml \
    --out_dir ckpt_convnext_norm_emb
```

This produces:

```text
ckpt_convnext_norm_emb/triplet_convnext-mlp_norm_emb.pth
```

For non-FG-OVD training, `train.py` explicitly ignores `initial_weights`, so this stage starts from scratch.

## Stage 2: FG-OVD fine-tuning

Then fine-tune the same text MLP on FG-OVD crop/category features:

```bash
cd clip_models/FG_CLIP
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python train.py \
    --train_config configs/train/fg-ovd.yaml \
    --model_config configs/model/convnext-mlp_norm_emb.yaml \
    --out_dir ckpt_convnext_norm_emb
```

Because `train.py` names checkpoints as:

```text
{train_config_name}_{model_config_name}.pth
```

this produces the GUIDED checkpoint:

```text
ckpt_convnext_norm_emb/fg-ovd_convnext-mlp_norm_emb.pth
```

## FG-OVD training details

`configs/train/fg-ovd.yaml` uses:

```yaml
lr: 0.00001
ltype: 'triplet'
margin: 0.05
max_violation: False
fgovd: True
pacco_validation_sets_dir: 'fg-ovd_feature_extraction/val_sets'
num_epochs: 5
batch_size: 256
shuffle: True
save_best_model: True
```

In the current `train.py`, the FG-OVD fine-tuning stage uses the following training subsets:

```python
sub_set_name = ['1_attributes', 'shuffle_negatives']
```

The validation set is:

```text
fg-ovd_feature_extraction/val_sets/1_attributes.pt
```

and COCO validation features are also evaluated as an additional validation set:

```text
features/convnext_large_d_320/val.json
```

## Notes

- If `configs/train/fg-ovd_20epoch.yaml` is used instead, the output filename becomes `fg-ovd_20epoch_convnext-mlp_norm_emb.pth`, not the default GUIDED checkpoint name.
- The optional ViT-B/16 checkpoint `checkpoints/fg-ovd_linear-mlp_ve-freezed.pt` is not used by default GUIDED inference. It is only needed if the optional `FGCLIP` ViT text encoder is explicitly used.
