# Third-Party Libraries

This project includes modified versions of the following libraries. We keep them as part of the repository because they contain custom modifications required by GUIDED.

## detectron2/

**Source**: [facebookresearch/detectron2](https://github.com/facebookresearch/detectron2)

Modified version of Detectron2 with custom changes for open-vocabulary detection support.

## detrex/

**Source**: [IDEA-Research/detrex](https://github.com/IDEA-Research/detrex)

Modified version of Detrex with custom DINO detector and ZeroShotClassifier implementations.


## clip_models/OpenClip/

**Source**: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip)

Custom OpenCLIP integration used for CLIP model loading and text encoding.


## clip_models/FG_CLIP/

Fine-grained CLIP module with MLP projection heads. The FG-CLIP MLP weights used by default FG-OVD inference are from the official FG-CLIP repository.

**Source**: [lorebianchi98/FG-CLIP](https://github.com/lorebianchi98/FG-CLIP)

Related benchmark codebase: [therosFG/FG-OVD](https://github.com/therosFG/FG-OVD)

---

> **Note**: If you want to use the official upstream versions instead, you will need to apply the custom modifications manually. Key changes include ZeroShotClassifier integration, FG text encoder support, and Attribute Attention module hooks.
