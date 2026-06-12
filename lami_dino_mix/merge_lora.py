from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from peft import LoraConfig, get_peft_model
import argparse
import torch
import os

def main(args):
    cfg = LazyConfig.load(args.config_file)
    base_model = instantiate(cfg.model)
    peft_model = get_peft_model(
        model=base_model,
        peft_config=instantiate(cfg.peft_config),
    )
    
    DetectionCheckpointer(peft_model).load(args.fulllora_ckpt)
    peft_model.merge_and_unload()
    
    # save
    torch.save(peft_model.base_model.model.state_dict(), 
        os.path.join(
            args.save_dir,
            f"{os.path.basename(args.fulllora_ckpt).split('.')[0]}_merged_lora.pt"
        )
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA")
    parser.add_argument(
        "--config_file",
        default="lami_dino_mix/configs/dino_convnext_large_4scale_12ep_lvis.py",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--fulllora_ckpt",
        default="output/A800/lami_dino_mix_split_loss/model_final.pth",
        help="path to fulllora checkpoint",
        type=str,
    )
    parser.add_argument(
        "--save_dir",
        default="output/A800/lami_dino_mix_split_loss",
        help="path to save dir",
        type=str,
    )
    args = parser.parse_args()
    main(args)