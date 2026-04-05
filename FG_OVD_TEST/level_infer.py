import logging
import os
import sys
import time
import numpy as np
import torch
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
from torch.nn.parallel import DistributedDataParallel
from detectron2.engine.defaults import create_ddp_model
import AEsir_utils.data_utils.data_visual as vis
from AEsir_utils.detection_utils import show_bboxes

from torch.utils.data import DataLoader

from clip_models.enc_text import getClip_model
import torch.nn.functional as F
from typing import Optional, List
import cv2
# import detectron2.data.transforms as T
import matplotlib.pyplot as plt





def do_test(
        cfg, 
        model, 
        clip_model: torch.nn.Module, 
        text_tokenizer
    ):
    loader = instantiate(cfg.dataloader.test)
    clip_model.eval()
    model.module.eval() if isinstance(model, DistributedDataParallel) else model.eval()
    visual_dir = cfg.train.output_dir + f"/{cfg.train.output_dir.split('/')[-1]}_visual"
    os.makedirs(visual_dir, exist_ok=True)

    if isinstance(model, DistributedDataParallel):
        model.module.base_idx = [False]
        model.module.novel_idx = [True]
    else:
        model.base_idx = [False]
        model.novel_idx = [True]
    
    with torch.no_grad():
        total_len = len(loader)
        for t, ip in enumerate(loader):
            # vis.print_json_structure(ip)
            image_filepath = ip[0]["file_name"]
            gt_img = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)
            
            img_tensor = ip[0]["image"]
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_id = ip[0]["image_id"]
            
            plt.close()
            fig, ax = plt.subplots(3,3, figsize=(20, 20))
            ax[0][0].imshow(gt_img)
            ax[0][0].set_title("gt")
            show_bboxes(ax[0][0], torch.tensor([ip[0]["gt_boxes"]]), [['gt']], colors=['b'], is_xyxy=False)
            
            level_vocab_lst = ip[0]["level_vocab"]
            neg_vocab_lst = ip[0]["neg_vocab"][:3]
            all_vocab_lst = level_vocab_lst + neg_vocab_lst

            eval_cont_embs = clip_model.encode_text(text_tokenizer(all_vocab_lst).to(next(clip_model.parameters()).device))
            eval_cont_embs = F.normalize(eval_cont_embs, p=2, dim=1)

            for idx, emb in enumerate(eval_cont_embs):
                emb = emb.unsqueeze(0)

                if isinstance(model, DistributedDataParallel):
                    model.module.eval_content_query_embedding = emb
                    model.module.vlm_content_query_embedding = emb
                    # model.module.transformer.decoder.class_embed.eval_zs_weight = emb
                    for class_emb in model.module.transformer.decoder.class_embed:
                        class_emb.eval_zs_weight = emb.permute(1, 0).contiguous()
                else:
                    model.eval_content_query_embedding = emb
                    model.vlm_content_query_embedding = emb
                    # model.transformer.decoder.class_embed.eval_zs_weight = emb
                    for class_emb in model.transformer.decoder.class_embed:
                        class_emb.eval_zs_weight = emb.permute(1, 0).contiguous()
                op = model(ip)[0]["instances"].to("cpu")
                # sigmoid
                # print(f"show op: {op}")
                # print(op.pred_boxes.tensor.shape)
                # print(op.scores.shape)
                # print attr
                op.scores = op.scores.sigmoid()
                topk = 1
                
                # print(f"{torch.tensor(op.pred_boxes).shape}")
                ax[(idx+1)//3][(idx+1)%3].imshow(gt_img)
                ax[(idx+1)//3][(idx+1)%3].set_title(f"{all_vocab_lst[idx]}")
                colors = ['g'] if idx+1 <= len(level_vocab_lst) else ['r']
                show_bboxes(ax[(idx+1)//3][(idx+1)%3], op.pred_boxes.tensor[:topk], [f"{score.item():.2f}" for score in op.scores[:topk]], colors=colors)
            # break
            # print(f"op: {op}")
            plt.savefig(os.path.join(visual_dir, f"{img_id}.png"))
            
            # break
            if (t+1)%10 == 0 and torch.cuda.current_device() == 0:
                print(f"{t+1}/{total_len}")
    # return


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    
    
    if args.ddebug:
        cfg.train.max_iter = 8
        cfg.train.eval_period = 8
        cfg.train.log_period = 4
        cfg.train.checkpointer.period = 8
        cfg.dataloader.train.num_workers = 0
        cfg.dataloader.train.total_batch_size = 1
        cfg.train.output_dir = 'FG_OVD_TEST/output/debug'
        cfg.dataloader.evaluator.output_dir = 'FG_OVD_TEST/output/debug'
        if cfg.model.save_dir:
            cfg.model.save_dir = cfg.model.save_dir + '_debug'
        cfg.DDEBUG = True
    else:
        cfg.DDEBUG = False
    default_setup(cfg, args)

    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
    
    clip_model, tokenizer = getClip_model(model_name = 'convnext_large_d_320', ckpt_file = "pretrain_models/timm_clip_convnext_large_trans.pth")

    clip_model = clip_model.to(cfg.train.device)
    # clip_model.eval()
    
    do_test(cfg, model, clip_model, tokenizer)



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    
    # 先验影响比较大，比如要检测镜子会倾向于框出镜子中的人，要检测桌子会倾向于框出桌子上的物品（150877）
    # 可能可以使用gbc10m训练