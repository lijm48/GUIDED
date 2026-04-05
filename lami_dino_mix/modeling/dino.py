# coding=utf-8
# Copyright 2022 The IDEA Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import copy
import math
import json
from typing import List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from detrex.layers import MLP, box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from detrex.utils import (inverse_sigmoid, is_dist_avail_and_initialized,
                          load_class_freq, get_fed_loss_inds, get_cluster_fed_loss_inds)

from detectron2.modeling import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

from clip_models.FG_clip_model import FG_convext_clip
import clip_models.OpenClip.src.open_clip as open_clip
import logging

logger = logging.getLogger(__name__)




class DINO(nn.Module):
    """Implement DAB-Deformable-DETR in `DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR
    <https://arxiv.org/abs/2203.03605>`_.

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        backbone (nn.Module): backbone module
        position_embedding (nn.Module): position embedding module
        neck (nn.Module): neck module to handle the intermediate outputs features
        transformer (nn.Module): transformer module
        embed_dim (int): dimension of embedding
        num_classes (int): Number of total categories.
        num_queries (int): Number of proposal dynamic anchor boxes in Transformer
        criterion (nn.Module): Criterion for calculating the total losses.
        pixel_mean (List[float]): Pixel mean value for image normalization.
            Default: [123.675, 116.280, 103.530].
        pixel_std (List[float]): Pixel std value for image normalization.
            Default: [58.395, 57.120, 57.375].
        aux_loss (bool): Whether to calculate auxiliary loss in criterion. Default: True.
        select_box_nums_for_evaluation (int): the number of topk candidates
            slected at postprocess for evaluation. Default: 300.
        device (str): Training device. Default: "cuda".
    """

    def __init__(
        self,
        backbone: nn.Module,
        position_embedding: nn.Module,
        neck: nn.Module,
        transformer: nn.Module,
        embed_dim: int,
        num_classes: int,
        num_queries: int,
        criterion: nn.Module,
        classifier,
        query_path,
        eval_query_path,
        vlm_query_path,
        pixel_mean: List[float] = [123.675, 116.280, 103.530],
        pixel_std: List[float] = [58.395, 57.120, 57.375],
        aux_loss: bool = True,
        select_box_nums_for_evaluation: int = 300,
        device="cuda",
        dn_number: int = 100,
        label_noise_ratio: float = 0.2,
        box_noise_scale: float = 1.0,
        use_fed_loss: bool = False,
        cluster_fed_loss: bool = False,
        cluster_label_path=None,
        fed_loss_num_cat: int = 50,
        cat_freq_path = None,
        fed_loss_freq_weight = 0.5,
        score_ensemble: bool = False,
        unseen_classes=None,
        seen_classes=None,
        all_classes=None,
        save_dir=None,
        vlm_temperature: float =100.0,
        alpha: float =0.3,
        beta: float =0.7,
        novel_scale: float =5.0,
        clip_head_path=None,

        class_cnt_per_ann = 11, # total len of category and neg_categories per annotation
        CG_class_cnt = 100,
        lvis_query_path = None,
        lvis_eval_query_path = None,
        lvis_vlm_query_path = None,
        cluster_emb_path = None,
        fg_text_clip: FG_convext_clip = None,
        fgovd_contrastive_margin: float = 0.2
    ):
        super().__init__()
        self.vlm_temperature = vlm_temperature
        self.alpha = alpha
        self.beta = beta
        self.novel_scale = novel_scale
        # define backbone and position embedding module
        self.backbone = backbone
        self.position_embedding = position_embedding

        # define neck module
        self.neck = neck

        # number of dynamic anchor boxes and embedding dimension
        self.num_queries = num_queries
        self.embed_dim = embed_dim

        # define transformer module
        self.transformer = transformer

        # define classification head and box head
        # self.class_embed = nn.Linear(embed_dim, num_classes)
        self.class_embed = classifier
        self.bbox_embed = MLP(embed_dim, embed_dim, 4, 3)
        self.num_classes = num_classes

        # where to calculate auxiliary loss in criterion
        self.aux_loss = aux_loss
        self.criterion = criterion

        # denoising
        # self.label_enc = nn.Embedding(num_classes, embed_dim)
        self.dn_number = dn_number
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # normalizer for input raw images
        self.device = device
        pixel_mean = torch.Tensor(pixel_mean).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(pixel_std).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # initialize weights
        # prior_prob = 0.01
        # bias_value = -math.log((1 - prior_prob) / prior_prob)
        # self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for _, neck_layer in self.neck.named_modules():
            if isinstance(neck_layer, nn.Conv2d):
                nn.init.xavier_uniform_(neck_layer.weight, gain=1)
                nn.init.constant_(neck_layer.bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = transformer.decoder.num_layers + 1
        self.class_embed = nn.ModuleList([copy.deepcopy(self.class_embed) for i in range(num_pred)])
        self.bbox_embed = nn.ModuleList([copy.deepcopy(self.bbox_embed) for i in range(num_pred)])
        nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

        # two-stage
        self.transformer.decoder.class_embed = self.class_embed
        self.transformer.decoder.bbox_embed = self.bbox_embed

        # hack implementation for two-stage
        for bbox_embed_layer in self.bbox_embed:
            nn.init.constant_(bbox_embed_layer.layers[-1].bias.data[2:], 0.0)

        # set topk boxes selected for inference
        self.select_box_nums_for_evaluation = select_box_nums_for_evaluation

        content_query_embedding = torch.tensor(np.load(query_path), dtype=torch.float32, device=device).contiguous()
        self.content_query_embedding = F.normalize(content_query_embedding, p=2, dim=1)

        eval_content_query_embedding = torch.tensor(np.load(eval_query_path), dtype=torch.float32, device=device).contiguous()
        self.eval_content_query_embedding = F.normalize(eval_content_query_embedding, p=2, dim=1)
        # self.eval_content_id = torch.tensor(np.load(eval_id_path), dtype=torch.int64, device=device)
        if vlm_query_path:
            vlm_content_query_embedding = torch.tensor(np.load(vlm_query_path), dtype=torch.float32, device=device).contiguous()# [1203, 768]
            self.vlm_content_query_embedding = F.normalize(vlm_content_query_embedding, p=2, dim=1)
        
        _, feat_dim = self.content_query_embedding.shape
        self.content_layer = nn.Linear(feat_dim, embed_dim)

        self.use_fed_loss = use_fed_loss
        self.cluster_fed_loss = cluster_fed_loss
        self.fed_loss_num_cat = fed_loss_num_cat
        if self.use_fed_loss:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        if self.cluster_fed_loss:
            self.cluster_label = np.load(cluster_label_path)

        self.score_ensemble = score_ensemble
        if self.score_ensemble:
            clip_head = torch.load(clip_head_path)
            self.identical, self.thead = clip_head[0]
            self.head = clip_head[1]

            self.seen_classes = json.load(open(seen_classes))
            self.all_classes = json.load(open(all_classes))
            idx = [self.all_classes.index(seen) for seen in self.seen_classes]
            self.base_idx = torch.zeros(len(self.all_classes), dtype=bool)
            self.base_idx[idx] = True
            if unseen_classes:
                self.unseen_classes = json.load(open(unseen_classes))
                idx_novel = [self.all_classes.index(unseen) for unseen in self.unseen_classes]
                self.novel_idx = torch.zeros(len(self.all_classes), dtype=bool)
                self.novel_idx[idx_novel] = True
            else:
                self.novel_idx = self.base_idx == False
        self.save_dir = save_dir
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        self.fg_text_clip = fg_text_clip
        self.class_cnt_per_ann = class_cnt_per_ann
        self.CG_class_cnt = CG_class_cnt
        
        
        # freeze clip model (only train the final mlp weights)
        for param in self.fg_text_clip.base_clip_model.parameters():
            param.requires_grad = False
        
        content_query_embedding = torch.tensor(np.load(lvis_query_path), dtype=torch.float32, device=self.device).contiguous()
        self.lvis_content_query_embedding = F.normalize(content_query_embedding, p=2, dim=1)

        eval_content_query_embedding = torch.tensor(np.load(lvis_eval_query_path), dtype=torch.float32, device=self.device).contiguous()
        self.lvis_eval_content_query_embedding = F.normalize(eval_content_query_embedding, p=2, dim=1)

        lvis_class_name_json = json.load(open("/apdcephfs_cq12/share_1150325/jiaminghli/data/lvis/lvis_v1_all_classes.json"))
        lvis_class_name_lst = ["A " + name.replace("_", " ") for name in lvis_class_name_json]
        batch_lvis_class_name_emb = []
        
        encode_batch_size = 128
        for i in range(0, len(lvis_class_name_lst), encode_batch_size):
            batch_lvis_class_name_emb.append(self.fg_text_clip.get_batch_text_embs(lvis_class_name_lst[i:i+encode_batch_size], use_mlp=False).detach())

        lvis_class_name_emb = torch.cat(batch_lvis_class_name_emb, dim=0)
        self.lvis_class_name_emb = F.normalize(lvis_class_name_emb, p=2, dim=1).to(self.device)
        
        # self.lvis_content_query_embedding = lvis_class_name_emb
        # self.lvis_eval_content_query_embedding = lvis_class_name_emb

        vlm_content_query_embedding = torch.tensor(np.load(lvis_vlm_query_path), dtype=torch.float32, device=self.device).contiguous()# [1203, 768]
        # vlm_content_query_embedding = self.fg_text_clip.get_batch_text_embs(flatten_instances_name_lst, use_mlp=True)
        self.lvis_vlm_content_query_embedding = F.normalize(vlm_content_query_embedding, p=2, dim=1)
        self.vlm_content_query_embedding2 = self.lvis_vlm_content_query_embedding

        self.lvis_vlm_content_query_embedding = lvis_class_name_emb
        # self.vlm_content_query_embedding = self.lvis_class_name_emb
        
        cluster_id_cluster_emb_map = torch.load(cluster_emb_path)
        
        self.cluster_emb_matrix = torch.concat([emb.unsqueeze(0) for idx, emb in cluster_id_cluster_emb_map.items() if emb is not None], dim=0).to(self.device)
        self.cluster_idx_lst = [cluster_id for cluster_id, emb in cluster_id_cluster_emb_map.items() if emb is not None]
            
        self.fgovd_contrastive_margin = fgovd_contrastive_margin
        self.multi = False
        self.soft_min = False

    def set_lvis_classifier_emb(self):
        self.content_query_embedding = self.lvis_content_query_embedding
        self.eval_content_query_embedding = self.lvis_eval_content_query_embedding
        # self.vlm_content_query_embedding = self.lvis_vlm_content_query_embedding 
        self.vlm_content_query_embedding = self.lvis_class_name_emb 

        for class_emb in self.transformer.decoder.class_embed:
            class_emb.eval_zs_weight = self.eval_content_query_embedding.permute(1, 0).contiguous()
            class_emb.zs_weight = self.content_query_embedding.permute(1, 0).contiguous()

        for class_emb in self.class_embed:
            class_emb.eval_zs_weight = self.eval_content_query_embedding.permute(1, 0).contiguous()
            class_emb.zs_weight = self.content_query_embedding.permute(1, 0).contiguous()

    def set_lvis_classifier_batch_inputs(self,batched_inputs):
        #! 重新实现的 cluster fed loss，通过直接更改 content_query_embedding等
        appear_lvis_class = torch.unique(torch.tensor([item for batch in batched_inputs for item in batch["instances"].gt_classes]))
        lvis_class_name_lst = json.load(open("dataset/lvis/lvis_v1_all_classes.json"))
        appear_lvis_class_name = ["A " + lvis_class_name_lst[idx].replace("_", " ") for idx in appear_lvis_class]
        appear_lvis_class_emb = self.fg_text_clip.get_batch_text_embs(appear_lvis_class_name, use_mlp=False).detach()
        appear_lvis_class_emb = F.normalize(appear_lvis_class_emb, p=2, dim=1)
        # appear_lvis_class_emb = self.lvis_content_query_embedding[appear_lvis_class]
        
        old_gt_new_gt_map = {int(gt): new_gt for new_gt, gt in enumerate(appear_lvis_class)}
        
        lvis_ann_gt_classes_cluster_sim = appear_lvis_class_emb @ self.cluster_emb_matrix.t()
        lvis_ann_gt_classes_cluster_id = torch.argmax(lvis_ann_gt_classes_cluster_sim, dim=1)
        lvis_ann_gt_classes_cluster_id = torch.tensor([self.cluster_idx_lst[idx] for idx in lvis_ann_gt_classes_cluster_id])
        lvis_ann_gt_classes_cluster_id = torch.unique(lvis_ann_gt_classes_cluster_id)
        
        freq_weight = self.freq_weight if self.freq_weight is not None else torch.ones(self.num_classes, device=self.device)
        prob = lvis_ann_gt_classes_cluster_id.new_ones(self.num_classes + 1).float()
        prob[-1] = 0; prob[:self.num_classes] = freq_weight.float().clone()
        
        lvis_cluster_label = torch.tensor(self.cluster_label)
        same_cluster_lvis_class = torch.nonzero(torch.isin(lvis_cluster_label, lvis_ann_gt_classes_cluster_id)).squeeze()
        prob[same_cluster_lvis_class] = 0
        more_appeared_lvis_class_id = torch.multinomial(
            prob, self.CG_class_cnt - appear_lvis_class.shape[0],
            replacement=False)
        
        more_appeared_lvis_class_emb = self.lvis_content_query_embedding[more_appeared_lvis_class_id]
        new_lvis_emb = torch.cat([appear_lvis_class_emb, more_appeared_lvis_class_emb], dim=0)
        self.content_query_embedding = new_lvis_emb
        self.eval_content_query_embedding = new_lvis_emb

        for class_emb in self.transformer.decoder.class_embed:
            class_emb.eval_zs_weight = new_lvis_emb.permute(1, 0).contiguous()
            class_emb.zs_weight = new_lvis_emb.permute(1, 0).contiguous()

        for class_emb in self.class_embed:
            class_emb.eval_zs_weight = new_lvis_emb.permute(1, 0).contiguous()
            class_emb.zs_weight = new_lvis_emb.permute(1, 0).contiguous()
            
        # adjust gt_classes index per annotation of each batch(image)
        for batch in batched_inputs:
            new_gt_class = [old_gt_new_gt_map[int(gt)] for gt in batch["instances"].gt_classes]
            batch['instances'].gt_classes = torch.tensor(new_gt_class, dtype=torch.int64)
            
        return batched_inputs

    def adjust_embeddings_to_threshold(self, a, b, threshold=1):
        """
        移动向量 b，使其与 a 的 cosine similarity 至少为 threshold。
        
        Args:
            a (torch.Tensor): 形状为 (N, D) 的 embedding，假设 norm 为 1。
            b (torch.Tensor): 形状为 (N, D) 的 embedding，假设 norm 为 1。
            threshold (float): 目标最小余弦相似度。
            
        Returns:
            torch.Tensor: 调整后的 b，形状为 (N, D)，norm 为 1。
        """
        
        # 2. 计算当前的余弦相似度
        # shape: (N, 1)
        sim = torch.sum(a * b, dim=-1, keepdim=True)
        
        # 3. 找到需要调整的索引 (当前相似度 < 阈值)
        mask = sim < threshold
        
        # 如果所有向量都满足条件，直接返回 b
        if not mask.any():
            return b
        
        # 4. 计算垂直于 a 的分量 (Gram-Schmidt 正交化)
        # b_perp = b - proj_a(b) = b - (b·a)a
        # 这里 sim 就是 b·a
        b_perp = b - sim * a
        
        # 5. 对垂直分量进行归一化
        # 注意：如果 b 和 a 平行，b_perp 会接近 0，需要加一个 epsilon 防止除零
        # 但由于我们只处理 sim < threshold 的情况，只要 threshold < 1.0，b_perp 就不会是 0
        b_perp_norm = torch.norm(b_perp, p=2, dim=-1, keepdim=True)
        b_perp_normalized = b_perp / (b_perp_norm + 1e-8)
        
        # 6. 构造目标向量
        # 新向量 = (目标投影长度 * a) + (剩余长度 * 垂直方向单位向量)
        # 目标投影长度 = threshold
        # 剩余长度 = sqrt(1 - threshold^2)
        coeff_parallel = threshold
        coeff_perp = (1 - threshold**2)**0.5
        
        b_target = coeff_parallel * a + coeff_perp * b_perp_normalized
        
        # 7. 组合结果：不需要调整的保持原样，需要调整的使用 b_target
        b_final = torch.where(mask, b_target, b)
        
        # 最后再次归一化以消除浮点数计算误差
        b_final = F.normalize(b_final, p=2, dim=-1)
        
        return b_final

    def clip_process_batch_inputs(self, batched_inputs,fg_n_neg=None ,debug=False):
        #! 在粗粒度CG_emb中有batch * 11（1正例+10负例）个emb，每个batch的11个emb为相同的主语
        #! 在细粒度FG_flatten_instances_name_emb中有batch * 11（1正例+10负例）个细粒度emb
        if fg_n_neg==None:
            fg_n_neg = 10

        flatten_instances_name_lst = [name for batch in batched_inputs for item in batch["instances"].gt_instances_name for name in item[:(fg_n_neg+1)]]
        gt_pos_multi_vocab = [ batch["instances"].gt_pos_multi_vocab[0] for batch in batched_inputs ]
        gt_neg_multi_vocab = [ batch["instances"].gt_neg_multi_vocab[0][:fg_n_neg] for batch in batched_inputs ]
        #gt_all_multi_vocab = [ batch["instances"].gt_all_multi_vocab[0] for batch in batched_inputs ]
        # if debug:
        #     import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        max_cnt = 1
        for i in range(len(gt_pos_multi_vocab)):
            if len(gt_pos_multi_vocab[i]) > max_cnt:
                max_cnt = len(gt_pos_multi_vocab[i])
            for j in range(len(gt_neg_multi_vocab[i])):
                if len(gt_neg_multi_vocab[i][j]) > max_cnt:
                    max_cnt = len(gt_neg_multi_vocab[i][j])

        padded_embs = []   # list of (max_cnt, D)
        masks = []         # list of (max_cnt,), 1 for real, 0 for pad
        diff_masks = [] 
        ones_vec = None
        # import pdb;pdb.set_trace()

        for i in range(len(gt_pos_multi_vocab)):
            # pos (a list of phrases)
            pos_embs = self.fg_text_clip.get_batch_text_embs(gt_pos_multi_vocab[i], use_mlp=True)  # (Pi, D)
            pos_embs = F.normalize(pos_embs, p=2, dim=1)
            pos_vocab_set = set(gt_pos_multi_vocab[i])
            
            D = pos_embs.shape[1]
            if ones_vec is None:
                ones_vec = F.normalize(torch.ones(1, D, device=pos_embs.device), p=2, dim=1)  # (1, D)

            pad_len = max_cnt - pos_embs.shape[0]
            if pad_len > 0:
                pos_padded = torch.cat([pos_embs, ones_vec.repeat(pad_len, 1)], dim=0)
            else:
                pos_padded = pos_embs[:max_cnt]
            mask_pos = torch.zeros(max_cnt, device=pos_embs.device)
            mask_pos[:pos_embs.shape[0]] = 1.0
            padded_embs.append(pos_padded)
            masks.append(mask_pos)
            diff_masks.append(mask_pos)

            # negs: each j is a list of phrases
            for j in range(len(gt_neg_multi_vocab[i])):
                neg_embs = self.fg_text_clip.get_batch_text_embs(gt_neg_multi_vocab[i][j], use_mlp=True)  # (Sj, D)
                neg_embs = F.normalize(neg_embs, p=2, dim=1)
                pad_len = max_cnt - neg_embs.shape[0]
                if pad_len > 0:
                    neg_padded = torch.cat([neg_embs, ones_vec.repeat(pad_len, 1)], dim=0)
                else:
                    neg_padded = neg_embs[:max_cnt]
                mask_neg = torch.zeros(max_cnt, device=neg_embs.device)
                mask_neg[:neg_embs.shape[0]] = 1.0

                mask_diff_neg = torch.zeros(max_cnt, device=neg_embs.device)
                
                # 遍历当前负样本的每一个描述短语
                flag = 0
                for k, phrase in enumerate(gt_neg_multi_vocab[i][j]):
                    # 如果该短语不在正样本集合中，标记为 1 (错误属性)
                    # 且必须在 max_cnt 范围内（虽然逻辑上不会越界，但为了安全）
                    if k < max_cnt and phrase not in pos_vocab_set:
                        mask_diff_neg[k] = 1.0
                        flag = 1
                if flag == 0:
                    # import pdb;pdb.set_trace()
                    # print(batched_inputs[i]['instances'].gt_instances_name)
                    # print(gt_pos_multi_vocab[i])
                    # print(gt_neg_multi_vocab[i][j])
                    mask_diff_neg[0] = 1.0
                    print("bbb")
                padded_embs.append(neg_padded)
                masks.append(mask_neg)
                diff_masks.append(mask_diff_neg)
        


        # (N, max_cnt, D) and (N, max_cnt), ordered by batch, and within batch: [pos, neg1..negK]
        vlm_all_query_embedding = torch.stack(padded_embs, dim=0)
        vlm_all_query_mask = torch.stack(masks, dim=0)
        vlm_diff_mask = torch.stack(diff_masks, dim=0)

        FG_flatten_instances_name_emb = self.fg_text_clip.get_batch_text_embs(flatten_instances_name_lst, use_mlp=True)
        FG_flatten_instances_name_emb = F.normalize(FG_flatten_instances_name_emb, p=2, dim=1)
        
        CG_subject_name_lst = [item for batch in batched_inputs for item in [batch["instances"].subject[0]]*(fg_n_neg+1)]
        CG_subject_emb = self.fg_text_clip.get_batch_text_embs(CG_subject_name_lst, use_mlp=False).detach()
        CG_subject_emb = F.normalize(CG_subject_emb, p=2, dim=1)
        CG_emb = CG_subject_emb
        self.CG_class_cnt = CG_emb.shape[0]


        # CG_subject_emb2 = self.fg_text_clip.get_batch_text_embs(CG_subject_name_lst, use_mlp2=True).detach()
        # CG_subject_emb2 = F.normalize(CG_subject_emb2, p=2, dim=1)
        # import pdb;pdb.set_trace()
        


        # padded_embs_coarse = []   # list of (max_cnt, D)
        # masks_coarse = []         # list of (max_cnt,), 1 for real, 0 for pad
        # ones_vec = None
        # cnt = 0

        # for i in range(len(gt_pos_multi_vocab)):
        #     # pos (a list of phrases)
        #     pos_embs = self.fg_text_clip.get_batch_text_embs(gt_pos_multi_vocab[i], use_mlp=False)  # (Pi, D)
        #     pos_embs= F.normalize(pos_embs, p=2, dim=1)
        #     pos_embs = self.adjust_embeddings_to_threshold(CG_emb[cnt], pos_embs)
        #     # import pdb;pdb.set_trace()
            
        #     D = pos_embs.shape[1]
        #     if ones_vec is None:
        #         ones_vec = F.normalize(torch.zeros(1, D, device=pos_embs.device), p=2, dim=1)  # (1, D)

        #     pad_len = max_cnt - pos_embs.shape[0]
        #     if pad_len > 0:
        #         pos_padded = torch.cat([pos_embs, ones_vec.repeat(pad_len, 1)], dim=0)
        #     else:
        #         pos_padded = pos_embs[:max_cnt]
        #     mask_pos = torch.zeros(max_cnt, device=pos_embs.device)
        #     mask_pos[:pos_embs.shape[0]] = 1.0
        #     padded_embs_coarse.append(pos_padded)
        #     masks_coarse.append(mask_pos)
        #     cnt+=1

        #     # negs: each j is a list of phrases
        #     for j in range(len(gt_neg_multi_vocab[i])):
        #         neg_embs = self.fg_text_clip.get_batch_text_embs(gt_neg_multi_vocab[i][j], use_mlp=False)  # (Sj, D)
        #         neg_embs = F.normalize(neg_embs, p=2, dim=1)
        #         # import pdb;pdb.set_trace()
        #         neg_embs = self.adjust_embeddings_to_threshold(CG_emb[cnt], neg_embs)
                
        #         pad_len = max_cnt - neg_embs.shape[0]
        #         if pad_len > 0:
        #             neg_padded = torch.cat([neg_embs, ones_vec.repeat(pad_len, 1)], dim=0)
        #         else:
        #             neg_padded = neg_embs[:max_cnt]
        #         mask_neg = torch.zeros(max_cnt, device=neg_embs.device)
        #         mask_neg[:neg_embs.shape[0]] = 1.0
        #         padded_embs_coarse.append(neg_padded)
        #         masks_coarse.append(mask_neg)
        #         cnt += 1
        # assert cnt == CG_emb.shape[0]
        # vlm_all_query_embedding_coarse = torch.stack(padded_embs_coarse, dim=0)
        # vlm_all_query_mask_coarse = torch.stack(masks_coarse, dim=0)
        # # vlm_all_query_embedding_coarse 
        # CG_emb = ( (vlm_all_query_embedding_coarse * vlm_all_query_mask_coarse.unsqueeze(2)).sum(dim=1)  + CG_subject_emb) / (vlm_all_query_mask_coarse.sum(dim=-1,keepdim=True)+1)
        # CG_emb = F.normalize(CG_emb, p=2, dim=1)



        # import pdb;pdb.set_trace()
        # adjust model classifier weights
        self.vlm_content_query_embedding = FG_flatten_instances_name_emb
        self.eval_content_query_embedding = CG_emb
        self.content_query_embedding = CG_emb
        # self.content_query_embedding2 = CG_emb2
        self.CG_subject_name_lst = CG_subject_name_lst 
        self.max_cnt = max_cnt
        self.vlm_all_query_embedding = vlm_all_query_embedding          # (N, max_cnt, D)
        self.vlm_all_query_mask = vlm_all_query_mask
        self.vlm_diff_mask = vlm_diff_mask
        # if  self.vlm_all_query_embedding.shape[0] != self.vlm_content_query_embedding.shape[0]:
        #     pass




        for class_emb in self.transformer.decoder.class_embed:
            class_emb.eval_zs_weight = CG_emb.permute(1, 0).contiguous()
            class_emb.zs_weight = CG_emb.permute(1, 0).contiguous()

        for class_emb in self.class_embed:
            class_emb.eval_zs_weight = CG_emb.permute(1, 0).contiguous()
            class_emb.zs_weight = CG_emb.permute(1, 0).contiguous()
            
        # adjust gt_classes index per annotation of each batch(image)
        for idx, batch in enumerate(batched_inputs):
            batch['instances'].gt_classes = torch.tensor([idx*(fg_n_neg+1)], dtype=torch.int64)
            
        return batched_inputs



    def filter_content_info(self, batched_inputs):
        freq_weight = self.freq_weight if self.freq_weight is not None else torch.ones(self.num_classes, device=self.device)
        inner_gt = []
        for target in batched_inputs:
            target = target['instances'].gt_classes
            inner_gt.append(target)
        inner_gt = torch.cat(inner_gt)

        if self.cluster_fed_loss:
            content_inds = get_cluster_fed_loss_inds(
                inner_gt,
                num_sample_cats=self.fed_loss_num_cat,
                C=self.num_classes,
                weight=freq_weight,
                cluster_label=self.cluster_label)
        else:
            content_inds = get_fed_loss_inds(
                inner_gt,
                num_sample_cats=self.fed_loss_num_cat,
                C=self.num_classes,
                weight=freq_weight)

        convert_map = torch.ones(self.num_classes, dtype=torch.int64) * -1
        for idx, content_id in enumerate(content_inds):
            convert_map[content_id.item()] = idx
        for idx, target in enumerate(batched_inputs):
            cats = target['instances'].gt_classes
            batched_inputs[idx]['instances'].gt_classes = convert_map[batched_inputs[idx]['instances'].gt_classes]

        return content_inds, batched_inputs
 
    def forward(self, batched_inputs, fg_n_neg=None, use_multi=True):
        """Forward function of `DINO` which excepts a list of dict as inputs.

        Args:
            batched_inputs (List[dict]): A list of instance dict, and each instance dict must consists of:
                - dict["image"] (torch.Tensor): The unnormalized image tensor.
                - dict["height"] (int): The original image height.
                - dict["width"] (int): The original image width.
                - dict["instance"] (detectron2.structures.Instances):
                    Image meta informations and ground truth boxes and labels during training.
                    Please refer to
                    https://detectron2.readthedocs.io/en/latest/modules/structures.html#detectron2.structures.Instances
                    for the basic usage of Instances.

        Returns:
            dict: Returns a dict with the following elements:
                - dict["pred_logits"]: the classification logits for all queries (anchor boxes in DAB-DETR).
                            with shape ``[batch_size, num_queries, num_classes]``
                - dict["pred_boxes"]: The normalized boxes coordinates for all queries in format
                    ``(x, y, w, h)``. These values are normalized in [0, 1] relative to the size of
                    each individual image (disregarding possible padding). See PostProcess for information
                    on how to retrieve the unnormalized bounding box.
                - dict["aux_outputs"]: Optional, only returned when auxilary losses are activated. It is a list of
                            dictionnaries containing the two above keys for each decoder layer.
        """

        if self.training:
            # if input data is from FG-OVD
            if batched_inputs[0].get("FG_OVD_FLAG" , None) == 'fgovd':
                batched_inputs = self.clip_process_batch_inputs(batched_inputs,fg_n_neg)
                self.use_fed_loss = False
            else:
                self.CG_subject_name_lst = None
                # self.fed_loss_num_cat=100
                self.set_lvis_classifier_emb()
                self.use_fed_loss = True
                # batched_inputs = self.set_lvis_classifier_batch_inputs(batched_inputs)
                # self.use_fed_loss = False
        else:
            if batched_inputs[0].get("FG_OVD_FLAG" , None) == 'fgovd':
                batched_inputs = self.clip_process_batch_inputs(batched_inputs,fg_n_neg, debug = True)
            elif batched_inputs[0].get("FG_OVD_FLAG" , None) == 'modify':
                # import pdb;pdb.set_trace()
                pass
            else:
                # self.fed_loss_num_cat=100
                self.set_lvis_classifier_emb()


        if self.save_dir:
            filename = batched_inputs[0]['file_name'].split('/')[-1].replace('jpg', 'pth')

        images = self.preprocess_image(batched_inputs)

        content_inds = None
        if self.training:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_ones(batch_size, H, W)
            for img_id in range(batch_size):
                img_h, img_w = batched_inputs[img_id]["instances"].image_size
                img_masks[img_id, :img_h, :img_w] = 0
            if self.use_fed_loss:
                content_inds, batched_inputs = self.filter_content_info(batched_inputs)
            
        else:
            batch_size, _, H, W = images.tensor.shape
            img_masks = images.tensor.new_zeros(batch_size, H, W)

        # original features
        if self.score_ensemble:
            features, features_wonorm = self.backbone(images.tensor)  # output feature dict
        else:
            features = self.backbone(images.tensor)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.position_embedding(multi_level_masks[-1]))

        if self.training:
            content_query_embeds = self.content_query_embedding[content_inds] if content_inds is not None else self.content_query_embedding
            content_query_embeds = self.content_layer(content_query_embeds)
            content_query_embeds = F.normalize(content_query_embeds, p=2, dim=1)
        else:
            content_query_embeds = self.content_layer(self.eval_content_query_embedding)
            content_query_embeds = F.normalize(content_query_embeds, p=2, dim=1)
 
        # denoising preprocessing
        # prepare label query embedding
        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if batched_inputs[0].get("FG_OVD_FLAG" , None):
                cdn_num_classes = self.CG_class_cnt
            else:
                cdn_num_classes = self.fed_loss_num_cat #if self.use_fed_loss else self.num_classes
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=cdn_num_classes,
                hidden_dim=self.embed_dim,
                # label_enc=self.label_enc,
                content_query_embeds=content_query_embeds,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
            content_query_embeds=content_query_embeds,
            content_inds=content_inds, 
        )
        # hack implementation for distributed training
        # inter_states[0] += self.label_enc.weight[0, 0] * 0.0
        inter_states[0] += self.content_layer.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        outputs_classes = []
        outputs_coords = []
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl], content_inds=content_inds)
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        outputs_coord = torch.stack(outputs_coords)
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # denoising postprocessing
        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        # prepare for loss computation
        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        # prepare two stage output
        interm_coord = enc_reference
        interm_class = self.transformer.decoder.class_embed[-1](enc_state, content_inds=content_inds)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        if self.training:
            #################################### FG-OVD vlm backward ####################################
            if batched_inputs[0].get("FG_OVD_FLAG" , None):
            # if True:
                box_cls = output["pred_logits"].clone()
                box_pred = output["pred_boxes"].clone()
                roi_features_ori = []
                for box in box_pred:
                    roi_features_ori.append(self.extract_region_feature(features_wonorm, box.unsqueeze(0), 'p3')) # 1, 900, 768
                roi_features_ori = torch.cat(roi_features_ori, dim=0)
                cls_score = box_cls.sigmoid()

                
                fg_cate_cnt = self.vlm_content_query_embedding.shape[0]
                # batch_size, 900, batch_size * 11
                # vlm_content_query_embedding = self.vlm_content_query_embedding
                if batched_inputs[0].get("FG_OVD_FLAG" , None): 
                    vlm_content_query_embedding = self.vlm_content_query_embedding
                else:
                    vlm_content_query_embedding = self.vlm_content_query_embedding[content_inds]

                vlm_logit = roi_features_ori @ vlm_content_query_embedding.t() * self.vlm_temperature
                vlm_score = F.softmax(vlm_logit, dim=-1)


                new_cls_score = cls_score ** (1 - self.beta) * vlm_score ** self.beta

                output["pred_logits"] = inverse_sigmoid(new_cls_score)
                
                loss_dict = self.criterion(output, targets, dn_meta)
            else:
                loss_dict = self.criterion(output, targets, dn_meta)
            ###############################################################################################
            
            
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict
        else:
            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            if self.save_dir and not self.score_ensemble:
                save_output = {}
                save_output["pred_logits"] = copy.deepcopy(output["pred_logits"]).cpu()
                save_output["pred_boxes"] = copy.deepcopy(output["pred_boxes"]).cpu()
                torch.save(save_output, os.path.join(self.save_dir, filename))
            if self.score_ensemble:
                roi_features_ori = self.extract_region_feature(features_wonorm, box_pred, 'p3')

                cls_score = box_cls.sigmoid()

                vlm_score = roi_features_ori @ self.vlm_content_query_embedding.t() * self.vlm_temperature
                vlm_score = vlm_score.softmax(dim=-1)

                if batched_inputs[0].get("FG_OVD_FLAG" , None): # FG-OVD data
                    cls_score = cls_score ** (1-self.beta) * vlm_score ** self.beta
                else:
                    cls_score[:, :, self.base_idx] = cls_score[:, :, self.base_idx] ** (
                            1 - self.alpha) * vlm_score[:, :, self.base_idx] ** self.alpha
                    cls_score[:, :, self.novel_idx] = cls_score[:, :, self.novel_idx] ** (
                            1 - self.beta) * vlm_score[:, :, self.novel_idx] ** self.beta 
                    cls_score[:, :, self.novel_idx] = cls_score[:, :, self.novel_idx] * self.novel_scale
                box_cls = cls_score
                results = self.inference(box_cls, box_pred, images.image_sizes, wo_sigmoid=True)
            else:
                results = self.inference(box_cls, box_pred, images.image_sizes)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
    
    def extract_region_feature(self, features, bbox, layer_name):
        if layer_name == 'p2':
            h, w = features['p2'].shape[-2:]# 50 75
        elif layer_name == 'p3':
            h, w = features['p3'].shape[-2:]# 50 75

        rpn_boxes = box_cxcywh_to_xyxy(bbox)
        rpn_boxes = torch.clamp(rpn_boxes, min=0, max=1)
        for i in range(len(rpn_boxes)):
            rpn_boxes[i][:,[0,2]] = rpn_boxes[i][:,[0,2]] * w
            rpn_boxes[i][:,[1,3]] = rpn_boxes[i][:,[1,3]] * h
        rpn_boxes = [rpn_box for rpn_box in rpn_boxes]
       
        bs = len(rpn_boxes)
        roi_features = torchvision.ops.roi_align(
            # hid,# [2, 768, 50, 66]
            features['p2'] if layer_name == 'p2' else features['p3'],
            rpn_boxes,
            output_size=(15, 15),
            spatial_scale=1.0,
            aligned=True)  # (bs * num_queries, c, 14, 14) [1800, 768, 30, 30]

        if layer_name == 'p2':
            roi_features = self.backbone.downsample_layers[3](roi_features)# [33, 768, 30, 30]->[33, 1536, 15, 15] 
            roi_features = self.backbone.stages[3](roi_features)# [33, 1536, 15, 15]->[33, 1536, 15, 15]
        roi_features = self.identical(roi_features)# [900, 1536, 15, 15]
        roi_features = self.thead(roi_features)# [900, 1536]
        roi_features = self.head(roi_features)# [900, 768] TODO:
        roi_features = roi_features.reshape(bs, -1, roi_features.shape[-1])
        roi_features = nn.functional.normalize(roi_features, dim=-1)# [1, 900, 768]
        return roi_features


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]

    def prepare_for_cdn(
        self,
        targets,
        dn_number,
        label_noise_ratio,
        box_noise_scale,
        num_queries,
        num_classes,
        hidden_dim,
        label_enc=None,
        content_query_embeds=None,
        convert_map=None,
    ):
        """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding
            in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
        if dn_number <= 0:
            return None, None, None, None
            # positive and negative dn queries
        dn_number = dn_number * 2
        known = [(torch.ones_like(t["labels"])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known]
        if int(max(known_num)) == 0:
            return None, None, None, None

        dn_number = dn_number // (int(max(known_num) * 2))

        if dn_number == 0:
            dn_number = 1
        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat(
            [torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)]
        )

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
        known_bboxs = boxes.repeat(2 * dn_number, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(
                -1
            )  # half of bbox prob
            new_label = torch.randint_like(
                chosen_indice, 0, num_classes
            )  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        single_padding = int(max(known_num))

        pad_size = int(single_padding * 2 * dn_number)
        positive_idx = (
            torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        )
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(boxes)
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = (
                torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            )
            rand_part = torch.rand_like(known_bboxs)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to("cuda")
        # input_label_embed = label_enc(m)
        
        if content_query_embeds is not None:
            input_label_content = content_query_embeds[m]
            input_label_embed = input_label_content

        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to("cuda")
        if len(known_num):
            map_known_indice = torch.cat(
                [torch.tensor(range(num)) for num in known_num]
            )  # [1,2, 1,2,3]
            map_known_indice = torch.cat(
                [map_known_indice + single_padding * i for i in range(2 * dn_number)]
            ).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to("cuda") < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
            if i == dn_number - 1:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * i * 2
                ] = True
            else:
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1),
                    single_padding * 2 * (i + 1) : pad_size,
                ] = True
                attn_mask[
                    single_padding * 2 * i : single_padding * 2 * (i + 1), : single_padding * 2 * i
                ] = True

        dn_meta = {
            "single_padding": single_padding * 2,
            "dn_num": dn_number,
        }

        return input_query_label, input_query_bbox, attn_mask, dn_meta

    def dn_post_process(self, outputs_class, outputs_coord, dn_metas):
        if dn_metas and dn_metas["single_padding"] > 0:
            padding_size = dn_metas["single_padding"] * dn_metas["dn_num"]
            output_known_class = outputs_class[:, :, :padding_size, :]
            output_known_coord = outputs_coord[:, :, :padding_size, :]
            outputs_class = outputs_class[:, :, padding_size:, :]
            outputs_coord = outputs_coord[:, :, padding_size:, :]

            out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1]}
            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_coord)
            dn_metas["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord

    def preprocess_image(self, batched_inputs):
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images)
        return images

    def inference(self, box_cls, box_pred, image_sizes, wo_sigmoid=False):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_queries, K).
                The tensor predicts the classification probability for each query.
            box_pred (Tensor): tensors of shape (batch_size, num_queries, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every queryx
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        # box_cls.shape: 1, 300, 80
        # box_pred.shape: 1, 300, 4
        if wo_sigmoid:
            prob = box_cls
        else:
            prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # For each box we assign the best class or the second best if the best on is `no_object`.
        # scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

        for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(
            zip(scores, labels, boxes, image_sizes)
        ):
            result = Instances(image_size)
            result.pred_boxes = Boxes(box_cxcywh_to_xyxy(box_pred_per_image))

            result.pred_boxes.scale(scale_x=image_size[1], scale_y=image_size[0])
            result.scores = scores_per_image
            result.pred_classes = labels_per_image
            results.append(result)
        return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_scores = targets_per_image.gt_scores
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes, "scores": gt_scores})
        return new_targets

    def compute_fgovd_contrastive_loss(self, scores):
        # if we are in the FG-OVD case, the score to maximize is always the first (positive caption)
        # in addition we perform only triplet loss keeping the image as anchor (row-wise optimization)
        positive_scores = scores[:, 0].view(scores.size(0), 1)
        d1 = positive_scores.expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.fgovd_contrastive_margin + scores - d1).clamp(min=0)

        # mask with all elements True in the first column
        mask = torch.cat((torch.ones(scores.shape[0], 1), torch.zeros(scores.shape[0], scores.shape[1] - 1)), dim=1) > 0.5
        I = mask
        if torch.cuda.is_available():
            I = I.to(scores.device)
        cost_s = cost_s.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        # if self.max_violation:
        #     cost_s = cost_s.max(1)[0]

        loss = cost_s.sum()
        return loss