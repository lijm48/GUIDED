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

import torch
import torch.nn as nn

from detrex.layers import (
    FFN,
    MLP,
    BaseTransformerLayer,
    MultiheadAttention,
    MultiScaleDeformableAttention,
    TransformerLayerSequence,
    get_sine_pos_embed,
)
from detrex.utils import inverse_sigmoid
import torch.nn.functional as F
import math


class DINOTransformerEncoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        post_norm: bool = False,
        num_feature_levels: int = 4,
    ):
        super(DINOTransformerEncoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=MultiScaleDeformableAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=attn_dropout,
                    batch_first=True,
                    num_levels=num_feature_levels,
                ),
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    num_fcs=2,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.embed_dim = self.layers[0].embed_dim
        self.pre_norm = self.layers[0].pre_norm

        if post_norm:
            self.post_norm_layer = nn.LayerNorm(self.embed_dim)
        else:
            self.post_norm_layer = None

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        **kwargs,
    ):

        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs,
            )

        if self.post_norm_layer is not None:
            query = self.post_norm_layer(query)
        return query


class DINOTransformerDecoder(TransformerLayerSequence):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        num_layers: int = 6,
        return_intermediate: bool = True,
        num_feature_levels: int = 4,
        look_forward_twice=True,
    ):
        super(DINOTransformerDecoder, self).__init__(
            transformer_layers=BaseTransformerLayer(
                attn=[
                    MultiheadAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        attn_drop=attn_dropout,
                        batch_first=True,
                    ),
                    MultiScaleDeformableAttention(
                        embed_dim=embed_dim,
                        num_heads=num_heads,
                        dropout=attn_dropout,
                        batch_first=True,
                        num_levels=num_feature_levels,
                    ),
                ],
                ffn=FFN(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    output_dim=embed_dim,
                    ffn_drop=ffn_dropout,
                ),
                norm=nn.LayerNorm(embed_dim),
                operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
            ),
            num_layers=num_layers,
        )
        self.return_intermediate = return_intermediate

        self.ref_point_head = MLP(2 * embed_dim, embed_dim, embed_dim, 2)

        self.bbox_embed = None
        self.class_embed = None
        self.look_forward_twice = look_forward_twice
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query,
        key,
        value,
        query_pos=None,
        key_pos=None,
        attn_masks=None,
        query_key_padding_mask=None,
        key_padding_mask=None,
        reference_points=None,  # num_queries, 4. normalized.
        valid_ratios=None,
        **kwargs,
    ):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)  # bs, num_queries, 4

        intermediate = []
        intermediate_reference_points = []
        for layer_idx, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

            query_sine_embed = get_sine_pos_embed(reference_points_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)

            output = layer(
                output,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                query_sine_embed=query_sine_embed,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                reference_points=reference_points_input,
                **kwargs,
            )

            if self.bbox_embed is not None:
                tmp = self.bbox_embed[layer_idx](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))
                if self.look_forward_twice:
                    intermediate_reference_points.append(new_reference_points)
                else:
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points




class DINOTransformer(nn.Module):
    """Transformer module for DINO

    Args:
        encoder (nn.Module): encoder module.
        decoder (nn.Module): decoder module.
        as_two_stage (bool): whether to use two-stage transformer. Default False.
        num_feature_levels (int): number of feature levels. Default 4.
        two_stage_num_proposals (int): number of proposals in two-stage transformer. Default 900.
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        two_stage_num_proposals=900,
        category_visual_vocabs_emb_path='dataset/metadata/lvis_visual_descs_convnext_large_d_320.pt',
        # learnt_init_query=False,
    ):
        super(DINOTransformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals

        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dim))

        # self.learnt_init_query = learnt_init_query
        # if self.learnt_init_query:
        #     self.tgt_embed = nn.Embedding(self.two_stage_num_proposals, self.embed_dim)
        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)

        self.init_weights()

        ##################### new modules ############################
        self.content_query_attention = Attribute_Attention(
            head_query_dim=256,
            Vocab_embed_dim=768,
            expert_cnt=6,
        )

        category_visual_vocabs_emb = torch.load(category_visual_vocabs_emb_path)
        max_vocab_cnt = 0
        for idx, vocab_emb in category_visual_vocabs_emb.items():
            max_vocab_cnt = max(max_vocab_cnt, vocab_emb.size(0))
        self.category_visual_vocabs_emb = []
        self.category_visual_vocabs_emb_mask = torch.zeros(len(category_visual_vocabs_emb.keys()), max_vocab_cnt, dtype=torch.bool).cuda()
        for idx, vocab_emb in category_visual_vocabs_emb.items():
            paddeed_vocab_emb = torch.cat([
                    vocab_emb,
                    torch.zeros(max_vocab_cnt - vocab_emb.size(0), vocab_emb.size(1), dtype=torch.float32, device=vocab_emb.device)
                ],
                dim=0
            )
            self.category_visual_vocabs_emb.append(paddeed_vocab_emb.unsqueeze(0))
            self.category_visual_vocabs_emb_mask[idx, :vocab_emb.size(0)] = True
        # cate_cnt, max_vocab_cnt, 768
        self.category_visual_vocabs_emb = torch.cat(self.category_visual_vocabs_emb, dim=0).cuda()
        ##############################################################

    def get_batch_vocab_emb_and_mask(self, real_enc_outputs_class_idx):
        """
        use lvis category idx to get category visual vocab emb set
        input:
            real_enc_outputs_class_idx: bs, num_queries, 1
        output:
            batch_vocab_emb: bs, num_queries, max_vocab_cnt(12), 768
            batch_vocab_emb_mask: bs, num_queries, max_vocab_cnt(12)
        """
        bs, num_queries = real_enc_outputs_class_idx.size()

        batch_vocab_emb = []
        query_batch = 10
        for q_idx in range(0, num_queries, query_batch):
            # bs, 1, max_vocab_cnt, 768
            sub_batch_vocab_emb = torch.gather(
                # bs, query_batch, cate_cnt, max_vocab_cnt, 768
                self.category_visual_vocabs_emb[None,None,:,:,:].repeat(bs, query_batch, 1, 1, 1),
                2,
                # bs, query_batch, 1, max_vocab_cnt, 768
                real_enc_outputs_class_idx[:,q_idx:q_idx+query_batch,None,None,None].repeat(1, 1, 1, self.category_visual_vocabs_emb.size(1), self.category_visual_vocabs_emb.size(2)),
            )
            batch_vocab_emb.append(sub_batch_vocab_emb)
        # bs, num_queries, max_vocab_cnt, 768
        batch_vocab_emb = torch.cat(batch_vocab_emb, dim=1).squeeze(2)
        
        # bs, num_queries, max_vocab_cnt
        batch_vocab_emb_mask = torch.gather(
            # bs, num_queries, cate_cnt, max_vocab_cnt
            self.category_visual_vocabs_emb_mask[None,None,:,:].repeat(bs, num_queries, 1, 1),
            2,
            # bs, num_queries, 1, max_vocab_cnt
            real_enc_outputs_class_idx[:,:,None,None].repeat(1, 1, 1, self.category_visual_vocabs_emb_mask.size(1)),
        )
        batch_vocab_emb_mask = batch_vocab_emb_mask.squeeze(2)
        return batch_vocab_emb, batch_vocab_emb_mask

    def get_vocab_emb_and_mask(self, content_inds=None):
        """
        return:
            category_visual_vocabs_emb: [cate_cnt, max_vocab_cnt, 768]
            category_visual_vocabs_emb_mask: [cate_cnt, max_vocab_cnt]
        """
        if content_inds is None:
            return self.category_visual_vocabs_emb, self.category_visual_vocabs_emb_mask
        else:
            return self.category_visual_vocabs_emb[content_inds], self.category_visual_vocabs_emb_mask[content_inds]


    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(N, -1, 4)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(
            -1, keepdim=True
        )
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float("inf"))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Get the reference points used in decoder.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The ratios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            #  TODO  check this 0.5
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        """Get the valid ratios of feature maps of all levels."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        attn_masks,
        content_query_embeds,
        content_inds=None,
        **kwargs,
    ):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            feat = feat.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device
        )

        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,  # bs, num_token, num_level, 2
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        # output_memory: bs, num_tokens, c
        # output_proposals: bs, num_tokens, 4. unsigmoided.

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory, content_inds=content_inds)
        enc_outputs_coord_unact = (
            self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )  # unsigmoided.
        # import pdb; pdb.set_trace()
        max_scores, max_labels = torch.max(enc_outputs_class, dim=-1)

        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(max_scores, topk, dim=1)[1]

        # extract region proposal boxes
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )  # unsigmoided.
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points

        # extract region features
        target_unact = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )

        content_ids = torch.gather(max_labels, 1, topk_proposals)

        ###################################### get real lvis ids ######################################
        # bs, num_queries
        if query_embed[2] is not None:
            all_content_ids = torch.cat([query_embed[2], content_ids], 1)
        else:
            all_content_ids = content_ids

        if content_inds is not None:
            real_enc_outputs_class_idx = torch.gather(
                content_inds[None, None, :].repeat(bs, all_content_ids.shape[1], 1).to(all_content_ids.device), 
                2, 
                all_content_ids.unsqueeze(-1)
            ).squeeze(-1) # bs, num_queries
        else:
            real_enc_outputs_class_idx = content_ids


        vocab_query_emb, vocab_query_emb_mask = self.get_vocab_emb_and_mask(content_inds=content_inds)
        # cate_cnt, query_dim(256)
        attn_content_query_embeds = self.content_query_attention(vocab_query_emb, vocab_query_emb_mask)
        # 


        #############################################################################################
        content_query = torch.gather(
            attn_content_query_embeds.unsqueeze(0).repeat(bs, 1, 1), 1,
            all_content_ids.unsqueeze(-1).repeat(1, 1, 256)) 

        if query_embed[0] is not None:
            all_target = torch.cat([query_embed[0], target_unact], 1)
        else:
            all_target = target_unact

        #! change
        target = all_target.detach() + content_query

        # decoder
        inter_states, inter_references = self.decoder(
            query=target,  # bs, num_queries, embed_dims
            key=memory,  # bs, num_tokens, embed_dims
            value=memory,  # bs, num_tokens, embed_dims
            query_pos=None,
            key_padding_mask=mask_flatten,  # bs, num_tokens
            reference_points=reference_points,  # num_queries, 4
            spatial_shapes=spatial_shapes,  # nlvl, 2
            level_start_index=level_start_index,  # nlvl
            valid_ratios=valid_ratios,  # bs, nlvl, 2
            attn_masks=attn_masks,
            **kwargs,
        )

        inter_references_out = inter_references
        return (
            inter_states,
            init_reference_out,
            inter_references_out,
            target_unact,
            topk_coords_unact.sigmoid(),
        )




class Attribute_Attention(nn.Module):
    def __init__(self, 
            head_query_dim: int = 256,
            Vocab_embed_dim: int = 768,
            expert_cnt: int = 1,
            dropout: float = 0.1,
            ffn_dropout: float = 0.1
        ):
        super().__init__()
        self.head_query_dim = head_query_dim
        self.Vocab_embed_dim = Vocab_embed_dim
        self.expert_cnt = expert_cnt
        self.expert_query = nn.Parameter(torch.empty(expert_cnt, head_query_dim))
        torch.nn.init.xavier_normal_(self.expert_query)
        
        self.k_proj = nn.Linear(Vocab_embed_dim, head_query_dim)
        self.v_proj = nn.Linear(Vocab_embed_dim, head_query_dim)
        torch.nn.init.xavier_normal_(self.k_proj.weight)
        torch.nn.init.xavier_normal_(self.v_proj.weight)
        




    def forward(self, vocab_embed, vocab_embed_mask=None):
        """
        mask type: 1 = valid, 0 = invalid
        args:
            vocab_embed: cate_cnt, vocab_cnt, Vocab_embed_dim(768)
            vocab_embed_mask: cate_cnt, vocab_cnt
        return:
            expert_cat_emb: num_experts, cate_cnt, head_query_dim
        """
        cate_cnt, vocab_cnt, Vocab_embed_dim = vocab_embed.size()
        q = self.expert_query 
        
        mix_vocab_embed = vocab_embed - vocab_embed[:, 0:1]
        # import pdb;pdb.set_trace()
        # k = self.k_proj(vocab_embed)
        k = self.k_proj(mix_vocab_embed) # cate_cnt, vocab_cnt, head_query_dim
        v = self.v_proj(vocab_embed) # cate_cnt, vocab_cnt, head_query_dim

        
        k = k.reshape(cate_cnt * vocab_cnt, self.head_query_dim)
        # num_experts, cate_cnt * vocab_cnt
        attn_score = torch.matmul(q, k.transpose(0,1)) / (self.head_query_dim ** 0.5) 
        # num_experts, cate_cnt, vocab_cnt
        attn_score = attn_score.reshape(self.expert_cnt, cate_cnt, vocab_cnt) 
        # mask opt
        if vocab_embed_mask is not None:
            attn_score = attn_score.masked_fill(vocab_embed_mask.unsqueeze(0).expand(self.expert_cnt, -1, -1) == 0, float('-inf'))
        
        attn_score = attn_score.softmax(dim=-1)
        # num_experts, cate_cnt, head_query_dim
        attn_out = torch.matmul(
            attn_score.unsqueeze(-1).transpose(2, 3), # num_experts, cate_cnt, 1, vocab_cnt
            v.unsqueeze(0).expand(self.expert_cnt, -1, -1, -1) # num_experts, cate_cnt, vocab_cnt, head_query_dim
        ).squeeze(-2)
        attn_out = attn_out.sum(dim=0)
        return attn_out
        
