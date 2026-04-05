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

"""
DINOTransformerAttr: DINO Transformer with Attribute_Attention.

Key difference vs. dino_transformer.py:
  - Loads LVIS multi-grained vocab embeddings from a .pt dict file
    (produced by gen_MG_vocab3.py).
  - Builds an Attribute_Attention module that aggregates per-category
    atomic-phrase CLIP embeddings into a 256-d residual.
  - In forward(), the residual is *added* (via a zero-initialised gate)
    to the content_query before it is combined with region features.

  Two call modes:
    LVIS branch  : fg_vocab_emb=None  -> looks up lvis_vocab_emb[content_inds]
    FG-OVD branch: fg_vocab_emb=[N,V,768], fg_vocab_mask=[N,V]
                   are passed from DINOAttr.forward()
"""

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


# ---------------------------------------------------------------------------
# Encoder / Decoder  (unchanged from dino_transformer.py)
# ---------------------------------------------------------------------------

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
        reference_points=None,
        valid_ratios=None,
        **kwargs,
    ):
        output = query
        bs, num_queries, _ = output.size()
        if reference_points.dim() == 2:
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)

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
                    new_reference_points[..., :2] = (
                        tmp[..., :2] + inverse_sigmoid(reference_points)
                    )
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


# ---------------------------------------------------------------------------
# Attribute_Attention
# ---------------------------------------------------------------------------

class Attribute_Attention(nn.Module):
    """Multi-expert cross-attention that aggregates per-category atomic-phrase
    CLIP embeddings (768-d) into a single head_query_dim-d residual.

    Design:
        Q : expert_query  [expert_cnt, head_query_dim]          (learnable)
        K : k_proj(vocab - vocab[:,0:1])  (attribute-delta encoding)
        V : v_proj(vocab)

    The delta encoding makes Keys focus on how each phrase *differs* from the
    whole-object description (index 0), mirroring the idea from gen_MG_vocab3.py
    where the first vocab entry is the full sentence and subsequent entries are
    single-attribute atomic phrases.
    """

    def __init__(
        self,
        head_query_dim: int = 256,
        Vocab_embed_dim: int = 768,
        expert_cnt: int = 6,
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

        self.out_norm = nn.LayerNorm(head_query_dim)

    def forward(self, vocab_embed, vocab_embed_mask=None):
        """
        Args:
            vocab_embed      : [cate_cnt, vocab_cnt, Vocab_embed_dim]
            vocab_embed_mask : [cate_cnt, vocab_cnt]  1/True=valid, 0/False=pad
        Returns:
            attn_out : [cate_cnt, head_query_dim]
        """
        cate_cnt, vocab_cnt, _ = vocab_embed.size()

        # attribute-delta: subtract the full-sentence embedding (index 0)
        # BUG-FIX 1: 当某个类别的所有vocab都是padding(全0)时，
        # vocab_embed[:,0:1]也是0向量，delta后仍全0，不会引入NaN。
        # 但如果vocab_embed本身含NaN(极少数情况)，在这里用nan_to_num兜底。
        # vocab_embed = torch.nan_to_num(vocab_embed, nan=0.0)
        mix_vocab = vocab_embed - vocab_embed[:, 0:1]          # [C, V, 768]

        k = self.k_proj(mix_vocab)                             # [C, V, D]
        v = self.v_proj(vocab_embed)                           # [C, V, D]

        # attention scores: [expert_cnt, C*V]
        k_flat = k.reshape(cate_cnt * vocab_cnt, self.head_query_dim)
        attn_score = torch.matmul(
            self.expert_query, k_flat.transpose(0, 1)
        ) / (self.head_query_dim ** 0.5)                       # [E, C*V]
        attn_score = attn_score.reshape(self.expert_cnt, cate_cnt, vocab_cnt)  # [E, C, V]

        # BUG-FIX 2: 当某行的valid slot数为0时，softmax(-inf,...,-inf)会产生NaN。
        # 解决方案：先确保每行至少有一个有效位置（强制index 0有效），
        # 再用masked_fill，这样softmax永远不会遇到全-inf的行。
        if vocab_embed_mask is not None:
            # 强制第0个位置有效，防止全-inf导致softmax产生NaN
            vocab_embed_mask = vocab_embed_mask.clone()
            vocab_embed_mask[:, 0] = 1.0
            invalid = (vocab_embed_mask.unsqueeze(0).expand(
                self.expert_cnt, -1, -1) == 0)
            attn_score = attn_score.masked_fill(invalid, float('-inf'))

        attn_score = attn_score.softmax(dim=-1)                # [E, C, V]

        # # BUG-FIX 3: softmax后可能仍残留NaN（极端情况），再做一次nan_to_num
        # attn_score = torch.nan_to_num(attn_score, nan=0.0)

        # weighted sum of values
        attn_out = torch.matmul(
            attn_score.unsqueeze(-2),                          # [E, C, 1, V]
            v.unsqueeze(0).expand(self.expert_cnt, -1, -1, -1) # [E, C, V, D]
        ).squeeze(-2)                                          # [E, C, D]

        attn_out = attn_out.sum(dim=0)                         # [C, D]
        attn_out = self.out_norm(attn_out)
        return attn_out


# ---------------------------------------------------------------------------
# DINOTransformerAttr
# ---------------------------------------------------------------------------

class DINOTransformerAttr(nn.Module):
    """DINO Transformer with Attribute_Attention for content-query enrichment.

    Additional constructor args:
        lvis_vocab_emb_path : str
            Path to .pt dict produced by gen_MG_vocab3.py.
            Format: {category_id (int): Tensor[vocab_cnt, 768]}
        attr_expert_cnt     : int  number of expert queries
        attr_head_query_dim : int  = embed_dim (256)
        attr_vocab_embed_dim: int  = CLIP dim (768)
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        num_feature_levels=4,
        two_stage_num_proposals=900,
        lvis_vocab_emb_path: str = 'dataset/metadata/lvis_visual_vocabs_convnext_large_d_320.pt',
        attr_expert_cnt: int = 6,
        attr_head_query_dim: int = 256,
        attr_vocab_embed_dim: int = 768,
    ):
        super(DINOTransformerAttr, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dim = self.encoder.embed_dim

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dim)
        )
        self.enc_output = nn.Linear(self.embed_dim, self.embed_dim)
        self.enc_output_norm = nn.LayerNorm(self.embed_dim)

        # ---- Attribute_Attention ----
        self.attr_attention = Attribute_Attention(
            head_query_dim=attr_head_query_dim,
            Vocab_embed_dim=attr_vocab_embed_dim,
            expert_cnt=attr_expert_cnt,
        )
        # Zero-initialised gate so training starts as the original model.
        # Learnt residual = gate(attr_out) is added to content_query.
        self.attr_gate = nn.Linear(attr_head_query_dim, attr_head_query_dim, bias=True)
        nn.init.zeros_(self.attr_gate.weight)
        nn.init.zeros_(self.attr_gate.bias)

        # ---- Load & pad LVIS vocab embeddings ----
        self._load_lvis_vocab_emb(lvis_vocab_emb_path, attr_vocab_embed_dim)

        self.init_weights()

    # ------------------------------------------------------------------
    def _load_lvis_vocab_emb(self, path, vocab_embed_dim):
        """Load and pad LVIS atomic-phrase CLIP embeddings.

        Stores:
            lvis_vocab_emb      [num_categories, max_vocab_cnt, vocab_embed_dim]
            lvis_vocab_emb_mask [num_categories, max_vocab_cnt]  bool
        as non-parameter buffers (automatically moved with .to(device)).
        """
        category_vocab_emb = torch.load(path, map_location='cpu')

        num_categories = max(category_vocab_emb.keys()) + 1
        max_vocab_cnt = max(v.size(0) for v in category_vocab_emb.values())

        lvis_vocab_emb = torch.zeros(
            num_categories, max_vocab_cnt, vocab_embed_dim, dtype=torch.float32
        )
        lvis_vocab_emb_mask = torch.zeros(
            num_categories, max_vocab_cnt, dtype=torch.bool
        )
        for idx, vocab_emb in category_vocab_emb.items():
            vi = vocab_emb.size(0)
            lvis_vocab_emb[idx, :vi] = vocab_emb.float().cpu()
            lvis_vocab_emb_mask[idx, :vi] = True

        self.register_buffer('lvis_vocab_emb', lvis_vocab_emb)
        self.register_buffer('lvis_vocab_emb_mask', lvis_vocab_emb_mask)

    # ------------------------------------------------------------------
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.normal_(self.level_embeds)
        # Reset gate to zero after xavier_uniform_ above
        nn.init.zeros_(self.attr_gate.weight)
        nn.init.zeros_(self.attr_gate.bias)

    # ------------------------------------------------------------------
    def _enrich_content_query(
        self,
        content_query_embeds,
        content_inds=None,
        fg_vocab_emb=None,
        fg_vocab_mask=None,
    ):
        """Add Attribute_Attention residual to content_query_embeds.

        Args:
            content_query_embeds : [num_cats, 256]  (projected & normalised)
            content_inds         : LongTensor[num_cats] LVIS category ids (LVIS branch)
            fg_vocab_emb         : [num_cats, V, 768]  (FG-OVD branch, or None)
            fg_vocab_mask        : [num_cats, V]        (FG-OVD branch, or None)

        Returns:
            enriched : [num_cats, 256]
        """
        num_cats = content_query_embeds.size(0)

        if fg_vocab_emb is not None:
            # FG-OVD 训练路径：per-entry 原子短语 embeddings 在运行时提供
            vocab_emb  = fg_vocab_emb    # [N, V, 768]
            vocab_mask = fg_vocab_mask   # [N, V]

        elif hasattr(self, '_infer_vocab_emb') and self._infer_vocab_emb is not None:
            # 推理路径（FG_OVD_FLAG='modify'）：
            # FG_inf.py 在调用 forward 前把推理时的 vocab 挂载到
            # transformer._infer_vocab_emb / _infer_vocab_mask。
            # 用完后立即清除，避免下次推理复用错误数据。
            vocab_emb  = self._infer_vocab_emb
            vocab_mask = self._infer_vocab_mask
            self._infer_vocab_emb  = None
            self._infer_vocab_mask = None

        else:
            # LVIS 训练路径：查预加载的 per-category vocab embeddings
            if content_inds is not None:
                vocab_emb  = self.lvis_vocab_emb[content_inds]
                vocab_mask = self.lvis_vocab_emb_mask[content_inds]
            else:
                vocab_emb  = self.lvis_vocab_emb
                vocab_mask = self.lvis_vocab_emb_mask

        # 尺寸守卫：vocab 行数必须与 content_query 行数一致，否则跳过
        # if vocab_emb.size(0) == 0
        if vocab_emb.size(0) == 0 or vocab_emb.size(0) != num_cats:
            return content_query_embeds

        attr_residual = self.attr_attention(vocab_emb, vocab_mask)   # [C, 256]
        enriched = content_query_embeds + self.attr_gate(attr_residual)
        return enriched

    # ------------------------------------------------------------------
    # Standard DINO geometry helpers (unchanged)
    # ------------------------------------------------------------------

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N, S, C = memory.shape
        proposals, _cur = [], 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W - 1, W, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposals.append(torch.cat((grid, wh), -1).view(N, -1, 4))
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)
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
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_W.float() / W, valid_H.float() / H], -1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        multi_level_feats,
        multi_level_masks,
        multi_level_pos_embeds,
        query_embed,
        attn_masks,
        content_query_embeds,          # [num_cats, 256]
        content_inds=None,             # LongTensor[num_cats] or None  (LVIS branch)
        fg_vocab_emb=None,             # [N, V, 768]  (FG-OVD branch)
        fg_vocab_mask=None,            # [N, V]        (FG-OVD branch)
        **kwargs,
    ):
        """
        Extra args compared to DINOTransformer.forward():
            fg_vocab_emb  : atomic-phrase CLIP embeddings for FG-OVD.
                            Shape [N, V, 768] where N = batch*(1+fg_n_neg).
                            Pass None for LVIS data (uses pre-loaded lvis_vocab_emb).
            fg_vocab_mask : validity mask for fg_vocab_emb  [N, V].
        """
        # ---- 1. Flatten multi-scale image features ----
        feat_flatten, mask_flatten, lvl_pos_embed_flatten, spatial_shapes = [], [], [], []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(multi_level_feats, multi_level_masks, multi_level_pos_embeds)
        ):
            bs, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat       = feat.flatten(2).transpose(1, 2)
            mask       = mask.flatten(1)
            pos_embed  = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed_flatten.append(pos_embed + self.level_embeds[lvl].view(1, 1, -1))
            feat_flatten.append(feat)
            mask_flatten.append(mask)

        feat_flatten          = torch.cat(feat_flatten, 1)
        mask_flatten          = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in multi_level_masks], 1)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat_flatten.device
        )

        # ---- 2. Encoder ----
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs,
        )

        # ---- 3. Two-stage proposal generation ----
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )

        enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](
            output_memory, content_inds=content_inds
        )
        enc_outputs_coord_unact = (
            self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
        )

        max_scores, max_labels = torch.max(enc_outputs_class, dim=-1)
        topk = self.two_stage_num_proposals
        topk_proposals = torch.topk(max_scores, topk, dim=1)[1]

        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        )
        reference_points = topk_coords_unact.detach().sigmoid()
        if query_embed[1] is not None:
            reference_points = torch.cat([query_embed[1].sigmoid(), reference_points], 1)
        init_reference_out = reference_points

        target_unact = torch.gather(
            output_memory, 1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, output_memory.shape[-1])
        )
        # content_ids: predicted category index for each top-k proposal
        content_ids = torch.gather(max_labels, 1, topk_proposals)   # [bs, topk]

        # ---- 4. Enrich content queries with Attribute_Attention ----
        # enriched_cq: [num_cats, 256]
        enriched_cq = self._enrich_content_query(
            content_query_embeds,
            content_inds=content_inds,
            fg_vocab_emb=fg_vocab_emb,
            fg_vocab_mask=fg_vocab_mask,
        )

        # ---- 5. Assemble decoder input (mirrors original dino_transformer.py) ----
        # Gather enriched content query for each two-stage proposal
        content_query = torch.gather(
            enriched_cq.unsqueeze(0).repeat(bs, 1, 1), 1,
            content_ids.unsqueeze(-1).repeat(1, 1, self.embed_dim)
        )                                                            # [bs, topk, 256]

        # target = region_feature + enriched_content_query
        target = target_unact.detach() + content_query

        # Prepend DN queries (already have content embedded by prepare_for_cdn)
        if query_embed[0] is not None:
            target = torch.cat([query_embed[0], target], 1)

        # ---- 6. Decoder ----
        inter_states, inter_references = self.decoder(
            query=target,
            key=memory,
            value=memory,
            query_pos=None,
            key_padding_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            attn_masks=attn_masks,
            **kwargs,
        )

        return (
            inter_states,
            init_reference_out,
            inter_references,
            target_unact,
            topk_coords_unact.sigmoid(),
        )
