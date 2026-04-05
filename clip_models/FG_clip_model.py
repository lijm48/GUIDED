import clip
import torch
from torch import nn
from clip_models.FG_CLIP.src.model import CrossAttentionModule, MLPs
from clip_models.enc_text import getClip_model_preprocess_tokenizer

from typing import List

class Text_mlp(MLPs):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def proj_text_embeds(self, text_embeds: torch.Tensor):
        text_linear_layers = self.linear_layers[1]
        
        x = text_embeds
        for layer in text_linear_layers:
            x = layer(x)
            if self.act is not None:
                x = self.act(x)
        if self.rescaling is not None:
            x = self.rescaling.weight * x
        return x

class FG_CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_clip_model, _ = clip.load("ViT-B/16", device="cuda")
        
        self.text_mlp = Text_mlp(
            mlp_dims = [],
            act = None,
            keep_embeds = [True, False],
            # initial_weights = '/data1/liangzhijia/my_LaMI/Vis_test/FG_CLIP/checkpoints/triplet_linear-mlp_ve-freezed.pt',
            cosine = True
        )
        mlp_weight = '/apdcephfs_cq12/share_1150325/jiaminghli/code/OVD/my_LaMI/Vis_test/FG_CLIP/checkpoints/fg-ovd_linear-mlp_ve-freezed.pt'
        print(f"clip mlp loading: {mlp_weight}")
        self.text_mlp.load_state_dict(torch.load(mlp_weight))
        
        self.tokenizer = clip.tokenize
        
    def get_batch_text_embs(self, texts: List[str], use_mlp: bool = False) -> torch.Tensor:
        text_embeds = self.base_clip_model.encode_text(self.tokenizer(texts).to(next(self.base_clip_model.parameters()).device))
        if use_mlp:
            text_embeds = self.text_mlp.proj_text_embeds(text_embeds.to(next(self.base_clip_model.parameters()).dtype))
        return text_embeds.cpu().detach()
    
class FG_convext_clip(nn.Module):
    def __init__(self,
            base_clip_ckpt_file = "/apdcephfs_cq12/share_1150325/jiaminghli/code/OVD/pretrain_models/timm_clip_convnext_large_trans.pth",
            mlp_weight = "/apdcephfs_cq12/share_1150325/jiaminghli/code/OVD/my_LaMI-main/Vis_test/FG_CLIP/ckpt_convnext_large_d_320/fg-ovd_convnext-mlp.pth",
            multi_mlp = False,   
        ):
        super().__init__()
        self.base_clip_model, _, self.tokenizer = getClip_model_preprocess_tokenizer(
            ckpt_file=base_clip_ckpt_file
        )
        
        del self.base_clip_model.visual
        
        self.text_mlp = Text_mlp(
            mlp_dims = [],
            act = None,
            keep_embeds = [True, False],
            # initial_weights = '/data1/liangzhijia/my_LaMI/Vis_test/FG_CLIP/checkpoints/triplet_linear-mlp_ve-freezed.pt',
            cosine = True,
            embedding_dim = 768
        )
        self.text_mlp2 = None
        if multi_mlp:

            self.text_mlp2 = Text_mlp(
                mlp_dims = [],
                act = None,
                keep_embeds = [True, False],
                # initial_weights = '/data1/liangzhijia/my_LaMI/Vis_test/FG_CLIP/checkpoints/triplet_linear-mlp_ve-freezed.pt',
                cosine = True,
                embedding_dim = 768
            )        # print(f"clip mlp loading: {mlp_weight}")
        if mlp_weight:
            self.text_mlp.load_state_dict(torch.load(mlp_weight))
            if multi_mlp:
                self.text_mlp2.load_state_dict(torch.load(mlp_weight))
        
        
    def get_batch_text_embs(self, texts: List[str], use_mlp: bool = False, use_mlp2: bool = False) -> torch.Tensor:
        text_embeds = self.base_clip_model.encode_text(self.tokenizer(texts).to(next(self.base_clip_model.parameters()).device))
        if use_mlp:
            text_embeds = self.text_mlp.proj_text_embeds(text_embeds.to(next(self.base_clip_model.parameters()).dtype))
        elif use_mlp2:
            assert self.text_mlp2 is not None
            text_embeds = self.text_mlp2.proj_text_embeds(text_embeds.to(next(self.base_clip_model.parameters()).dtype))

        return text_embeds