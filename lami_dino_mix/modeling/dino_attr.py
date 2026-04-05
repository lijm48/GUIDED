# coding=utf-8
"""
DINOAttr: DINO with Attribute_Attention enabled.

Inherits everything from lami_dino_mix.modeling.dino.DINO and
overrides only the transformer call so that:

  - LVIS branch  : the transformer looks up lvis_vocab_emb[content_inds]
                   internally and enriches content queries via Attribute_Attention.
  - FG-OVD branch: the per-batch atomic-phrase embeddings built in
                   clip_process_batch_inputs() (vlm_all_query_embedding,
                   vlm_all_query_mask) are forwarded to the transformer so
                   Attribute_Attention can work on them too.

The underlying DINOTransformerAttr handles both modes via its
fg_vocab_emb / fg_vocab_mask kwargs.
"""

from .dino import DINO


class DINOAttr(DINO):
    """DINO + Attribute_Attention for content query enrichment.

    No new constructor arguments are needed beyond those already accepted by DINO.
    The transformer must be a DINOTransformerAttr instance (configured in the
    corresponding model config file).
    """

    def forward(self, batched_inputs, fg_n_neg=None, use_multi=True):
        """Identical to DINO.forward() except that the self.transformer(...)
        call is augmented with fg_vocab_emb / fg_vocab_mask when processing
        FG-OVD data so that Attribute_Attention inside DINOTransformerAttr can
        use the per-batch atomic-phrase CLIP embeddings.
        """
        import torch
        import torch.nn.functional as F
        import copy
        import os
        from detrex.utils import inverse_sigmoid
        from detectron2.modeling import detector_postprocess

        # ----------------------------------------------------------------
        # The block below is identical to DINO.forward() up until the
        # self.transformer(...) call.  We duplicate it here so we can
        # inject the two new kwargs without monkey-patching the base class.
        # ----------------------------------------------------------------

        if self.training:
            if batched_inputs[0].get("FG_OVD_FLAG", None) == 'fgovd':
                batched_inputs = self.clip_process_batch_inputs(batched_inputs, fg_n_neg)
                self.use_fed_loss = False
            else:
                self.CG_subject_name_lst = None
                self.set_lvis_classifier_emb()
                self.use_fed_loss = True
        else:
            if batched_inputs[0].get("FG_OVD_FLAG", None) == 'fgovd':
                batched_inputs = self.clip_process_batch_inputs(batched_inputs, fg_n_neg, debug=True)
            elif batched_inputs[0].get("FG_OVD_FLAG", None) == 'modify':
                pass
            else:
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

        if self.score_ensemble:
            features, features_wonorm = self.backbone(images.tensor)
        else:
            features = self.backbone(images.tensor)

        multi_level_feats = self.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(
                self.position_embedding(multi_level_masks[-1])
            )

        if self.training:
            content_query_embeds = (
                self.content_query_embedding[content_inds]
                if content_inds is not None
                else self.content_query_embedding
            )
            content_query_embeds = self.content_layer(content_query_embeds)
            content_query_embeds = F.normalize(content_query_embeds, p=2, dim=1)
        else:
            content_query_embeds = self.content_layer(self.eval_content_query_embedding)
            content_query_embeds = F.normalize(content_query_embeds, p=2, dim=1)

        if self.training:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if batched_inputs[0].get("FG_OVD_FLAG", None):
                cdn_num_classes = self.CG_class_cnt
            else:
                cdn_num_classes = self.fed_loss_num_cat
            input_query_label, input_query_bbox, attn_mask, dn_meta = self.prepare_for_cdn(
                targets,
                dn_number=self.dn_number,
                label_noise_ratio=self.label_noise_ratio,
                box_noise_scale=self.box_noise_scale,
                num_queries=self.num_queries,
                num_classes=cdn_num_classes,
                hidden_dim=self.embed_dim,
                content_query_embeds=content_query_embeds,
            )
        else:
            input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # ----------------------------------------------------------------
        # Determine Attribute_Attention vocab inputs
        #   FG-OVD : use atomic-phrase embeddings built in clip_process_batch_inputs
        #   LVIS   : pass None -> transformer uses pre-loaded lvis_vocab_emb
        # ----------------------------------------------------------------
        is_fgovd = batched_inputs[0].get("FG_OVD_FLAG", None)
        if is_fgovd:
            fg_vocab_emb  = getattr(self, 'vlm_all_query_embedding', None)
            fg_vocab_mask = getattr(self, 'vlm_all_query_mask', None)
        else:
            fg_vocab_emb  = None
            fg_vocab_mask = None

        # ----------------------------------------------------------------
        # Feed into DINOTransformerAttr (note the two new kwargs)
        # ----------------------------------------------------------------
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,
        ) = self.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
            content_query_embeds=content_query_embeds,
            content_inds=content_inds,
            fg_vocab_emb=fg_vocab_emb,
            fg_vocab_mask=fg_vocab_mask,
        )

        # ----------------------------------------------------------------
        # The rest is identical to DINO.forward()
        # ----------------------------------------------------------------
        inter_states[0] += self.content_layer.weight[0, 0] * 0.0

        outputs_classes, outputs_coords = [], []
        for lvl in range(inter_states.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](inter_states[lvl], content_inds=content_inds)
            tmp = self.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coords.append(tmp.sigmoid())
            outputs_classes.append(outputs_class)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        if dn_meta is not None:
            outputs_class, outputs_coord = self.dn_post_process(
                outputs_class, outputs_coord, dn_meta
            )

        output = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        if self.aux_loss:
            output["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)

        interm_coord  = enc_reference
        interm_class  = self.transformer.decoder.class_embed[-1](enc_state, content_inds=content_inds)
        output["enc_outputs"] = {"pred_logits": interm_class, "pred_boxes": interm_coord}

        if self.training:
            if is_fgovd:
                box_cls  = output["pred_logits"].clone()
                box_pred = output["pred_boxes"].clone()
                roi_features_ori = []
                for box in box_pred:
                    roi_features_ori.append(
                        self.extract_region_feature(features_wonorm, box.unsqueeze(0), 'p3')
                    )
                roi_features_ori = torch.cat(roi_features_ori, dim=0)
                cls_score = box_cls.sigmoid()

                vlm_content_query_embedding = self.vlm_content_query_embedding


                vlm_logit = (
                    roi_features_ori
                    @ vlm_content_query_embedding.t()
                    * self.vlm_temperature
                )
                vlm_score = F.softmax(vlm_logit, dim=-1)

                new_cls_score = cls_score ** (1 - self.beta) * vlm_score ** self.beta
                output["pred_logits"] = inverse_sigmoid(new_cls_score)

            loss_dict = self.criterion(output, targets, dn_meta)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]
            return loss_dict

        else:
            box_cls  = output["pred_logits"]
            box_pred = output["pred_boxes"]
            if self.save_dir and not self.score_ensemble:
                save_output = {
                    "pred_logits": copy.deepcopy(output["pred_logits"]).cpu(),
                    "pred_boxes":  copy.deepcopy(output["pred_boxes"]).cpu(),
                }
                torch.save(save_output, os.path.join(self.save_dir, filename))

            if self.score_ensemble:
                roi_features_ori = self.extract_region_feature(features_wonorm, box_pred, 'p3')
                cls_score = box_cls.sigmoid()

                vlm_score = (
                    roi_features_ori
                    @ self.vlm_content_query_embedding.t()
                    * self.vlm_temperature
                ).softmax(dim=-1)

                if is_fgovd:
                    cls_score = cls_score ** (1 - self.beta) * vlm_score ** self.beta
                else:
                    cls_score[:, :, self.base_idx] = (
                        cls_score[:, :, self.base_idx] ** (1 - self.alpha)
                        * vlm_score[:, :, self.base_idx] ** self.alpha
                    )
                    cls_score[:, :, self.novel_idx] = (
                        cls_score[:, :, self.novel_idx] ** (1 - self.beta)
                        * vlm_score[:, :, self.novel_idx] ** self.beta
                    )
                    cls_score[:, :, self.novel_idx] *= self.novel_scale
                box_cls = cls_score
                results = self.inference(box_cls, box_pred, images.image_sizes, wo_sigmoid=True)
            else:
                results = self.inference(box_cls, box_pred, images.image_sizes)

            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width  = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
