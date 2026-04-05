#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os
import sys
import time
import json
import numpy as np
import torch
import torch.distributed
from torch.nn.parallel import DataParallel, DistributedDataParallel

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
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
import datetime
from detectron2.evaluation.evaluator import DatasetEvaluators, inference_context, log_every_n_seconds
from torch import nn
from lvis import LVIS
# import AEsir_utils.data_utils.data_visual as vis

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from torchvision.ops import batched_nms
from detectron2.data import DatasetCatalog, MetadataCatalog
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
try:
    from torchmetrics.detection import MeanAveragePrecision
except ImportError:
    from torchmetrics.detection import MAP
    MeanAveragePrecision = MAP
import itertools

import warnings
warnings.filterwarnings(
    "ignore", 
    message=".*Converting mask without torch.bool dtype to bool.*", 
    category=UserWarning
)
logger = logging.getLogger("main")

def get_model_parameter_count(model):
    """
    计算模型的总参数量、可训练参数量和可训练参数比例。

    Args:
        model (torch.nn.Module): PyTorch 模型。

    Returns:
        tuple: 包含总参数量 (total_params)、可训练参数量 (trainable_params) 和可训练参数比例 (trainable_ratio) 的元组。
    """
    total_params = sum(p.numel() for name,p in model.named_parameters() if not "clip_model" in name)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params if total_params != 0 else 0
    return total_params, trainable_params, trainable_ratio

def apply_NMS(preds, iou=0.5):
    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']
    
    indexes_to_keep = batched_nms(boxes, 
                                  scores, 
                                  torch.IntTensor([0] * len(boxes)),
                                  iou)
    
    filtered_boxes = []
    filtered_scores = []
    filtered_labels = []
    
    for x in range(len(boxes)):
        if x in indexes_to_keep:
            filtered_boxes.append(boxes[x])
            filtered_scores.append(scores[x])
            filtered_labels.append(labels[x])
    
    preds['boxes'] = torch.stack(filtered_boxes, dim=0)
    preds['scores'] = torch.stack(filtered_scores, dim=0)
    preds['labels'] = torch.stack(filtered_labels, dim=0)
    return preds

def transform_predslist_to_dict(preds):
    result = {}
    for pred in preds:
        image = pred['image_filepath']
        if image not in result:
            result[image] = []
        result[image].append(pred)
    return result  

def get_image_preds(preds, DEVICE='cpu'):
    labels = []
    scores = []
    boxes = []
    for pred in preds:
        labels += [x for x in pred['labels']]
        scores += [x for x in pred['scores']]
        boxes += ([x for x in pred['boxes']])
        assert_box(boxes)

    boxes = boxes if boxes != [] else [[0,0,0,0]]
    if type(boxes[0]) != torch.Tensor:
        return {
            'boxes': torch.Tensor(boxes).to(DEVICE),
            'labels': torch.IntTensor(labels).to(DEVICE),
            'scores': torch.Tensor(scores).to(DEVICE)
        }
    else:
        return {
            'boxes': torch.stack(boxes, dim=0).to(DEVICE),
            'labels': torch.IntTensor(labels).to(DEVICE),
            'scores': torch.Tensor(scores).to(DEVICE)
        }

def assert_box(boxes):
    """Check that the box is in [xmin, ymin, xmax, ymax] format"""
    for box in boxes:
        assert box[0] <= box[2] and box[1] <= box[3]

def convert_format(boxes):
    for box in boxes:
        box[2] += box[0]
        box[3] += box[1]
    return boxes

def get_image_ground_truth(data, image_id):
    """
    Given a dictionary 'data' and an 'image_id', returns a dictionary with 'boxes' and 'categories' information for
    that image.

    Args:
        data (dict): The data dictionary containing 'annotations'.
        image_id (int): The image_id for which to retrieve data.

    Returns:
        dict: A dictionary with 'boxes' and 'categories' information for the given image_id.
    """
    image_data = {'boxes': [], 'labels': []}  # Initialize the dictionary to store image data

    # Loop through each annotation in the 'annotations' list
    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            # If the 'image_id' in the annotation matches the given 'image_id', append bbox and category_id to the lists
            image_data['boxes'].append(annotation['bbox'])
            image_data['labels'].append(annotation['category_id'])

    image_data['boxes'] = convert_format(image_data['boxes'])
    assert_box(image_data['boxes'])
    # tensorize elements
    image_data['boxes'] = torch.Tensor(image_data['boxes']).cpu()
    image_data['labels'] = torch.IntTensor(image_data['labels']).cpu()
    
    return image_data


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
            self.grad_scaler = grad_scaler

        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[Trainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If you want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        with autocast(enabled=self.amp):
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if self.amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

def inference_on_FG_OVD_dataset(
    model,
    dataloader,
    fg_name,
    MAX_PREDICTIONS=100,
    n_neg=5,
    fg_meta=None
):  
    def adjust_out_id(output, vocabulary_id):
        for i in range(len(output['labels'])):
            output['labels'][i] = vocabulary_id[output['labels'][i]]
        return output

    logger.info(f"inference on FG {fg_name} dataset...")

    out_lst = []
    contiguous_id_to_thing_dataset_id = fg_meta.contiguous_id_to_thing_dataset_id
    with ExitStack() as stack:
        if isinstance(model, torch.nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        # 仅主进程展示进度条
        iterator = dataloader
        if comm.is_main_process():
            iterator = tqdm(dataloader)

        for batch in iterator:
            gt_instance = batch[0]["instances"]
            category_id = gt_instance.all_class_id[0]
            category_id = [contiguous_id_to_thing_dataset_id[idx] for idx in category_id]

            file_name = batch[0]["file_name"]
            file_name = '/'.join(file_name.split('/')[-2:])
            out = model(batch, n_neg)[0]["instances"].to("cpu")

            final_out = {
                'scores': out.scores[:MAX_PREDICTIONS].cpu().detach().numpy(),
                'labels': out.pred_classes[:MAX_PREDICTIONS].cpu().detach().numpy(),
                'boxes': out.pred_boxes.tensor[:MAX_PREDICTIONS].cpu().detach().numpy(),
                'category_id': category_id[0],
                'vocabulary':  category_id[:n_neg+1],
                'image_filepath': file_name
            }
            final_out = adjust_out_id(final_out, category_id[:n_neg+1])
            out_lst.append(final_out)

    # —— 新增：分布式或单卡通用聚合逻辑 —— #
    comm.synchronize()  # 等待所有进程完成推理
    all_out = comm.gather(out_lst, dst=0)  # 收集到主进程

    if not comm.is_main_process():
        # 非主进程不做 eval，直接返回
        return None

    # 主进程：flatten 结果并计算 mAP
    all_out = list(itertools.chain(*all_out)) if all_out is not None else []
    eval_FG_1attribute_map(all_out, n_neg=n_neg, fg_name=fg_name)
    return None


def eval_FG_1attribute_map(preds_list, n_neg=5, fg_name=None):
    assert isinstance(fg_name,str)
    dataset_path = f"/apdcephfs_cq12/share_1150325/jiaminghli/data/FG_OVD/benchmarks/{fg_name}_with_subject.json"
    
    test_set = json.load(open(dataset_path))
    
    # Initialize metric
    metric = MeanAveragePrecision(dist_sync_on_step=False,sync_on_compute=False).cpu()
    test_set['annotations'] = [ann for ann in test_set['annotations'] if len(ann['neg_category_ids']) >= n_neg]
    preds_per_image = transform_predslist_to_dict(preds_list)
    
    targets = []
    preds = []
    
    n_images = 0
    # for imm in tqdm(test_set['images']):
    for imm in test_set['images']:
        target = get_image_ground_truth(test_set, imm['id'])
        # skipping image if empty after eliminating since the number of negatives was low
        if len(target['labels']) == 0:
            continue
        
        if imm['file_name'] in preds_per_image:
            # in case the ground truth for the image includes captions not processed by the detector, we remove them
            # relevant_cats = [predictions['category_id'] for predictions in preds_per_image[imm['file_name']]]
            relevant_cats = []
            predictions = preds_per_image[imm['file_name']]
            for prediction in predictions:
                cate_id = prediction['category_id']
                relevant_cats.append(cate_id)
            mask = torch.isin(target['labels'], torch.tensor(relevant_cats))
            target['labels'] = target['labels'][mask]
            target['boxes'] = target['boxes'][mask]
            preds_per_cat = [get_image_preds([pred_per_cat]) for pred_per_cat in preds_per_image[imm['file_name']]]
            preds_per_cat = [apply_NMS(pred_per_cat) for pred_per_cat in preds_per_cat]
            pred = {
                'boxes': torch.cat([x['boxes'] for x in preds_per_cat], dim=0),
                'labels': torch.cat([x['labels'] for x in preds_per_cat], dim=0),
                'scores': torch.cat([x['scores'] for x in preds_per_cat], dim=0),
            }
        else:
            continue
        n_images += 1
        targets.append(target)
        preds.append(pred)
        
        
    # Update metric with predictions and respective ground truth
    metric.update(preds, targets)
    
    # getting time of execution of the mAP
    print("Starting mAP computation")
    start_time = time.time()
    # Compute the results
    result = metric.compute()
    # print("--- %s seconds ---" % (time.time() - start_time))
    result['n_images'] = n_images
    # de-tensorize the results:
    result = {
        'map': float(result['map']),
        'map_50': float(result['map_50']),
        'map_75': float(result['map_75']),
        'map_small': float(result['map_small']),
        'map_medium': float(result['map_medium']),
        'map_large': float(result['map_large']),
        'mar_1': float(result['mar_1']),
        'mar_10': float(result['mar_10']),
        'mar_100': float(result['mar_100']),
        'mar_small': float(result['mar_small']),
        'mar_medium': float(result['mar_medium']),
        'mar_large': float(result['mar_large']),
        'map_per_class': float(result['map_per_class']),
        'mar_100_per_class': float(result['mar_100_per_class']),
        'n_images': int(result['n_images'])  
    }
    
    # print(result)
    logger = logging.getLogger("detectron2")
    logger.info(f"{fg_name}, n_neg={n_neg} mAP: {result['map']}")

def do_test(cfg, model):
    # loader = instantiate(cfg.dataloader.train)
    
    # for ip in loader:
    #     vis.print_json_structure(ip)
    #     break
    # return
    
    if "evaluator" in cfg.dataloader:

        fg_meta = MetadataCatalog.get("FG_OVD_bench_1_attr_with_subject")
        inference_on_FG_OVD_dataset(
            model = model,
            dataloader=instantiate(cfg.dataloader.test_FG_OVD_bench_1_attr_with_subject),
            fg_name="1_attributes",
            MAX_PREDICTIONS=100,
            n_neg=5,
            fg_meta=fg_meta
        )
        comm.synchronize()
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator), cfg.DDEBUG
        )
        print_csv_format(ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    logger = logging.getLogger("detectron2")
    # lora
    model = instantiate(cfg.model)
    if args.use_lora:
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        logger.info("Using LoRA to fine-tune")
        model = get_peft_model(
            model=model,
            peft_config=instantiate(cfg.peft_config),
        )

    logger.info("Model:\n{}".format(model))
    total_params, trainable_params, trainable_ratio = get_model_parameter_count(model)
    logger.info(f"Total params: {total_params}")
    logger.info(f"Trainable params: {trainable_params}")
    logger.info(f"Trainable ratio: {trainable_ratio}")
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)
    # train_loader = instantiate(cfg.dataloader.train_fg)

    model = create_ddp_model(model, **cfg.train.ddp)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    torch.autograd.set_detect_anomaly(True)
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)

    if args.ddebug:
        cfg.train.max_iter = 8
        cfg.train.eval_period = 8
        cfg.train.log_period = 4
        cfg.train.checkpointer.period = 8
        cfg.dataloader.train.num_workers = 0
        cfg.dataloader.train.total_batch_size = 3
        cfg.train.output_dir = 'FG_OVD_TRAIN/output/debug'
        cfg.dataloader.evaluator.output_dir = 'FG_OVD_TRAIN/output/debug'
        if cfg.model.save_dir:
            cfg.model.save_dir = cfg.model.save_dir + '_debug'
        cfg.DDEBUG = True
    else:
        cfg.DDEBUG = False
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


if __name__ == "__main__":
    args = default_argument_parser()
    args.add_argument("--use-lora", action="store_true")
    args = args.parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

