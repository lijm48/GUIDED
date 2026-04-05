from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation import LVISEvaluator

from detrex.data import DetrDatasetMapper
from .data_utils import (
    CombinedDataLoader,
    MG_DatasetMapper
)

dataloader = OmegaConf.create()

dataloader.train_lvis = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="lvis_v1_train_norare"),
    sampler="RepeatFactorTrainingSampler",
    repeat_threshold=0.001,
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.train_fg = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="FG_OVD_train_1_attr_with_subject"),
    sampler="RepeatFactorTrainingSampler",
    repeat_threshold=0.001,
    mapper=L(MG_DatasetMapper)(
        augmentation=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(),
            L(T.ResizeShortestEdge)(
                short_edge_length=(400, 500, 600),
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute_range",
                crop_size=(384, 600),
            ),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice",
            ),
        ],
        is_train=True,
        mask_on=False,
        img_format="RGB",
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.train = L(CombinedDataLoader)(
    loader_1 = L(build_detection_train_loader)(
        dataset=L(get_detection_dataset_dicts)(names="lvis_v1_train_norare"),
        sampler="RepeatFactorTrainingSampler",
        repeat_threshold=0.001,
        mapper=L(DetrDatasetMapper)(
            augmentation=[
                L(T.RandomFlip)(),
                L(T.ResizeShortestEdge)(
                    short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                    max_size=1333,
                    sample_style="choice",
                ),
            ],
            augmentation_with_crop=[
                L(T.RandomFlip)(),
                L(T.ResizeShortestEdge)(
                    short_edge_length=(400, 500, 600),
                    sample_style="choice",
                ),
                L(T.RandomCrop)(
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                ),
                L(T.ResizeShortestEdge)(
                    short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                    max_size=1333,
                    sample_style="choice",
                ),
            ],
            is_train=True,
            mask_on=False,
            img_format="RGB",
        ),
        total_batch_size="${..total_batch_size}",
        num_workers="${..num_workers}",
    ),
    loader_2 = L(build_detection_train_loader)(
        dataset=L(get_detection_dataset_dicts)(names="FG_OVD_train_1_attr_with_subject_multi_vocab"),
        sampler="RepeatFactorTrainingSampler",
        repeat_threshold=0.001,
        mapper=L(MG_DatasetMapper)(
            augmentation=[
                L(T.RandomFlip)(),
                L(T.ResizeShortestEdge)(
                    short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                    max_size=1333,
                    sample_style="choice",
                ),
            ],
            augmentation_with_crop=[
                L(T.RandomFlip)(),
                L(T.ResizeShortestEdge)(
                    short_edge_length=(400, 500, 600),
                    sample_style="choice",
                ),
                L(T.RandomCrop)(
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                ),
                L(T.ResizeShortestEdge)(
                    short_edge_length=(480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800),
                    max_size=1333,
                    sample_style="choice",
                ),
            ],
            is_train=True,
            mask_on=False,
            img_format="RGB",
        ),
        total_batch_size="${..total_batch_size}",
        num_workers="${..num_workers}",
    ),
    ratio=(1, 1),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="lvis_v1_val", filter_empty=False),
    mapper=L(DetrDatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.test_FG_OVD_bench_1_attr_with_subject = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="FG_OVD_bench_1_attr_with_subject_multi_vocab"),
    mapper=L(MG_DatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.test_FG_OVD_bench_transparency_with_subject = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="FG_OVD_bench_transparency_with_subject"),
    mapper=L(MG_DatasetMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=800,
                max_size=1333,
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
        img_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(LVISEvaluator)(
    dataset_name="${..test.dataset.names}",
)