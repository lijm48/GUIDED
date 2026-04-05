# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .builtin_meta import ADE20K_SEM_SEG_CATEGORIES, _get_builtin_metadata
from .cityscapes import load_cityscapes_instances, load_cityscapes_semantic
from .cityscapes_panoptic import register_all_cityscapes_panoptic
from .coco import load_sem_seg, register_coco_instances
from .coco_panoptic import register_coco_panoptic, register_coco_panoptic_separated
from .lvis import get_lvis_instances_meta, register_lvis_instances
from .pascal_voc import register_pascal_voc

from detectron2.structures import BoxMode
from detectron2.utils.file_io import PathManager
from fvcore.common.timer import Timer
import logging
logger = logging.getLogger(__name__)

# ==== Predefined datasets and splits for COCO ==========

_PREDEFINED_SPLITS_COCO = {}
_PREDEFINED_SPLITS_COCO["coco"] = {
    "coco_2014_train": ("coco/train2014", "coco/annotations/instances_train2014.json"),
    "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
    "coco_2014_minival": ("coco/val2014", "coco/annotations/instances_minival2014.json"),
    "coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/instances_valminusminival2014.json",
    ),
    "coco_2017_train": ("coco/train2017", "coco/annotations/instances_train2017.json"),
    "coco_2017_val": ("coco/val2017", "coco/annotations/instances_val2017.json"),
    "coco_2017_test": ("coco/test2017", "coco/annotations/image_info_test2017.json"),
    "coco_2017_test-dev": ("coco/test2017", "coco/annotations/image_info_test-dev2017.json"),
    "coco_2017_val_100": ("coco/val2017", "coco/annotations/instances_val2017_100.json"),
}

# _PREDEFINED_SPLITS_COCO["obj365v2"] = {
#     "obj365v2_train": ("object365/train/", "object365/annotations/obj365v2_train_filtered.json"),
#     "obj365v2_val": ("object365/val/", "object365/annotations/zhiyuan_objv2_val.json")
# }

_PREDEFINED_SPLITS_COCO["coco_zeroshot"] = {
    "zeroshot_coco_2017_train": ("coco/train2017", "coco/zero-shot/instances_train2017_seen_2_proposal.json"),
    # "zeroshot_coco_2017_train": ("coco/train2017", "coco/zero-shot/instances_train2017_seen.json"),
    # "zeroshot_coco_2017_train": ("coco/train2017", "coco/zero-shot/instances_train2017_all_2.json"),
    "zeroshot_coco_2017_val": ("coco/val2017", "coco/zero-shot/instances_val2017_all_2.json"),
    "zeroshot_coco_2017_val_unseen": ("coco/val2017", "coco/zero-shot/zeroshot_unseen.json"),
}
_PREDEFINED_SPLITS_COCO["coco_zeroshot_seen"] = {
    "zeroshot_coco_2017_train_seen": ("coco/train2017", "coco/zero-shot/instances_train2017_seen_2.json"),
    "zeroshot_coco_2017_val_seen": ("coco/val2017", "coco/zero-shot/instances_val2017_seen_2.json"),
}
_PREDEFINED_SPLITS_COCO["coco_zeroshot_unseen"] = {
    # "zeroshot_coco_2017_unseen": ("coco/val2017", "coco/zero-shot/zeroshot_seen.json"),
    "zeroshot_coco_2017_unseen": ("coco/val2017", "coco/zero-shot/instances_val2017_unseen_2.json"),
}

_PREDEFINED_SPLITS_COCO["coco_person"] = {
    "keypoints_coco_2014_train": (
        "coco/train2014",
        "coco/annotations/person_keypoints_train2014.json",
    ),
    "keypoints_coco_2014_val": ("coco/val2014", "coco/annotations/person_keypoints_val2014.json"),
    "keypoints_coco_2014_minival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_minival2014.json",
    ),
    "keypoints_coco_2014_valminusminival": (
        "coco/val2014",
        "coco/annotations/person_keypoints_valminusminival2014.json",
    ),
    "keypoints_coco_2017_train": (
        "coco/train2017",
        "coco/annotations/person_keypoints_train2017.json",
    ),
    "keypoints_coco_2017_val": ("coco/val2017", "coco/annotations/person_keypoints_val2017.json"),
    "keypoints_coco_2017_val_100": (
        "coco/val2017",
        "coco/annotations/person_keypoints_val2017_100.json",
    ),
}


_PREDEFINED_SPLITS_COCO_PANOPTIC = {
    "coco_2017_train_panoptic": (
        # This is the original panoptic annotation directory
        "coco/panoptic_train2017",
        "coco/annotations/panoptic_train2017.json",
        # This directory contains semantic annotations that are
        # converted from panoptic annotations.
        # It is used by PanopticFPN.
        # You can use the script at detectron2/datasets/prepare_panoptic_fpn.py
        # to create these directories.
        "coco/panoptic_stuff_train2017",
    ),
    "coco_2017_val_panoptic": (
        "coco/panoptic_val2017",
        "coco/annotations/panoptic_val2017.json",
        "coco/panoptic_stuff_val2017",
    ),
    "coco_2017_val_100_panoptic": (
        "coco/panoptic_val2017_100",
        "coco/annotations/panoptic_val2017_100.json",
        "coco/panoptic_stuff_val2017_100",
    ),
}


def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

    for (
        prefix,
        (panoptic_root, panoptic_json, semantic_root),
    ) in _PREDEFINED_SPLITS_COCO_PANOPTIC.items():
        prefix_instances = prefix[: -len("_panoptic")]
        instances_meta = MetadataCatalog.get(prefix_instances)
        image_root, instances_json = instances_meta.image_root, instances_meta.json_file
        # The "separated" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic FPN
        register_coco_panoptic_separated(
            prefix,
            _get_builtin_metadata("coco_panoptic_separated"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            os.path.join(root, semantic_root),
            instances_json,
        )
        # The "standard" version of COCO panoptic segmentation dataset,
        # e.g. used by Panoptic-DeepLab
        register_coco_panoptic(
            prefix,
            _get_builtin_metadata("coco_panoptic_standard"),
            image_root,
            os.path.join(root, panoptic_root),
            os.path.join(root, panoptic_json),
            instances_json,
        )


# ==== Predefined datasets and splits for LVIS ==========


_PREDEFINED_SPLITS_LVIS = {
    "lvis_v1": {
        "lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
        "lvis_v1_train_norare": ("coco/", "lvis/lvis_v1_train_norare.json"),
        "lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
        "lvis_v1_minival": ("coco/", "lvis/lvis_v1_minival.json"),
        # "lvis_v1_val_level": ("coco/", "FG_OVD/level_3_attributes_val.json"),
        "FG_OVD_val": ("coco/", "FG_OVD/FG_OVD_val.json"),
        "lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
        "lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
    },
    "lvis_v0.5": {
        "lvis_v0.5_train": ("coco/", "lvis/lvis_v0.5_train.json"),
        "lvis_v0.5_val": ("coco/", "lvis/lvis_v0.5_val.json"),
        "lvis_v0.5_val_rand_100": ("coco/", "lvis/lvis_v0.5_val_rand_100.json"),
        "lvis_v0.5_test": ("coco/", "lvis/lvis_v0.5_image_info_test.json"),
    },
    "lvis_v0.5_cocofied": {
        "lvis_v0.5_train_cocofied": ("coco/", "lvis/lvis_v0.5_train_cocofied.json"),
        "lvis_v0.5_val_cocofied": ("coco/", "lvis/lvis_v0.5_val_cocofied.json"),
    },
}

FG_OVD = {
    "FG_OVD_val_1_attr": ("coco/", "FG_OVD/validation_sets/1_attributes.json"),
    # "FG_OVD_val_2_attr": ("coco/", "FG_OVD/validation_sets/2_attributes.json"),
    # "FG_OVD_val_3_attr": ("coco/", "FG_OVD/validation_sets/3_attributes.json"),
    # "FG_OVD_val_color": ("coco/", "FG_OVD/validation_sets/color.json"),
    # "FG_OVD_val_material": ("coco/", "FG_OVD/validation_sets/material.json"),
    # "FG_OVD_val_pattern": ("coco/", "FG_OVD/validation_sets/pattern.json"),
    # "FG_OVD_val_shuffle_negatives": ("coco/", "FG_OVD/validation_sets/shuffle_negatives.json"),
    # "FG_OVD_val_transparency": ("coco/", "FG_OVD/validation_sets/transparency.json"),

    "FG_OVD_train_1_attr": ("coco/", "FG_OVD/training_sets/1_attributes.json"),
    "FG_OVD_train_2_attr": ("coco/", "FG_OVD/training_sets/2_attributes.json"),
    "FG_OVD_train_3_attr": ("coco/", "FG_OVD/training_sets/3_attributes.json"),
    # "FG_OVD_train_color": ("coco/", "FG_OVD/training_sets/color.json"),
    # "FG_OVD_train_material": ("coco/", "FG_OVD/training_sets/material.json"),
    # "FG_OVD_train_pattern": ("coco/", "FG_OVD/training_sets/pattern.json"),   
    # "FG_OVD_train_shuffle_negatives": ("coco/", "FG_OVD/training_sets/shuffle_negatives.json"),
    # "FG_OVD_train_transparency": ("coco/", "FG_OVD/training_sets/transparency.json"),
    
    "FG_OVD_train_1_attr_with_subject": ("coco/", "FG_OVD/training_sets/with_subject/1_attributes_with_subject.json"),
    # "FG_OVD_train_1_attr_with_subject_multi_vocab": ("coco/", "FG_OVD/training_sets/with_multi_vocab2/1_attributes_with_multi_vocab_ver.json"),
    # "FG_OVD_train_1_attr_with_subject_multi_vocab": ("coco/", "FG_OVD/training_sets/with_multi_vocab/1_attributes_with_subject_with_multi_vocab_single.json"),
    "FG_OVD_train_1_attr_with_subject_multi_vocab": ("coco/", "FG_OVD/training_sets/with_multi_vocab_single_final/1_attributes_with_subject_with_multi_vocab_single.json"),
    "FG_OVD_bench_1_attr_with_subject": ("coco/", "FG_OVD/benchmarks/1_attributes_with_subject.json"),
    "FG_OVD_bench_1_attr_with_subject_multi_vocab": ("coco/", "FG_OVD/benchmarks/with_multi_vocab_with_subject/1_attributes_with_multi_vocab.json"),
    "FG_OVD_bench_transparency_with_subject": ("coco/", "FG_OVD/benchmarks/transparency_with_subject.json"),
    
    # "FG_OVD_val_1_attr_level": ("coco/", "FG_OVD/level_3_attributes_val.json")
}

def register_all_lvis(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_LVIS.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_lvis_instances(
                key,
                get_lvis_instances_meta(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

def get_FG_OVD_instances_meta(json_file):
    import json
    json_data = json.load(open(json_file, "r"))
    cat_item_lst = sorted(json_data["categories"], key=lambda x: x["id"])
    thing_classes = [k["name"] for k in cat_item_lst]
    cls_img_count = [k["image_count"] for k in cat_item_lst]

    meta = {"thing_classes": thing_classes, "class_image_count": cls_img_count}
    meta["thing_dataset_id_to_contiguous_id"] = {
        k["id"]: i for i, k in enumerate(cat_item_lst)
    }
    meta["contiguous_id_to_thing_dataset_id"] = {
        v: k for k, v in meta["thing_dataset_id_to_contiguous_id"].items()
    }

    return meta

def register_FG_OVD_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_FG_OVD_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="lvis", **metadata
    )

def load_FG_OVD_json(json_file, image_root, dataset_name):
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # if dataset_name is not None:
    meta = get_FG_OVD_instances_meta(json_file)
    MetadataCatalog.get(dataset_name).set(**meta)

    # sort indices for reproducible results
    img_ids = sorted(lvis_api.imgs.keys())
    # imgs is a list of dicts, each looks something like:
    # {'license': 4,
    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
    #  'file_name': 'COCO_val2014_000000001268.jpg',
    #  'height': 427,
    #  'width': 640,
    #  'date_captured': '2013-11-17 05:57:24',
    #  'id': 1268}
    imgs = lvis_api.load_imgs(img_ids)
    # anns is a list[list[dict]], where each dict is an annotation
    # record for an object. The inner list enumerates the objects in an image
    # and the outer list enumerates over images. Example of anns[0]:
    # [{'segmentation': [[192.81,
    #     247.09,
    #     ...
    #     219.03,
    #     249.06]],
    #   'area': 1035.749,
    #   'image_id': 1268,
    #   'bbox': [192.81, 224.8, 74.73, 33.43],
    #   'category_id': 16,
    #   'id': 42986},
    #  ...]
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    # Sanity check that each annotation has a unique id
    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        json_file
    )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in the LVIS format from {}".format(len(imgs_anns), json_file))

    def get_file_name(img_root, img_dict):
        # Determine the path including the split folder ("train2017", "val2017", "test2017") from
        # the coco_url field. Example:
        #   'coco_url': 'http://images.cocodataset.org/train2017/000000155379.jpg'
        split_folder, file_name = img_dict["coco_url"].split("/")[-2:]
        return os.path.join(img_root + split_folder, file_name)

    dataset_dicts = []

    for (img_dict, anno_dict_list) in imgs_anns:
        # record = {}
        # record["file_name"] = get_file_name(image_root, img_dict)
        # record["height"] = img_dict["height"]
        # record["width"] = img_dict["width"]
        # record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
        # record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        # image_id = record["image_id"] = img_dict["id"]

        # objs = []
        for anno in anno_dict_list:
            record = {}
            record["file_name"] = get_file_name(image_root, img_dict)
            record["height"] = img_dict["height"]
            record["width"] = img_dict["width"]
            record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
            record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
            
            image_id = record["image_id"] = img_dict["id"]


            assert anno["image_id"] == image_id
            if len(anno.get("neg_category_ids", [])) != 10:
                continue
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}

            obj["category_id"] = meta["thing_dataset_id_to_contiguous_id"][anno["category_id"]]
            obj["neg_category_ids"] = [meta["thing_dataset_id_to_contiguous_id"][idx] for idx in anno["neg_category_ids"]]

            obj["category_name"] = meta["thing_classes"][obj["category_id"]]
            obj["neg_category_name"] = [meta["thing_classes"][idx] for idx in obj["neg_category_ids"]]

            if anno.get("level_vocab", None) is not None:
                obj["level_vocab"] = anno["level_vocab"]
            if anno.get("subject", None) is not None:
                obj["subject"] = anno["subject"]
            if anno.get("category_multi_vocab_lst", None) is not None:
                # import pdb;pdb.set_trace()
                obj["category_multi_vocab_lst"] = anno["category_multi_vocab_lst"]
            if anno.get("neg_category_multi_vocab_lsts", None) is not None:
                # import pdb;pdb.set_trace()
                obj["neg_category_multi_vocab_lsts"] = anno["neg_category_multi_vocab_lsts"]
            record["annotations"] = [obj]
            dataset_dicts.append(record)
            # objs.append(obj)

        # record["annotations"] = objs

    return dataset_dicts

def register_all_FG_OVD(root):
    for key, (image_root, json_file) in FG_OVD.items():
        register_FG_OVD_instances(
            key,
            get_FG_OVD_instances_meta(os.path.join(root, json_file)),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# ==== Predefined splits for raw cityscapes images ===========
_RAW_CITYSCAPES_SPLITS = {
    "cityscapes_fine_{task}_train": ("cityscapes/leftImg8bit/train/", "cityscapes/gtFine/train/"),
    "cityscapes_fine_{task}_val": ("cityscapes/leftImg8bit/val/", "cityscapes/gtFine/val/"),
    "cityscapes_fine_{task}_test": ("cityscapes/leftImg8bit/test/", "cityscapes/gtFine/test/"),
}


def register_all_cityscapes(root):
    for key, (image_dir, gt_dir) in _RAW_CITYSCAPES_SPLITS.items():
        meta = _get_builtin_metadata("cityscapes")
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)

        inst_key = key.format(task="instance_seg")
        DatasetCatalog.register(
            inst_key,
            lambda x=image_dir, y=gt_dir: load_cityscapes_instances(
                x, y, from_json=True, to_polygons=True
            ),
        )
        MetadataCatalog.get(inst_key).set(
            image_dir=image_dir, gt_dir=gt_dir, evaluator_type="cityscapes_instance", **meta
        )

        sem_key = key.format(task="sem_seg")
        DatasetCatalog.register(
            sem_key, lambda x=image_dir, y=gt_dir: load_cityscapes_semantic(x, y)
        )
        MetadataCatalog.get(sem_key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="cityscapes_sem_seg",
            ignore_label=255,
            **meta,
        )


# ==== Predefined splits for PASCAL VOC ===========
def register_all_pascal_voc(root):
    SPLITS = [
        ("voc_2007_trainval", "VOC2007", "trainval"),
        ("voc_2007_train", "VOC2007", "train"),
        ("voc_2007_val", "VOC2007", "val"),
        ("voc_2007_test", "VOC2007", "test"),
        ("voc_2012_trainval", "VOC2012", "trainval"),
        ("voc_2012_train", "VOC2012", "train"),
        ("voc_2012_val", "VOC2012", "val"),
    ]
    for name, dirname, split in SPLITS:
        year = 2007 if "2007" in name else 2012
        register_pascal_voc(name, os.path.join(root, dirname), split, year)
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"


def register_all_ade20k(root):
    root = os.path.join(root, "ADEChallengeData2016")
    for name, dirname in [("train", "training"), ("val", "validation")]:
        image_dir = os.path.join(root, "images", dirname)
        gt_dir = os.path.join(root, "annotations_detectron2", dirname)
        name = f"ade20k_sem_seg_{name}"
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            stuff_classes=ADE20K_SEM_SEG_CATEGORIES[:],
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
        )

_PREDEFINED_SPLITS_ANP = {
    "ANP_traffic": {
    "ANP_traffic_train": ("ANP_traffic/", "ANP_traffic/annotations/anp_train.json"),
    "ANP_traffic_finetune": ("ANP_traffic/", "ANP_traffic/annotations/anp_train_filter_unknown.json"),
    "ANP_traffic_val": ("ANP_traffic/JIDU_small_testset/", "ANP_traffic/annotations/anp_val_allcone.json"),
    }
}

def register_all_anp(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_ANP.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )

_PREDEFINED_SPLITS_VG_PSEUDO = {
    "vg_pseudo": {
        "vg_filter_rare": ("VisualGenome/images", "VisualGenome/vg_filter_rare.json",)
    }
}


def register_all_vg(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_VG_PSEUDO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
                extra_annotation_keys=["score"],
            )


# True for open source;
# Internally at fb, we register them elsewhere
if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "dataset"))
    register_all_coco(_root)
    register_all_lvis(_root)
    register_all_FG_OVD(_root)
    # register_all_cityscapes(_root)
    # register_all_cityscapes_panoptic(_root)
    # register_all_pascal_voc(_root)
    # register_all_ade20k(_root)
    # register_all_anp(_root)
    # register_all_vg(_root)
