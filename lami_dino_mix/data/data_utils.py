import torch
from torch.utils.data import Dataset, DataLoader
from peft import LoraConfig, get_peft_model
import random
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)

import copy
import numpy as np
from detrex.data import DetrDatasetMapper
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
import os
from torch.utils.data import get_worker_info


class CombinedDataset(Dataset):
    def __init__(self, dataset_names_a, dataset_names_b, b_bias: int = 5):
        """
        Args:
            dataset_names_a (str or list[str]): Dataset names for the first dataset (e.g., "lvis_v1_train_norare").
            dataset_names_b (str or list[str]): Dataset names for the second dataset (e.g., ["FG_OVD_train_1_attr"]).
        """
        self.dataset_a_dicts = get_detection_dataset_dicts(names=dataset_names_a)
        self.dataset_b_dicts = get_detection_dataset_dicts(names=dataset_names_b)
        self.len_a = len(self.dataset_a_dicts)
        self.len_b = len(self.dataset_b_dicts)
        if os.environ.get("DEBUG_PDB") == "1" and get_worker_info() is None:
            import pdb; pdb.set_trace()
        self.total_length = max(self.len_a, self.len_b) #  你可以根据需要调整总长度的计算方式
        self.dataset_choice = ['a'] * self.len_a + ['b'] * self.len_b * b_bias #  创建数据集选择列表，用于随机采样

    def __len__(self):
        return self.total_length # 返回总长度，可以根据需求调整

    def __getitem__(self, idx):
        dataset_selector = random.choice(self.dataset_choice) # 随机选择数据集 'a' 或 'b'
        if dataset_selector == 'a':
            data = self.dataset_a_dicts[random.randint(0, self.len_a - 1)] # 从 dataset_a 中随机采样
            data['dataset_name'] = 'dataset_a' #  添加 dataset 名称标识，如果需要在 DatasetMapper 中区分
        else:
            data = self.dataset_b_dicts[random.randint(0, self.len_b - 1)] # 从 dataset_b 中随机采样
            data['dataset_name'] = 'dataset_b' #  添加 dataset 名称标识，如果需要在 DatasetMapper 中区分
        return data



class CombinedDataLoader:
    def __init__(self,
            loader_1: DataLoader, 
            loader_2: DataLoader, 
            ratio=(4, 1),
            **kwargs
        ):
        # asser batch size is same
        assert loader_1.batch_size == loader_2.batch_size

        self.loader_1_iter = iter(loader_1)
        self.loader_2_iter = iter(loader_2)
        self.ratio = ratio
        self.total_batches_per_cycle = sum(ratio)
        self.batches_yielded_in_cycle = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches_yielded_in_cycle < self.ratio[0]:
            batch = next(self.loader_1_iter)
            self.batches_yielded_in_cycle += 1
            return batch

        elif self.batches_yielded_in_cycle < self.total_batches_per_cycle:
            batch = next(self.loader_2_iter)
            self.batches_yielded_in_cycle += 1
            if self.batches_yielded_in_cycle == self.total_batches_per_cycle:
                self.batches_yielded_in_cycle = 0 # 完成一个循环，重置计数器
            return batch
        else:
            self.batches_yielded_in_cycle = 0 # 理论上不会到这里，以防万一重置
            raise StopIteration


class MG_DatasetMapper(DetrDatasetMapper):
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """

        #! in FG-OVD dataset, assert each image has only one annotation
        assert len(dataset_dict["annotations"]) == 1, f"len(dataset_dict['annotations']) = {len(dataset_dict['annotations'])}"

        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if self.augmentation_with_crop is None:
            image, transforms = T.apply_transform_gens(self.augmentation, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.augmentation, image)
            else:
                image, transforms = T.apply_transform_gens(self.augmentation_with_crop, image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["FG_OVD_FLAG"] = "fgovd"

        # if not self.is_train:
        #     # USER: Modify this if you want to keep them for some reason.
        #     dataset_dict.pop("annotations", None)
        #     return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            
            instances_name_lst, instances_all_name_id_list = [], []
            pos_multi_vocab_lst = []
            neg_multi_vocab_lsts = []
            for item in annos:
                instances_name_lst.append([item["category_name"]] + item["neg_category_name"])
                instances_all_name_id_list.append([item["category_id"]] + item["neg_category_ids"])
                # 1. 读取正类的分解文本 (category_multi_vocab_lst)
                pos_vocabs = item.get("category_multi_vocab_lst", [])
                pos_multi_vocab_lst.append(pos_vocabs)
                
                # 2. 读取负类的分解文本 (neg_category_multi_vocab_lsts)
                neg_vocabs = item.get("neg_category_multi_vocab_lsts", [])
                neg_multi_vocab_lsts.append(neg_vocabs)
            
            instances = utils.annotations_to_instances(annos, image_shape)
            instances.gt_instances_name = instances_name_lst
            instances.all_class_id = instances_all_name_id_list
            instances.gt_pos_multi_vocab = pos_multi_vocab_lst
            instances.gt_neg_multi_vocab = neg_multi_vocab_lsts
            # pos_multi_vocab_lst.extend(neg_multi_vocab_lsts)
            # instances.gt_all_multi_vocab =  pos_multi_vocab_lst
            # print(pos_multi_vocab_lst)
            # import pdb;pdb.set_trace()

            if annos[0].get("subject", None) is not None:
                instances.subject = ["A " + annos[0]["subject"]]


            
            dataset_dict["instances"] = instances
        return dataset_dict



