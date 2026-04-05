import argparse
import torch

from tqdm import tqdm

# Use GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from torchvision.ops import batched_nms

# import some common libraries
import sys
import numpy as np
import os, json, cv2, random

# from utils.pickle_handler import saveObject, loadObject
# from utils.json_handler import read_json, write_json

import torch.nn.functional as F
SCORE_THRESH = 0.1
import numpy as np
import pickle, json
from detectron2.config import LazyConfig, instantiate
import detectron2.data.transforms as T
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.checkpoint import DetectionCheckpointer
from clip_models.enc_text import getClip_model
from clip_models.FG_clip_model import FG_convext_clip

import warnings
warnings.filterwarnings(
    "ignore", 
    message=".*Converting mask without torch.bool dtype to bool.*", 
    category=UserWarning
)

def print_json_structure(data, indent='', level=0):
    """
    Recursively prints the structure of a JSON-like data object.

    This function prints the keys and types of values in a dictionary, the length
    and an example element for lists, and the shape for numpy arrays and torch tensors.
    Other data types are printed with their type name and value.

    Args:
        data: The data object to be analyzed. Can be a dictionary, list, numpy array,
              torch tensor, or other data type.
        indent (str): The indentation string used for formatting the output.
        level (int): The current level of recursion, used for formatting the output.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}|-- \033[31m{key}\033[0m:")
            print_json_structure(value, indent + "    ", level + 1)
    
    # 处理列表
    elif isinstance(data, list):
        print(f"{indent}\033[33mList of length {len(data)}\033[0m")
        if len(data) > 0:
            print(f"\033[33m{indent}(Example element):\033[0m")
            print_json_structure(data[-1], indent, level + 1)
    
    # 处理 NumPy 数组
    elif isinstance(data, np.ndarray):
        print(f"{indent}\033[32mnp.ndarray with shape\033[0m {data.shape}")
    
    # 处理 PyTorch 张量
    elif isinstance(data, torch.Tensor):
        print(f"{indent}\033[32mtorch.Tensor with shape\033[0m {data.shape}")
    
    # 处理其他类型
    else:
        print(f"\033[32m{indent}({type(data).__name__})\033[0m{data}")

def saveObject(obj, path):
    """"Save an object using the pickle library on a file
    
    :param obj: undefined. Object to save
    :param fileName: str. Name of the file of the object to save
    """
    print("Saving " + path)
    with open(path, 'wb') as fid:
        pickle.dump(obj, fid)
        
def loadObject(path):
    """"Load an object from a file
    
    :param fileName: str. Name of the file of the object to load
    :return: obj: undefined. Object loaded
    """
    try:
        with open(path, 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None   
    
def read_json(file_name):
    #Read JSON file
    with open(file_name) as infile:
        data = json.load(infile)
    return data

def write_json(data, file_name):
    # Write JSON file
    with open(file_name, "w") as outfile:
        json.dump(data, outfile)

def convert_to_x1y1x2y2(bbox, img_width, img_height):
    """
    Convert bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format.

    Args:
        bbox (np.array): NumPy array of bounding boxes in the format [cx, cy, w, h].
        img_width (int): Width of the image.
        img_height (int): Height of the image.

    Returns:
        np.array: NumPy array of bounding boxes in the format [x1, y1, x2, y2].
    """
    cx, cy, w, h = bbox
    x1 = (cx - w / 2) * img_width
    y1 = (cy - h / 2) * img_height
    x2 = (cx + w / 2) * img_width
    y2 = (cy + h / 2) * img_height
    return np.array([x1, y1, x2, y2])



skipped_categories = 0

def evaluate_image(
        lami_model: torch.nn.Module, 
        clip_model, 
        FG_clip,
        tokenizer,
        ann,
        im, 
        vocabulary, 
        MAX_PREDICTIONS=100,
        len_vocabulary=6,
        categories = None,
    ):
    global skipped_categories
    # preparing the inputs

    def adjust_out_id(output):
        for i in range(len(output['labels'])):
            output['labels'][i] = output['vocabulary'][output['labels'][i]]
        return output

    height = im.shape[0]
    width = im.shape[1]
    argumentations = [T.ResizeShortestEdge(short_edge_length=800, max_size=1333)]
    image, transforms = T.apply_transform_gens(argumentations, im)

    ann["bbox_mode"] = BoxMode.XYWH_ABS
    ann = utils.transform_instance_annotations(ann, transforms, image.shape[:2])
    anns = [utils.transform_instance_annotations(ann, transforms, image.shape[:2])]
    # lami model input:
    # file_name, height, width, image, image_id, instance
    multi = False
    if multi:
        gt_pos_multi_vocab = ann['category_multi_vocab_lst']
        gt_neg_multi_vocab = ann['neg_category_multi_vocab_lsts'][:len_vocabulary-1]

        # print(ann['subject'])
        # print(categories[ann['category_id']]['name'])
        # print(gt_pos_multi_vocab)
        gt_pos_multi_vocab = list(dict.fromkeys(gt_pos_multi_vocab))
        gt_neg_multi_vocab = [list(dict.fromkeys(x)) for x in gt_neg_multi_vocab]
        # print(categories[ann['neg_category_ids'][0]]['name'], categories[ann['neg_category_ids'][1]]['name'])
        # print(gt_neg_multi_vocab)

        use_multi = True
        if gt_pos_multi_vocab[0][0] != 'A':
            use_multi = False
            print(gt_pos_multi_vocab[0][0])
        max_cnt = 1
        for i in range(1):
            if len(gt_pos_multi_vocab) > max_cnt:
                max_cnt = len(gt_pos_multi_vocab)
            for j in range(len(gt_neg_multi_vocab)):
                if len(gt_neg_multi_vocab[j]) > max_cnt:
                    max_cnt = len(gt_neg_multi_vocab[j])

        padded_embs = []   # list of (max_cnt, D)
        masks = []         # list of (max_cnt,), 1 for real, 0 for pad
        diff_masks = [] 
        ones_vec = None

        for i in range(1):
            # pos (a list of phrases)
            pos_embs =  FG_clip.get_batch_text_embs(gt_pos_multi_vocab, use_mlp=True)  # (Pi, D)
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
            for j in range(len(gt_neg_multi_vocab)):
                neg_embs =  FG_clip.get_batch_text_embs(gt_neg_multi_vocab[j], use_mlp=True)  # (Sj, D)
                neg_embs = F.normalize(neg_embs, p=2, dim=1)
                pad_len = max_cnt - neg_embs.shape[0]
                if pad_len > 0:
                    neg_padded = torch.cat([neg_embs, ones_vec.repeat(pad_len, 1)], dim=0)
                else:
                    neg_padded = neg_embs[:max_cnt]
                mask_neg = torch.zeros(max_cnt, device=neg_embs.device)
                mask_neg[:neg_embs.shape[0]] = 1.0
                mask_diff_neg = torch.zeros(max_cnt, device=neg_embs.device)

                flag = 0
                for k, phrase in enumerate(gt_neg_multi_vocab[j]):
                    # 如果该短语不在正样本集合中，标记为 1 (错误属性)
                    # 且必须在 max_cnt 范围内（虽然逻辑上不会越界，但为了安全）
                    if k < max_cnt and phrase not in pos_vocab_set:
                        mask_diff_neg[k] = 1.0
                        flag = 1
                if flag == 0:
                    mask_diff_neg[0] = 1.0
                    print("bbb")

                padded_embs.append(neg_padded)
                masks.append(mask_neg)
                diff_masks.append(mask_diff_neg)

        # (N, max_cnt, D) and (N, max_cnt), ordered by batch, and within batch: [pos, neg1..negK]
        vlm_all_query_embedding = torch.stack(padded_embs, dim=0)
        vlm_all_query_mask = torch.stack(masks, dim=0)
        vlm_diff_mask = torch.stack(diff_masks, dim=0)
        lami_model.vlm_all_query_embedding  =  vlm_all_query_embedding
        lami_model.vlm_all_query_mask = vlm_all_query_mask
        lami_model.vlm_diff_mask = vlm_diff_mask
        # import pdb; pdb.set_trace()

    input_dict = {
        'file_name': ann['file_name'],
        'height': height,
        'width': width,
        'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))),
        'image_id': ann['image_id'],
        'instance': utils.annotations_to_instances(anns, image.shape[:2]),
        "FG_OVD_FLAG": 'modify'
    }
    batch_input = [input_dict]

    if isinstance(ann['subject'], list):
        subject_name = ["A " + item.replace(".","") + "." for item in ann['subject']]
        subject_name = subject_name[:len(vocabulary)]
    else:
        assert isinstance(ann['subject'], str)
        subject_name = ["A " + ann['subject'] for _ in range(len(vocabulary))]
    
    # if the tokens length is above 77, the model can't handle them
    # if  inputs['input_ids'].shape[1] > 77:
    #     skipped_categories += 1
    #     return None
    
    # Get predictions
    with torch.no_grad():
        # subject embedding
        query_emb = FG_clip.get_batch_text_embs(subject_name, use_mlp=False)

        # query_emb = FG_clip.get_batch_text_embs(vocabulary, use_mlp=False)
        query_emb = F.normalize(query_emb, p=2, dim=-1)
        # query_emb = query_emb.repeat(len(vocabulary), 1)

        # FG embedding
        eval_embs = FG_clip.get_batch_text_embs(vocabulary, use_mlp=True)
        eval_embs = F.normalize(eval_embs, p=2, dim=1).to(device)

        # Set the eval embeddings
        lami_model.eval_content_query_embedding = query_emb
        lami_model.content_query_embedding = query_emb
        lami_model.vlm_content_query_embedding = eval_embs
        for class_emb in lami_model.transformer.decoder.class_embed:
            class_emb.eval_zs_weight = query_emb.permute(1, 0).contiguous()
            class_emb.zs_weight = query_emb.permute(1, 0).contiguous()

        # Set the eval indices
        lami_model.base_idx = [False for _ in range(eval_embs.shape[0])]
        lami_model.novel_idx = [True for _ in range(eval_embs.shape[0])]

        # ----------------------------------------------------------------
        # 为 DINOTransformerAttr 的 Attribute_Attention 准备推理时的 vocab emb。
        # vocabulary 里每个词条（正例+负例）的 multi-vocab 原子短语嵌入。
        # 每个词条用它自己的 eval_emb 作为唯一一个 vocab，shape=[1,768]，
        # 这样 Attribute_Attention 退化为对单短语做 attention（安全兜底行为）。
        # 如果推理数据里提供了 multi_vocab，可以在此处替换为多短语版本。
        # ----------------------------------------------------------------
        if hasattr(lami_model, 'transformer') and hasattr(lami_model.transformer, 'attr_attention'):
            # 收集每个词条（正例 + 所有负例）的原子短语列表
            # vocabulary[0] = 正例, vocabulary[1..V-1] = 负例
            V = eval_embs.size(0)
            D = eval_embs.size(1)

            # 逐词条编码多短语；无 multi_vocab 字段时退化为单短语
            pos_phrases_lst  = ann.get('category_multi_vocab_lst', None)
            neg_phrases_lsts = ann.get('neg_category_multi_vocab_lsts', [])

            # 构造每个词条的去重短语列表
            all_phrase_lists = []
            if pos_phrases_lst is not None:
                all_phrase_lists.append(list(dict.fromkeys(pos_phrases_lst)))
            else:
                all_phrase_lists.append(None)   # 用单条 eval_emb 兜底

            for j in range(V - 1):
                if j < len(neg_phrases_lsts) and neg_phrases_lsts[j]:
                    all_phrase_lists.append(list(dict.fromkeys(neg_phrases_lsts[j])))
                else:
                    all_phrase_lists.append(None)

            # 确定最大短语数（决定列维度）
            max_phrases = 1
            for phrases in all_phrase_lists:
                if phrases is not None:
                    max_phrases = max(max_phrases, len(phrases))

            # 初始化为零（无效位置），mask 全 0
            infer_vocab_emb  = torch.zeros(V, max_phrases, D, device=device)
            infer_vocab_mask = torch.zeros(V, max_phrases, dtype=torch.float32, device=device)

            for idx, phrases in enumerate(all_phrase_lists):
                if phrases is not None:
                    emb = FG_clip.get_batch_text_embs(phrases, use_mlp=True)
                    emb = F.normalize(emb, p=2, dim=1).to(device)   # [Pi, D]
                    n   = emb.size(0)
                    infer_vocab_emb[idx, :n]  = emb
                    infer_vocab_mask[idx, :n] = 1.0
                else:
                    # 退化：用该词条的 FG embedding 作为唯一一条 vocab
                    infer_vocab_emb[idx, 0]  = eval_embs[idx]
                    infer_vocab_mask[idx, 0] = 1.0

            lami_model.transformer._infer_vocab_emb  = infer_vocab_emb
            lami_model.transformer._infer_vocab_mask = infer_vocab_mask
        # ----------------------------------------------------------------

        outputs = lami_model(batch_input)[0]["instances"].to("cpu")
    
    return {
        'scores': outputs.scores[:MAX_PREDICTIONS].cpu().detach().numpy(),
        'labels': outputs.pred_classes[:MAX_PREDICTIONS].cpu().detach().numpy(),
        'boxes': outputs.pred_boxes.tensor[:MAX_PREDICTIONS].cpu().detach().numpy()
    }
        
    

def get_category_name(id, categories):
    for category in categories:
        if id == category['id']:
            return category['name']
        
def get_image_filepath(id, images):
    for image in images:
        if id == image['id']:
            return image['file_name']

def create_vocabulary(ann, categories):
    vocabulary_id = [ann['category_id']] + ann['neg_category_ids']
    vocabulary = [get_category_name(id, categories) for id in vocabulary_id]
    
    return vocabulary, vocabulary_id

def adjust_out_id(output, vocabulary_id):
    for i in range(len(output['labels'])):
        output['labels'][i] = vocabulary_id[output['labels'][i]]
    return output

def get_lami_detr_models(args: argparse.Namespace):
    """
    Returns:
        model, clip_model, tokenizer
    """
    cfg = LazyConfig.load(args.lami_config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    model = instantiate(cfg.model)
    model = model.to(device)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

    clip_model, tokenizer = getClip_model(model_name = 'convnext_large_d_320', ckpt_file = "/apdcephfs_cq12/share_1150325/jiaminghli/code/OVD/pretrain_models/timm_clip_convnext_large_trans.pth")
    clip_model = clip_model.to(device)
    model.eval(); clip_model.eval()
    return model, clip_model, tokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to process')
    parser.add_argument('--out', type=str, required=True, help='Out path')
    parser.add_argument('--nms', default=False, action='store_true', help='If set it will be applied NMS with iou=0.5')
    parser.add_argument('--disentangled_inferences', default=False, action='store_true', help='If set, a vocabulary is decomposed in single captions')
    parser.add_argument('--large', default=False, action='store_true', help='If set, it will be loaded the large model')
    parser.add_argument('--n_hardnegatives', type=int, default=10, help="Number of hardnegatives in each vocabulary")
    parser.add_argument('--lami_config_file', type=str, default='lami_dino_mix/configs/dino_convnext_large_4scale_12ep_lvis_attr.py', help='LAMI config file')
    parser.add_argument(
        "opts",
        help="""
            Modify config options at the end of the command. For Yacs configs, use
            space-separated "PATH.KEY VALUE" pairs.
            For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument('--only_neg_categories', default=False, action='store_true', help='If set, only negative categories are considered')
    parser.add_argument('--coco_path', type=str, default='/apdcephfs_cq12/share_1150325/jiaminghli/data/coco/', help='Path to COCO dataset root directory')

    args = parser.parse_args()
    global skipped_categories

    coco_path = args.coco_path

    
    # data = read_json('/home/lorenzobianchi/PacoDatasetHandling/jsons/captioned_%s.json' % dataset_name)
    data = read_json(args.dataset)
    
    lami_model, clip_model, tokenizer = get_lami_detr_models(args)

    if hasattr(lami_model, "fg_text_clip"):
        print("use lami fg clip")
        FG_clip = lami_model.fg_text_clip
    else:
        FG_clip = FG_convext_clip().to(device)
    
    complete_outputs = []
    categories_done = []
    for ann in tqdm(data['annotations']):
        # if the category is not done, we add it to the list
        if ann['category_id'] not in categories_done:
            categories_done.append(ann['category_id'])
        else:
            continue
        vocabulary, vocabulary_id = create_vocabulary(ann, data['categories'])
        # check if a number of hardnegatives is setted to non-default values
        # if it is, the vocabulary is clipped and if it is too short, we skip that image
        len_vocabulary = args.n_hardnegatives + 1
        if len(vocabulary) < len_vocabulary:
            continue

        vocabulary = vocabulary[:len_vocabulary]
        vocabulary_id = vocabulary_id[:len_vocabulary]
        if args.only_neg_categories:
            vocabulary = vocabulary[1:]
            vocabulary_id = vocabulary_id[1:]

        image_filepath = coco_path + get_image_filepath(ann['image_id'], data['images'])
        ann["file_name"] = image_filepath
        imm = cv2.cvtColor(cv2.imread(image_filepath), cv2.COLOR_BGR2RGB)

        output = evaluate_image(
            lami_model=lami_model,
            clip_model=clip_model,
            FG_clip=FG_clip,
            tokenizer=tokenizer,
            ann=ann,
            im=imm,
            vocabulary=vocabulary,
            len_vocabulary=len_vocabulary,
            categories = data['categories']
        )

        if output == None:
            continue
        output['category_id'] = ann['category_id']
        output['vocabulary'] = vocabulary_id
        output['image_filepath'] = get_image_filepath(ann['image_id'], data['images'])
        output = adjust_out_id(output, vocabulary_id)
        complete_outputs.append(output)
        
    saveObject(complete_outputs, args.out)
    print("Skipped categories: %d/%d" % (skipped_categories, len(categories_done)))
if __name__ == '__main__':
    main()