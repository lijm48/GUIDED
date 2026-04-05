import json
import argparse
import clip_models.OpenClip.src.open_clip as open_clip
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def make_descriptor_sentence(descriptor):
    descriptor = descriptor[0].lower() + descriptor[1:]
    if descriptor.startswith('a') or descriptor.startswith('an'):
        return f"which is {descriptor}"
    elif descriptor.startswith('has') or descriptor.startswith('often') or descriptor.startswith('typically') or descriptor.startswith('may') or descriptor.startswith('can'):
        return f"which {descriptor}"
    elif descriptor.startswith('used'):
        return f"which is {descriptor}"
    else:
        return f"which has {descriptor}"

def wordify(string):
    word = string.replace('_', ' ')
    return word

def getClip_model(model_name = 'convnext_large_d_320', ckpt_file = "pretrain_models/timm_clip_convnext_large_trans.pth"):
    """
    args:
        model_name: the name of the model
        ckpt_file: the checkpoint file of the model
    returns:
        model, tokenizer
    """
    logger.info(f"Loading clip model {model_name} from {ckpt_file}")
    model, _ = open_clip.create_model_from_pretrained(
            model_name='convnext_large_d_320', 
            pretrained=ckpt_file,
            # model_name=args.model_name, 
            # pretrained=f'/xxx/{args.model_name}.pth',
            precision='fp32',
        )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.load_state_dict(torch.load(ckpt_file))
    return model, tokenizer

def getClip_model_preprocess_tokenizer(model_name = 'convnext_large_d_320', ckpt_file = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-basecv/lijiaming/OVD/pretrain_models/timm_clip_convnext_large_trans.pth", precision='fp32'):
    """
    args:
        model_name: the name of the model
        ckpt_file: the checkpoint file of the model
    returns:
        model, preprocess, tokenizer
    """
    logger.info(f"Loading clip model {model_name} from {ckpt_file}")
    model, preprocess = open_clip.create_model_from_pretrained(
            model_name='convnext_large_d_320', 
            pretrained=ckpt_file,
            # model_name=args.model_name, 
            # pretrained=f'/xxx/{args.model_name}.pth',
            precision=precision,
        )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.load_state_dict(torch.load(ckpt_file))
    return model, preprocess, tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default = 'convnext_large_d_320', choices=['convnext_large_d_320', 'RN50'])
    args = parser.parse_args()

    lvisval = json.load(open('dataset/lvis/lvis_v1_val.json'))
    names = [wordify(cat['name']) for cat in lvisval['categories']]
    # replace with your generated visual descriptions. for json format, you can refer to https://github.com/sachit-menon/classify_by_description_release/blob/master/generate_descriptors.py
    visual_descs = json.load(open('dataset/visual_desc/lvis_confuse_names_lvis0122_generated.json'))
    for i, key in enumerate(visual_descs.keys()):
        if not isinstance(visual_descs[key], list):
            print(key)
            visual_descs[key] = visual_descs_base[names.index(key)] 
    # import ipdb;ipdb.set_trace()

    # get descriptions
    gpt_descriptions = []
    for i, name in enumerate(names):
        descriptions = visual_descs[wordify(name)]
        for j, desc in enumerate(descriptions):
            if desc == '': continue
            descriptions[j] = wordify(name) + ', ' + make_descriptor_sentence(desc)
        if descriptions == ['']:
            descriptions = [name]
        gpt_descriptions.append(descriptions)
    # import ipdb;ipdb.set_trace()

    # extract features
    device = "cuda"
    model, _ = open_clip.create_model_from_pretrained(
            model_name='convnext_large_d_320', 
            pretrained='/xxx/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin',
            # model_name=args.model_name, 
            # pretrained=f'/xxx/{args.model_name}.pth',
            device=device,
            precision='fp16' if args.model_name == "EVA02-L-14-336" else 'fp32'
        )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model_name)

    description_encodings = []
    with torch.no_grad():
        for i, v in enumerate(gpt_descriptions):
            tokens = tokenizer(v).to(device)
            embeddings = F.normalize(model.encode_text(tokens)).cpu()
            description_encodings.append(embeddings.mean(dim=0))
    # import ipdb;ipdb.set_trace()
    description_encodings = torch.stack(description_encodings, dim=0)
    np.save(f'dataset/metadata/lvis_visual_desc_{args.model_name.lower()}.npy', description_encodings.numpy())