from detrex.config import get_config
from .models.dino_convnextl import model
from ..data.MG_detr import dataloader
from peft import LoraConfig, get_peft_model
from detectron2.config import LazyCall as L

dataset_path = "/apdcephfs_cq12/share_1150325/jiaminghli/data/"
model.vlm_query_path = "dataset/metadata/lvis_visual_desc_confuse_lvis_convnextl.npy"
model.score_ensemble = True
model.backbone.score_ensemble = model.score_ensemble
model.seen_classes = dataset_path + '/lvis/lvis_v1_seen_classes.json'
model.all_classes = dataset_path + '/lvis/lvis_v1_all_classes.json'
model.vlm_temperature = 100.0 # keep same with f-vlm
model.alpha = 0.0
model.beta = 0.6
model.novel_scale = 5.0

peft_config=L(LoraConfig)(
    r=16,
    lora_alpha=16,
    target_modules=[
        # decoder decoder.layers.*.attentions.*.
        # "attn.in_proj_weight",
        # "attn.in_proj_bias",
        "attn.out_proj",
        "sampling_offsets",
        "attention_weights",
        "value_proj",
        "output_proj",
        
        # class embedding projs
        "linear",

        # FG text encoder last proj
        "fg_text_clip.text_mlp.linear_layers.1.0",
    ],
    exclude_modules="fg_text_clip.base_clip_model.*",
    lora_dropout=0.1,
    bias="none",
)

# get default config
# dataloader = get_config("common/data/lvis_detr.py").dataloader
optimizer = get_config("common/optim.py").AdamW
lr_multiplier = get_config("common/lvis_schedule.py").lr_multiplier_12ep_warmup
train = get_config("common/train.py").train


# modify training config
# train.init_checkpoint = "clip_convnext_large_trans.pth"
train.init_checkpoint = "pretrain_models/model_final.pth"
train.output_dir = "./output/MG_lora_FG_OVD"

# max training iterations
train.max_iter = 85200# TODO

# run evaluation every 5000 iters
train.eval_period = 85200

# log training infomation every 20 iters
train.log_period = 200

# save checkpoint every 5000 iters
train.checkpointer.period = 7100//2

# gradient clipping for training
train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

# set training devices
train.device = "cuda"
model.device = train.device

model.num_classes = 1203
model.query_path = "dataset/metadata/lvis_visual_desc_convnextl.npy"
model.eval_query_path = "dataset/metadata/lvis_visual_desc_convnextl.npy"

model.use_fed_loss = True
model.cluster_fed_loss = True
model.cluster_label_path = 'dataset/cluster/lvis_cluster_128.npy'
model.cat_freq_path = dataset_path+"/lvis/lvis_v1_train_norare_cat_info.json"
model.fed_loss_num_cat=100
model.select_box_nums_for_evaluation = 300

# modify optimizer config
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

# modify dataloader config
dataloader.train.num_workers = 8

# please notice that this is total batch size.
# surpose you're using 4 gpus for training and the batch size for
# each gpu is 16/4 = 4
dataloader.train.total_batch_size = 32

# dump the testing results into output_dir for visualization
dataloader.evaluator.output_dir = train.output_dir
dataloader.test.dataset.names = "lvis_v1_val"
