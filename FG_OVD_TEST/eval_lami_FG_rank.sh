python eval_rank.py \
    --predictions /data1/liangzhijia/my_LaMI/FG_OVD_TEST/FG_results/FG_lvis_pretrain/lami_3_attributes.pkl \
    --ground_truth /data1/liangzhijia/FG-OVD/benchmarks/3_attributes.json \

python eval_rank.py \
    --predictions /data1/liangzhijia/my_LaMI/FG_OVD_TEST/FG_results/FG_lvis_pretrain/lami_1_attributes.pkl \
    --ground_truth /data1/liangzhijia/FG-OVD/benchmarks/1_attributes.json \

python eval_rank.py \
    --predictions /data1/liangzhijia/my_LaMI/FG_OVD_TEST/FG_results/FG_lvis_pretrain/lami_2_attributes.pkl \
    --ground_truth /data1/liangzhijia/FG-OVD/benchmarks/2_attributes.json \

python eval_rank.py \
    --predictions /data1/liangzhijia/my_LaMI/FG_OVD_TEST/FG_results/FG_lvis_pretrain/lami_color.pkl \
    --ground_truth /data1/liangzhijia/FG-OVD/benchmarks/color.json \

python eval_rank.py \
    --predictions /data1/liangzhijia/my_LaMI/FG_OVD_TEST/FG_results/FG_lvis_pretrain/material.pkl \
    --ground_truth /data1/liangzhijia/FG-OVD/benchmarks/material.json \

python eval_rank.py \
    --predictions /data1/liangzhijia/my_LaMI/FG_OVD_TEST/FG_results/FG_lvis_pretrain/pattern.pkl \
    --ground_truth /data1/liangzhijia/FG-OVD/benchmarks/pattern.json \
    
python eval_rank.py \
    --predictions /data1/liangzhijia/my_LaMI/FG_OVD_TEST/FG_results/FG_lvis_pretrain/shuffle_negatives.pkl \
    --ground_truth /data1/liangzhijia/FG-OVD/benchmarks/shuffle_negatives.json \

# python eval_rank.py \
#     --predictions /data1/liangzhijia/my_LaMI/FG_OVD_TEST/FG_results/FG_lvis_pretrain/transparency.pkl \
#     --ground_truth /data1/liangzhijia/FG-OVD/benchmarks/transparency.json \