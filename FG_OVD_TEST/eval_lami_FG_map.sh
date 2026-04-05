Name=multi_diff3
python eval_map.py \
    --predictions ../../output/FG_bench/$Name/lami_3_attributes.pkl \
    --ground_truth /apdcephfs_cq12/share_1150325/jiaminghli/data/FG_OVD/benchmarks/3_attributes.json \
    --out  ../../output/FG_bench/$Name/lami_3_attributes.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ../../output/FG_bench/$Name/lami_1_attributes.pkl \
    --ground_truth /apdcephfs_cq12/share_1150325/jiaminghli/data/FG_OVD/benchmarks/1_attributes.json \
    --out  ../../output/FG_bench/$Name/lami_1_attributes.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ../../output/FG_bench/$Name/lami_2_attributes.pkl \
    --ground_truth /apdcephfs_cq12/share_1150325/jiaminghli/data/FG_OVD/benchmarks/2_attributes.json \
    --out  ../../output/FG_bench/$Name/lami_2_attributes.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ../../output/FG_bench/$Name/lami_color.pkl \
    --ground_truth /apdcephfs_cq12/share_1150325/jiaminghli/data/FG_OVD/benchmarks/color.json \
    --out  ../../output/FG_bench/$Name/lami_color.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ../../output/FG_bench/$Name/material.pkl \
    --ground_truth /apdcephfs_cq12/share_1150325/jiaminghli/data/FG_OVD/benchmarks/material.json \
    --out  ../../output/FG_bench/$Name/material.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ../../output/FG_bench/$Name/pattern.pkl \
    --ground_truth /apdcephfs_cq12/share_1150325/jiaminghli/data/FG_OVD/benchmarks/pattern.json \
    --out  ../../output/FG_bench/$Name/pattern.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ../../output/FG_bench/$Name/shuffle_negatives.pkl \
    --ground_truth /apdcephfs_cq12/share_1150325/jiaminghli/data/FG_OVD/benchmarks/shuffle_negatives.json \
    --out  ../../output/FG_bench/$Name/shuffle_negatives.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ../../output/FG_bench/$Name/transparency.pkl \
    --ground_truth /apdcephfs_cq12/share_1150325/jiaminghli/data/FG_OVD/benchmarks/transparency.json \
    --out  ../../output/FG_bench/$Name/transparency.txt \
    --evaluate_all_vocabulary \