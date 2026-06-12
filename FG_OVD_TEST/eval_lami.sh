python eval_map.py \
    --predictions ./FG_results/FG_no_ensemble/lami_3_attributes.pkl \
    --ground_truth ../../FG-OVD-main/benchmarks/3_attributes.json \
    --out ./FG_results/FG_no_ensemble/lami_3_attributes.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ./FG_results/FG_no_ensemble/lami_1_attributes.pkl \
    --ground_truth ../../FG-OVD-main/benchmarks/1_attributes.json \
    --out ./FG_results/FG_no_ensemble/lami_1_attributes.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ./FG_results/FG_no_ensemble/lami_2_attributes.pkl \
    --ground_truth ../../FG-OVD-main/benchmarks/2_attributes.json \
    --out ./FG_results/FG_no_ensemble/lami_2_attributes.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ./FG_results/FG_no_ensemble/lami_color.pkl \
    --ground_truth ../../FG-OVD-main/benchmarks/color.json \
    --out ./FG_results/FG_no_ensemble/lami_color.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ./FG_results/FG_no_ensemble/material.pkl \
    --ground_truth ../../FG-OVD-main/benchmarks/material.json \
    --out ./FG_results/FG_no_ensemble/material.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ./FG_results/FG_no_ensemble/pattern.pkl \
    --ground_truth ../../FG-OVD-main/benchmarks/pattern.json \
    --out ./FG_results/FG_no_ensemble/pattern.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ./FG_results/FG_no_ensemble/shuffle_negatives.pkl \
    --ground_truth ../../FG-OVD-main/benchmarks/shuffle_negatives.json \
    --out ./FG_results/FG_no_ensemble/shuffle_negatives.txt \
    --evaluate_all_vocabulary \

python eval_map.py \
    --predictions ./FG_results/FG_no_ensemble/transparency.pkl \
    --ground_truth ../../FG-OVD-main/benchmarks/transparency.json \
    --out ./FG_results/FG_no_ensemble/transparency.txt \
    --evaluate_all_vocabulary \