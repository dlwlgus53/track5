#!/bin/bash


# training and validation for knowledge selection
conda activate track5
model_name=microsoft/deberta-v3-base
model_name_exp=deberta-v3-base_baseline
cuda_id=1

CUDA_VISIBLE_DEVICES=${cuda_id} python baseline.py \
        --task selection \
        --dataroot data \
        --model_name_or_path ${model_name} \
        --negative_sample_method "oracle" \
        --knowledge_file knowledge.json \
        --params_file baseline/configs/selection/params.json \
        --exp_name ks-review-${model_name_exp}-oracle-baseline

