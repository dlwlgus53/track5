#!/bin/bash

# training and validation for turn detection
model_name=microsoft/deberta-v3-base
model_name_exp=deberta-v3-base
cuda_id=0

CUDA_VISIBLE_DEVICES=${cuda_id} python baseline.py \
        --task detection \
        --dataroot data \
        --model_name_or_path ${model_name} \
        --params_file baseline/configs/detection/params.json \
        --exp_name td-review-${model_name_exp}-baseline \
        --knowledge_file knowledge.json

