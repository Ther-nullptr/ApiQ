#!/bin/bash

model_dir=/home/yujin-wa20/projects/LoftQ/model_zoo/loftq
model_name=Mistral-7B-v0.1-4bit-16rank
model_name_full=${model_dir}/${model_name}

TASK=math # or "math" for arithmetic reasoning
MODEL=$model_name_full
DATA_DIR=./dataset
OUTPUT_DIR=./results/${TASK}/${model_name}
TOOL=./train_multitask.py

mkdir -p $OUTPUT_DIR
python $TOOL \
    --do_train \
    --do_eval \
    --model_name_or_path $MODEL \
    --task $TASK \
    --data_dir $DATA_DIR \
    --test_split test \
    --use_normalized_template \
    --max_length 512 \
    --seed 42 \
    --learning_rate 3e-4 \
    --max_grad_norm 1 \
    --num_train_epochs 3 \
    --gradient_accumulation_steps 16 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --warmup_ratio 0.1 \
    --greedy_decoding \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --disable_tqdm false \
    --report_to "none" \
    --remove_unused_columns false \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir 2>&1 | tee $OUTPUT_DIR/out