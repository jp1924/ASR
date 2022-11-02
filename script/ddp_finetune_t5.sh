#!/bin/bash

script_path=""

train_data=""
valid_data=""

model_name_or_path=""
checkpoint_path=""
output_dir=""
cache_dir=""

num_gpu = 1

export CUDA_VISIBLE_DEVICES=""
export WANDB_DISABLED=""
export WANDB_PROJECT=""
export WANDB_ENTITY=""
export WANDB_CACHE_DIR=$cache_dir
export WANDB_USERNAME=""
export WANDB_RUN_GROUP=""
export WANDB_TAGS=""
export WANDB_DISABLE_CODE=""
# export WANDB_RESUME=""
# export WANDB_RUN_ID=""
export OMP_NUM_THREADS=8

# --resume_from_checkpoint=$checkpoint_path \
python -m torch.distributed.launch --standalone --nnodes=1 --nproc_per_node=$num_gpu \
    $script_path \
    --run_name="" \
    --model_name=$model_name_or_path \
    --train_csv=$train_data \
    --valid_csv=$valid_data \
    --output_dir=$output_dir \
    --cache=$cache_dir \
    --per_device_train_batch_size= \
    --per_device_eval_batch_size= \
    --gradient_accumulation_steps= \
    --eval_accumulation_steps= \
    --learning_rate= \
    --warmup_steps= \
    --num_train_epochs= \
    --lr_scheduler_type= \
    --logging_strategy= \
    --logging_steps= \
    --evaluation_strategy= \
    --eval_steps= \
    --save_strategy= \
    --save_steps= \
    --do_train true \
    --do_eval true \
    --group_by_length true \
    --fp16 true \
    --num_proc=