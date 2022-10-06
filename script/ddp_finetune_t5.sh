#!/bin/bash

script_path = ""

model_name_or_path = ""
checkpoint_path = ""
data_name_or_script = ""
output_dir = ""
cache_dir = ""

export CUDA_VISIBLE_DEVICES = ""
export WANDB_DISABLED = ""
export WANDB_PROJECT = ""
export WANDB_ENTITY = ""
export WANDB_CACHE_DIR = cache_dir
export WANDB_USERNAME = ""
export WANDB_RUN_GROUP = ""
export WANDB_TAGS = ""
export WANDB_DISABLE_CODE = ""
export WANDB_RESUME = ""
export WANDB_RUN_ID = ""

python torch.distributed.launch \
    --standalone \
    --nnodes = 1 \
    --nproc_per_node = 3 \
    $script_path \
    --model_name_or_path = $model_name_or_path \
    --resume_from_checkpoint = $checkpoint_path
    --data_name_or_script = $data_name_or_script \
    --output_dir = $output_dir \
    --cache = $cache_dir \
    --per_device_train_batch_size = 12 \
    --per_device_eval_batch_size = 4 \
    --gradient_accumulation_steps = 2 \
    --eval_accumulation_steps = 2 \
    --learning_rate = 2e-5 \
    --warmup_steps = 1000 \
    --num_train_epochs = 2 \
    --lr_scheduler_type = "linear" \
    --logging_strategy = "steps" \
    --logging_steps = 50 \
    --evaluation_strategy = "steps" \
    --eval_steps = 1000 \
    --save_strategy = "steps" \
    --save_steps = 1000 \
    --do_train = true \
    --do_eval = true \
    --predict_with_generate = false
    --num_proc = 10