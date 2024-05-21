export WANDB_WATCH="none"
export WANDB_DISABLED="false"

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"

export OMP_NUM_THREADS=2
torchrun --nproc_per_node=4 \
    /root/workspace/wav2vec2_pretrain.py \
    --output_dir=/root/output_dir/pretrain \
    --run_name=wav2vec2-pretrain \
    --model_name_or_path=/root/init_model \
    --preprocessing_num_workers=20 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --per_device_eval_batch_size=2 \
    --overwrite_cache=false \
    --cache_dir=false \
    --num_train_epochs=10 \
    --seed=42 \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --report_to=wandb \
    --learning_rate=0.001 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.4 \
    --weight_decay=0.01 \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --eval_steps=10000 \
    --save_steps=10000 \
    --logging_strategy=steps \
    --logging_steps=1 \
    --fp16=true \
    --dataset_names jp1924/KsponSpeech jp1924/KoreanSpeech jp1924/KconfSpeech jp1924/KrespSpeech jp1924/MeetingSpeech \
    --train_dataset_prefix train \
    --valid_dataset_prefix dev validation \
    --test_dataset_prefix eval_clean eval_other \
    --cache_file_name=preprocessor.arrow \
    --cache_dir=/root/.cache/.preprocessor_cache_dir \
    --gradient_checkpointing=false \
    --remove_unused_columns=true \
    --group_by_length=true \
    --torch_compile=true
    