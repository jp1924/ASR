export WANDB_PROJECT="Wav2Vec2"
export WANDB_RUN_GROUP="finetune"

export WANDB_WATCH="none"
export WANDB_DISABLED="false"
export WANDB_DISABLE_CODE="false"

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"
export CUDA_VISIBLE_DEVICES="0"

export OMP_NUM_THREADS=2
python3 \
    /root/workspace/wav2vec2_finetune.py \
    --output_dir=/root/output_dir/finetune \
    --run_name=wav2vec2-finetune \
    --model_name_or_path=/root/output_dir/pretrain/last_model \
    --preprocessing_num_workers=20 \
    --per_device_train_batch_size=8 \
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
    --lr_scheduler_type=tri_stage \
    --warmup_ratio=0.4 \
    --weight_decay=0.01 \
    --evaluation_strategy=steps \
    --save_strategy=steps \
    --eval_steps=5000 \
    --save_steps=5000 \
    --logging_strategy=steps \
    --logging_steps=1 \
    --fp16=true \
    --dataset_names jp1924/KsponSpeech jp1924/KoreanSpeech jp1924/KconfSpeech jp1924/KrespSpeech jp1924/MeetingSpeech jp1924/BroadcastSpeech \
    --train_dataset_prefix train \
    --valid_dataset_prefix validation dev \
    --test_dataset_prefix eval_clean eval_other \
    --cache_file_name=preprocessor.arrow \
    --cache_dir=/root/.cache/.preprocessor_cache_dir \
    --gradient_checkpointing=false \
    --remove_unused_columns=true \
    --group_by_length=true \
    --torch_compile=true \
    --lr_scheduler_kwargs='{"num_hold_steps":0.1,"num_decay_steps":0.5,"final_learning_rate":0.00001}' \