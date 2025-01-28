export WANDB_PROJECT="STT"
export WANDB_RUN_GROUP="Wav2Vec2-finetune"

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export OMP_NUM_THREADS=2

    # --lr_scheduler_type="cosine" \

deepspeed --include=localhost:0,1,2,3 --master_port=3218 \
    '/root/workspace/src/finetune_ctc.py' \
    --output_dir="/root/output_dir/wav2vec2/ksponspeech/pack" \
    --cache_dir="/root/.cache/.[KoWav2Vec2Base]preprocessor/finetune" \
    --model_name_or_path="/root/output_dir/wav2vec2/ksponspeech/fix-pack-2/checkpoint-11811" \
    --run_name="wav2vec2-base" \
    --data_preprocessor_type="wav2vec2_finetune_ctc" \
    --audio_min_seq=1 \
    --audio_max_seq=512 \
    --per_device_train_batch_size=32 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=2 \
    --eval_accumulation_steps=1 \
    --preprocessing_num_workers=10 \
    --do_data_main_process_first=true \
    --num_train_epochs=3 \
    --seed=42 \
    --do_train=true \
    --do_eval=true \
    --do_predict=false \
    --report_to='wandb' \
    --learning_rate=0.0005 \
    --warmup_ratio=0.4 \
    --lr_scheduler_type='tri_stage' \
    --lr_scheduler_kwargs='{"num_hold_steps":0.1,"num_decay_steps":0.5,"final_learning_rate":0.00001}' \
    --weight_decay=0.01 \
    --eval_strategy=steps \
    --eval_steps=1000 \
    --save_strategy=no \
    --logging_strategy=steps \
    --logging_steps=1 \
    --bf16=true \
    --tf32=false \
    --attn_implementation="flash_attention_2" \
    --dataset_repo_ls \
        jp1924/KsponSpeech \
    --train_dataset_prefix \
        train \
    --valid_dataset_prefix \
        validation \
        dev \
    --test_dataset_prefix \
        eval_clean \
        eval_other \
    --torch_compile=true \
    --dataloader_num_workers=4 \
    --dataloader_pin_memory=true \
    --dataloader_prefetch_factor=5 \
    --ddp_timeout=1800000000 \
    --torch_empty_cache_steps=100 \
    --include_tokens_per_second=true \
    --include_num_input_tokens_seen=true \
    --remove_unused_columns=false \
    --config_kwargs='{"ctc_loss_reduction": "mean"}' \
    --deepspeed="/root/workspace/config/ZeRO_2_act_check.json" 

