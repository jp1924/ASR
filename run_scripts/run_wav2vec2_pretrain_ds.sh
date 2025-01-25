export WANDB_PROJECT="STT"
export WANDB_RUN_GROUP="Wav2Vec2-pretrain"

export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"
# export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

export OMP_NUM_THREADS=2

    # --auto_find_batch_size=true \
deepspeed --include=localhost:0,1,2,3 --master_port=3218 \
    '/root/workspace/src/pretrain.py' \
    --output_dir='/root/output_dir/wav2vec2/ksponspeech/pack' \
    --cache_dir='/root/.cache/.[KoWav2Vec2Base]preprocessor/pretraining' \
    --model_name_or_path='jp1924/KoWav2Vec2Base' \
    --run_name='packed-wav2vec2' \
    --audio_min_seq=1 \
    --audio_max_seq=512 \
    --per_device_train_batch_size=12 \
    --preprocessing_num_workers=10 \
    --do_data_main_process_first=true \
    --data_preprocessor_type="wav2vec2-pretrain" \
    --seed=42 \
    --do_train=true \
    --do_eval=true \
    --num_train_epochs=3 \
    --do_predict=false \
    --lr_scheduler_type='cosine' \
    --report_to='none' \
    --learning_rate=3e-4 \
    --warmup_ratio=0.3 \
    --weight_decay=0.001 \
    --eval_strategy='steps' \
    --eval_steps=500 \
    --save_strategy='epoch' \
    --logging_strategy='steps' \
    --logging_steps=1 \
    --bf16=true \
    --tf32=true \
    --attn_implementation="flash_attention_2" \
    --dataset_repo_ls \
        'jp1924/KsponSpeech' \
    --train_dataset_prefix \
        'train' \
    --valid_dataset_prefix \
        'validation' \
        'dev' \
    --test_dataset_prefix \
        'eval_clean' \
        'eval_other' \
    --torch_compile=true \
    --do_packing=true \
    --packing_max_elem=20 \
    --dataloader_num_workers=4 \
    --dataloader_pin_memory=true \
    --dataloader_prefetch_factor=10 \
    --ddp_timeout=1800000000 \
    --torch_empty_cache_steps=100 \
    --deepspeed='/root/workspace/config/ZeRO_2_act_check.json'