export TORCH_DISTRIBUTED_DEBUG="DETAIL"
export TORCHDYNAMO_DISABLE="1"

torchrun --nproc_per_node=4 \
    /root/workspace/wav2vec2_pretrain.py \
    --output_dir=/root/output_dir \
    --run_name=wav2vec2-pretrain-debug \
    --model_name_or_path=/root/model \
    --preprocessing_num_workers=10 \
    --per_device_train_batch_size=6 \
    --gradient_accumulation_steps=1 \
    --per_device_eval_batch_size=2 \
    --overwrite_cache=false \
    --cache_dir=false \
    --seed=42 \
    --do_train=true \
    --do_eval=true \
    --do_predict=true \
    --report_to=none \
    --learning_rate=5e-4 \
    --warmup_ratio=0.3 \
    --weight_decay=0.01 \
    --evaluation_strategy=no \
    --save_strategy=no \
    --logging_strategy=steps \
    --logging_steps=1 \
    --fp16=true \
    --dataset_names \
        jp1924/KsponSpeech \
    --train_dataset_prefix=train \
    --valid_dataset_prefix validation dev \
    --test_dataset_prefix eval_clean eval_other \
    --cache_file_name=preprocessor.arrow \
    --cache_dir=/root/.preprocessor_cache_dir \
    --gradient_checkpointing=true \
    --remove_unused_columns=false \
    --group_by_length=false \
    --torch_compile=true