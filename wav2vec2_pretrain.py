#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

""" Pre-Training a 🤗 Wav2Vec2 model on unlabeled audio data """

import os
from typing import Any, Dict, List, Optional, Union

import torch
from data import DataCollatorForWav2Vec2Pretraining
from datasets import Dataset, concatenate_datasets, load_dataset
from models import Wav2Vec2ForPreTraining
from setproctitle import setproctitle

# Wav2Vec2ForPreTraining,
from transformers import (
    HfArgumentParser,
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    is_torch_xla_available,
    is_wandb_available,
)
from transformers import logging as hf_logging
from transformers import set_seed
from utils import (
    Wav2Vec2PretrainingArguments,
    default_sentence_norm,
    get_feat_extract_output_lengths,
    librosa_silence_filter,
)
from wav2vec2_pretrainer import Wav2Vec2Pretrainer

hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")

global GLOBAL_LOGGER
GLOBAL_LOGGER = None


def main(train_args: Wav2Vec2PretrainingArguments):
    def preprocessor(example: Dict[str, Union[List[Any], List[List[Any]]]]) -> Dict[str, List[Any]]:
        sentence_ls = example["sentence"]
        sentence_ls = sentence_ls if isinstance(sentence_ls, list) else [sentence_ls]

        audio_ls = example["audio"]
        audio_ls = audio_ls if isinstance(audio_ls, list) else [audio_ls]
        audio_ls = [audio["array"] for audio in audio_ls]

        normalized_sentence_ls = list()
        normalized_audio_ls = list()
        length_ls = list()
        for sentence, audio in zip(sentence_ls, audio_ls):
            audio = librosa_silence_filter(audio)
            audio_length = audio.shape[0]

            if not audio.any():
                continue

            if not train_args.min_duration_in_seconds <= audio_length <= train_args.max_duration_in_seconds:
                continue

            sentence = default_sentence_norm(sentence)
            if not sentence:
                continue

            sentence = tokenizer(sentence, return_attention_mask=False)["input_ids"]
            label_length = len(sentence)

            # NOTE: for CTC loss
            feature_length = get_feat_extract_output_lengths(audio_length, config)
            if label_length > feature_length:
                continue

            audio = feature_extractor(audio, sampling_rate=16000)["input_values"]

            normalized_sentence_ls.append(sentence)
            normalized_audio_ls.append(audio)
            length_ls.append(audio_length)

        return {
            "labels": normalized_sentence_ls,
            "input_values": normalized_audio_ls,
            "length": length_ls,
        }

    def collect_dataset(prefix_ls: List[str]) -> Dataset:
        data_ls = list()
        for prefix in prefix_ls:
            check_key: str = lambda key: (prefix in key)
            filter_data = [concatenate_datasets(data_dict.pop(key)) for key in list(data_dict.keys()) if check_key(key)]
            data_ls.extend(filter_data)
        return concatenate_datasets(data_ls)

    def set_wandb() -> None:
        # TODO: 이건 나중에 args로 바꿀 것
        GLOBAL_LOGGER.run.log_code(
            "/root/workspace",
            include_fn=lambda path: path.endswith(".py") or path.endswith(".json"),
        )
        # logging args
        combined_dict = {**train_args.to_dict()}
        if hasattr(model, "config") and model.config is not None:
            model_config = model.config.to_dict()
            combined_dict = {**model_config, **combined_dict}

        GLOBAL_LOGGER.config.update(combined_dict, allow_val_change=True)

        # set default metrics
        if getattr(GLOBAL_LOGGER, "define_metric", None):
            GLOBAL_LOGGER.define_metric("train/global_step")
            GLOBAL_LOGGER.define_metric("*", step_metric="train/global_step", step_sync=True)

        # set model watch
        _watch_model = os.getenv("WANDB_WATCH", "false")
        if not is_torch_xla_available() and _watch_model in ("all", "parameters", "gradients"):
            GLOBAL_LOGGER.watch(model, log=_watch_model, log_freq=max(100, train_args.logging_steps))
        GLOBAL_LOGGER.run._label(code="transformers_trainer")

    # load model, feature_extractor, tokenizer
    config = Wav2Vec2Config.from_pretrained(
        train_args.model_name_or_path,
        attn_implementation=train_args.attn_implementation,
    )
    model = Wav2Vec2ForPreTraining(config)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(train_args.model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(train_args.model_name_or_path)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    # for vscode intellisence
    model: Wav2Vec2ForPreTraining
    config: Wav2Vec2Config
    feature_extractor: Wav2Vec2FeatureExtractor
    tokenizer: Wav2Vec2CTCTokenizer
    processor: Wav2Vec2Processor

    # NOTE: Trainer에서 자동으로 해줌, 하지만 확인을 위해 이렇게 선언 함.
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # set logger
    if GLOBAL_LOGGER and (train_args.local_rank == 0):
        set_wandb()

    # load dataset & preprocess
    data_dict = dict()
    for dataset_name in train_args.dataset_names:
        logger.info(f"load-{dataset_name}")
        dataset = load_dataset(dataset_name)

        # DatasetDict이라서 이런식으로 해줘야 함.
        column_names = set(sum(dataset.column_names.values(), []))
        with train_args.main_process_first(desc="data preprocess"):

            cache_file_name = None
            if train_args.cache_file_name:
                get_cache_path: str = lambda x: os.path.join(
                    train_args.cache_dir, f"{name}-{x}_{train_args.cache_file_name}"
                )
                name = dataset_name.split("/")[-1]
                cache_file_name = {x: get_cache_path(x) for x in dataset}

            # NOTE: finetune에서 사용할 데이터 Pretrain에서 전처리 함
            # 만약 순수 음성만 넣을 거라면 sentence 부분을 ""로 비워든 상태로 돌리면 정상적으로 진행 됨
            dataset = dataset.map(
                preprocessor,
                num_proc=train_args.preprocessing_num_workers,
                load_from_cache_file=True,
                batched=train_args.preprocessing_batched,
                cache_file_names=cache_file_name,
                batch_size=train_args.preprocessing_batch_size,
                remove_columns=column_names,
                desc=f"preprocess-{dataset_name}",
            )
        for data_key in dataset:
            if data_key not in data_dict:
                data_dict[data_key] = []

            specific_dataset = dataset[data_key]

            added_data = [f"{dataset_name}-{data_key}"] * len(specific_dataset)
            specific_dataset = specific_dataset.add_column("dataset_name", added_data)

            data_dict[data_key].append(specific_dataset)

    train_dataset = None
    check_containe_dataset = any([x in data_dict for x in train_args.train_dataset_prefix])
    if train_args.do_train and check_containe_dataset:
        train_dataset = collect_dataset(train_args.train_dataset_prefix)
        train_dataset.set_format("torch")
        logger.info("train_dataset")
        logger.info(train_dataset)

    valid_dataset = None
    check_containe_dataset = any([x in data_dict for x in train_args.valid_dataset_prefix])
    if train_args.do_eval and check_containe_dataset:
        valid_dataset = collect_dataset(train_args.valid_dataset_prefix)
        valid_dataset.set_format("torch")
        logger.info("valid_dataset")
        logger.info(valid_dataset)

    test_dataset = None
    check_containe_dataset = any([x in data_dict for x in train_args.test_dataset_prefix])
    if train_args.do_predict and check_containe_dataset:
        test_dataset = collect_dataset(train_args.test_dataset_prefix)
        test_dataset.set_format("torch")
        logger.info("test_dataset")
        logger.info(test_dataset)

    # 6898663 >> 3.3% 정도 필터링 됨. 여기엔 tokenizing할 수 없는 문자, 음성 길이가 맞지 않는 문자 등 여러 요인으로 인해 필터링 된 데이터가 포함
    # 7136987
    # 238324

    valid_dataset_dict = dict()
    valid_name_ls = valid_dataset["dataset_name"]
    for dataset_name in set(valid_name_ls):
        part_idx = [idx for idx, x in enumerate(valid_name_ls) if x == dataset_name]
        part_dataset = valid_dataset.select(part_idx, keep_in_memory=False)

        # 'jp1924/KconfSpeech-validation'
        start = dataset_name.rindex("/") + 1
        end = dataset_name.rindex("-")

        valid_dataset_dict[dataset_name[start:end]] = part_dataset

    # set collator
    collator = DataCollatorForWav2Vec2Pretraining(
        model=model,
        feature_extractor=feature_extractor,
        pad_to_multiple_of=train_args.pad_to_multiple_of,
        mask_time_prob=train_args.mask_time_prob or config.mask_time_prob,
        mask_time_length=train_args.mask_time_length or config.mask_time_length,
        mask_time_min_masks=config.mask_time_min_masks,
    )

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
        )

    # set trainer
    trainer = Wav2Vec2Pretrainer(
        model=model,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset_dict,
        tokenizer=processor,
        args=train_args,
    )
    if train_args.do_train and train_dataset:
        train(trainer)

    if train_args.do_eval and valid_dataset:
        valid(trainer)

    if train_args.do_predict and test_dataset:
        predict(trainer, test_dataset)


def train(trainer: Wav2Vec2Pretrainer) -> None:
    train_args: Wav2Vec2PretrainingArguments = trainer.args
    outputs = trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)
    # trainer 특성 때문에 save_metrics 안됨.


@torch.no_grad()
def valid(trainer: Wav2Vec2Pretrainer, valid_datasets: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    metrics = trainer.evaluate(valid_datasets)


@torch.no_grad()
def predict(trainer: Wav2Vec2Pretrainer, test_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    test_dataset_dict = dict()
    test_name_ls = test_dataset["dataset_name"]
    for dataset_name in set(test_name_ls):
        part_idx = [idx for idx, x in enumerate(test_name_ls) if x == dataset_name]
        part_dataset = test_dataset.select(part_idx, keep_in_memory=False)

        # 'jp1924/KconfSpeech-validation'
        start = dataset_name.rindex("/") + 1
        end = dataset_name.rindex("-")

        outputs = trainer.predict(part_dataset, metric_key_prefix=f"test/{dataset_name[start:]}")
        if GLOBAL_LOGGER:
            GLOBAL_LOGGER.log(outputs.metrics)
        test_dataset_dict[dataset_name[start:end]] = part_dataset


if __name__ == "__main__":
    parser = HfArgumentParser([Wav2Vec2PretrainingArguments])
    train_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    check_wandb = ("wandb" in train_args.report_to) and (train_args.local_rank == 0)
    if is_wandb_available() and check_wandb:
        import wandb

        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            group=os.getenv("WANDB_RUN_GROUP"),
            name=train_args.run_name,
            save_code=True,
        )
        GLOBAL_LOGGER = wandb

    main(train_args)

    if GLOBAL_LOGGER:
        GLOBAL_LOGGER.finish()
        GLOBAL_LOGGER.finish()
