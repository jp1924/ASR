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
from typing import Any, Dict, List, Union

import torch
from accelerate.logging import get_logger
from data import DataCollatorForWav2Vec2Pretraining
from datasets import Dataset, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from transformers import (
    HfArgumentParser,
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Processor,
    is_wandb_available,
    set_seed,
)
from utils import (
    Wav2Vec2PretrainingArguments,
    default_sentence_norm,
    get_feat_extract_output_lengths,
    librosa_silence_filter,
)
from wav2vec2_pretrainer import Wav2Vec2Pretrainer

logger = get_logger(__name__)


def main(train_args: Wav2Vec2PretrainingArguments):
    def preprocessor(audio_ls, sentence_ls) -> Dict[str, List[Any]]:
        sentence_ls = sentence_ls if isinstance(sentence_ls, list) else [sentence_ls]

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

    def collect_dataset(prefix_ls: List[str]) -> List[Dataset]:
        data_ls = list()
        for prefix in prefix_ls:
            check_key: str = lambda key: (prefix in key)
            filter_data = [data_dict.pop(key) for key in list(data_dict.keys()) if check_key(key)]
            data_ls.extend(filter_data)
        return data_ls

    config = Wav2Vec2Config.from_pretrained(train_args.model_name_or_path)
    model = Wav2Vec2ForPreTraining(config)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(train_args.model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(train_args.model_name_or_path)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    # NOTE: Trainer에서 자동으로 해줌, 하지만 확인을 위해 이렇게 선언 함.
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_dict = dict()
    for dataset in train_args.dataset_names:
        dataset = load_dataset(dataset)
        with train_args.main_process_first(desc="data preprocess"):
            dataset = dataset.map(
                preprocessor,
                num_proc=train_args.preprocessing_num_workers,
                input_columns=[train_args.audio_column_name, train_args.sentence_column_name],
                load_from_cache_file=True,
                batched=train_args.preprocessing_batched,
                batch_size=train_args.preprocessing_batch_size,
            )
        for data_key in dataset:
            if data_key not in data_dict:
                data_dict[data_key] = []

            data_dict[data_key].append(dataset)

    train_dataset = collect_dataset(train_args.train_dataset_prefix)
    valid_dataset = collect_dataset(train_args.valid_dataset_prefix)
    test_dataset = collect_dataset(train_args.test_dataset_prefix)

    collator = DataCollatorForWav2Vec2Pretraining(
        model=model,
        feature_extractor=feature_extractor,
        pad_to_multiple_of=train_args.pad_to_multiple_of,
        mask_time_prob=train_args.mask_time_prob,
        mask_time_length=train_args.mask_time_length,
    )

    trainer = Wav2Vec2Pretrainer(
        model=model,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor,
        args=train_args,
    )
    if train_args.do_train:
        train(trainer)
    if train_args.do_eval:
        valid(trainer)
    if train_args.do_predict:
        predict(trainer, test_dataset)


def train(trainer: Wav2Vec2Pretrainer) -> None:
    trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)

    save_dir = os.path.join(trainer.args.output_dir, "last_model")
    trainer.save_model(save_dir)
    trainer.save_metrics()
    trainer.save_state()


@torch.no_grad()
def valid(trainer: Wav2Vec2Pretrainer) -> None:
    pass


@torch.no_grad()
def predict(trainer: Wav2Vec2Pretrainer, test_dataset: Dataset) -> None:
    pass


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

    main(train_args)
