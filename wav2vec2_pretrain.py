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

import datasets
import torch
from accelerate.logging import get_logger
from data import DataCollatorForWav2Vec2Pretraining
from datasets import DatasetDict, concatenate_datasets, load_dataset
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

AUDIO_MIN_LENGTH = os.getenv("AUDIO_MIN_LENGTH", 3584)  # 이거 어떻게 계산 하더라?
AUDIO_MAX_LENGTH = os.getenv("AUDIO_MAX_LENGTH", 448512)  # 이거 어떻게 계산 하더라?


def main(train_args: Wav2Vec2PretrainingArguments):
    def preprocessor(example):
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

            if not AUDIO_MIN_LENGTH <= audio_length <= AUDIO_MAX_LENGTH:
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

    if is_wandb_available() and (("wandb" in train_args.report_to) and (train_args.local_rank == 0)):
        import wandb

        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            group=os.getenv("WANDB_RUN_GROUP"),
            name=train_args.run_name,
            save_code=True,
        )
    config = Wav2Vec2Config.from_pretrained(train_args.model_name_or_path)
    model = Wav2Vec2ForPreTraining(config)

    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(train_args.model_name_or_path)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(train_args.model_name_or_path)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

    # NOTE: Trainer에서 자동으로 해줌, 하지만 확인을 위해 이렇게 선언 함.
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

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
        tokenizer=processor,
        args=train_args,
    )
    if train_args.do_train:
        train(trainer)
    if train_args.do_eval:
        predict(trainer)
    if train_args.do_predict:
        predict(trainer)


def train(trainer: Wav2Vec2Pretrainer) -> None:
    trainer.train(resume_from_checkpoint=trainer.args.resume_from_checkpoint)

    save_dir = os.path.join(trainer.args.output_dir, "last_model")
    trainer.save_model(save_dir)
    trainer.save_metrics()
    trainer.save_state()


def predict(trainer: Wav2Vec2Pretrainer) -> None:
    pass


if __name__ == "__main__":
    parser = HfArgumentParser([Wav2Vec2PretrainingArguments])
    train_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    main(train_args)
