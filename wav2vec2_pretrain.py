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
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForPreTraining,
    is_wandb_available,
    set_seed,
)
from utils import Wav2Vec2PretrainingArguments
from wav2vec2_pretrainer import Wav2Vec2Pretrainer

logger = get_logger(__name__)


def main(train_args: Wav2Vec2PretrainingArguments):

    if is_wandb_available() and (("wandb" in train_args.report_to) and (train_args.local_rank == 0)):
        import wandb

        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            entity=os.getenv("WANDB_ENTITY"),
            group=os.getenv("WANDB_RUN_GROUP"),
            name=train_args.run_name,
            save_code=True,
        )

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(train_args.model_name_or_path)
    config = Wav2Vec2Config.from_pretrained(train_args.model_name_or_path)
    model = Wav2Vec2ForPreTraining(config)

    # NOTE: Trainer에서 자동으로 해줌, 하지만 확인을 위해 이렇게 선언 함.
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    collator = DataCollatorForWav2Vec2Pretraining(
        model=model,
        feature_extractor=feature_extractor,
        padding=train_args.padding,
        pad_to_multiple_of=train_args.pad_to_multiple_of,
        mask_time_prob=train_args.mask_time_prob,
        mask_time_length=train_args.mask_time_length,
    )

    trainer = Wav2Vec2Pretrainer(
        model=model,
        data_collator=collator,
        tokenizer=feature_extractor,
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
