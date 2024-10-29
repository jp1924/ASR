import os
import time
from typing import Dict, Optional, Tuple, Union

import torch
from data import DataCollatorForWav2Vec2Pretraining
from datasets import Dataset, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from utils import Wav2Vec2PretrainingArguments, librosa_silence_filter
from wav2vec2_pretrainer import Wav2Vec2Pretrainer

from transformers import (
    HfArgumentParser,
    Wav2Vec2Config,
    Wav2Vec2ForPreTraining,
    Wav2Vec2Processor,
    set_seed,
)
from transformers import logging as hf_logging
from transformers.trainer_utils import is_main_process


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def main(train_args: Wav2Vec2PretrainingArguments) -> None:
    def preprocessor(example):
        audio_ls = example[train_args.audio_column_name]

        finish_length_ls, finish_audio_ls = list(), list()
        for audio in audio_ls:
            audio = librosa_silence_filter(audio["array"])

            if not audio.any():
                continue

            outputs = processor(audio=audio, sampling_rate=train_args.sampling_rate, return_tensors="np")
            audio, length = outputs["input_values"][0], outputs["input_values"][0].shape[0]

            finish_audio_ls.append(audio)
            finish_length_ls.append(length)

        outputs = {
            "input_values": finish_audio_ls,
            train_args.length_column_name: finish_length_ls,
        }
        return outputs

    def length_filter(length_ls):
        return [
            train_args.min_duration_in_seconds <= length <= train_args.max_duration_in_seconds for length in length_ls
        ]

    def prepare_datasets() -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        train_dataset_ls, valid_dataset_ls, test_dataset_ls = list(), list(), list()
        for repo_name in train_args.dataset_repo_ls:
            start_time = time.time()

            if is_main_process(train_args.local_rank):
                logger.info(f"load-{repo_name}")

            data_name = train_args.data_name_map.get(repo_name, None)
            truncate_map = train_args.data_truncate_map.get(repo_name, {})

            datasets = load_dataset(repo_name, data_name)

            map_cache_file_name = None
            filter_cache_file_name = None
            if train_args.cache_file_name:
                name = repo_name.split("/")[-1]
                map_cache_file_name = {
                    x: train_args.cache_dir.joinpath(f"map_{name}-{x}_{train_args.cache_file_name}").as_posix()
                    for x in datasets
                }
                filter_cache_file_name = {
                    x: train_args.cache_dir.joinpath(
                        f"filter_{train_args.min_duration_in_seconds}-{train_args.max_duration_in_seconds}_{name}-{x}_{train_args.cache_file_name}"
                    ).as_posix()
                    for x in datasets
                }

            # DatasetsDict이라서 이런식으로 해줘야 함.
            datasets = datasets.map(
                preprocessor,
                num_proc=train_args.preprocessing_num_workers,
                load_from_cache_file=True,
                batched=train_args.preprocessing_batched,
                cache_file_names=map_cache_file_name,
                batch_size=train_args.preprocessing_batch_size,
                remove_columns=set(sum(datasets.column_names.values(), [])),
                desc=f"preprocess-{repo_name}",
            )

            datasets = datasets.filter(
                length_filter,
                num_proc=train_args.preprocessing_num_workers,
                input_columns=[train_args.length_column_name],
                cache_file_names=filter_cache_file_name,
                batched=train_args.preprocessing_batched,
                batch_size=train_args.preprocessing_batch_size,
                desc=f"length-filtering-{repo_name}",
            )

            for data_type in truncate_map:
                truncate_size = truncate_map[data_type]
                data = datasets[data_type].shuffle()
                if len(data) <= truncate_size:
                    if is_main_process(train_args.local_rank):
                        logger.info(
                            f"{repo_name}의 {data_type}크기는 {len(data)}이지만"
                            f"truncate_size는 {truncate_size} 크기를 조절하셈."
                        )
                    continue

                datasets[data_type] = data.select(range(truncate_size))

            if is_main_process(train_args.local_rank):
                logger.info(datasets)
                logger.info(f"{repo_name}-load time: {time.time() - start_time}")

            for dataset_key in datasets:
                dataset = None
                if dataset_key in train_args.train_dataset_prefix and train_args.do_train:
                    dataset = datasets[dataset_key]
                    train_dataset_ls.append(dataset)

                if dataset_key in train_args.valid_dataset_prefix and train_args.do_eval:
                    dataset = datasets[dataset_key]
                    valid_dataset_ls.append(dataset)

                if dataset_key in train_args.test_dataset_prefix and train_args.do_predict:
                    dataset = datasets[dataset_key]
                    test_dataset_ls.append(dataset)

                if dataset and is_main_process(train_args.local_rank):
                    length_ls = sorted(dataset[train_args.length_column_name], reverse=True)
                    logger.info(f"{repo_name}/{dataset_key}-length: {length_ls[:100]}")
                    logger.info(
                        f"{dataset_key}_total_hour: {(sum(length_ls) / train_args.sampling_rate) / 60**2:.2f}h"
                    )

        train_dataset = None
        if train_dataset_ls:
            train_dataset = concatenate_datasets(train_dataset_ls)
            train_dataset.set_format("pt")
            if is_main_process(train_args.local_rank):
                logger.info(f"train_dataset:\n{train_dataset}")

        valid_dataset = None
        if valid_dataset_ls:
            valid_dataset = concatenate_datasets(valid_dataset_ls)
            valid_dataset.set_format("pt")
            if is_main_process(train_args.local_rank):
                logger.info(f"valid_dataset:\n{valid_dataset}")

        test_dataset = None
        if test_dataset_ls:
            test_dataset = concatenate_datasets(test_dataset_ls)
            test_dataset.set_format("pt")
            if is_main_process(train_args.local_rank):
                logger.info(f"test_dataset:\n{test_dataset}")

        return (train_dataset, valid_dataset, test_dataset)

    model_name_or_path = train_args.resume_from_checkpoint or train_args.model_name_or_path
    config = Wav2Vec2Config.from_pretrained(
        model_name_or_path,
        attn_implementation=train_args.attn_implementation,
    )
    model = Wav2Vec2ForPreTraining.from_pretrained(model_name_or_path, config=config)
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path)

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    # load dataset & preprocess
    train_dataset, valid_dataset, test_dataset = prepare_datasets()

    # set collator
    collator = DataCollatorForWav2Vec2Pretraining(
        model=model,
        feature_extractor=processor.feature_extractor,
        pad_to_multiple_of=train_args.pad_to_multiple_of,
        mask_time_prob=config.mask_time_prob,
        mask_time_length=config.mask_time_length,
        mask_time_min_masks=config.mask_time_min_masks,
    )

    # set trainer
    trainer = Wav2Vec2Pretrainer(
        model=model,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=processor,
        args=train_args,
    )

    if train_args.do_train and train_dataset:
        train(trainer, train_args)

    if train_args.do_eval and valid_dataset:
        valid(trainer)

    if train_args.do_predict and test_dataset:
        logger.info("do_predict 코드는 아직 작성 중")


def train(trainer: Wav2Vec2Pretrainer, train_args: Wav2Vec2PretrainingArguments) -> None:
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)


@torch.no_grad()
def valid(trainer: Wav2Vec2Pretrainer, valid_datasets: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


if __name__ == "__main__":
    parser = HfArgumentParser([Wav2Vec2PretrainingArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if remain_args and is_main_process(train_args.local_rank):
        logger.info(f"remain_args: {remain_args}")

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    main(train_args)
