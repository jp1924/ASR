import os
import time
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from data_processor import wav2vec2_ctc_finetune_preprocessor
from datasets import Dataset, concatenate_datasets, load_dataset
from evaluate import load
from setproctitle import setproctitle
from trainer import DataCollatorCTCWithPadding

from transformers import (
    AutoConfig,
    AutoModelForCTC,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2Processor,
    set_seed,
)
from transformers import logging as hf_logging
from transformers.trainer_utils import EvalPrediction, is_main_process


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def main(train_args: Wav2Vec2FinetuningArguments) -> None:
    def processing_datasets(func: Callable) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        def process_dataset(
            dataset: Dataset,
            dataset_key: str,
            repo_name: str,
            truncate_map: dict,
            filter_cache_file_name: str,
        ) -> None:
            original_size = len(dataset)
            if dataset_key in truncate_map:
                truncate_size = truncate_map[dataset_key]
                dataset_size = len(dataset)
                dataset = dataset if dataset_size <= truncate_size else dataset.shuffle().select(range(truncate_size))
                if dataset_size <= truncate_size and train_args.is_world_process_zero:
                    logger.info(
                        f"{repo_name}의 {dataset_key}크기는 {dataset_size}이지만 truncate_size는 {truncate_size} 크기를 조절하셈."
                    )

            if dataset_key in train_args.train_dataset_prefix and train_args.do_train:
                dataset = dataset.filter(
                    lambda length_ls: [
                        train_args.audio_min_seq <= length <= train_args.audio_max_seq for length in length_ls
                    ],  # type: ignore
                    num_proc=train_args.preprocessing_num_workers,
                    input_columns=[train_args.length_column_name],
                    cache_file_name=filter_cache_file_name[dataset_key],
                    load_from_cache_file=True,
                    batched=True,
                    batch_size=train_args.preprocessing_batch_size,
                    desc=f"length-filtering-{repo_name}/{dataset_key}",
                )
                train_dataset_ls.append(dataset)

            if dataset_key in train_args.valid_dataset_prefix and train_args.do_eval:
                valid_dataset_ls.append(dataset)

            if dataset_key in train_args.test_dataset_prefix and train_args.do_predict:
                test_dataset_ls.append(dataset)

            if train_args.is_world_process_zero:
                length_ls = sorted(dataset[train_args.length_column_name], reverse=True)[:100]
                length_ls = [int(length) for length in length_ls]
                logger.info(f"{repo_name}/{dataset_key}-length: {length_ls}")
                logger.info(f"{repo_name}/{dataset_key}-size: {original_size} -> {len(dataset)}")

        def concat(datasets_ls: List[Dataset], dataset_type: str) -> Optional[Dataset]:
            if datasets_ls:
                dataset = concatenate_datasets(datasets_ls)
                dataset.set_format("pt")
                if train_args.is_world_process_zero:
                    logger.info(f"{dataset_type}_dataset:\n{dataset}")
                return dataset
            return None

        start_time = time.time()
        train_dataset_ls, valid_dataset_ls, test_dataset_ls = [], [], []
        for repo_name in train_args.dataset_repo_ls:
            if train_args.is_world_process_zero:
                logger.info(f"load-{repo_name}")

            data_name = train_args.data_name_map.get(repo_name, None)
            truncate_map = train_args.data_truncate_map.get(repo_name, {})
            datasets = load_dataset(repo_name, data_name)

            map_cache_file_name, filter_cache_file_name = None, None
            if train_args.cache_dir is not None:
                name = repo_name.split("/")[-1]
                name = f"{name}-{data_name}" if data_name else name

                map_cache_file_name = {
                    x: train_args.cache_dir.joinpath(f"map_{name}-{x}_preprocessor.arrow").as_posix() for x in datasets
                }
                filter_cache_file_name = {
                    x: train_args.cache_dir.joinpath(
                        f"filter_{f'{truncate_map[x]}-' if x in truncate_map else ''}{train_args.audio_min_seq}-{train_args.audio_max_seq}_{name}-{x}_preprocessor.arrow"
                    ).as_posix()
                    for x in datasets
                }

            datasets = datasets.map(
                func,
                num_proc=train_args.preprocessing_num_workers,
                load_from_cache_file=True,
                batched=True,
                cache_file_names=map_cache_file_name,
                batch_size=train_args.preprocessing_batch_size,
                remove_columns=set(sum(datasets.column_names.values(), [])),
                desc=f"preprocess-{repo_name}",
                fn_kwargs={"processor": processor, "args": train_args},
            )

            for dataset_key in datasets:
                process_dataset(datasets[dataset_key], dataset_key, repo_name, truncate_map, filter_cache_file_name)

        train_dataset = concat(train_dataset_ls, "train")
        valid_dataset = concat(valid_dataset_ls, "valid")
        test_dataset = concat(test_dataset_ls, "test")

        if train_args.is_world_process_zero:
            logger.info(f"load_dataset_time: {time.time() - start_time:.2f}")

        return train_dataset, valid_dataset, test_dataset

    def compute_metrics(pred: EvalPrediction):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = config.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
        pred_str = [unicodedata.normalize("NFC", x) for x in pred_str]
        label_str = [unicodedata.normalize("NFC", x) for x in label_str]

        wer_score = wer_metric.compute(predictions=pred_str, references=label_str)
        cer_score = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer_score, "cer": cer_score}

    model_name_or_path = train_args.resume_from_checkpoint or train_args.model_name_or_path
    processor = Wav2Vec2Processor.from_pretrained(model_name_or_path, **train_args.processor_kwargs)
    config = AutoConfig.from_pretrained(model_name_or_path, **train_args.config_kwargs)
    model_kwargs = {"config": config, **train_args.model_kwargs}
    model = AutoModelForCTC.from_pretrained(model_name_or_path, **model_kwargs)

    model.freeze_feature_extractor()

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    match train_args.data_preprocessor_type:
        case "wav2vec2_ctc_finetune":
            preprocessor_func = wav2vec2_ctc_finetune_preprocessor

    with (
        train_args.main_process_first(desc="main_process_first")
        if train_args.do_data_main_process_first
        else nullcontext()
    ):
        # load datasets
        train_dataset, valid_dataset, test_dataset = processing_datasets(preprocessor_func)

    collator = DataCollatorCTCWithPadding(
        processor=processor,
        pad_to_multiple_of=train_args.pad_to_multiple_of,
    )

    # set metrics
    wer_metric, cer_metric = load("wer"), load("cer")

    # set trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        processing_class=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    if train_args.do_train and train_dataset:
        train(trainer, train_args)

    if train_args.do_eval and valid_dataset:
        valid(trainer)

    if train_args.do_predict and test_dataset:
        logger.info("do_predict 코드는 아직 작성 중")


def train(trainer: Trainer, train_args: Wav2Vec2FinetuningArguments) -> None:
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)
    trainer.save_metrics(save_dir)


@torch.no_grad()
def valid(trainer: Trainer, valid_datasets: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


if __name__ == "__main__":
    parser = HfArgumentParser([Wav2Vec2FinetuningArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if remain_args and is_main_process(train_args.local_rank):
        logger.info(f"remain_args: {remain_args}")

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(train_args.run_name)

    main(train_args)
