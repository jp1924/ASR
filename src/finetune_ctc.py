import json
import logging
import os
import sys
import time
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from datasets import Dataset
from datasets import logging as ds_logging
from evaluate import load
from setproctitle import setproctitle

import optimization
from preprocessor import PROCESSOR_REGISTRY, processing_datasets
from trainer import DataCollatorCTCWithPadding
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    Wav2Vec2ConformerForCTC,
    Wav2Vec2Processor,
    set_seed,
)
from transformers import logging as hf_logging
from transformers.trainer_utils import EvalPrediction, is_main_process


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


@dataclass
class DataPipelineArguments:
    # data
    dataset_repo_ls: List[str] = field(
        default_factory=list,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    preprocessing_batch_size: int = field(
        default=1000,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'audio'"},
    )
    sentence_column_name: str = field(
        default="sentence",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'audio'"},
    )

    audio_min_seq: int = field(
        default=1,
        metadata={"help": "Filter out audio files that are longer than `audio_min_seq` seconds"},
    )
    audio_max_seq: int = field(
        default=512,
        metadata={"help": "Filter out audio files that are shorter than `audio_max_seq` seconds"},
    )
    train_dataset_prefix: List[str] = field(
        default_factory=list,
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    valid_dataset_prefix: List[str] = field(
        default_factory=list,
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    test_dataset_prefix: List[str] = field(
        default_factory=list,
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    data_truncate_map: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": "A map to truncate part of the data. {'repo_name': {'train': 3000, 'validation': 1500}}."},
    )
    data_name_map: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": "A map to config_name of the data. {'repo_name': 'data_config_name'"},
    )
    do_data_main_process_first: bool = field(
        default=False,
        metadata={"help": "Whether to process the main data first."},
    )
    cache_dir: str = field(
        default="",
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    data_preprocessor_type: str = field(
        default="wav2vec2-pretrain",
        metadata={"help": "Data preprocessor type."},
    )


@dataclass
class TrainPipelineArguments:
    # 이걸 해야 스케줄러가 정상적으로 적용됨.

    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )

    attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "Length of each vector mask span to mask along the time axis in the contrastive task. If omitted, will pull value from model config."
        },
    )
    sampling_rate: int = field(
        default=16000,
        metadata={"help": ""},
    )

    profiling: bool = field(
        default=False,
        metadata={"help": "Enable profiling."},
    )
    profile_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    config_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    model_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )
    processor_kwargs: Optional[Union[dict, str]] = field(
        default="{}",
        metadata={"help": ""},
    )


@dataclass
class FinetuneCTCArguments(TrainingArguments, DataPipelineArguments, TrainPipelineArguments):
    lr_scheduler_type: Union[optimization.NewSchedulerType, str] = field(default="linear")

    def __post_init__(self):
        super().__post_init__()

        def _convert_str_dict(passed_value: dict):
            "Safely checks that a passed value is a dictionary and converts any string values to their appropriate types."
            for key, value in passed_value.items():
                if isinstance(value, dict):
                    passed_value[key] = _convert_str_dict(value)
                elif isinstance(value, str):
                    # First check for bool and convert
                    if value.lower() in ("true", "false"):
                        passed_value[key] = value.lower() == "true"
                    # Check for digit
                    elif value.isdigit():
                        passed_value[key] = int(value)
                    elif value.replace(".", "", 1).isdigit():
                        passed_value[key] = float(value)

            return passed_value

        _ADDITIONAL_VALID_DICT_FILEDS = [
            "data_truncate_map",
            "data_name_map",
            "config_kwargs",
            "model_kwargs",
            "processor_kwargs",
            "profile_kwargs",
        ]
        _VALID_LIST_FIELDS = [
            "dataset_repo_ls",
            "train_dataset_prefix",
            "valid_dataset_prefix",
            "test_dataset_prefix",
        ]

        # copied from: transformers/training_args.py/__post_init__()
        for field in _ADDITIONAL_VALID_DICT_FILEDS:
            passed_value = getattr(self, field)
            # We only want to do this if the str starts with a bracket to indiciate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, field, loaded_dict)
            elif isinstance(passed_value, dict) or passed_value is None:
                pass
            else:
                raise ValueError(f"{field}은 dict로 설정해야 함.")

        for field in _VALID_LIST_FIELDS:
            passed_value = getattr(self, field)
            if isinstance(passed_value, str) and passed_value.startswith("["):
                loaded_list = json.loads(passed_value)
                setattr(self, field, loaded_list)
            elif isinstance(passed_value, list) or passed_value is None:
                pass
            else:
                raise ValueError(f"{field}은 list로 설정해야 함.")

        self.config_kwargs = {
            **self.config_kwargs,
            "attn_implementation": self.attn_implementation,
        }

        self.cache_dir = Path(self.cache_dir) if self.cache_dir else None
        self.model_name_or_path = self.resume_from_checkpoint or self.model_name_or_path

    @property
    def is_local_process_zero(self) -> bool:
        return self.local_process_index == 0

    @property
    def is_world_process_zero(self) -> bool:
        from transformers.utils import is_sagemaker_mp_enabled

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp  # type: ignore

            return smp.rank() == 0
        else:
            return self.process_index == 0


def main(train_args: FinetuneCTCArguments) -> None:
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

    model.freeze_feature_encoder()

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    with (
        train_args.main_process_first(desc="main_process_first")
        if train_args.do_data_main_process_first
        else nullcontext()
    ):
        # load datasets
        train_dataset, valid_dataset, test_dataset = processing_datasets(
            PROCESSOR_REGISTRY[train_args.data_preprocessor_type],
            train_args,
            processor,
            config,
        )

    collator = DataCollatorCTCWithPadding(processor=processor)

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


def train(trainer: Trainer, train_args: FinetuneCTCArguments) -> None:
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)
    trainer.save_metrics(save_dir)


@torch.no_grad()
def valid(trainer: Trainer, valid_datasets: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


if __name__ == "__main__":
    parser = HfArgumentParser([FinetuneCTCArguments])
    train_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if remain_args and is_main_process(train_args.local_rank):
        logger.info(f"remain_args: {remain_args}")

    if train_args.seed is not None:
        set_seed(train_args.seed)

    if train_args.run_name is not None:
        setproctitle(f"{train_args.run_name}-{train_args.local_process_index}")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    ds_logging.set_verbosity(log_level)
    hf_logging.set_verbosity(log_level)
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    main(train_args)
