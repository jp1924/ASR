import os
from typing import Any, Dict, List, Optional, Union

import torch
from data import DataCollatorForWav2Vec2Pretraining
from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from utils import Wav2Vec2PretrainingArguments, librosa_silence_filter
from wav2vec2_pretrainer import Wav2Vec2Pretrainer

from transformers import (
    AutoConfig,
    AutoModelForPreTraining,
    AutoProcessor,
    HfArgumentParser,
    is_torch_xla_available,
    is_wandb_available,
    set_seed,
)
from transformers import logging as hf_logging


hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")

global GLOBAL_LOGGER
GLOBAL_LOGGER = None


def main(train_args: Wav2Vec2PretrainingArguments) -> None:
    def preprocessor(example: Dict[str, Union[List[Any], List[List[Any]]]]) -> Dict[str, List[Any]]:
        audio_ls = example[train_args.audio_column_name]
        audio_ls = audio_ls if isinstance(audio_ls, list) else [audio_ls]
        audio_ls = [audio["array"] for audio in audio_ls]

        length_ls = list()
        norm_audio_ls = list()
        for audio in audio_ls:
            audio = librosa_silence_filter(audio)
            audio_length = audio.shape[0]

            duration_check = train_args.min_duration_in_seconds <= audio_length <= train_args.max_duration_in_seconds
            if not (audio.any() and duration_check):
                continue

            audio = processor(audio=audio, sampling_rate=train_args.sampling_rate)
            audio = audio[main_input_name][0].tolist()

            norm_audio_ls.append(audio)
            length_ls.append(audio_length)

        outputs = {
            main_input_name: norm_audio_ls,
            train_args.length_column_name: length_ls,
        }
        return outputs

    def collect_dataset(prefix_ls: List[str]) -> Optional[Dataset]:
        if not prefix_ls:
            return None

        data_ls = list()
        for prefix in prefix_ls:
            check_key: str = lambda key: (prefix in key)
            filter_data = [
                concatenate_datasets(data_dict.pop(key)) for key in list(data_dict.keys()) if check_key(key)
            ]
            data_ls.extend(filter_data)
        dataset = concatenate_datasets(data_ls)
        dataset.set_format("torch")

        return dataset

    def set_wandb() -> None:
        GLOBAL_LOGGER.run.log_code(
            train_args.wandb_code_log_dir,
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

    model_path = train_args.resume_from_checkpoint or train_args.model_name_or_path
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForPreTraining.from_pretrained(model_path, config=config)
    processor = AutoProcessor.from_pretrained(model_path)

    main_input_name = model.main_input_name

    # set logger
    if GLOBAL_LOGGER and (train_args.local_rank == 0):
        set_wandb()

    # load dataset & preprocess
    data_dict = dict()
    for dataset_name in train_args.dataset_repo_ls:
        logger.info(f"load-{dataset_name}")
        dataset = load_dataset(dataset_name)

        # DatasetDict이라서 이런식으로 해줘야 함.
        column_names = set(sum(dataset.column_names.values(), []))
        with train_args.main_process_first(desc="data preprocess"):
            cache_file_name = None
            if train_args.cache_file_name:
                get_cache_path: str = lambda x: os.path.join(
                    train_args.cache_dir,
                    f"{name}-{x}_{train_args.cache_file_name}",
                )
                name = dataset_name.split("/")[-1]
                cache_file_name = {x: get_cache_path(x) for x in dataset}

            # NOTE: 어떤 이슈 였는지 기억은 나질 않지만, 이렇게 하면 속도가 2배 이상 더 빨라진다는 datasets 개발자의 오피셜이 있었음.
            features = Features(
                {
                    main_input_name: [Value("float32")],
                    train_args.length_column_name: Value("int32"),
                }
            )

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
                features=features,
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
    if train_args.do_train:
        train_dataset = collect_dataset(train_args.train_dataset_prefix)
        if (train_args.local_rank == 0) and train_dataset:
            train_total_length = sum(train_dataset[train_args.length_column_name])
            logger.info("train_dataset")
            logger.info(train_dataset)
            logger.info(f"train_total_hour: {(train_total_length / 16000) / 60**2:.2f}h")

    valid_dataset = None
    if train_args.do_eval:
        valid_dataset = collect_dataset(train_args.valid_dataset_prefix)

        valid_exclude_ls = train_args.valid_exclude_ls or []
        valid_dataset_dict = dict()
        valid_name_ls = valid_dataset["dataset_name"]
        if train_args.split_valid:
            for dataset_name in set(valid_name_ls):
                part_idx = [idx for idx, x in enumerate(valid_name_ls) if x == dataset_name]
                part_dataset = valid_dataset.select(part_idx, keep_in_memory=False)

                # 'jp1924/KconfSpeech-validation'
                start = dataset_name.rindex("/") + 1
                end = dataset_name.rindex("-")

                if dataset_name[start:end] in valid_exclude_ls:
                    continue

                if len(part_dataset) > train_args.valid_truncate_num:
                    part_dataset = part_dataset.shuffle(train_args.seed)
                    part_dataset = part_dataset.select(range(train_args.valid_truncate_num))

                if (train_args.local_rank == 0) and valid_dataset:
                    valid_total_length = sum(part_dataset[train_args.length_column_name])
                    logger.info(f"{dataset_name[start:end]}-valid_dataset")
                    logger.info(part_dataset)
                    logger.info(f"valid_total_hour: {(valid_total_length / 16000) / 60**2:.2f}h")
                valid_dataset_dict[dataset_name[start:end]] = part_dataset
            valid_dataset = valid_dataset_dict
        else:
            if (train_args.local_rank == 0) and valid_dataset:
                valid_total_length = sum(valid_dataset[train_args.length_column_name])
                logger.info("valid_dataset")
                logger.info(valid_dataset)
                logger.info(f"valid_total_hour: {(valid_total_length / 16000) / 60**2:.2f}h")

    test_dataset = None
    if train_args.do_predict:
        test_dataset = collect_dataset(train_args.test_dataset_prefix)
        if (train_args.local_rank == 0) and test_dataset:
            test_total_length = sum(test_dataset[train_args.length_column_name])
            logger.info("test_dataset")
            logger.info(test_dataset)
            logger.info(f"test_total_hour: {(test_total_length / 16000) / 60**2:.2f}h")

    # set collator
    collator = DataCollatorForWav2Vec2Pretraining(
        model=model,
        feature_extractor=processor.feature_extractor,
        pad_to_multiple_of=train_args.pad_to_multiple_of,
        mask_time_prob=config.mask_time_prob,
        mask_time_length=config.mask_time_length,
        mask_time_min_masks=config.mask_time_min_masks,
    )

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
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
        train(trainer)

    if train_args.do_eval and valid_dataset:
        valid(trainer)

    if train_args.do_predict and test_dataset:
        predict(trainer, test_dataset)


def train(trainer: Wav2Vec2Pretrainer) -> None:
    train_args: Wav2Vec2PretrainingArguments = trainer.args
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)
    # custom trainer는 save_metrics 안됨.


@torch.no_grad()
def valid(trainer: Wav2Vec2Pretrainer, valid_datasets: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


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
        # NOTE: trainer.log를 사용하면 train/test 처럼 찍혀서 나와서 wandb로 직접 찍음
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
