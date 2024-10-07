import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from data import DataCollatorForWav2Vec2Pretraining
from datasets import Dataset, concatenate_datasets, load_dataset
from setproctitle import setproctitle
from utils import Wav2Vec2PretrainingArguments, librosa_silence_filter
from wav2vec2_pretrainer import Wav2Vec2Pretrainer

from transformers import (
    AutoConfig,
    AutoModelForPreTraining,
    AutoProcessor,
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
    def preprocessor(example: Dict[str, Union[List[Any], List[List[Any]]]]) -> Dict[str, List[Any]]:
        audio_ls = example[train_args.audio_column_name]
        audio_ls = audio_ls if isinstance(audio_ls, list) else [audio_ls]
        audio_ls = [audio["array"] for audio in audio_ls]

        finish_length_ls, finish_audio_ls = list(), list()
        for audio in audio_ls:
            audio = librosa_silence_filter(audio)
            audio_length = audio.shape[0]

            duration_check = train_args.min_duration_in_seconds <= audio_length <= train_args.max_duration_in_seconds
            if not (audio.any() and duration_check):
                continue

            audio = processor(audio=audio, sampling_rate=train_args.sampling_rate)
            audio = audio[main_input_name][0].tolist()

            finish_audio_ls.append(audio)
            finish_length_ls.append(audio_length)

        outputs = {
            main_input_name: finish_audio_ls,
            train_args.length_column_name: finish_length_ls,
        }
        return outputs

    def prepare_datasets() -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
        train_dataset_ls = valid_dataset_ls = test_dataset_ls = list()
        for repo_name in train_args.dataset_repo_ls:
            logger.info(f"load-{repo_name}")
            datasets = load_dataset(repo_name)

            if repo_name in train_args.data_truncate_map:
                for data_type in train_args.data_truncate_map[repo_name]:
                    truncate_size = train_args.data_truncate_map[repo_name][data_type]
                    data = datasets[data_type].shuffle()
                    if len(data) <= truncate_size:
                        continue

                    datasets[data_type] = data.select(range(truncate_size))

            if train_args.cache_file_name:
                get_cache_path: str = lambda x: os.path.join(  # noqa: E731
                    train_args.cache_dir,
                    f"{name}-{x}_{train_args.cache_file_name}",
                )
                name = repo_name.split("/")[-1]
                train_args.cache_file_name = {x: get_cache_path(x) for x in datasets}

            # DatasetsDict이라서 이런식으로 해줘야 함.
            with train_args.main_process_first(desc="data preprocess"):
                datasets = datasets.map(
                    preprocessor,
                    num_proc=train_args.preprocessing_num_workers,
                    load_from_cache_file=True,
                    batched=train_args.preprocessing_batched,
                    cache_file_names=train_args.cache_file_name,
                    batch_size=train_args.preprocessing_batch_size,
                    remove_columns=set(sum(datasets.column_names.values(), [])),
                    desc=f"preprocess-{repo_name}",
                )
                datasets.set_format("pt")

            for dataset_key in datasets:
                if dataset_key in train_args.train_dataset_prefix and train_args.do_train:
                    train_dataset_ls.append(datasets[dataset_key])
                if dataset_key in train_args.valid_dataset_prefix and train_args.do_eval:
                    valid_dataset_ls.append(datasets[dataset_key])
                if dataset_key in train_args.test_dataset_prefix and train_args.do_predict:
                    test_dataset_ls.append(datasets[dataset_key])

        train_dataset = None
        if train_dataset_ls:
            train_dataset = concatenate_datasets(train_dataset_ls)
            if is_main_process(train_args.local_rank):
                train_total_length = sum(train_dataset[train_args.length_column_name])
                logger.info(f"train_dataset:\n{train_dataset}")
                logger.info(f"train_total_hour: {(train_total_length / 16000) / 60**2:.2f}h")

        valid_dataset = None
        if valid_dataset_ls:
            valid_dataset = concatenate_datasets(valid_dataset_ls)
            if is_main_process(train_args.local_rank):
                valid_total_length = sum(valid_dataset[train_args.length_column_name])
                logger.info(f"valid_dataset:\n{valid_dataset}")
                logger.info(f"valid_total_hour: {(valid_total_length / 16000) / 60**2:.2f}h")

        test_dataset = None
        if test_dataset_ls:
            test_dataset = concatenate_datasets(test_dataset_ls)
            if is_main_process(train_args.local_rank):
                test_total_length = sum(test_dataset[train_args.length_column_name])
                logger.info(f"test_dataset:\n{test_dataset}")
                logger.info(f"test_total_hour: {(test_total_length / 16000) / 60**2:.2f}h")

        return (train_dataset, valid_dataset, test_dataset)

    model_path = train_args.resume_from_checkpoint or train_args.model_name_or_path
    config: Wav2Vec2Config = AutoConfig.from_pretrained(model_path)
    model: Wav2Vec2ForPreTraining = AutoModelForPreTraining.from_pretrained(model_path, config=config)
    processor: Wav2Vec2Processor = AutoProcessor.from_pretrained(model_path)

    main_input_name = model.main_input_name

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
        predict(trainer, test_dataset)


def train(trainer: Wav2Vec2Pretrainer, train_args: Wav2Vec2PretrainingArguments) -> None:
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)


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
        test_dataset_dict[dataset_name[start:end]] = part_dataset


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
