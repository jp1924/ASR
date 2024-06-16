import os
import unicodedata
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from data import DataCollatorCTCWithPadding, HFAddBackgroundNoise
from datasets import Dataset, Features, Value, concatenate_datasets, load_dataset
from evaluate import load
from setproctitle import setproctitle
from utils import (
    Wav2Vec2FinetuningArguments,
    default_sentence_norm,
    get_feat_extract_output_lengths,
    librosa_silence_filter,
    set_scheduler,
)

from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForCTC,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    Wav2Vec2Processor,
    is_torch_xla_available,
    is_wandb_available,
    set_seed,
)
from transformers import logging as hf_logging
from transformers.trainer_utils import EvalPrediction


# NOTE: 이걸 해야 tri-stage scheduler를 사용할 수 있음.
set_scheduler()

hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")

global GLOBAL_LOGGER


def main(train_args: Wav2Vec2FinetuningArguments) -> None:
    def preprocessor(example: Dict[str, Union[List[Any], List[List[Any]]]]) -> Dict[str, List[Any]]:
        sentence_ls = example[train_args.sentence_column_name]
        sentence_ls = sentence_ls if isinstance(sentence_ls, list) else [sentence_ls]

        audio_ls = example[train_args.audio_column_name]
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

    def collect_dataset(prefix_ls: List[str]) -> Optional[Dataset]:
        if not prefix_ls:
            return None

        data_ls = list()
        for prefix in prefix_ls:
            check_key: str = lambda key: (prefix in key)  # noqa: E731
            filter_data = [
                concatenate_datasets(data_dict.pop(key)) for key in list(data_dict.keys()) if check_key(key)
            ]
            data_ls.extend(filter_data)
        dataset = concatenate_datasets(data_ls)
        dataset.set_format("torch")

        return dataset

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

    def audio_augmentate(audio_batch: torch.Tensor) -> torch.Tensor:
        audio_batch = augmentator(audio_batch)
        return audio_batch

    def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

        pred_str = tokenizer.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = tokenizer.batch_decode(pred.label_ids, group_tokens=False)
        pred_str = [unicodedata.normalize("NFC", x) for x in pred_str]
        label_str = [unicodedata.normalize("NFC", x) for x in label_str]

        wer_score = wer_metric.compute(predictions=pred_str, references=label_str)
        cer_score = cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer_score, "cer": cer_score}

    # load model, feature_extractor, tokenizer
    model_path = train_args.resume_from_checkpoint or train_args.model_name_or_path
    config = AutoConfig.from_pretrained(
        model_path,
        attn_implementation=train_args.attn_implementation,
    )
    model = AutoModelForCTC.from_pretrained(model_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)

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
                get_cache_path: str = lambda x: os.path.join(  # noqa: E731
                    train_args.cache_dir,
                    f"{name}-{x}_{train_args.cache_file_name}",
                )
                name = dataset_name.split("/")[-1]
                cache_file_name = {x: get_cache_path(x) for x in dataset}

            # NOTE: 어떤 이슈 였는지 기억은 나질 않지만, 이렇게 하면 속도가 2배 이상 더 빨라진다는 개발자의 오피셜이 있었음.
            features = Features(
                {
                    "labels": Value("string"),
                    "input_values": Value("float32"),
                    "length": Value("int32"),
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

    noise_dataset = None
    for dataset_name in train_args.noise_dataset_repo_ls:
        logger.info(f"load-{dataset_name}")
        dataset = load_dataset(dataset_name)
        dataset = dataset.select_columns(train_args.audio_column_name)
        dataset = [dataset]

        if noise_dataset:
            dataset.append(noise_dataset)

        noise_dataset = concatenate_datasets(dataset)

    if noise_dataset:
        noise_dataset.set_format("torch")

        augmentator = HFAddBackgroundNoise()

    train_dataset = None
    if train_args.do_train:
        train_dataset = collect_dataset(train_args.train_dataset_prefix)

        if noise_dataset:
            train_dataset.set_transform(audio_augmentate, columns=train_args.audio_column_name)

        if (train_args.local_rank == 0) and train_dataset:
            train_total_length = sum(train_dataset["length"])
            logger.info("train_dataset")
            logger.info(train_dataset)
            logger.info(f"train_total_hour: {(train_total_length / 16000) / 60**2:.2f}h")

    valid_dataset = None
    if train_args.do_eval:
        valid_dataset = collect_dataset(train_args.valid_dataset_prefix)

        # if noise_dataset:
        #     valid_dataset.set_transform(audio_augmentate, columns=train_args.audio_column_name)

        valid_dataset_dict = dict()
        valid_name_ls = valid_dataset["dataset_name"]
        for dataset_name in set(valid_name_ls):
            part_idx = [idx for idx, x in enumerate(valid_name_ls) if x == dataset_name]
            part_dataset = valid_dataset.select(part_idx, keep_in_memory=False)

            # 'jp1924/KconfSpeech-validation'
            start = dataset_name.rindex("/") + 1
            end = dataset_name.rindex("-")

            if dataset_name[start:end] in train_args.valid_exclude_ls:
                continue

            if len(part_dataset) > train_args.valid_truncate_num:
                part_dataset = part_dataset.shuffle(train_args.seed)
                part_dataset = part_dataset.select(range(train_args.valid_truncate_num))

            valid_dataset_dict[dataset_name[start:end]] = part_dataset
        valid_dataset = valid_dataset_dict

        if (train_args.local_rank == 0) and valid_dataset:
            valid_total_length = sum(valid_dataset["length"])
            logger.info("valid_dataset")
            logger.info(valid_dataset)
            logger.info(f"valid_total_hour: {(valid_total_length / 16000) / 60**2:.2f}h")

    test_dataset = None
    if train_args.do_predict:
        test_dataset = collect_dataset(train_args.test_dataset_prefix)

        # if noise_dataset:
        #     test_dataset.set_transform(audio_augmentate, columns=train_args.audio_column_name)

        if (train_args.local_rank == 0) and test_dataset:
            test_total_length = sum(test_dataset["length"])
            logger.info("test_dataset")
            logger.info(test_dataset)
            logger.info(f"test_total_hour: {(test_total_length / 16000) / 60**2:.2f}h")

    # 6898663 >> 3.3% 정도 필터링 됨. 여기엔 tokenizing할 수 없는 문자, 음성 길이가 맞지 않는 문자 등 여러 요인으로 인해 필터링 된 데이터가 포함
    # 7136987
    # 238324

    # set collator
    collator = DataCollatorCTCWithPadding(
        processor=processor,
        pad_to_multiple_of=train_args.pad_to_multiple_of,
    )

    wer_metric = load("wer")
    cer_metric = load("cer")

    if train_args.torch_compile:
        model = torch.compile(
            model,
            backend=train_args.torch_compile_backend,
            mode=train_args.torch_compile_mode,
            fullgraph=True,
        )

    # set trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset_dict,
        compute_metrics=compute_metrics,
    )
    if train_args.do_train and train_dataset:
        train(trainer)

    if train_args.do_eval and valid_dataset:
        valid(trainer)

    if train_args.do_predict and test_dataset:
        predict(trainer, test_dataset)


def train(trainer: Trainer) -> None:
    train_args: Wav2Vec2FinetuningArguments = trainer.args
    trainer.train(resume_from_checkpoint=train_args.resume_from_checkpoint)

    save_dir = os.path.join(train_args.output_dir, "last_model")
    trainer.save_model(save_dir)
    trainer.save_metrics(save_dir)


@torch.no_grad()
def valid(trainer: Trainer, valid_datasets: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
    valid_datasets = valid_datasets if valid_datasets else trainer.eval_dataset
    trainer.evaluate(valid_datasets)


@torch.no_grad()
def predict(trainer: Trainer, test_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None) -> None:
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
    parser = HfArgumentParser([Wav2Vec2FinetuningArguments])
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
