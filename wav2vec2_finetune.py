import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from data import DataCollatorCTCWithPadding
from datasets import Dataset, concatenate_datasets, load_dataset
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
    AutoModelForCTC,
    AutoProcessor,
    HfArgumentParser,
    Trainer,
    set_seed,
)
from transformers import logging as hf_logging
from transformers.trainer_utils import EvalPrediction, is_main_process


# NOTE: 이걸 해야 tri-stage scheduler를 사용할 수 있음.
set_scheduler()

hf_logging.set_verbosity_info()
logger = hf_logging.get_logger("transformers")


def main(train_args: Wav2Vec2FinetuningArguments) -> None:
    def preprocessor(example: Dict[str, Union[List[Any], List[List[Any]]]]) -> Dict[str, List[Any]]:
        sentence_ls = example[train_args.sentence_column_name]
        sentence_ls = sentence_ls if isinstance(sentence_ls, list) else [sentence_ls]

        audio_ls = example[train_args.audio_column_name]
        audio_ls = audio_ls if isinstance(audio_ls, list) else [audio_ls]
        audio_ls = [audio["array"] for audio in audio_ls]

        finish_label_ls, finish_audio_ls, finish_length_ls = list(), list(), list()
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

            sentence = processor.tokenizer(sentence, return_attention_mask=False)["input_ids"]
            label_length = len(sentence)

            # NOTE: for CTC loss
            feature_length = get_feat_extract_output_lengths(audio_length, config)
            if label_length > feature_length:
                continue

            audio = processor.feature_extractor(audio, sampling_rate=train_args.sampling_rate)["input_values"]

            finish_label_ls.append(sentence)
            finish_audio_ls.append(audio[0].tolist())
            finish_length_ls.append(audio_length)

        return {
            "labels": finish_label_ls,
            main_input_name: finish_audio_ls,
            train_args.length_column_name: finish_length_ls,
        }

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

    def compute_metrics(pred: EvalPrediction) -> Dict[str, float]:
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

    # load model, feature_extractor, tokenizer
    model_path = train_args.resume_from_checkpoint or train_args.model_name_or_path
    config = AutoConfig.from_pretrained(model_path, attn_implementation=train_args.attn_implementation)
    model = AutoModelForCTC.from_pretrained(model_path, config=config)
    processor = AutoProcessor.from_pretrained(model_path)

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
    collator = DataCollatorCTCWithPadding(
        processor=processor,
        pad_to_multiple_of=train_args.pad_to_multiple_of,
    )

    wer_metric, cer_metric = load("wer"), load("cer")

    # set trainer
    trainer = Trainer(
        model=model,
        args=train_args,
        tokenizer=processor,
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
        predict(trainer, test_dataset)


def train(trainer: Trainer, train_args: Wav2Vec2FinetuningArguments) -> None:
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

        test_dataset_dict[dataset_name[start:end]] = part_dataset


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
