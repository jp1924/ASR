import os
from typing import Dict
import numpy as np

from datasets import Dataset, load_dataset, load_metric
from setproctitle import setproctitle
from transformers import (
    HfArgumentParser,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers.trainer_utils import EvalPrediction
from transformers.integrations import WandbCallback
from utils import DataArgument, ModelArgument


def main(parser: HfArgumentParser) -> None:
    train_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    setproctitle(train_args.run_name)

    tokenizer = T5TokenizerFast.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache)
    config = T5Config.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache)
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache
    )
    model.resize_token_embeddings(len(tokenizer))  # ??

    # [NOTE]: datasets에서 csv를 불러올 때 무조건 columns이 명시 되어 있어야 한다.
    #         datasets은 상단의 열을 columns으로 인식하기 때문에 잘못하면 이상한 columns이 될 수 있다.
    loaded_data = load_dataset(
        "csv", data_files=data_args.data_name_or_script, cache_dir=model_args.cache, split="train"
    )

    def preprocess(input_values: Dataset) -> dict:
        """"""

        train_prompt = "translation_num_to_text"
        train_input = f"""{train_prompt}: {input_values["num_col"]}"""
        label_input = input_values["sen_col"]

        # [NOTE]: train이라는 이름은 나중에 바꾸는 게 좋을 듯 valid, test도 있어서 맞지가 않는다.
        train_encoded = tokenizer(train_input, return_attention_mask=False)
        label_encoded = tokenizer(label_input, return_attention_mask=False)

        result = {"num_col": train_encoded["input_ids"], "sen_col": label_encoded["input_ids"]}
        return result

    # [NOTE]: data preprocess
    desc_name = "T5_preprocess"
    loaded_data = loaded_data.map(preprocess, num_proc=data_args.num_proc, desc=desc_name)
    loaded_data = loaded_data.rename_columns({"num_col": "input_ids", "sen_col": "labels"})

    # [NOTE]: check data
    if train_args.do_eval or train_args.do_predict:
        # 들어오는 데이터 파일을 train, valid, test로 구분해야 할지
        # 아님 하나의 data에서 train, valid, test를 분리 해야할 지 모르겠다.
        splited_data = loaded_data.train_test_split(0.001)
        train_data = splited_data["train"]
        valid_data = splited_data["test"]
    else:
        train_data = loaded_data
        valid_data = None

    wer = load_metric("wer")
    blue = load_metric("bleu")
    rouge = load_metric("rouge")

    def metrics(evaluation_result: EvalPrediction) -> Dict[str, float]:
        predicts = evaluation_result.predictions[0]
        predicts = predicts.argmax(2)
        decoded_preds = tokenizer.batch_decode(predicts, skip_special_tokens=True)

        labels = evaluation_result.label_ids
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        wer_score = wer._compute(decoded_preds, decoded_labels)
        blue_score = blue._compute(decoded_preds, decoded_labels)
        rouge_score = rouge._compute(decoded_preds, decoded_labels)

        result = {"wer": wer_score, "blue": blue_score, "rouge": rouge_score}
        return result

    collator = DataCollatorForSeq2Seq(tokenizer, model)
    callbacks = [WandbCallback] if os.getenv("WANDB_DISABLED") == "false" else None
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_data,
        compute_metrics=metrics,
        args=train_args,
        eval_dataset=valid_data,
        data_collator=collator,
        callbacks=callbacks,
    )

    if train_args.do_train:
        train(trainer)
    if train_args.do_eval:
        eval(trainer)
    if train_args.do_predict:
        predict(trainer, valid_data)


def train(trainer: Seq2SeqTrainer) -> None:
    """"""
    outputs = trainer.train()
    metrics = outputs.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def eval(trainer: Seq2SeqTrainer) -> None:
    """"""
    outputs = trainer.evaluate()
    metrics = outputs.metrics

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict(trainer: Seq2SeqTrainer, test_data) -> None:
    """"""
    outputs = trainer.predict(test_dataset=test_data)
    metrics = outputs.metrics

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    # example_source: https://github.com/huggingface/transformers/tree/main/examples/pytorch/translation
    parser = HfArgumentParser([Seq2SeqTrainingArguments, ModelArgument, DataArgument])
    # [NOTE]: check wandb env variable
    # -> 환경 변수를 이용해 조작이 가능함.
    #    https://docs.wandb.ai/guides/track/advanced/environment-variables
    main(parser)
