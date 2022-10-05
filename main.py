import os
from typing import Dict, Union
import numpy as np

from datasets import Dataset, load_dataset, load_metric
from setproctitle import setproctitle
from transformers import (
    HfArgumentParser,
    T5Config,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from transformers.integrations import WandbCallback
from utils import DataArgument, ModelArgument


def main(parser: HfArgumentParser) -> None:
    train_args, model_args, data_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    tokenizer = T5TokenizerFast.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache)
    config = T5Config.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache)
    model = T5ForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path, config=config, cache_dir=model_args.cache
    )

    # [NOTE]: datasets에서 csv를 불러올 때 무조건 columns이 명시 되어 있어야 한다.
    #         datasets은 상단의 열을 columns으로 인식하기 때문에 잘못하면 이상한 columns이 될 수 있다.
    loaded_data = load_dataset(
        "csv", data_files=data_args.data_name_or_script, cache_dir=model_args.cache, split="train"
    )

    def preprocess(input_values: Dataset) -> dict:
        """
        Text To Text: train, label 데이터도 전부 text임.
        """

        # [NOTE]: 이런 prompt의 경우 config에서 설정해야 할 듯 하다.
        #         config에 task_specific_params라는 값이 있는데 이 값을 이용하는게 HF 개발자가 의도한 사용법이 아닐까 생각함.
        # [NOTE]: 임시로 설정한 이름들, 나중에 데이터를 받아보면 삭제됨.
        train_col_name = "sentence"
        label_col_name = "label"

        train_prompt = "translation_num_to_text"
        label_prompt = "label"

        train_input = f"{train_prompt}: {input_values[train_col_name]}"
        label_input = f"{label_prompt}: {input_values[label_col_name]}"

        # [NOTE]: train이라는 이름은 나중에 바꾸는 게 좋을 듯 valid, test도 있어서 맞지가 않는다.
        train_encoded = tokenizer(train_input, return_attention_mask=False)
        label_encoded = tokenizer(label_input, return_attention_mask=False)

        result = {"sentence": train_encoded["input_ids"], "label": label_encoded["input_ids"]}
        return result

    # [NOTE]: data preprocess
    desc_name = "T5_preprocess"
    loaded_data = loaded_data.map(preprocess, num_proc=data_args.num_proc, desc=desc_name)
    loaded_data = loaded_data.rename_columns({"sentence": "input_ids", "label": "labels"})
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
    def metrics(predictions: np.ndarray, references: np.ndarray) -> Dict[str, Union[int, float]]:

        print

        wer_score = wer._compute()
        blue_score = blue._compute()
        rouge_score = rouge._compute()
        result = {"wer": wer_score, "blue": blue_score, "rouge": rouge_score}
        return result

    collator = DataCollatorForSeq2Seq(tokenizer, model)
    callbacks = [WandbCallback] if os.getenv("WANDB_DISABLED") == "false" else None
    trainer = Trainer(
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
        train(trainer, label_name="label")
    if train_args.do_eval:
        eval(trainer)
    if train_args.do_predict:
        predict(trainer, valid_data)


def train(trainer: Trainer, label_name: str) -> None:
    """"""
    outputs = trainer.train()
    metrics = outputs.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


def eval(trainer: Trainer) -> None:
    """"""
    outputs = trainer.evaluate()
    metrics = outputs.metrics

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


def predict(trainer: Trainer, test_data) -> None:
    """"""
    outputs = trainer.predict(test_dataset=test_data)
    metrics = outputs.metrics

    trainer.log_metrics("predict", metrics)
    trainer.save_metrics("predict", metrics)


def set_wandb_env(process_name) -> None:
    """"""
    # [NOTE]: 혹은 launch.json에서 설정하는 것도 가능.
    # [NOTE]: json -> https://stackoverflow.com/questions/36231681/how-to-perform-os-environ-join-in-python
    os.environ["WANDB_CACHE_DIR"] = "/data/jsb193/github/t5/.cache"
    os.environ["WANDB_DIR"] = "/data/jsb193/github/t5/T5"
    os.environ["WANDB_NAME"] = process_name
    os.environ["WANDB_NOTEBOOK_NAME"] = "run for t5 testing, please ignore this project"  # --check name
    os.environ["WANDB_USERNAME"] = "jp_42maru"
    os.environ["WANDB_RUN_GROUP"] = "tadev"
    os.environ["WANDB_TAGS"] = "T5, finetune, test"
    os.environ["WANDB_DISABLE_CODE"] = "false"


if __name__ == "__main__":
    parser = HfArgumentParser([TrainingArguments, ModelArgument, DataArgument])

    process_name = "T5"
    setproctitle(process_name)

    os.environ["WANDB_DISABLED"] = "true"
    if os.getenv("WANDB_DISABLED") == "false":
        # [NOTE]: check wandb env variable
        # -> 환경 변수를 이용해 조작이 가능함.
        #    https://docs.wandb.ai/guides/track/advanced/environment-variables
        set_wandb_env(process_name)

    main(parser)
