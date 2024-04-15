from dataclasses import dataclass, field
from typing import List

from transformers import TrainingArguments


@dataclass
class TNTTrainingArguments(TrainingArguments):
    # data
    # List[str]
    dataset_names: str = field(
        default=None,
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
    preprocessing_batched: bool = field(
        default=True,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    text_column_name: str = field(
        default="text",
        metadata={"help": "Column in the dataset that contains text file path"},
    )
    train_dataset_prefix: List[str] = field(default=None)
    valid_dataset_prefix: List[str] = field(default=None)
    test_dataset_prefix: List[str] = field(default=None)
    cache_file_name: str = field(
        default=None,
        metadata={"help": "Path to cached file name"},
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )

    # model
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models.", "required": True},
    )

    quantization_config_path: str = field(default=None)

    peft_config_name_or_path: str = field(default="")
    torch_dtype: str = field(default=None)
    low_cpu_mem_usage: bool = field(default=None)
    do_peft: bool = field(default=False)
    use_auth_token: str = field(default=None)
    trust_remote_code: str = field(default=None)
    device_map: str = field(default=None)
    attn_implementation: str = field(default=None)
