from dataclasses import dataclass, field
from typing import List

from transformers import TrainingArguments


@dataclass
class Wav2Vec2PretrainingArguments(TrainingArguments):
    # data
    dataset_repo_ls: List[str] = field(
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
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'audio'"},
    )
    min_duration_in_seconds: float = field(
        default=8000.0,
        metadata={"help": "Filter out audio files that are longer than `min_duration_in_seconds` seconds"},
    )
    max_duration_in_seconds: float = field(
        default=448512.0,
        metadata={"help": "Filter out audio files that are shorter than `max_duration_in_seconds` seconds"},
    )
    train_dataset_prefix: List[str] = field(
        default="train",
        metadata={"help": ""},
    )
    valid_dataset_prefix: List[str] = field(
        default="validation",
        metadata={"help": ""},
    )
    test_dataset_prefix: List[str] = field(
        default="eval_other",
        metadata={"help": ""},
    )
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
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."},
    )

    pad_to_multiple_of: int = field(
        default=None,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        },
    )
    max_gumbel_temperature: float = field(
        default=2.0,
        metadata={"help": "Maximum temperature for gumbel softmax."},
    )
    min_gumbel_temperature: float = field(
        default=0.5,
        metadata={"help": "Minimum temperature for gumbel softmax."},
    )
    gumbel_temperature_decay: float = field(
        default=0.999995, metadata={"help": "Decay of gumbel temperature during training."}
    )
    mask_time_prob: float = field(
        default=None,
        metadata={
            "help": "Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked in the contrastive task. If omitted, will pull value from model config."
        },
    )
    mask_time_length: int = field(
        default=None,
        metadata={
            "help": "Length of each vector mask span to mask along the time axis in the contrastive task. If omitted, will pull value from model config."
        },
    )
    attn_implementation: str = field(default=None)

    wandb_code_log_dir: str = field(
        default="",
        metadata={"help": ""},
    )
    sampling_rate: int = field(
        default=16000,
        metadata={"help": ""},
    )
    valid_exclude_ls: List[str] = field(
        default=None,
        metadata={"help": ""},
    )
    valid_truncate_num: int = field(
        default=3000,
        metadata={"help": ""},
    )
    split_valid: bool = field(
        default=False,
        metadata={"help": ""},
    )
