import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

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
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    valid_dataset_prefix: List[str] = field(
        default="validation",
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    test_dataset_prefix: List[str] = field(
        default="eval_other",
        metadata={"help": "A prefix required to distinguish splits in the data loaded by load_dataset."},
    )
    data_truncate_map: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "A map to truncate part of the data. {'repo_name': {'train': 3000, 'validation': 1500}}."},
    )
    data_name_map: Optional[Union[dict, str]] = field(
        default=None,
        metadata={"help": "A map to config_name of the data. {'repo_name': 'data_config_name'"},
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

    pad_to_multiple_of: Optional[int] = field(
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
        default=0.999995,
        metadata={"help": "Decay of gumbel temperature during training."},
    )
    mask_time_prob: Optional[float] = field(
        default=None,
        metadata={
            "help": "Percentage (between 0 and 1) of all feature vectors along the time axis which will be masked in the contrastive task. If omitted, will pull value from model config."
        },
    )
    mask_time_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Length of each vector mask span to mask along the time axis in the contrastive task. If omitted, will pull value from model config."
        },
    )
    attn_implementation: str = field(
        default=None,
        metadata={
            "help": "Length of each vector mask span to mask along the time axis in the contrastive task. If omitted, will pull value from model config."
        },
    )

    sampling_rate: int = field(
        default=16000,
        metadata={"help": ""},
    )

    def __post_init__(self):
        super().__post_init__()
        self.data_truncate_map = json.loads(self.data_truncate_map) if self.data_truncate_map else {}
        self.data_name_map = json.loads(self.data_name_map) if self.data_name_map else {}

        self.train_dataset_prefix = self.train_dataset_prefix if self.train_dataset_prefix else []
        self.valid_dataset_prefix = self.valid_dataset_prefix if self.valid_dataset_prefix else []
        self.test_dataset_prefix = self.test_dataset_prefix if self.test_dataset_prefix else []

        self.cache_dir = Path(self.cache_dir) if self.cache_dir else None
