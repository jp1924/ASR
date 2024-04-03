from dataclasses import dataclass, field

from transformers import TrainingArguments


@dataclass
class Wav2Vec2PretrainingArguments(TrainingArguments):
    dataset_name: str = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_names: list[str] = field(
        default=None,
        metadata={
            "help": "The configuration names of the dataset to use (via the datasets library).",
            "required": True,
            "nargs": "+",
        },
    )
    dataset_split_names: list[str] = field(
        default=None,
        metadata={
            "help": "The names of the training data set splits to use (via the datasets library).",
            "required": True,
            "nargs": "+",
        },
    )
    preprocessing_num_workers: int = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    audio_column_name: str = field(
        default="audio", metadata={"help": "Column in the dataset that contains speech file path. Defaults to 'audio'"}
    )
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models.", "required": True},
    )
    train_cache_file_name: str = field(default=None, metadata={"help": "Path to the train cached file name"})
    validation_cache_file_name: str = field(default=None, metadata={"help": "Path to the validation cached file name"})
    output_dir: str = field(default=None, metadata={"help": "Where to store the final model."})
    seed: int = field(default=0, metadata={"help": "A seed for reproducible training."})
    max_duration_in_seconds: float = field(
        default=5.0, metadata={"help": "Filter out audio files that are longer than `max_duration_in_seconds` seconds"}
    )
    min_duration_in_seconds: float = field(
        default=3.0, metadata={"help": "Filter out audio files that are shorter than min_duration_in_seconds seconds"}
    )

    pad_to_multiple_of: int = field(
        default=None,
        metadata={
            "help": "If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta)."
        },
    )
    max_gumbel_temperature: float = field(default=2.0, metadata={"help": "Maximum temperature for gumbel softmax."})
    min_gumbel_temperature: float = field(default=0.5, metadata={"help": "Minimum temperature for gumbel softmax."})
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
