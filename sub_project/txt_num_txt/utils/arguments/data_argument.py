from dataclasses import dataclass, field


@dataclass
class DataArgument:
    train_csv: str = field(default=None)
    valid_csv: str = field(default=None)
    max_length: int = field(default=512)
    num_proc: int = field(default=1)
