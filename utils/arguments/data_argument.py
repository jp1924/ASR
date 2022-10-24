from dataclasses import dataclass, field


@dataclass
class DataArgument:
    train_data: str = field(default=None)
    valid_data: str = field(default=None)
    num_proc: int = field(default=1)
