from dataclasses import dataclass, field


@dataclass
class DataArgument:
    name_or_script: str = field(default=None)
