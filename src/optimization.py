import importlib
import math
from enum import EnumMeta
from functools import partial
from typing import Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

import transformers.trainer_utils
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION, SchedulerType
from transformers.utils import ExplicitEnum


class NewSchedulerMeta(EnumMeta):
    def __new__(metacls, name, bases, class_dict):
        # SchedulerType의 멤버를 동적으로 추가
        for member in SchedulerType:
            class_dict[member.name] = member.value
        return super().__new__(metacls, name, bases, class_dict)


class NewSchedulerType(ExplicitEnum, metaclass=NewSchedulerMeta):
    TRI_STAGE = "tri_stage"  # 추가됨


def _get_tri_stage_schedule_with_warmup_lr_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_hold_steps: int,
    num_decay_steps: int,
    decay_factor: float,
    final_learning_rate: float,
) -> float:
    if current_step < num_warmup_steps:  # stage 1: warmup
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step < (num_warmup_steps + num_hold_steps):  # stage 2: hold
        return 1.0
    elif current_step < (num_warmup_steps + num_hold_steps + num_decay_steps):  # stage 3: decay
        decay_steps = current_step - (num_warmup_steps + num_hold_steps)
        decay_rate = math.log(decay_factor) / num_decay_steps
        return float(math.exp(decay_rate * decay_steps))
    else:  # stage 4: final
        return final_learning_rate


def get_tri_stage_schedule_with_warmup_lr_lambda(
    optimizer: Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    num_hold_steps: Union[int, float],
    num_decay_steps: Union[int, float],
    final_learning_rate: float,
    last_epoch: int = -1,
) -> LambdaLR:
    learning_rate = optimizer.defaults["lr"]
    if isinstance(num_hold_steps, float):
        num_hold_steps = int(num_training_steps * num_hold_steps)

    if isinstance(num_decay_steps, float):
        num_decay_steps = int(num_training_steps * num_decay_steps)

    decay_factor = final_learning_rate / learning_rate

    lr_lambda = partial(
        _get_tri_stage_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_hold_steps=num_hold_steps,
        num_decay_steps=num_decay_steps,
        final_learning_rate=final_learning_rate / learning_rate,
        decay_factor=decay_factor,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


NEW_TYPE_TO_SCHEDULER_FUNCTION = TYPE_TO_SCHEDULER_FUNCTION
NEW_TYPE_TO_SCHEDULER_FUNCTION.update({NewSchedulerType.TRI_STAGE: get_tri_stage_schedule_with_warmup_lr_lambda})

transformers.trainer_utils.SchedulerType = NewSchedulerType
transformers.training_args.SchedulerType = NewSchedulerType
transformers.optimization.SchedulerType = NewSchedulerType
transformers.optimization.TYPE_TO_SCHEDULER_FUNCTION = TYPE_TO_SCHEDULER_FUNCTION
