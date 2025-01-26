import importlib
import math
from functools import partial
from typing import Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.trainer_utils import SchedulerType


class NewSchedulerType(SchedulerType):
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


def set_scheduler():
    NEW_TYPE_TO_SCHEDULER_FUNCTION = TYPE_TO_SCHEDULER_FUNCTION
    NEW_TYPE_TO_SCHEDULER_FUNCTION.update({NewSchedulerType.TRI_STAGE: get_tri_stage_schedule_with_warmup_lr_lambda})

    # NOTE: 빼놓고 추가하지 않은 곳이 있으면 정상동작 안할 가능성이 존재함. 확인 필요
    module = importlib.import_module("transformers.optimization")
    setattr(module, "TYPE_TO_SCHEDULER_FUNCTION", NEW_TYPE_TO_SCHEDULER_FUNCTION)

    module = importlib.import_module("transformers.trainer_utils")
    setattr(module, "SchedulerType", NewSchedulerType)

    module = importlib.import_module("transformers.training_args")
    setattr(module, "SchedulerType", NewSchedulerType)

    module = importlib.import_module("transformers.optimization")
    setattr(module, "SchedulerType", NewSchedulerType)
