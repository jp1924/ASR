import importlib
import math
from functools import partial
from typing import Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
from transformers.utils import ExplicitEnum


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
    learning_rate = optimizer.lr
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


def set_scheduler() -> None:
    module = importlib.import_module("transformers.trainer_utils")
    module = importlib.import_module("transformers.optimization")

    class SchedulerType(ExplicitEnum):
        LINEAR = "linear"
        COSINE = "cosine"
        COSINE_WITH_RESTARTS = "cosine_with_restarts"
        POLYNOMIAL = "polynomial"
        CONSTANT = "constant"
        CONSTANT_WITH_WARMUP = "constant_with_warmup"
        INVERSE_SQRT = "inverse_sqrt"
        REDUCE_ON_PLATEAU = "reduce_lr_on_plateau"
        COSINE_WITH_MIN_LR = "cosine_with_min_lr"
        WARMUP_STABLE_DECAY = "warmup_stable_decay"
        TRI_STAGE = "tri_stage"

    NEW_TYPE_TO_SCHEDULER_FUNCTION = TYPE_TO_SCHEDULER_FUNCTION
    NEW_TYPE_TO_SCHEDULER_FUNCTION.update({SchedulerType.TRI_STAGE: get_tri_stage_schedule_with_warmup_lr_lambda})

    setattr(module, "TYPE_TO_SCHEDULER_FUNCTION", NEW_TYPE_TO_SCHEDULER_FUNCTION)
    setattr(module, "SchedulerType", SchedulerType)
