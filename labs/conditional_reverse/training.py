"""Python-owned Adam loop for the conditional-reverse learned variants."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

from riftco_transformer import Adam, cross_entropy_time_range

from .config import TrainingConfig
from .data import Batch, TokenCodec, iter_batches
from .evaluation import EvaluationMetrics, evaluate_model
from .model import ModelRuntime
from .protocol import Example


@dataclass(frozen=True, slots=True)
class TrainingStepMetrics:
    step: int
    epoch: int
    batch_index: int
    example_count: int
    target_loss: float
    gradient_norm: float
    clip_scale: float


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    final_step: int
    training_example_count: int
    mean_training_loss: float
    validation: EvaluationMetrics


@dataclass(frozen=True, slots=True)
class TrainingHistory:
    steps: tuple[TrainingStepMetrics, ...]
    epochs: tuple[EpochMetrics, ...]
    final_step: int


def target_half_loss(logits: object, batch: Batch) -> object:
    """Return CE over exactly the task logits interval ``[L, 2L)``."""

    if not isinstance(batch, Batch):
        raise TypeError("batch must be a Batch")
    return cross_entropy_time_range(
        logits,
        batch.target_rows,
        batch.sequence_length,
        batch.sequence_length,
    )


def train_model(
    runtime: ModelRuntime,
    training_examples: Sequence[Example],
    validation_examples: Sequence[Example],
    codec: TokenCodec,
    config: TrainingConfig,
    *,
    step_sink: Callable[[TrainingStepMetrics], None] | None = None,
) -> TrainingHistory:
    """Fit one runtime while Python owns epochs, shuffling, and validation."""

    if not training_examples:
        raise ValueError("training_examples must not be empty")
    if not validation_examples:
        raise ValueError("validation_examples must not be empty")
    if not isinstance(codec, TokenCodec):
        raise TypeError("codec must be a TokenCodec")
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig")
    if step_sink is not None and not callable(step_sink):
        raise TypeError("step_sink must be callable")

    steps: list[TrainingStepMetrics] = []
    epochs: list[EpochMetrics] = []
    with runtime.model.parameters() as parameters:
        with Adam(
            parameters,
            learning_rate=config.learning_rate,
            beta1=config.beta1,
            beta2=config.beta2,
            epsilon=config.epsilon,
            maximum_gradient_norm=config.maximum_gradient_norm,
        ) as optimizer:
            if optimizer.backend != runtime.backend:
                raise RuntimeError("model and Adam backends differ")
            reached_limit = False
            for epoch in range(1, config.epochs + 1):
                weighted_loss = 0.0
                trained_examples = 0
                for batch_index, batch in enumerate(
                    iter_batches(
                        training_examples,
                        codec,
                        config.batch_size,
                        shuffle=config.shuffle,
                        seed=config.seed,
                        epoch=epoch - 1,
                    ),
                    start=1,
                ):
                    if (
                        config.maximum_steps is not None
                        and optimizer.step_count >= config.maximum_steps
                    ):
                        reached_limit = True
                        break
                    optimizer.zero_grad()
                    forward = runtime.model.forward(batch.input_rows)
                    with forward.logits as logits:
                        with target_half_loss(logits, batch) as loss:
                            loss_value = loss.item()
                            if not math.isfinite(loss_value):
                                raise ValueError("training loss must be finite")
                            loss.backward()
                            statistics = optimizer.step()
                    metric = TrainingStepMetrics(
                        step=statistics.step,
                        epoch=epoch,
                        batch_index=batch_index,
                        example_count=batch.batch_size,
                        target_loss=loss_value,
                        gradient_norm=statistics.gradient_norm,
                        clip_scale=statistics.clip_scale,
                    )
                    steps.append(metric)
                    if step_sink is not None:
                        step_sink(metric)
                    weighted_loss += loss_value * batch.batch_size
                    trained_examples += batch.batch_size
                    if (
                        config.maximum_steps is not None
                        and optimizer.step_count >= config.maximum_steps
                    ):
                        reached_limit = True
                        break

                if trained_examples == 0:
                    break
                validation = evaluate_model(
                    runtime,
                    validation_examples,
                    codec,
                    config.evaluation_batch_size,
                    retain_per_example=False,
                )
                epochs.append(
                    EpochMetrics(
                        epoch=epoch,
                        final_step=optimizer.step_count,
                        training_example_count=trained_examples,
                        mean_training_loss=weighted_loss / trained_examples,
                        validation=validation.metrics,
                    )
                )
                if reached_limit:
                    break
    return TrainingHistory(
        tuple(steps),
        tuple(epochs),
        steps[-1].step if steps else 0,
    )


__all__ = [
    "EpochMetrics",
    "TrainingHistory",
    "TrainingStepMetrics",
    "target_half_loss",
    "train_model",
]
