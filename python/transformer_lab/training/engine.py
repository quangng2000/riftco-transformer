"""Shared, stage-neutral causal language-model training machinery."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import dataclass
import math
import random
from typing import Callable, Protocol, Sequence

from ..native import (
    Adam,
    DecoderOnlyTransformer,
    backend_available,
    cross_entropy,
)


TokenRow = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    """A rectangular next-token batch."""

    inputs: tuple[TokenRow, ...]
    targets: tuple[TokenRow, ...]

    def __post_init__(self) -> None:
        try:
            inputs = tuple(tuple(row) for row in self.inputs)
            targets = tuple(tuple(row) for row in self.targets)
        except TypeError as error:
            raise TypeError(
                "training inputs and targets must be iterables of rows"
            ) from error
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "targets", targets)

        if not inputs:
            raise ValueError("training batch must contain at least one row")
        if len(inputs) != len(targets):
            raise ValueError(
                "training inputs and targets must have the same row count"
            )
        width = len(inputs[0])
        if width == 0:
            raise ValueError("training batch rows must not be empty")
        for row_index, (input_row, target_row) in enumerate(
            zip(inputs, targets)
        ):
            if len(input_row) != width or len(target_row) != width:
                raise ValueError(
                    "training inputs and targets must be rectangular and "
                    "have matching widths"
                )
            for column_index, token in enumerate(input_row):
                _token_id(
                    token,
                    f"inputs[{row_index}][{column_index}]",
                )
            for column_index, token in enumerate(target_row):
                _token_id(
                    token,
                    f"targets[{row_index}][{column_index}]",
                )

    @property
    def batch_size(self) -> int:
        return len(self.inputs)

    @property
    def context_size(self) -> int:
        return len(self.inputs[0])


class BatchSource(Protocol):
    """Strategy interface for interchangeable training-data sources."""

    def next_batch(self) -> TrainingBatch:
        """Return the next rectangular causal-language-model batch."""


class MetricSink(Protocol):
    """Adapter interface for console, CSV, or experiment metric outputs."""

    def __call__(self, metric: TrainingMetric) -> None:
        """Consume one immutable metric record."""


@dataclass(frozen=True, slots=True)
class TrainingLoopConfig:
    """Stage-neutral optimizer-loop controls."""

    steps: int
    evaluation_interval: int = 10
    loss_average_window: int = 10

    def __post_init__(self) -> None:
        for name, value in (
            ("steps", self.steps),
            ("evaluation_interval", self.evaluation_interval),
            ("loss_average_window", self.loss_average_window),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True, slots=True)
class TrainingMetric:
    """One training or validation observation."""

    phase: str
    step: int
    loss: float
    perplexity: float
    loss_average: float | None = None
    gradient_norm: float | None = None
    clip_scale: float | None = None


def selected_backend(requested: str) -> str:
    """Resolve ``auto`` and reject an unavailable explicitly requested GPU."""

    if requested not in {"auto", "cpu", "metal"}:
        raise ValueError("backend must be 'auto', 'cpu', or 'metal'")
    if requested == "auto":
        return "metal" if backend_available("metal") else "cpu"
    if requested == "metal" and not backend_available("metal"):
        raise RuntimeError("Metal was requested but is unavailable")
    return requested


def loss_perplexity(loss: float) -> float:
    """Convert mean cross-entropy to perplexity without overflowing."""

    try:
        return math.exp(loss)
    except OverflowError:
        return math.inf


class RandomWindowBatchSource:
    """Uniformly sample shifted windows from one token sequence."""

    def __init__(
        self,
        tokens: Sequence[int],
        *,
        batch_size: int,
        context_size: int,
        random_seed: int,
    ) -> None:
        self._tokens = tuple(tokens)
        self._batch_size = _positive_integer(batch_size, "batch_size")
        self._context_size = _positive_integer(
            context_size,
            "context_size",
        )
        if len(self._tokens) <= self._context_size:
            raise ValueError(
                "tokens must contain at least context_size + 1 values"
            )
        self._random = random.Random(
            _nonnegative_integer(random_seed, "random_seed")
        )

    def next_batch(self) -> TrainingBatch:
        start_count = len(self._tokens) - self._context_size
        inputs: list[TokenRow] = []
        targets: list[TokenRow] = []
        for _ in range(self._batch_size):
            start = self._random.randrange(start_count)
            stop = start + self._context_size
            inputs.append(self._tokens[start:stop])
            targets.append(self._tokens[start + 1 : stop + 1])
        return TrainingBatch(tuple(inputs), tuple(targets))


class SequenceWindowBatchSource:
    """Uniformly sample windows without crossing sequence boundaries."""

    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        batch_size: int,
        context_size: int,
        random_seed: int,
    ) -> None:
        self._batch_size = _positive_integer(batch_size, "batch_size")
        self._context_size = _positive_integer(
            context_size,
            "context_size",
        )
        self._random = random.Random(
            _nonnegative_integer(random_seed, "random_seed")
        )

        usable_sequences: list[tuple[int, ...]] = []
        cumulative_window_counts: list[int] = []
        total = 0
        for sequence in sequences:
            copied = tuple(sequence)
            window_count = len(copied) - self._context_size
            if window_count <= 0:
                continue
            usable_sequences.append(copied)
            total += window_count
            cumulative_window_counts.append(total)
        if total == 0:
            raise ValueError(
                "no sequence contains at least context_size + 1 tokens"
            )
        self._sequences = tuple(usable_sequences)
        self._cumulative_window_counts = tuple(
            cumulative_window_counts
        )
        self._total_window_count = total

    def next_batch(self) -> TrainingBatch:
        inputs: list[TokenRow] = []
        targets: list[TokenRow] = []
        for _ in range(self._batch_size):
            flat_index = self._random.randrange(self._total_window_count)
            sequence_index = bisect_right(
                self._cumulative_window_counts,
                flat_index,
            )
            previous_total = (
                0
                if sequence_index == 0
                else self._cumulative_window_counts[sequence_index - 1]
            )
            start = flat_index - previous_total
            stop = start + self._context_size
            sequence = self._sequences[sequence_index]
            inputs.append(sequence[start:stop])
            targets.append(sequence[start + 1 : stop + 1])
        return TrainingBatch(tuple(inputs), tuple(targets))


class ExampleWindowBatchSource:
    """Sample examples uniformly, then sample a window inside each example.

    This strategy prevents a long instruction from receiving proportionally
    more probability solely because it contains more possible windows.
    """

    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        batch_size: int,
        context_size: int,
        random_seed: int,
    ) -> None:
        self._batch_size = _positive_integer(batch_size, "batch_size")
        self._context_size = _positive_integer(
            context_size,
            "context_size",
        )
        self._random = random.Random(
            _nonnegative_integer(random_seed, "random_seed")
        )
        usable_sequences: list[TokenRow] = []
        for sequence in sequences:
            copied = tuple(sequence)
            if len(copied) > self._context_size:
                usable_sequences.append(copied)
        self._sequences = tuple(usable_sequences)
        if not self._sequences:
            raise ValueError(
                "no sequence contains at least context_size + 1 tokens"
            )

    def next_batch(self) -> TrainingBatch:
        inputs: list[TokenRow] = []
        targets: list[TokenRow] = []
        for _ in range(self._batch_size):
            sequence = self._sequences[
                self._random.randrange(len(self._sequences))
            ]
            start = self._random.randrange(
                len(sequence) - self._context_size
            )
            stop = start + self._context_size
            inputs.append(sequence[start:stop])
            targets.append(sequence[start + 1 : stop + 1])
        return TrainingBatch(tuple(inputs), tuple(targets))


def fixed_batches(source: BatchSource, count: int) -> tuple[TrainingBatch, ...]:
    """Materialize a deterministic evaluation set from its own source."""

    checked_count = _positive_integer(count, "count")
    return tuple(source.next_batch() for _ in range(checked_count))


class CausalLanguageModelTrainer:
    """Reusable forward/backward/Adam loop shared by training stages."""

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        optimizer: Adam,
    ) -> None:
        if not isinstance(model, DecoderOnlyTransformer):
            raise TypeError("model must be a DecoderOnlyTransformer")
        if not isinstance(optimizer, Adam):
            raise TypeError("optimizer must be an Adam")
        if model.backend != optimizer.backend:
            raise ValueError("model and optimizer backends must match")
        self._model = model
        self._optimizer = optimizer

    def evaluate(
        self,
        batches: Sequence[TrainingBatch],
    ) -> float:
        if not batches:
            raise ValueError("evaluation batches must not be empty")
        loss_total = 0.0
        for batch in batches:
            with self._model(batch.inputs) as logits:
                with cross_entropy(logits, batch.targets) as loss:
                    loss_total += loss.item()
        return loss_total / len(batches)

    def run(
        self,
        source: BatchSource,
        config: TrainingLoopConfig,
        *,
        validation_batches: Sequence[TrainingBatch] = (),
        metric_sink: Callable[[TrainingMetric], None] | None = None,
    ) -> tuple[TrainingMetric, ...]:
        if not isinstance(config, TrainingLoopConfig):
            raise TypeError("config must be a TrainingLoopConfig")
        metrics: list[TrainingMetric] = []

        def publish(metric: TrainingMetric) -> None:
            metrics.append(metric)
            if metric_sink is not None:
                metric_sink(metric)

        initial_step = self._optimizer.step_count
        final_step = initial_step + config.steps

        if validation_batches:
            validation_loss = self.evaluate(validation_batches)
            publish(
                TrainingMetric(
                    phase="validation",
                    step=initial_step,
                    loss=validation_loss,
                    perplexity=loss_perplexity(validation_loss),
                )
            )

        recent_losses: deque[float] = deque(
            maxlen=config.loss_average_window
        )
        for _ in range(config.steps):
            batch = source.next_batch()
            with self._model(batch.inputs) as logits:
                with cross_entropy(logits, batch.targets) as loss:
                    loss_value = loss.item()
                    loss.backward()
                    statistics = self._optimizer.step()

            recent_losses.append(loss_value)
            publish(
                TrainingMetric(
                    phase="train",
                    step=statistics.step,
                    loss=loss_value,
                    loss_average=(
                        sum(recent_losses) / len(recent_losses)
                    ),
                    perplexity=loss_perplexity(loss_value),
                    gradient_norm=statistics.gradient_norm,
                    clip_scale=statistics.clip_scale,
                )
            )

            should_evaluate = (
                statistics.step % config.evaluation_interval == 0
                or statistics.step == final_step
            )
            if validation_batches and should_evaluate:
                validation_loss = self.evaluate(validation_batches)
                publish(
                    TrainingMetric(
                        phase="validation",
                        step=statistics.step,
                        loss=validation_loss,
                        perplexity=loss_perplexity(validation_loss),
                    )
                )
        return tuple(metrics)


def _positive_integer(value: object, name: str) -> int:
    checked = _nonnegative_integer(value, name)
    if checked == 0:
        raise ValueError(f"{name} must be greater than zero")
    return checked


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _token_id(value: object, name: str) -> int:
    checked = _nonnegative_integer(value, name)
    if checked > (1 << 32) - 1:
        raise ValueError(f"{name} must be at most 2**32 - 1")
    return checked


__all__ = [
    "BatchSource",
    "CausalLanguageModelTrainer",
    "ExampleWindowBatchSource",
    "MetricSink",
    "RandomWindowBatchSource",
    "SequenceWindowBatchSource",
    "TrainingBatch",
    "TrainingLoopConfig",
    "TrainingMetric",
    "fixed_batches",
    "loss_perplexity",
    "selected_backend",
]
