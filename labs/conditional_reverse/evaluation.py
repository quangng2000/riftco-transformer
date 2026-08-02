"""Learned-model evaluation, interventions, and streaming probe analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from typing import Callable, Iterable, Iterator, Sequence

from riftco_transformer import cross_entropy_time_range
from riftco_transformer.programmed import (
    ProgramAugmentedForwardOptions,
    ProgramInputSteering,
    Representation,
)

from .analysis import (
    PairedEffect,
    PcaAccumulator,
    PcaResult,
    paired_effect,
    steering_specs_for_variant,
)
from .config import AnalysisConfig, Variant
from .data import Batch, TokenCodec, iter_batches
from .model import ModelRuntime
from .protocol import Example


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    loss: float
    example_count: int
    target_token_count: int
    correct_target_token_count: int
    correct_sequence_count: int
    target_token_accuracy: float
    exact_sequence_accuracy: float
    reverse_example_count: int
    reverse_target_token_accuracy: float
    reverse_exact_sequence_accuracy: float
    copy_example_count: int
    copy_target_token_accuracy: float
    copy_exact_sequence_accuracy: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: EvaluationMetrics
    predictions: tuple[tuple[int, ...], ...]
    per_example_token_accuracy: tuple[float, ...]
    per_example_exact_accuracy: tuple[float, ...]
    prediction_fingerprint: str

    @property
    def retains_per_example(self) -> bool:
        return bool(self.predictions)


@dataclass(frozen=True, slots=True)
class HypothesisAgreement:
    target_token_accuracy: float
    exact_sequence_accuracy: float


@dataclass(frozen=True, slots=True)
class HypothesisScores:
    conditional: HypothesisAgreement
    copy_only: HypothesisAgreement
    reverse_only: HypothesisAgreement


@dataclass(frozen=True, slots=True)
class AblationCondition:
    name: str
    metrics: EvaluationMetrics
    token_accuracy_effect: PairedEffect
    exact_accuracy_effect: PairedEffect


@dataclass(frozen=True, slots=True)
class AblationSuite:
    baseline: EvaluationMetrics
    conditions: tuple[AblationCondition, ...]


@dataclass(frozen=True, slots=True)
class SteeringCondition:
    name: str
    metrics: EvaluationMetrics
    hypotheses: HypothesisScores
    token_accuracy_effect: PairedEffect
    exact_accuracy_effect: PairedEffect


@dataclass(frozen=True, slots=True)
class SteeringSuite:
    example_count: int
    baseline: EvaluationMetrics
    baseline_hypotheses: HypothesisScores
    conditions: tuple[SteeringCondition, ...]


@dataclass(frozen=True, slots=True)
class ProbePcaResult:
    probe_metrics: EvaluationMetrics
    pca: PcaResult
    representation_name: str = "program.output.raw"


class _MetricAccumulator:
    __slots__ = (
        "copy_correct_sequences",
        "copy_correct_tokens",
        "copy_examples",
        "correct_sequences",
        "correct_tokens",
        "examples",
        "loss_sum",
        "reverse_correct_sequences",
        "reverse_correct_tokens",
        "reverse_examples",
        "sequence_length",
    )

    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length
        self.loss_sum = 0.0
        self.examples = 0
        self.correct_tokens = 0
        self.correct_sequences = 0
        self.reverse_examples = 0
        self.reverse_correct_tokens = 0
        self.reverse_correct_sequences = 0
        self.copy_examples = 0
        self.copy_correct_tokens = 0
        self.copy_correct_sequences = 0

    def update(
        self,
        batch: Batch,
        predictions: tuple[tuple[int, ...], ...],
        loss: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        if not math.isfinite(loss):
            raise ValueError("evaluation loss must be finite")
        if len(predictions) != batch.batch_size:
            raise ValueError("prediction rows must match batch size")
        self.loss_sum += loss * batch.batch_size
        self.examples += batch.batch_size
        token_accuracies: list[float] = []
        exact_accuracies: list[float] = []
        for prediction, target, reversed_branch in zip(
            predictions,
            batch.output_rows,
            batch.reversed,
        ):
            if len(prediction) != self.sequence_length:
                raise ValueError("prediction has the wrong sequence length")
            correct = sum(left == right for left, right in zip(prediction, target))
            exact = correct == self.sequence_length
            self.correct_tokens += correct
            self.correct_sequences += int(exact)
            if reversed_branch:
                self.reverse_examples += 1
                self.reverse_correct_tokens += correct
                self.reverse_correct_sequences += int(exact)
            else:
                self.copy_examples += 1
                self.copy_correct_tokens += correct
                self.copy_correct_sequences += int(exact)
            token_accuracies.append(correct / self.sequence_length)
            exact_accuracies.append(float(exact))
        return tuple(token_accuracies), tuple(exact_accuracies)

    def finish(self) -> EvaluationMetrics:
        if self.examples == 0:
            raise ValueError("evaluation examples must not be empty")
        token_count = self.examples * self.sequence_length
        return EvaluationMetrics(
            loss=self.loss_sum / self.examples,
            example_count=self.examples,
            target_token_count=token_count,
            correct_target_token_count=self.correct_tokens,
            correct_sequence_count=self.correct_sequences,
            target_token_accuracy=self.correct_tokens / token_count,
            exact_sequence_accuracy=self.correct_sequences / self.examples,
            reverse_example_count=self.reverse_examples,
            reverse_target_token_accuracy=_safe_ratio(
                self.reverse_correct_tokens,
                self.reverse_examples * self.sequence_length,
            ),
            reverse_exact_sequence_accuracy=_safe_ratio(
                self.reverse_correct_sequences,
                self.reverse_examples,
            ),
            copy_example_count=self.copy_examples,
            copy_target_token_accuracy=_safe_ratio(
                self.copy_correct_tokens,
                self.copy_examples * self.sequence_length,
            ),
            copy_exact_sequence_accuracy=_safe_ratio(
                self.copy_correct_sequences,
                self.copy_examples,
            ),
        )


def evaluate_model(
    runtime: ModelRuntime,
    examples: Sequence[Example],
    codec: TokenCodec,
    batch_size: int,
    *,
    options: ProgramAugmentedForwardOptions | None = None,
    capture_name: str | None = None,
    capture_sink: Callable[[Representation], None] | None = None,
    retain_per_example: bool = True,
) -> EvaluationResult:
    """Evaluate in bounded batches and optionally stream one named trace."""

    if not examples:
        raise ValueError("evaluation examples must not be empty")
    if not isinstance(codec, TokenCodec):
        raise TypeError("codec must be a TokenCodec")
    if capture_sink is not None and capture_name is None:
        raise ValueError("capture_sink requires capture_name")
    if capture_name is not None and (
        not isinstance(capture_name, str) or not capture_name
    ):
        raise ValueError("capture_name must be a nonempty str")
    if not isinstance(retain_per_example, bool):
        raise TypeError("retain_per_example must be a bool")
    configured = options or ProgramAugmentedForwardOptions()
    if capture_name is not None and not configured.capture_representations:
        configured = replace(configured, capture_representations=True)

    accumulator = _MetricAccumulator(runtime.protocol.sequence_length)
    predictions: list[tuple[int, ...]] = []
    token_accuracies: list[float] = []
    exact_accuracies: list[float] = []
    fingerprint = hashlib.sha256()
    for batch in iter_batches(examples, codec, batch_size):
        forward = runtime.model.forward(batch.input_rows, configured)
        with forward.logits as logits:
            expected_shape = (
                batch.batch_size,
                batch.context_length,
                codec.vocabulary_size,
            )
            if logits.shape != expected_shape:
                raise RuntimeError(
                    f"logits shape {logits.shape} does not match {expected_shape}"
                )
            values = logits.tolist()
            batch_predictions = target_argmax(
                values,
                batch,
                codec.vocabulary_size,
            )
            with cross_entropy_time_range(
                logits,
                batch.target_rows,
                batch.sequence_length,
                batch.sequence_length,
            ) as loss:
                loss_value = loss.item()

        batch_token, batch_exact = accumulator.update(
            batch,
            batch_predictions,
            loss_value,
        )
        for row in batch_predictions:
            for token in row:
                fingerprint.update(token.to_bytes(4, "little", signed=False))
        if retain_per_example:
            predictions.extend(batch_predictions)
            token_accuracies.extend(batch_token)
            exact_accuracies.extend(batch_exact)

        if capture_name is not None:
            try:
                captured = forward.representations.at(capture_name)
            except KeyError as error:
                raise RuntimeError(
                    f"native trace omitted {capture_name!r}"
                ) from error
            if capture_sink is not None:
                capture_sink(captured)

    return EvaluationResult(
        metrics=accumulator.finish(),
        predictions=tuple(predictions),
        per_example_token_accuracy=tuple(token_accuracies),
        per_example_exact_accuracy=tuple(exact_accuracies),
        prediction_fingerprint=fingerprint.hexdigest(),
    )


def target_argmax(
    logits: Sequence[float],
    batch: Batch,
    vocabulary_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Greedy predictions only for logits times ``[L, 2L)``."""

    if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int):
        raise TypeError("vocabulary_size must be an int")
    if vocabulary_size <= 0:
        raise ValueError("vocabulary_size must be positive")
    expected = batch.batch_size * batch.context_length * vocabulary_size
    if len(logits) != expected:
        raise ValueError("logits has the wrong flattened size")
    result: list[tuple[int, ...]] = []
    for row in range(batch.batch_size):
        predictions: list[int] = []
        for time in range(batch.sequence_length, batch.context_length):
            offset = (row * batch.context_length + time) * vocabulary_size
            best_token = 0
            best_value = float(logits[offset])
            if not math.isfinite(best_value):
                raise ValueError("logits must be finite")
            for token in range(1, vocabulary_size):
                value = float(logits[offset + token])
                if not math.isfinite(value):
                    raise ValueError("logits must be finite")
                if value > best_value:
                    best_token = token
                    best_value = value
            predictions.append(best_token)
        result.append(tuple(predictions))
    return tuple(result)


def score_hypotheses(
    examples: Sequence[Example],
    predictions: Sequence[Sequence[int]],
    codec: TokenCodec,
) -> HypothesisScores:
    """Measure agreement with conditional, copy-only, and reverse-only rules."""

    if not examples:
        raise ValueError("hypothesis scoring requires examples")
    if len(examples) != len(predictions):
        raise ValueError("predictions must match examples")
    conditional = tuple(codec.encode(example.target) for example in examples)
    copy_only = tuple(codec.encode(example.source) for example in examples)
    reverse_only = tuple(
        codec.encode(example.source[::-1]) for example in examples
    )
    copied_predictions = tuple(tuple(row) for row in predictions)
    return HypothesisScores(
        conditional=_hypothesis_agreement(copied_predictions, conditional),
        copy_only=_hypothesis_agreement(copied_predictions, copy_only),
        reverse_only=_hypothesis_agreement(copied_predictions, reverse_only),
    )


def run_paired_ablations(
    runtime: ModelRuntime,
    examples: Sequence[Example],
    codec: TokenCodec,
    batch_size: int,
    *,
    shift: int = 1,
) -> AblationSuite:
    """Run same-order learned/program/combined batch-roll interventions."""

    baseline = evaluate_model(runtime, examples, codec, batch_size)
    conditions: list[tuple[str, ProgramAugmentedForwardOptions]] = [
        (
            "learned_attention",
            ProgramAugmentedForwardOptions(
                batch_roll_learned_attention=True,
                batch_roll_shift=shift,
            ),
        )
    ]
    if runtime.has_program:
        conditions.extend(
            (
                (
                    "program_output",
                    ProgramAugmentedForwardOptions(
                        batch_roll_shift=shift,
                        ablate_program_output=True,
                    ),
                ),
                (
                    "combined",
                    ProgramAugmentedForwardOptions(
                        batch_roll_learned_attention=True,
                        batch_roll_shift=shift,
                        ablate_program_output=True,
                    ),
                ),
            )
        )
    results: list[AblationCondition] = []
    for name, options in conditions:
        changed = evaluate_model(
            runtime,
            examples,
            codec,
            batch_size,
            options=options,
        )
        results.append(
            AblationCondition(
                name=name,
                metrics=changed.metrics,
                token_accuracy_effect=paired_effect(
                    baseline.per_example_token_accuracy,
                    changed.per_example_token_accuracy,
                ),
                exact_accuracy_effect=paired_effect(
                    baseline.per_example_exact_accuracy,
                    changed.per_example_exact_accuracy,
                ),
            )
        )
    return AblationSuite(baseline.metrics, tuple(results))


def run_f_steering(
    runtime: ModelRuntime,
    examples: Sequence[Example],
    codec: TokenCodec,
    batch_size: int,
    analysis: AnalysisConfig,
) -> SteeringSuite:
    """Evaluate selector-basis masking/amplification on a balanced set."""

    if runtime.variant is not Variant.F or not runtime.has_program:
        raise ValueError("semantic selector steering is defined only for F")
    balanced = balanced_examples(examples)
    baseline = evaluate_model(runtime, balanced, codec, batch_size)
    baseline_hypotheses = score_hypotheses(
        balanced,
        baseline.predictions,
        codec,
    )
    conditions: list[SteeringCondition] = []
    for spec in steering_specs_for_variant(
        Variant.F,
        runtime.config.program_width,
        position=0,
        strength=analysis.steering_strength,
    ):
        native_steering = tuple(
            ProgramInputSteering(
                input_index=spec.input_index,
                position=position,
                scales=spec.scales,
                offsets=spec.offsets,
            )
            for position in spec.positions
        )
        changed = evaluate_model(
            runtime,
            balanced,
            codec,
            batch_size,
            options=ProgramAugmentedForwardOptions(steering=native_steering),
        )
        conditions.append(
            SteeringCondition(
                name=spec.label,
                metrics=changed.metrics,
                hypotheses=score_hypotheses(
                    balanced,
                    changed.predictions,
                    codec,
                ),
                token_accuracy_effect=paired_effect(
                    baseline.per_example_token_accuracy,
                    changed.per_example_token_accuracy,
                ),
                exact_accuracy_effect=paired_effect(
                    baseline.per_example_exact_accuracy,
                    changed.per_example_exact_accuracy,
                ),
            )
        )
    return SteeringSuite(
        example_count=len(balanced),
        baseline=baseline.metrics,
        baseline_hypotheses=baseline_hypotheses,
        conditions=tuple(conditions),
    )


def fit_program_output_pca(
    runtime: ModelRuntime,
    probe_examples: Sequence[Example],
    codec: TokenCodec,
    batch_size: int,
    analysis: AnalysisConfig,
) -> ProbePcaResult:
    """Fit probe-only PCA while retaining no full-split representation tuple."""

    if not runtime.has_program:
        raise ValueError("program-output PCA requires a program branch")
    accumulator = PcaAccumulator(runtime.config.program_width)

    def consume(representation: Representation) -> None:
        if representation.name != "program.output.raw":
            raise ValueError("unexpected representation name")
        accumulator.update(iter_representation_rows(representation))

    evaluation = evaluate_model(
        runtime,
        probe_examples,
        codec,
        batch_size,
        capture_name="program.output.raw",
        capture_sink=consume,
        retain_per_example=False,
    )
    return ProbePcaResult(
        probe_metrics=evaluation.metrics,
        pca=accumulator.finish(
            min(analysis.pca_components, runtime.config.program_width),
            tolerance=analysis.pca_tolerance,
            max_sweeps=analysis.pca_max_sweeps,
        ),
    )


def iter_representation_rows(
    representation: Representation,
) -> Iterator[tuple[float, ...]]:
    if not representation.shape:
        raise ValueError("representation must have a feature axis")
    width = representation.shape[-1]
    if width <= 0:
        raise ValueError("representation feature width must be positive")
    expected = math.prod(representation.shape)
    if len(representation.values) != expected:
        raise ValueError("representation values do not match its shape")
    for start in range(0, len(representation.values), width):
        yield representation.values[start : start + width]


def balanced_examples(examples: Sequence[Example]) -> tuple[Example, ...]:
    """Deterministically interleave equal reverse and copy subsets."""

    reverse = tuple(example for example in examples if example.reversed)
    copy = tuple(example for example in examples if not example.reversed)
    count = min(len(reverse), len(copy))
    if count == 0:
        raise ValueError("balanced steering needs both task branches")
    return tuple(
        example
        for pair in zip(reverse[:count], copy[:count])
        for example in pair
    )


def _hypothesis_agreement(
    predictions: tuple[tuple[int, ...], ...],
    targets: tuple[tuple[int, ...], ...],
) -> HypothesisAgreement:
    if any(len(left) != len(right) for left, right in zip(predictions, targets)):
        raise ValueError("hypothesis rows must have equal width")
    token_count = sum(len(row) for row in targets)
    correct_tokens = sum(
        predicted == expected
        for prediction, target in zip(predictions, targets)
        for predicted, expected in zip(prediction, target)
    )
    correct_sequences = sum(
        prediction == target
        for prediction, target in zip(predictions, targets)
    )
    return HypothesisAgreement(
        target_token_accuracy=correct_tokens / token_count,
        exact_sequence_accuracy=correct_sequences / len(targets),
    )


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


__all__ = [
    "AblationCondition",
    "AblationSuite",
    "EvaluationMetrics",
    "EvaluationResult",
    "HypothesisAgreement",
    "HypothesisScores",
    "ProbePcaResult",
    "SteeringCondition",
    "SteeringSuite",
    "balanced_examples",
    "evaluate_model",
    "fit_program_output_pca",
    "iter_representation_rows",
    "run_f_steering",
    "run_paired_ablations",
    "score_hypotheses",
    "target_argmax",
]
