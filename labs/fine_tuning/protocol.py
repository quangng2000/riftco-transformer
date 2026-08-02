"""Controlled comparison for full fine-tuning, LoRA, and QLoRA."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Callable

from riftco_transformer.artifacts import ModelBundle
from riftco_transformer.post_training import (
    CausalEvaluation,
    DatasetFingerprints,
    InstructionFormatter,
    InstructionSplits,
    PlainChatFormatter,
    PostTrainingConfig,
    evaluate_instruction_examples,
    freeze_instruction_formatter,
    post_train,
    selected_post_training_backend,
    validate_formatted_splits_disjoint,
)
from riftco_transformer.training import TrainingMetric


@dataclass(frozen=True, slots=True)
class FineTuningCandidate:
    """One fixed training recipe participating in a validation selection."""

    name: str
    config: PostTrainingConfig
    selection_group: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("candidate name must be a str")
        if not self.name.strip():
            raise ValueError("candidate name must not be blank")
        if not isinstance(self.config, PostTrainingConfig):
            raise TypeError("candidate config must be a PostTrainingConfig")
        group = self.selection_group
        if group is None:
            group = self.config.fine_tuning_method
        if not isinstance(group, str):
            raise TypeError("selection_group must be a str or None")
        if not group.strip():
            raise ValueError("selection_group must not be blank")
        object.__setattr__(self, "selection_group", group)

    @property
    def fine_tuning_method(self) -> str:
        return self.config.fine_tuning_method


def _default_candidates() -> tuple[FineTuningCandidate, ...]:
    return (
        FineTuningCandidate(
            "full",
            PostTrainingConfig(fine_tuning_method="full"),
        ),
        FineTuningCandidate(
            "lora-rank-4",
            PostTrainingConfig(fine_tuning_method="lora"),
        ),
    )


@dataclass(frozen=True, slots=True)
class FineTuningExperimentConfig:
    """Candidate recipes and one shared held-out evaluation policy.

    Candidates in the same ``selection_group`` compete on validation loss.
    Only the validation winner of each group is allowed to see test data. The
    default groups are the fine-tuning methods, so a full model and the best
    configured LoRA rank each receive one final held-out test measurement.
    """

    candidates: tuple[FineTuningCandidate, ...] = field(
        default_factory=_default_candidates
    )
    evaluation_context_size: int | None = None
    evaluation_batch_size: int = 1
    evaluation_attention: str = "materialized"
    reject_split_overlap: bool = True

    def __post_init__(self) -> None:
        try:
            candidates = tuple(self.candidates)
        except TypeError as error:
            raise TypeError("candidates must be an iterable") from error
        if not candidates:
            raise ValueError("candidates must not be empty")
        if any(
            not isinstance(candidate, FineTuningCandidate)
            for candidate in candidates
        ):
            raise TypeError(
                "candidates must contain only FineTuningCandidate values"
            )
        names = tuple(candidate.name for candidate in candidates)
        if len(set(names)) != len(names):
            raise ValueError("candidate names must be unique")
        object.__setattr__(self, "candidates", candidates)

        if self.evaluation_context_size is not None:
            _positive_integer(
                self.evaluation_context_size,
                "evaluation_context_size",
            )
        _positive_integer(
            self.evaluation_batch_size,
            "evaluation_batch_size",
        )
        if self.evaluation_attention not in {"materialized", "flash"}:
            raise ValueError(
                "evaluation_attention must be 'materialized' or 'flash'"
            )
        if not isinstance(self.reject_split_overlap, bool):
            raise TypeError("reject_split_overlap must be a bool")


@dataclass(frozen=True, slots=True)
class FineTuningTrial:
    """Training and generalization measurements for one fixed recipe."""

    candidate: FineTuningCandidate
    starting_artifact_id: str
    bundle: ModelBundle
    training_metrics: tuple[TrainingMetric, ...]
    trainable_parameter_count: int
    base_parameter_count: int
    train: CausalEvaluation
    validation: CausalEvaluation
    test: CausalEvaluation | None
    train_loss_delta_from_base: float
    validation_loss_delta_from_base: float
    test_loss_delta_from_base: float | None
    validation_generalization_gap: float
    test_generalization_gap: float | None
    training_seconds: float

    @property
    def name(self) -> str:
        return self.candidate.name

    @property
    def fine_tuning_method(self) -> str:
        return self.candidate.fine_tuning_method

    @property
    def selection_group(self) -> str:
        group = self.candidate.selection_group
        assert group is not None
        return group

    @property
    def trainable_fraction(self) -> float:
        return self.trainable_parameter_count / self.base_parameter_count

    @property
    def selected_for_test(self) -> bool:
        return self.test is not None


@dataclass(frozen=True, slots=True)
class FineTuningComparison:
    """Baseline and candidate measurements with validation-only selection."""

    base_artifact_id: str
    fingerprints: DatasetFingerprints
    resolved_backend: str
    evaluation_context_size: int
    config: FineTuningExperimentConfig
    baseline_train: CausalEvaluation
    baseline_validation: CausalEvaluation
    baseline_test: CausalEvaluation
    selected_candidate_names: tuple[str, ...]
    trials: tuple[FineTuningTrial, ...]

    @property
    def selected_trials(self) -> tuple[FineTuningTrial, ...]:
        selected = set(self.selected_candidate_names)
        return tuple(trial for trial in self.trials if trial.name in selected)

    @property
    def baseline_validation_generalization_gap(self) -> float:
        return self.baseline_validation.loss - self.baseline_train.loss

    @property
    def baseline_test_generalization_gap(self) -> float:
        return self.baseline_test.loss - self.baseline_train.loss


@dataclass(frozen=True, slots=True)
class _PendingTrial:
    candidate: FineTuningCandidate
    starting_artifact_id: str
    bundle: ModelBundle
    training_metrics: tuple[TrainingMetric, ...]
    trainable_parameter_count: int
    train: CausalEvaluation
    validation: CausalEvaluation
    training_seconds: float


def compare_fine_tuning(
    base_bundle: ModelBundle,
    splits: InstructionSplits,
    config: FineTuningExperimentConfig | None = None,
    *,
    formatter: InstructionFormatter | None = None,
    metric_sink: Callable[[str, TrainingMetric], None] | None = None,
) -> FineTuningComparison:
    """Compare full, LoRA, and/or QLoRA recipes on held-out data.

    Every candidate starts from the same immutable base and trains only on the
    train split. All candidates are exhaustively scored on train and
    validation with one shared metric. Validation selects one candidate per
    group; only then are the base and selected candidates scored on test.
    """

    if not isinstance(base_bundle, ModelBundle):
        raise TypeError("base_bundle must be a ModelBundle")
    if not isinstance(splits, InstructionSplits):
        raise TypeError("splits must be InstructionSplits")
    configured = FineTuningExperimentConfig() if config is None else config
    if not isinstance(configured, FineTuningExperimentConfig):
        raise TypeError("config must be a FineTuningExperimentConfig")
    if metric_sink is not None and not callable(metric_sink):
        raise TypeError("metric_sink must be callable")
    if configured.reject_split_overlap:
        splits.validate_disjoint()

    requires_qlora_backend = any(
        candidate.config.fine_tuning_method == "qlora"
        for candidate in configured.candidates
    )
    resolved_backends = tuple(
        selected_post_training_backend(
            candidate.config,
            require_qlora_capability=requires_qlora_backend,
        )
        for candidate in configured.candidates
    )
    if len(set(resolved_backends)) != 1:
        raise ValueError(
            "all candidates must resolve to the same evaluation backend"
        )
    resolved_backend = resolved_backends[0]

    training_contexts = {
        candidate.config.context_size
        for candidate in configured.candidates
    }
    if configured.evaluation_context_size is None:
        if len(training_contexts) != 1:
            raise ValueError(
                "evaluation_context_size is required when candidate "
                "training context sizes differ"
            )
        evaluation_context = next(iter(training_contexts))
    else:
        evaluation_context = configured.evaluation_context_size
    if evaluation_context > base_bundle.config.maximum_context:
        raise ValueError(
            "evaluation_context_size exceeds the model maximum_context"
        )

    configured_formatter = (
        PlainChatFormatter() if formatter is None else formatter
    )
    frozen_formatter = freeze_instruction_formatter(
        splits,
        configured_formatter,
    )
    if configured.reject_split_overlap:
        validate_formatted_splits_disjoint(splits, frozen_formatter)

    starting_artifact_id = base_bundle.artifact_id
    base_parameter_count = sum(
        parameter.value_count for parameter in base_bundle.parameters
    )
    for candidate in configured.candidates:
        if candidate.config.context_size > base_bundle.config.maximum_context:
            raise ValueError(
                f"candidate {candidate.name!r} training context_size "
                "exceeds the model maximum_context"
            )
    # Validate every parameter scope before starting any candidate. This keeps
    # an invalid late LoRA rank from wasting earlier full-training work.
    trainable_parameter_counts = tuple(
        _trainable_parameter_count(
            base_bundle,
            replace(candidate.config, backend=resolved_backend),
            resolved_backend,
        )
        for candidate in configured.candidates
    )
    with base_bundle.instantiate(resolved_backend) as baseline_runtime:
        baseline_runtime.model.set_full_sequence_attention(
            configured.evaluation_attention
        )
        baseline_train = evaluate_instruction_examples(
            baseline_runtime.model,
            baseline_runtime.tokenizer,
            splits.train,
            context_size=evaluation_context,
            batch_size=configured.evaluation_batch_size,
            formatter=frozen_formatter,
        )
        baseline_validation = evaluate_instruction_examples(
            baseline_runtime.model,
            baseline_runtime.tokenizer,
            splits.validation,
            context_size=evaluation_context,
            batch_size=configured.evaluation_batch_size,
            formatter=frozen_formatter,
        )

    pending_trials: list[_PendingTrial] = []
    for candidate, trainable_parameter_count in zip(
        configured.candidates,
        trainable_parameter_counts,
    ):
        training_config = replace(
            candidate.config,
            backend=resolved_backend,
        )
        candidate_sink = (
            None
            if metric_sink is None
            else lambda metric, name=candidate.name: metric_sink(name, metric)
        )
        training_started = perf_counter()
        trained = post_train(
            base_bundle,
            splits.train,
            training_config,
            formatter=frozen_formatter,
            metric_sink=candidate_sink,
        )
        training_seconds = perf_counter() - training_started
        if base_bundle.artifact_id != starting_artifact_id:
            raise RuntimeError(
                "base bundle changed during fine-tuning comparison"
            )
        if trained.bundle.parent_artifact_id != starting_artifact_id:
            raise RuntimeError(
                "candidate did not start from the configured base bundle"
            )

        with trained.bundle.instantiate(resolved_backend) as runtime:
            runtime.model.set_full_sequence_attention(
                configured.evaluation_attention
            )
            train_evaluation = evaluate_instruction_examples(
                runtime.model,
                runtime.tokenizer,
                splits.train,
                context_size=evaluation_context,
                batch_size=configured.evaluation_batch_size,
                formatter=frozen_formatter,
            )
            validation_evaluation = evaluate_instruction_examples(
                runtime.model,
                runtime.tokenizer,
                splits.validation,
                context_size=evaluation_context,
                batch_size=configured.evaluation_batch_size,
                formatter=frozen_formatter,
            )
        pending_trials.append(
            _PendingTrial(
                candidate=candidate,
                starting_artifact_id=starting_artifact_id,
                bundle=trained.bundle,
                training_metrics=trained.metrics,
                trainable_parameter_count=trainable_parameter_count,
                train=train_evaluation,
                validation=validation_evaluation,
                training_seconds=training_seconds,
            )
        )

    selected_indexes = _select_by_validation_group(pending_trials)
    selected_candidate_names = tuple(
        pending_trials[index].candidate.name for index in selected_indexes
    )

    # No test forward pass occurs until every group winner is fixed from
    # validation. Reporting more than one group consumes test data for a final
    # method comparison; callers should retire that test split afterward.
    with base_bundle.instantiate(resolved_backend) as baseline_runtime:
        baseline_runtime.model.set_full_sequence_attention(
            configured.evaluation_attention
        )
        baseline_test = evaluate_instruction_examples(
            baseline_runtime.model,
            baseline_runtime.tokenizer,
            splits.test,
            context_size=evaluation_context,
            batch_size=configured.evaluation_batch_size,
            formatter=frozen_formatter,
        )

    selected_set = set(selected_indexes)
    trials: list[FineTuningTrial] = []
    for index, pending in enumerate(pending_trials):
        test_evaluation: CausalEvaluation | None = None
        test_delta: float | None = None
        test_gap: float | None = None
        if index in selected_set:
            with pending.bundle.instantiate(resolved_backend) as runtime:
                runtime.model.set_full_sequence_attention(
                    configured.evaluation_attention
                )
                test_evaluation = evaluate_instruction_examples(
                    runtime.model,
                    runtime.tokenizer,
                    splits.test,
                    context_size=evaluation_context,
                    batch_size=configured.evaluation_batch_size,
                    formatter=frozen_formatter,
                )
            test_delta = test_evaluation.loss - baseline_test.loss
            test_gap = test_evaluation.loss - pending.train.loss

        trials.append(
            FineTuningTrial(
                candidate=pending.candidate,
                starting_artifact_id=pending.starting_artifact_id,
                bundle=pending.bundle,
                training_metrics=pending.training_metrics,
                trainable_parameter_count=(
                    pending.trainable_parameter_count
                ),
                base_parameter_count=base_parameter_count,
                train=pending.train,
                validation=pending.validation,
                test=test_evaluation,
                train_loss_delta_from_base=(
                    pending.train.loss - baseline_train.loss
                ),
                validation_loss_delta_from_base=(
                    pending.validation.loss - baseline_validation.loss
                ),
                test_loss_delta_from_base=test_delta,
                validation_generalization_gap=(
                    pending.validation.loss - pending.train.loss
                ),
                test_generalization_gap=test_gap,
                training_seconds=pending.training_seconds,
            )
        )

    return FineTuningComparison(
        base_artifact_id=starting_artifact_id,
        fingerprints=splits.fingerprints,
        resolved_backend=resolved_backend,
        evaluation_context_size=evaluation_context,
        config=configured,
        baseline_train=baseline_train,
        baseline_validation=baseline_validation,
        baseline_test=baseline_test,
        selected_candidate_names=selected_candidate_names,
        trials=tuple(trials),
    )


def _select_by_validation_group(
    trials: list[_PendingTrial],
) -> tuple[int, ...]:
    groups: list[str] = []
    for trial in trials:
        group = trial.candidate.selection_group
        assert group is not None
        if group not in groups:
            groups.append(group)
    selected: list[int] = []
    for group in groups:
        candidates = [
            index
            for index, trial in enumerate(trials)
            if trial.candidate.selection_group == group
        ]
        # Candidate declaration order is the deterministic tie-break. A CLI
        # rank sweep declares ranks in ascending order, preserving the existing
        # lower-rank tie-break.
        selected.append(
            min(
                candidates,
                key=lambda index: (
                    trials[index].validation.loss,
                    index,
                ),
            )
        )
    return tuple(selected)


def _trainable_parameter_count(
    bundle: ModelBundle,
    config: PostTrainingConfig,
    backend: str,
) -> int:
    if config.fine_tuning_method == "full":
        return sum(parameter.value_count for parameter in bundle.parameters)
    with bundle.instantiate(backend) as runtime:
        runtime.model.attach_lora(config.lora)
        with runtime.model.adapter_parameters() as parameters:
            return parameters.total_numel


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


__all__ = [
    "FineTuningCandidate",
    "FineTuningComparison",
    "FineTuningExperimentConfig",
    "FineTuningTrial",
    "compare_fine_tuning",
]
