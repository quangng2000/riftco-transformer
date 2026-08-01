"""Fair, reproducible LoRA-rank comparison for instruction tuning.

Every rank is trained from a fresh runtime instantiated from the same
immutable :class:`ModelBundle`. The training seed, adapter seed, batches,
optimizer settings, targets, and ``alpha / rank`` scale are shared. Validation
selects the best rank; the held-out test split is reported separately.

The evaluator deliberately performs no backward pass or optimizer step. It
scores every next-token target once in deterministic, non-overlapping causal
chunks and weights chunk means by their target-token counts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Protocol

from ..artifacts import ModelBundle, ParameterSpec
from ..data import PreparedDataset, verify_prepared_dataset
from ..native import (
    DecoderOnlyTransformer,
    LoraConfig,
    Tokenizer,
)
from ..post_training import (
    InstructionExample,
    InstructionFormatter,
    PlainChatFormatter,
    PostTrainingConfig,
    post_train,
)
from ..post_training.evaluation import (
    CausalEvaluation,
    DatasetFingerprints,
    InstructionSplits,
    evaluate_instruction_examples,
    freeze_instruction_formatter,
    load_instruction_splits,
    load_prepared_instruction_splits as _load_prepared_instruction_splits,
    validate_formatted_splits_disjoint,
)
from ..serving import GreedySampler, TextGenerator
from ..training import TrainingMetric, selected_backend


_DEFAULT_LORA_TARGETS = (
    "attention.query",
    "attention.value",
)
_TARGET_PARAMETER_SUFFIXES = {
    "attention.query": ".attention.query.weight",
    "attention.key": ".attention.key.weight",
    "attention.value": ".attention.value.weight",
    "attention.output": ".attention.output.weight",
    "feed_forward.expand": ".feed_forward.expand.weight",
    "feed_forward.project": ".feed_forward.project.weight",
}


class InferencePromptFormatter(Protocol):
    """Strategy for turning a user instruction into a generation prefix."""

    def format_prompt(self, prompt: str) -> str:
        """Return the exact text passed to the tokenizer for generation."""


class IdentityPromptFormatter:
    """Treat prompts as fully formatted model-ready text."""

    __slots__ = ()

    def format_prompt(self, prompt: str) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a str")
        if not prompt:
            raise ValueError("prompt must not be empty")
        return prompt


@dataclass(frozen=True, slots=True)
class LoraRankExperimentConfig:
    """Shared controls for every trial in a LoRA rank comparison."""

    ranks: tuple[int, ...] = (1, 2, 4, 8)
    alpha_over_rank: float = 2.0
    targets: tuple[str, ...] = _DEFAULT_LORA_TARGETS
    adapter_random_seed: int = 5489
    training_random_seed: int = 29
    steps: int = 20
    context_size: int = 16
    batch_size: int = 2
    evaluation_interval: int = 10
    loss_average_window: int = 10
    learning_rate: float = 1.0e-3
    backend: str = "auto"
    sampling_strategy: str = "example_uniform"
    evaluation_context_size: int | None = None
    reject_split_overlap: bool = True
    inference_max_new_tokens: int = 16
    kv_cache: str = "paged"
    kv_cache_block_size: int = 16
    # Appended to preserve the positional order of the original public
    # configuration fields.
    attention: str = "materialized"
    activation_checkpointing: str = "disabled"

    def __post_init__(self) -> None:
        try:
            ranks = tuple(self.ranks)
        except TypeError as error:
            raise TypeError("ranks must be an iterable of ints") from error
        if not ranks:
            raise ValueError("ranks must not be empty")
        for index, rank in enumerate(ranks):
            _positive_integer(rank, f"ranks[{index}]")
        if len(set(ranks)) != len(ranks):
            raise ValueError("ranks must not contain duplicates")
        object.__setattr__(self, "ranks", ranks)

        scale = _positive_real(self.alpha_over_rank, "alpha_over_rank")
        object.__setattr__(self, "alpha_over_rank", scale)
        try:
            targets = tuple(self.targets)
        except TypeError as error:
            raise TypeError(
                "targets must be an iterable of strings"
            ) from error
        # LoraConfig is the single source of truth for target, seed, and
        # native float32 scale validation.
        first_lora = LoraConfig(
            rank=ranks[0],
            alpha=scale * ranks[0],
            targets=targets,
            random_seed=self.adapter_random_seed,
        )
        for rank in ranks[1:]:
            LoraConfig(
                rank=rank,
                alpha=scale * rank,
                targets=first_lora.targets,
                random_seed=self.adapter_random_seed,
            )
        object.__setattr__(self, "targets", first_lora.targets)

        _nonnegative_integer(
            self.training_random_seed,
            "training_random_seed",
        )
        for name in (
            "steps",
            "context_size",
            "batch_size",
            "evaluation_interval",
            "loss_average_window",
            "kv_cache_block_size",
        ):
            _positive_integer(getattr(self, name), name)
        _positive_real(self.learning_rate, "learning_rate")
        if self.evaluation_context_size is not None:
            _positive_integer(
                self.evaluation_context_size,
                "evaluation_context_size",
            )
        if not isinstance(self.reject_split_overlap, bool):
            raise TypeError("reject_split_overlap must be a bool")
        _nonnegative_integer(
            self.inference_max_new_tokens,
            "inference_max_new_tokens",
        )
        if self.backend not in {"auto", "cpu", "metal", "cuda", "tpu"}:
            raise ValueError(
                "backend must be 'auto', 'cpu', 'metal', 'cuda', or 'tpu'"
            )
        if self.sampling_strategy not in {
            "example_uniform",
            "window_uniform",
        }:
            raise ValueError(
                "sampling_strategy must be 'example_uniform' or "
                "'window_uniform'"
            )
        if self.attention not in {"materialized", "flash"}:
            raise ValueError(
                "attention must be 'materialized' or 'flash'"
            )
        if self.activation_checkpointing not in {"disabled", "block"}:
            raise ValueError(
                "activation_checkpointing must be 'disabled' or 'block'"
            )
        if self.kv_cache not in {"contiguous", "paged"}:
            raise ValueError(
                "kv_cache must be 'contiguous' or 'paged'"
            )

    def alpha_for_rank(self, rank: int) -> float:
        """Return alpha while preserving the configured alpha/rank scale."""

        checked_rank = _positive_integer(rank, "rank")
        if checked_rank not in self.ranks:
            raise ValueError("rank is not configured for this experiment")
        return LoraConfig(
            rank=checked_rank,
            alpha=self.alpha_over_rank * checked_rank,
            targets=self.targets,
            random_seed=self.adapter_random_seed,
        ).alpha


@dataclass(frozen=True, slots=True)
class InferenceSample:
    """One deterministic greedy generation and its indicative latency.

    Generation produces exactly the configured token count; the lab has no
    EOS stopping policy yet. LoRA is merged before serving, so latency here is
    a smoke measurement, not evidence that one rank serves faster.
    """

    prompt: str
    formatted_prompt: str
    completion: str
    text: str
    generated_token_ids: tuple[int, ...]
    elapsed_seconds: float

    @property
    def tokens_per_second(self) -> float:
        if not self.generated_token_ids:
            return 0.0
        if self.elapsed_seconds == 0.0:
            return math.inf
        return len(self.generated_token_ids) / self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class LoraRankTrial:
    """Training, held-out evaluation, and inference results for one rank."""

    rank: int
    alpha: float
    alpha_over_rank: float
    adapter_parameter_count: int
    starting_artifact_id: str
    bundle: ModelBundle
    training_metrics: tuple[TrainingMetric, ...]
    validation: CausalEvaluation
    validation_loss_delta_from_base: float
    test: CausalEvaluation | None
    test_loss_delta_from_base: float | None
    inference_samples: tuple[InferenceSample, ...]
    training_seconds: float


@dataclass(frozen=True, slots=True)
class LoraRankComparison:
    """Complete rank sweep with validation-based selection."""

    base_artifact_id: str
    fingerprints: DatasetFingerprints
    resolved_backend: str
    config: LoraRankExperimentConfig
    baseline_validation: CausalEvaluation
    baseline_test: CausalEvaluation
    trials: tuple[LoraRankTrial, ...]

    @property
    def ranked_by_validation(self) -> tuple[LoraRankTrial, ...]:
        """Trials ordered by validation loss, breaking ties by lower rank."""

        return tuple(
            sorted(
                self.trials,
                key=lambda trial: (trial.validation.loss, trial.rank),
            )
        )

    @property
    def best_trial(self) -> LoraRankTrial:
        return self.ranked_by_validation[0]

    @property
    def best_rank(self) -> int:
        return self.best_trial.rank

    @property
    def selected_test(self) -> CausalEvaluation:
        """Held-out test result for the validation-selected trial."""

        result = self.best_trial.test
        if result is None:
            raise RuntimeError("selected trial is missing held-out test data")
        return result


def load_prepared_instruction_splits(
    source: str | Path | PreparedDataset,
    *,
    reject_overlap: bool = True,
) -> InstructionSplits:
    """Compatibility forwarding layer for the shared split loader."""

    prepared = (
        source
        if isinstance(source, PreparedDataset)
        else verify_prepared_dataset(source)
    )
    return _load_prepared_instruction_splits(
        prepared,
        reject_overlap=reject_overlap,
    )


def compare_lora_ranks(
    base_bundle: ModelBundle,
    splits: InstructionSplits,
    config: LoraRankExperimentConfig | None = None,
    *,
    formatter: InstructionFormatter | None = None,
    inference_prompts: Iterable[str] = (),
    inference_prompt_formatter: InferencePromptFormatter | None = None,
    metric_sink: Callable[[int, TrainingMetric], None] | None = None,
) -> LoraRankComparison:
    """Train and compare LoRA ranks under identical experimental controls.

    All candidates are scored on validation. Only after the best rank is fixed
    is the test split evaluated, once for the base model and once for the
    selected trial. Test data therefore cannot influence rank selection.
    """

    if not isinstance(base_bundle, ModelBundle):
        raise TypeError("base_bundle must be a ModelBundle")
    if not isinstance(splits, InstructionSplits):
        raise TypeError("splits must be InstructionSplits")
    configured = (
        LoraRankExperimentConfig() if config is None else config
    )
    if not isinstance(configured, LoraRankExperimentConfig):
        raise TypeError("config must be a LoraRankExperimentConfig")
    if configured.reject_split_overlap:
        splits.validate_disjoint()
    if metric_sink is not None and not callable(metric_sink):
        raise TypeError("metric_sink must be callable")

    prompt_tuple = _inference_prompts(inference_prompts)
    evaluation_context = (
        configured.context_size
        if configured.evaluation_context_size is None
        else configured.evaluation_context_size
    )
    if configured.context_size > base_bundle.config.maximum_context:
        raise ValueError(
            "training context_size exceeds the model maximum_context"
        )
    if evaluation_context > base_bundle.config.maximum_context:
        raise ValueError(
            "evaluation context_size exceeds the model maximum_context"
        )

    configured_formatter = (
        PlainChatFormatter() if formatter is None else formatter
    )
    if not callable(getattr(configured_formatter, "format", None)):
        raise TypeError("formatter must provide a callable format() method")
    frozen_formatter = freeze_instruction_formatter(
        splits,
        configured_formatter,
    )
    if configured.reject_split_overlap:
        validate_formatted_splits_disjoint(
            splits,
            frozen_formatter,
        )
    prepared_prompts = _prepare_inference_prompts(
        prompt_tuple,
        configured_formatter,
        inference_prompt_formatter,
    )
    _preflight_target_ranks(
        base_bundle,
        configured.ranks,
        configured.targets,
    )
    resolved_backend = selected_backend(configured.backend)
    starting_artifact_id = base_bundle.artifact_id
    trials: list[LoraRankTrial] = []

    with base_bundle.instantiate(resolved_backend) as baseline_runtime:
        baseline_runtime.model.set_full_sequence_attention(
            configured.attention
        )
        _validate_training_examples(
            baseline_runtime.tokenizer,
            splits.train,
            frozen_formatter,
            configured.context_size,
        )
        baseline_validation = evaluate_instruction_examples(
            baseline_runtime.model,
            baseline_runtime.tokenizer,
            splits.validation,
            context_size=evaluation_context,
            batch_size=configured.batch_size,
            formatter=frozen_formatter,
        )

    for rank in configured.ranks:
        alpha = configured.alpha_for_rank(rank)
        post_training_config = PostTrainingConfig(
            steps=configured.steps,
            context_size=configured.context_size,
            batch_size=configured.batch_size,
            evaluation_interval=configured.evaluation_interval,
            loss_average_window=configured.loss_average_window,
            learning_rate=configured.learning_rate,
            random_seed=configured.training_random_seed,
            backend=resolved_backend,
            fine_tuning_method="lora",
            sampling_strategy=configured.sampling_strategy,
            attention=configured.attention,
            activation_checkpointing=(
                configured.activation_checkpointing
            ),
            lora=LoraConfig(
                rank=rank,
                alpha=alpha,
                targets=configured.targets,
                random_seed=configured.adapter_random_seed,
            ),
        )
        per_rank_sink = (
            None
            if metric_sink is None
            else lambda metric, trial_rank=rank: metric_sink(
                trial_rank,
                metric,
            )
        )
        training_started = perf_counter()
        trained = post_train(
            base_bundle,
            splits.train,
            post_training_config,
            formatter=frozen_formatter,
            metric_sink=per_rank_sink,
        )
        training_seconds = perf_counter() - training_started

        if base_bundle.artifact_id != starting_artifact_id:
            raise RuntimeError("base bundle changed during rank experiment")
        if trained.bundle.parent_artifact_id != starting_artifact_id:
            raise RuntimeError(
                "rank trial did not start from the configured base bundle"
            )

        with trained.bundle.instantiate(resolved_backend) as runtime:
            runtime.model.set_full_sequence_attention(configured.attention)
            validation = evaluate_instruction_examples(
                runtime.model,
                runtime.tokenizer,
                splits.validation,
                context_size=evaluation_context,
                batch_size=configured.batch_size,
                formatter=frozen_formatter,
            )
            inference_samples = _generate_samples(
                runtime.model,
                runtime.tokenizer,
                prepared_prompts,
                configured,
            )

        trials.append(
            LoraRankTrial(
                rank=rank,
                alpha=alpha,
                alpha_over_rank=alpha / rank,
                adapter_parameter_count=_adapter_parameter_count(
                    base_bundle,
                    rank,
                    configured.targets,
                ),
                starting_artifact_id=starting_artifact_id,
                bundle=trained.bundle,
                training_metrics=trained.metrics,
                validation=validation,
                validation_loss_delta_from_base=(
                    validation.loss - baseline_validation.loss
                ),
                test=None,
                test_loss_delta_from_base=None,
                inference_samples=inference_samples,
                training_seconds=training_seconds,
            )
        )

    selected_index = min(
        range(len(trials)),
        key=lambda index: (
            trials[index].validation.loss,
            trials[index].rank,
        ),
    )
    # Test model evaluation begins only after validation has selected the
    # winning rank. Its formatted representation was computed earlier solely
    # for the cross-split leakage preflight.
    with base_bundle.instantiate(resolved_backend) as baseline_runtime:
        baseline_runtime.model.set_full_sequence_attention(
            configured.attention
        )
        baseline_test = evaluate_instruction_examples(
            baseline_runtime.model,
            baseline_runtime.tokenizer,
            splits.test,
            context_size=evaluation_context,
            batch_size=configured.batch_size,
            formatter=frozen_formatter,
        )
    selected = trials[selected_index]
    with selected.bundle.instantiate(resolved_backend) as selected_runtime:
        selected_runtime.model.set_full_sequence_attention(
            configured.attention
        )
        selected_test = evaluate_instruction_examples(
            selected_runtime.model,
            selected_runtime.tokenizer,
            splits.test,
            context_size=evaluation_context,
            batch_size=configured.batch_size,
            formatter=frozen_formatter,
        )
    trials[selected_index] = replace(
        selected,
        test=selected_test,
        test_loss_delta_from_base=(
            selected_test.loss - baseline_test.loss
        ),
    )

    return LoraRankComparison(
        base_artifact_id=starting_artifact_id,
        fingerprints=splits.fingerprints,
        resolved_backend=resolved_backend,
        config=configured,
        baseline_validation=baseline_validation,
        baseline_test=baseline_test,
        trials=tuple(trials),
    )


def _generate_samples(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    prompts: tuple[tuple[str, str], ...],
    config: LoraRankExperimentConfig,
) -> tuple[InferenceSample, ...]:
    if not prompts:
        return ()
    generator = TextGenerator(
        model,
        tokenizer,
        GreedySampler(),
        kv_cache=config.kv_cache,
        kv_cache_block_size=config.kv_cache_block_size,
    )
    samples: list[InferenceSample] = []
    for prompt, formatted_prompt in prompts:
        started = perf_counter()
        generated = generator.generate(
            formatted_prompt,
            max_new_tokens=config.inference_max_new_tokens,
        )
        elapsed = perf_counter() - started
        completion_bytes = tokenizer.decode_bytes(
            generated.generated_token_ids
        )
        samples.append(
            InferenceSample(
                prompt=prompt,
                formatted_prompt=formatted_prompt,
                completion=completion_bytes.decode(
                    "utf-8",
                    errors="replace",
                ),
                text=generated.text,
                generated_token_ids=generated.generated_token_ids,
                elapsed_seconds=elapsed,
            )
        )
    return tuple(samples)


def _adapter_parameter_count(
    bundle: ModelBundle,
    rank: int,
    targets: tuple[str, ...],
) -> int:
    total = 0
    for parameter in _selected_matrix_parameters(bundle, targets):
        total += rank * sum(parameter.shape)
    return total


def _selected_matrix_parameters(
    bundle: ModelBundle,
    targets: tuple[str, ...],
) -> tuple[ParameterSpec, ...]:
    selected_parameters: list[ParameterSpec] = []
    for parameter in bundle.parameters:
        selected = (
            parameter.name == "language_model_head.weight"
            if "language_model_head" in targets
            else False
        )
        if not selected:
            selected = any(
                parameter.name.endswith(_TARGET_PARAMETER_SUFFIXES[target])
                for target in targets
                if target != "language_model_head"
            )
        if not selected:
            continue
        if len(parameter.shape) != 2:
            raise RuntimeError(
                f"LoRA target {parameter.name!r} is not a matrix"
            )
        selected_parameters.append(parameter)
    if not selected_parameters:
        raise RuntimeError("LoRA targets did not match any model parameter")
    return tuple(selected_parameters)


def _preflight_target_ranks(
    bundle: ModelBundle,
    ranks: tuple[int, ...],
    targets: tuple[str, ...],
) -> None:
    parameters = _selected_matrix_parameters(bundle, targets)
    maximum_low_rank = min(
        min(parameter.shape) for parameter in parameters
    )
    oversized = tuple(rank for rank in ranks if rank > maximum_low_rank)
    if oversized:
        displayed = ", ".join(str(rank) for rank in oversized)
        raise ValueError(
            f"rank(s) {displayed} exceed the smallest selected projection "
            f"dimension {maximum_low_rank}; those adapters would not be "
            "low-rank"
        )


def _validate_training_examples(
    tokenizer: Tokenizer,
    examples: tuple[InstructionExample, ...],
    formatter: InstructionFormatter,
    context_size: int,
) -> None:
    too_short: list[int] = []
    for index, example in enumerate(examples):
        formatted = formatter.format(example)
        try:
            tokens = tokenizer.encode(formatted)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"could not encode training example {index}: {error}"
            ) from error
        if len(tokens) <= context_size:
            too_short.append(index)
    if too_short:
        displayed = ", ".join(str(index) for index in too_short[:8])
        suffix = "..." if len(too_short) > 8 else ""
        raise ValueError(
            "each formatted training example must encode to at least "
            "context_size + 1 tokens; too-short example indices: "
            f"{displayed}{suffix}"
        )


def _prepare_inference_prompts(
    prompts: tuple[str, ...],
    training_formatter: InstructionFormatter,
    prompt_formatter: InferencePromptFormatter | None,
) -> tuple[tuple[str, str], ...]:
    if not prompts:
        return ()
    configured_prompt_formatter = prompt_formatter
    if configured_prompt_formatter is None:
        candidate = getattr(training_formatter, "format_prompt", None)
        if not callable(candidate):
            raise ValueError(
                "the instruction formatter must provide format_prompt(), or "
                "inference_prompt_formatter must be supplied"
            )
        configured_prompt_formatter = training_formatter
    if not callable(
        getattr(configured_prompt_formatter, "format_prompt", None)
    ):
        raise TypeError(
            "inference_prompt_formatter must provide a callable "
            "format_prompt() method"
        )

    prepared: list[tuple[str, str]] = []
    for index, prompt in enumerate(prompts):
        formatted = configured_prompt_formatter.format_prompt(prompt)
        if not isinstance(formatted, str):
            raise TypeError(
                "inference_prompt_formatter.format_prompt() must return a str"
            )
        if not formatted:
            raise ValueError(
                f"formatted inference prompt {index} must not be empty"
            )
        prepared.append((prompt, formatted))
    return tuple(prepared)


def _inference_prompts(prompts: Iterable[str]) -> tuple[str, ...]:
    try:
        copied = tuple(prompts)
    except TypeError as error:
        raise TypeError("inference_prompts must be an iterable") from error
    for index, prompt in enumerate(copied):
        if not isinstance(prompt, str):
            raise TypeError(f"inference_prompts[{index}] must be a str")
        if not prompt:
            raise ValueError(
                f"inference_prompts[{index}] must not be empty"
            )
    return copied


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


def _positive_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


__all__ = [
    "CausalEvaluation",
    "DatasetFingerprints",
    "IdentityPromptFormatter",
    "InferenceSample",
    "InferencePromptFormatter",
    "InstructionSplits",
    "LoraRankComparison",
    "LoraRankExperimentConfig",
    "LoraRankTrial",
    "compare_lora_ranks",
    "evaluate_instruction_examples",
    "load_instruction_splits",
    "load_prepared_instruction_splits",
]
