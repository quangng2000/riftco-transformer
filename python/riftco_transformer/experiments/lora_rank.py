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
import hashlib
import json
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
    cross_entropy,
)
from ..post_training import (
    InstructionExample,
    InstructionFormatter,
    PlainChatFormatter,
    PostTrainingConfig,
    load_instruction_jsonl_bytes,
    post_train,
)
from ..serving import GreedySampler, TextGenerator
from ..training import TrainingMetric, loss_perplexity, selected_backend


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


@dataclass(frozen=True, slots=True)
class InstructionSplits:
    """Prepared, immutable train/validation/test instruction records."""

    train: tuple[InstructionExample, ...]
    validation: tuple[InstructionExample, ...]
    test: tuple[InstructionExample, ...]

    def __post_init__(self) -> None:
        for name in ("train", "validation", "test"):
            try:
                examples = tuple(getattr(self, name))
            except TypeError as error:
                raise TypeError(f"{name} must be an iterable") from error
            if not examples:
                raise ValueError(f"{name} split must not be empty")
            if any(
                not isinstance(example, InstructionExample)
                for example in examples
            ):
                raise TypeError(
                    f"{name} must contain only InstructionExample values"
                )
            object.__setattr__(self, name, examples)

    @property
    def fingerprints(self) -> DatasetFingerprints:
        """Content identities independent of JSONL whitespace and paths."""

        return DatasetFingerprints(
            train=_examples_fingerprint(self.train),
            validation=_examples_fingerprint(self.validation),
            test=_examples_fingerprint(self.test),
        )

    def validate_disjoint(self) -> None:
        """Reject an exact prompt/response record shared by two splits."""

        records = {
            "train": {_example_key(example) for example in self.train},
            "validation": {
                _example_key(example) for example in self.validation
            },
            "test": {_example_key(example) for example in self.test},
        }
        pairs = (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        for left, right in pairs:
            overlap = records[left] & records[right]
            if overlap:
                raise ValueError(
                    f"{left} and {right} splits overlap by "
                    f"{len(overlap)} exact instruction record(s)"
                )


@dataclass(frozen=True, slots=True)
class DatasetFingerprints:
    """SHA-256 identities of the three prepared instruction splits."""

    train: str
    validation: str
    test: str


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
        if self.backend not in {"auto", "cpu", "metal"}:
            raise ValueError("backend must be 'auto', 'cpu', or 'metal'")
        if self.sampling_strategy not in {
            "example_uniform",
            "window_uniform",
        }:
            raise ValueError(
                "sampling_strategy must be 'example_uniform' or "
                "'window_uniform'"
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
class CausalEvaluation:
    """Read-only held-out causal-language-model measurements."""

    example_count: int
    usable_example_count: int
    skipped_example_count: int
    target_token_count: int
    chunk_count: int
    forward_batch_count: int
    loss: float
    perplexity: float
    elapsed_seconds: float


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


def load_instruction_splits(
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    *,
    reject_overlap: bool = True,
) -> InstructionSplits:
    """Load three prepared JSONL files without shuffling or resplitting."""

    if not isinstance(reject_overlap, bool):
        raise TypeError("reject_overlap must be a bool")
    # Importing here keeps parsing behavior centralized in post_training.
    from ..post_training import load_instruction_jsonl

    splits = InstructionSplits(
        train=load_instruction_jsonl(train_path),
        validation=load_instruction_jsonl(validation_path),
        test=load_instruction_jsonl(test_path),
    )
    if reject_overlap:
        splits.validate_disjoint()
    return splits


def load_prepared_instruction_splits(
    source: str | Path | PreparedDataset,
    *,
    reject_overlap: bool = True,
) -> InstructionSplits:
    """Load split buffers bound to one verified prepared-dataset snapshot.

    When ``source`` is already a :class:`PreparedDataset`, its manifest and
    file metadata are reused. Each split is then read exactly once, and that
    immutable byte buffer is checked against the verified byte count and
    SHA-256 before the same buffer is parsed.
    """

    if not isinstance(reject_overlap, bool):
        raise TypeError("reject_overlap must be a bool")
    prepared = (
        source
        if isinstance(source, PreparedDataset)
        else verify_prepared_dataset(source)
    )
    expected_media_type = "application/x-ndjson"
    loaded: dict[str, tuple[InstructionExample, ...]] = {}
    for partition in ("train", "validation", "test"):
        prepared_file = prepared.files[partition]
        if prepared_file.media_type != expected_media_type:
            raise ValueError(
                "prepared instruction splits must use canonical JSONL"
            )
        content = prepared_file.read_verified_bytes()
        loaded[partition] = load_instruction_jsonl_bytes(
            content,
            source=prepared_file.path,
        )
    splits = InstructionSplits(
        train=loaded["train"],
        validation=loaded["validation"],
        test=loaded["test"],
    )
    if reject_overlap:
        splits.validate_disjoint()
    return splits


def evaluate_instruction_examples(
    model: DecoderOnlyTransformer,
    tokenizer: Tokenizer,
    examples: Iterable[InstructionExample],
    *,
    context_size: int,
    batch_size: int = 1,
    formatter: InstructionFormatter | None = None,
) -> CausalEvaluation:
    """Measure deterministic causal loss without changing model parameters.

    Each usable formatted sequence is partitioned into chunks containing at
    most ``context_size`` next-token targets. A one-token overlap carries the
    final token of one chunk into the next chunk as its first input.
    Consequently, every held-out target after the first token is scored
    exactly once.

    This is the same full-sequence causal objective used during post-training,
    not response-only loss. Every chunk starts a fresh model context and resets
    learned positions to zero; context is not carried across chunk boundaries.
    Rows encoding to fewer than two tokens are counted as skipped, while an
    entirely unusable evaluation split is rejected. Equal-width chunks are
    grouped into deterministic batches up to ``batch_size`` for efficiency.
    """

    if not isinstance(model, DecoderOnlyTransformer):
        raise TypeError("model must be a DecoderOnlyTransformer")
    if not isinstance(tokenizer, Tokenizer):
        raise TypeError("tokenizer must be a Tokenizer")
    checked_context = _positive_integer(context_size, "context_size")
    checked_batch_size = _positive_integer(batch_size, "batch_size")
    if checked_context > model.config.maximum_context:
        raise ValueError(
            "evaluation context_size exceeds the model maximum_context"
        )
    example_tuple = _instruction_examples(examples, "examples")
    configured_formatter = (
        PlainChatFormatter() if formatter is None else formatter
    )
    if not callable(getattr(configured_formatter, "format", None)):
        raise TypeError("formatter must provide a callable format() method")

    started = perf_counter()
    weighted_losses: list[float] = []
    target_token_count = 0
    chunk_count = 0
    forward_batch_count = 0
    usable_example_count = 0
    skipped_example_count = 0
    pending_by_width: dict[
        int,
        list[tuple[tuple[int, ...], tuple[int, ...]]],
    ] = {}

    def evaluate_pending(
        pending: list[tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> None:
        nonlocal forward_batch_count
        if not pending:
            return
        inputs = tuple(row[0] for row in pending)
        targets = tuple(row[1] for row in pending)
        evaluated_tokens = len(inputs) * len(inputs[0])
        with model(inputs) as logits:
            with cross_entropy(logits, targets) as loss:
                weighted_losses.append(loss.item() * evaluated_tokens)
        forward_batch_count += 1
        pending.clear()

    for example_index, example in enumerate(example_tuple):
        formatted = configured_formatter.format(example)
        if not isinstance(formatted, str):
            raise TypeError("formatter.format() must return a str")
        try:
            tokens = tuple(tokenizer.encode(formatted))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"could not encode evaluation example {example_index}: "
                f"{error}"
            ) from error
        if len(tokens) < 2:
            skipped_example_count += 1
            continue

        usable_example_count += 1
        for start in range(0, len(tokens) - 1, checked_context):
            stop = min(start + checked_context + 1, len(tokens))
            inputs = tokens[start : stop - 1]
            targets = tokens[start + 1 : stop]
            target_count = len(targets)
            pending = pending_by_width.setdefault(target_count, [])
            pending.append((inputs, targets))
            if len(pending) == checked_batch_size:
                evaluate_pending(pending)
            target_token_count += target_count
            chunk_count += 1

    for width in sorted(pending_by_width):
        evaluate_pending(pending_by_width[width])
    if target_token_count == 0:
        raise ValueError(
            "evaluation has no usable examples containing at least two tokens"
        )
    mean_loss = math.fsum(weighted_losses) / target_token_count
    return CausalEvaluation(
        example_count=len(example_tuple),
        usable_example_count=usable_example_count,
        skipped_example_count=skipped_example_count,
        target_token_count=target_token_count,
        chunk_count=chunk_count,
        forward_batch_count=forward_batch_count,
        loss=mean_loss,
        perplexity=loss_perplexity(mean_loss),
        elapsed_seconds=perf_counter() - started,
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
    frozen_formatter = _freeze_formatter(
        (splits.train, splits.validation, splits.test),
        configured_formatter,
    )
    if configured.reject_split_overlap:
        _validate_formatted_splits_disjoint(
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


class _FrozenFormatter:
    """Read-only formatted-text lookup shared by every rank trial."""

    __slots__ = ("_formatted",)

    def __init__(self, formatted: dict[tuple[str, str], str]) -> None:
        self._formatted = formatted

    def format(self, example: InstructionExample) -> str:
        if not isinstance(example, InstructionExample):
            raise TypeError("example must be an InstructionExample")
        try:
            return self._formatted[_example_key(example)]
        except KeyError as error:
            raise ValueError(
                "formatter received an example outside the prepared splits"
            ) from error


def _freeze_formatter(
    example_groups: Iterable[tuple[InstructionExample, ...]],
    formatter: InstructionFormatter,
) -> _FrozenFormatter:
    formatted: dict[tuple[str, str], str] = {}
    for examples in example_groups:
        for example in examples:
            key = _example_key(example)
            if key in formatted:
                continue
            value = formatter.format(example)
            if not isinstance(value, str):
                raise TypeError("formatter.format() must return a str")
            if not value:
                raise ValueError(
                    "formatter.format() must not return empty text"
                )
            formatted[key] = value
    return _FrozenFormatter(formatted)


def _validate_formatted_splits_disjoint(
    splits: InstructionSplits,
    formatter: InstructionFormatter,
) -> None:
    formatted = {
        "train": {
            formatter.format(example) for example in splits.train
        },
        "validation": {
            formatter.format(example) for example in splits.validation
        },
        "test": {
            formatter.format(example) for example in splits.test
        },
    }
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = formatted[left] & formatted[right]
        if overlap:
            raise ValueError(
                f"{left} and {right} splits overlap after instruction "
                f"formatting by {len(overlap)} canonical model input(s)"
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


def _instruction_examples(
    examples: Iterable[InstructionExample],
    name: str,
) -> tuple[InstructionExample, ...]:
    try:
        copied = tuple(examples)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable") from error
    if not copied:
        raise ValueError(f"{name} must not be empty")
    if any(not isinstance(example, InstructionExample) for example in copied):
        raise TypeError(
            f"{name} must contain only InstructionExample values"
        )
    return copied


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


def _examples_fingerprint(
    examples: tuple[InstructionExample, ...],
) -> str:
    digest = hashlib.sha256()
    for example in examples:
        record = json.dumps(
            {
                "prompt": example.prompt,
                "response": example.response,
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _example_key(example: InstructionExample) -> tuple[str, str]:
    return (example.prompt, example.response)


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
