"""Held-out evaluation contracts for supervised post-training.

The types in this module are independent of the fine-tuning method. Full
fine-tuning and merged LoRA artifacts are ordinary decoder-only models, so
they must be scored by exactly the same evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Iterable

from ..data import PreparedDataset, verify_prepared_dataset
from ..native import DecoderOnlyTransformer, Tokenizer, cross_entropy
from ..training import loss_perplexity
from .pipeline import (
    InstructionExample,
    InstructionFormatter,
    PlainChatFormatter,
    load_instruction_jsonl,
    load_instruction_jsonl_bytes,
)


@dataclass(frozen=True, slots=True)
class DatasetFingerprints:
    """SHA-256 identities of three canonical instruction splits."""

    train: str
    validation: str
    test: str


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
        """Return content identities independent of paths and whitespace."""

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
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            overlap = records[left] & records[right]
            if overlap:
                raise ValueError(
                    f"{left} and {right} splits overlap by "
                    f"{len(overlap)} exact instruction record(s)"
                )


@dataclass(frozen=True, slots=True)
class CausalEvaluation:
    """Read-only, target-token-weighted causal-model measurements."""

    example_count: int
    usable_example_count: int
    skipped_example_count: int
    target_token_count: int
    chunk_count: int
    forward_batch_count: int
    loss: float
    perplexity: float
    elapsed_seconds: float


class _FrozenInstructionFormatter:
    """Read-only formatted-text lookup shared by every candidate."""

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
    """Load split buffers bound to one verified prepared-dataset snapshot."""

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


def freeze_instruction_formatter(
    splits: InstructionSplits,
    formatter: InstructionFormatter | None = None,
) -> InstructionFormatter:
    """Format every split once and return an immutable lookup formatter."""

    if not isinstance(splits, InstructionSplits):
        raise TypeError("splits must be InstructionSplits")
    configured = PlainChatFormatter() if formatter is None else formatter
    if not callable(getattr(configured, "format", None)):
        raise TypeError("formatter must provide a callable format() method")

    formatted: dict[tuple[str, str], str] = {}
    for examples in (splits.train, splits.validation, splits.test):
        for example in examples:
            key = _example_key(example)
            if key in formatted:
                continue
            value = configured.format(example)
            if not isinstance(value, str):
                raise TypeError("formatter.format() must return a str")
            if not value:
                raise ValueError(
                    "formatter.format() must not return empty text"
                )
            formatted[key] = value
    return _FrozenInstructionFormatter(formatted)


def validate_formatted_splits_disjoint(
    splits: InstructionSplits,
    formatter: InstructionFormatter,
) -> None:
    """Reject records that become identical after model-input formatting."""

    if not isinstance(splits, InstructionSplits):
        raise TypeError("splits must be InstructionSplits")
    if not callable(getattr(formatter, "format", None)):
        raise TypeError("formatter must provide a callable format() method")
    formatted = {
        "train": {formatter.format(example) for example in splits.train},
        "validation": {
            formatter.format(example) for example in splits.validation
        },
        "test": {formatter.format(example) for example in splits.test},
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
    final token of one chunk into the next chunk as its first input, so every
    target after the first token is scored exactly once. Equal-width chunks
    are grouped into deterministic batches and the final mean is weighted by
    target-token count.

    This is the full-sequence objective used by post-training. It includes the
    prompt, delimiters, and response; it is not response-only loss.
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
    pending_chunk_count = 0

    def evaluate_pending(
        pending: list[tuple[tuple[int, ...], tuple[int, ...]]],
    ) -> None:
        nonlocal forward_batch_count, pending_chunk_count
        if not pending:
            return
        inputs = tuple(row[0] for row in pending)
        targets = tuple(row[1] for row in pending)
        evaluated_tokens = len(inputs) * len(inputs[0])
        with model(inputs) as logits:
            with cross_entropy(logits, targets) as loss:
                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    raise ValueError(
                        "causal evaluation loss must be finite"
                    )
                weighted_losses.append(loss_value * evaluated_tokens)
        forward_batch_count += 1
        pending_chunk_count -= len(pending)
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
            pending_chunk_count += 1
            if len(pending) == checked_batch_size:
                evaluate_pending(pending)
            elif pending_chunk_count >= checked_batch_size:
                # Ragged tails can otherwise retain one almost-full chunk for
                # every possible width. Flush the fullest deterministic bucket
                # to keep buffered token storage O(batch_size * context_size).
                flush_width = max(
                    pending_by_width,
                    key=lambda width: (
                        len(pending_by_width[width]),
                        -width,
                    ),
                )
                evaluate_pending(pending_by_width[flush_width])
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
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


__all__ = [
    "CausalEvaluation",
    "DatasetFingerprints",
    "InstructionSplits",
    "evaluate_instruction_examples",
    "freeze_instruction_formatter",
    "load_instruction_splits",
    "load_prepared_instruction_splits",
    "validate_formatted_splits_disjoint",
]
