"""Deterministic conditional string-reversal task and evaluation protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
from typing import Callable, Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ProtocolConfig:
    """Task vocabulary, condition, and deterministic data seed."""

    sequence_length: int = 15
    alphabet: str = "abcdefghijklmnopqrstuvwxyz"
    reverse_when_first_is: str = "aeiou"
    delimiter: str = "|"
    seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.sequence_length, bool) or not isinstance(
            self.sequence_length, int
        ):
            raise TypeError("sequence_length must be an int")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")
        if not isinstance(self.alphabet, str):
            raise TypeError("alphabet must be a str")
        if len(self.alphabet) < 2 or len(set(self.alphabet)) != len(
            self.alphabet
        ):
            raise ValueError("alphabet needs at least two unique symbols")
        if not isinstance(self.reverse_when_first_is, str):
            raise TypeError("reverse_when_first_is must be a str")
        triggers = self.reverse_when_first_is
        if len(set(triggers)) != len(triggers):
            raise ValueError("reverse trigger symbols must be unique")
        if any(symbol not in self.alphabet for symbol in triggers):
            raise ValueError("reverse trigger is absent from the alphabet")
        if not triggers or len(triggers) == len(self.alphabet):
            raise ValueError("task needs both reverse and copy branches")
        if not isinstance(self.delimiter, str) or len(self.delimiter) != 1:
            raise ValueError("delimiter must be one character")
        if self.delimiter in self.alphabet:
            raise ValueError("delimiter must not be in the alphabet")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")

    @property
    def context_length(self) -> int:
        """Teacher-forced input length for ``source|target``."""

        return 2 * self.sequence_length


@dataclass(frozen=True, slots=True)
class SplitSizes:
    train: int = 10_000
    probe: int = 5_000
    validation: int = 1_000
    test: int = 1_000

    def __post_init__(self) -> None:
        for name in ("train", "probe", "validation", "test"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")

    @property
    def total(self) -> int:
        return self.train + self.probe + self.validation + self.test


@dataclass(frozen=True, slots=True)
class Example:
    source: str
    target: str
    reversed: bool
    tokens: str
    inputs: str
    targets: str


@dataclass(frozen=True, slots=True)
class Evaluation:
    example_count: int
    target_token_count: int
    target_token_accuracy: float
    exact_sequence_accuracy: float
    reverse_target_token_accuracy: float
    copy_target_token_accuracy: float
    reverse_example_count: int
    copy_example_count: int


def make_example(source: str, config: ProtocolConfig) -> Example:
    if not isinstance(source, str):
        raise TypeError("source must be a str")
    if len(source) != config.sequence_length:
        raise ValueError("source has the wrong length")
    if any(symbol not in config.alphabet for symbol in source):
        raise ValueError("source contains a symbol outside the alphabet")
    reversed_branch = source[0] in config.reverse_when_first_is
    target = source[::-1] if reversed_branch else source
    tokens = source + config.delimiter + target
    return Example(
        source=source,
        target=target,
        reversed=reversed_branch,
        tokens=tokens,
        inputs=tokens[:-1],
        targets=tokens[1:],
    )


def generate_disjoint_splits(
    config: ProtocolConfig = ProtocolConfig(),
    sizes: SplitSizes = SplitSizes(),
) -> dict[str, tuple[Example, ...]]:
    """Sample source-disjoint splits with one seeded random stream.

    Floyd's algorithm keeps memory proportional to the requested examples and
    works even when ``len(alphabet) ** sequence_length`` exceeds platform
    sequence limits.
    """

    source_space = len(config.alphabet) ** config.sequence_length
    if sizes.total > source_space:
        raise ValueError("requested splits exceed the finite source space")
    generator = random.Random(config.seed)
    selected: set[int] = set()
    indices: list[int] = []
    for upper in range(source_space - sizes.total, source_space):
        candidate = generator.randrange(upper + 1)
        chosen = upper if candidate in selected else candidate
        selected.add(chosen)
        indices.append(chosen)
    generator.shuffle(indices)
    examples = tuple(
        make_example(_decode_source(index, config), config)
        for index in indices
    )
    result: dict[str, tuple[Example, ...]] = {}
    offset = 0
    for name in ("train", "probe", "validation", "test"):
        count = getattr(sizes, name)
        result[name] = examples[offset : offset + count]
        offset += count
    return result


def split_fingerprint(examples: Iterable[Example]) -> str:
    digest = hashlib.sha256()
    for example in examples:
        digest.update(example.source.encode("utf-8"))
        digest.update(b"\0")
        digest.update(example.target.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


Prediction = Callable[[Example], str]


def predict_oracle(example: Example) -> str:
    return example.target


def predict_copy(example: Example) -> str:
    return example.source


def predict_reverse(example: Example) -> str:
    return example.source[::-1]


def evaluate(
    examples: Sequence[Example],
    predictor: Prediction,
) -> Evaluation:
    if not examples:
        raise ValueError("evaluation examples must not be empty")
    if not callable(predictor):
        raise TypeError("predictor must be callable")
    correct_tokens = 0
    exact = 0
    branch_correct = {True: 0, False: 0}
    branch_tokens = {True: 0, False: 0}
    branch_examples = {True: 0, False: 0}
    for example in examples:
        prediction = predictor(example)
        if not isinstance(prediction, str):
            raise TypeError("predictor must return a str")
        if len(prediction) != len(example.target):
            raise ValueError("prediction has the wrong target length")
        matches = sum(
            predicted == expected
            for predicted, expected in zip(prediction, example.target)
        )
        correct_tokens += matches
        exact += prediction == example.target
        branch_correct[example.reversed] += matches
        branch_tokens[example.reversed] += len(example.target)
        branch_examples[example.reversed] += 1
    token_count = sum(len(example.target) for example in examples)
    return Evaluation(
        example_count=len(examples),
        target_token_count=token_count,
        target_token_accuracy=correct_tokens / token_count,
        exact_sequence_accuracy=exact / len(examples),
        reverse_target_token_accuracy=_safe_ratio(
            branch_correct[True], branch_tokens[True]
        ),
        copy_target_token_accuracy=_safe_ratio(
            branch_correct[False], branch_tokens[False]
        ),
        reverse_example_count=branch_examples[True],
        copy_example_count=branch_examples[False],
    )


def verify_disjoint(splits: Mapping[str, Sequence[Example]]) -> None:
    owner: dict[str, str] = {}
    for split_name, examples in splits.items():
        for example in examples:
            previous = owner.get(example.source)
            if previous is not None:
                raise ValueError(
                    f"source appears in both {previous} and {split_name}"
                )
            owner[example.source] = split_name


def _decode_source(index: int, config: ProtocolConfig) -> str:
    symbols = [config.alphabet[0]] * config.sequence_length
    base = len(config.alphabet)
    for position in range(config.sequence_length - 1, -1, -1):
        index, symbol = divmod(index, base)
        symbols[position] = config.alphabet[symbol]
    return "".join(symbols)


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


__all__ = [
    "Evaluation",
    "Example",
    "ProtocolConfig",
    "SplitSizes",
    "evaluate",
    "generate_disjoint_splits",
    "make_example",
    "predict_copy",
    "predict_oracle",
    "predict_reverse",
    "split_fingerprint",
    "verify_disjoint",
]
