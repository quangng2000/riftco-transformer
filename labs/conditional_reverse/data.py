"""Dependency-free token encoding and batching for conditional reversal."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Iterator, Sequence

from .protocol import Example, ProtocolConfig


@dataclass(frozen=True, slots=True)
class TokenCodec:
    """Stable alphabet-order token IDs with the delimiter last."""

    alphabet: str
    delimiter: str

    def __post_init__(self) -> None:
        if not isinstance(self.alphabet, str):
            raise TypeError("alphabet must be a str")
        if len(self.alphabet) < 2 or len(set(self.alphabet)) != len(
            self.alphabet
        ):
            raise ValueError("alphabet needs at least two unique symbols")
        if not isinstance(self.delimiter, str) or len(self.delimiter) != 1:
            raise ValueError("delimiter must be one character")
        if self.delimiter in self.alphabet:
            raise ValueError("delimiter must not be in the alphabet")

    @classmethod
    def from_protocol(cls, config: ProtocolConfig) -> TokenCodec:
        if not isinstance(config, ProtocolConfig):
            raise TypeError("config must be a ProtocolConfig")
        return cls(config.alphabet, config.delimiter)

    @property
    def vocabulary(self) -> tuple[str, ...]:
        return tuple(self.alphabet) + (self.delimiter,)

    @property
    def vocabulary_size(self) -> int:
        return len(self.alphabet) + 1

    @property
    def delimiter_id(self) -> int:
        return len(self.alphabet)

    def encode(self, text: str) -> tuple[int, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        token_ids = {token: index for index, token in enumerate(self.alphabet)}
        token_ids[self.delimiter] = self.delimiter_id
        try:
            return tuple(token_ids[token] for token in text)
        except KeyError as error:
            raise ValueError(
                f"text contains token outside the vocabulary: {error.args[0]!r}"
            ) from error

    def decode(self, token_ids: Iterable[int]) -> str:
        try:
            values = tuple(token_ids)
        except TypeError as error:
            raise TypeError("token_ids must be iterable") from error
        vocabulary = self.vocabulary
        decoded: list[str] = []
        for index, token_id in enumerate(values):
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError(f"token_ids[{index}] must be an int")
            if token_id < 0 or token_id >= len(vocabulary):
                raise ValueError(f"token_ids[{index}] is outside the vocabulary")
            decoded.append(vocabulary[token_id])
        return "".join(decoded)


@dataclass(frozen=True, slots=True)
class EncodedExample:
    """One teacher-forced example in native-friendly integer form."""

    source_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    reversed: bool

    def __post_init__(self) -> None:
        if not self.source_ids:
            raise ValueError("source_ids must not be empty")
        if len(self.output_ids) != len(self.source_ids):
            raise ValueError("source_ids and output_ids must have equal length")
        expected_context = 2 * len(self.source_ids)
        if len(self.input_ids) != expected_context:
            raise ValueError("input_ids has the wrong teacher-forced length")
        if len(self.target_ids) != expected_context:
            raise ValueError("target_ids has the wrong teacher-forced length")
        if self.target_ids[len(self.source_ids) :] != self.output_ids:
            raise ValueError("the supervised target half must equal output_ids")
        if not isinstance(self.reversed, bool):
            raise TypeError("reversed must be a bool")
        for name in ("source_ids", "output_ids", "input_ids", "target_ids"):
            for index, value in enumerate(getattr(self, name)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name}[{index}] must be an int")
                if value < 0:
                    raise ValueError(f"{name}[{index}] must be nonnegative")

    @property
    def sequence_length(self) -> int:
        return len(self.source_ids)

    @property
    def context_length(self) -> int:
        return len(self.input_ids)

    @property
    def supervised_time_range(self) -> tuple[int, int]:
        return (self.sequence_length, self.context_length)


@dataclass(frozen=True, slots=True)
class Batch:
    """Flattened row-major batch plus pairing and branch metadata."""

    example_indices: tuple[int, ...]
    input_ids: tuple[int, ...]
    target_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    reversed: tuple[bool, ...]
    batch_size: int
    sequence_length: int

    def __post_init__(self) -> None:
        _require_positive_int("batch_size", self.batch_size)
        _require_positive_int("sequence_length", self.sequence_length)
        if len(self.example_indices) != self.batch_size:
            raise ValueError("example_indices must have one value per row")
        if len(self.reversed) != self.batch_size:
            raise ValueError("reversed must have one value per row")
        if len(self.input_ids) != self.batch_size * self.context_length:
            raise ValueError("input_ids has the wrong flattened size")
        if len(self.target_ids) != self.batch_size * self.context_length:
            raise ValueError("target_ids has the wrong flattened size")
        if len(self.output_ids) != self.batch_size * self.sequence_length:
            raise ValueError("output_ids has the wrong flattened size")
        if any(not isinstance(value, bool) for value in self.reversed):
            raise TypeError("reversed values must be bools")
        for name in ("example_indices", "input_ids", "target_ids", "output_ids"):
            for index, value in enumerate(getattr(self, name)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{name}[{index}] must be an int")
                if value < 0:
                    raise ValueError(f"{name}[{index}] must be nonnegative")

    @property
    def context_length(self) -> int:
        return 2 * self.sequence_length

    @property
    def supervised_time_range(self) -> tuple[int, int]:
        """Half-open logits interval carrying the task loss."""

        return (self.sequence_length, self.context_length)

    @property
    def input_rows(self) -> tuple[tuple[int, ...], ...]:
        return _rows(self.input_ids, self.batch_size, self.context_length)

    @property
    def target_rows(self) -> tuple[tuple[int, ...], ...]:
        return _rows(self.target_ids, self.batch_size, self.context_length)

    @property
    def output_rows(self) -> tuple[tuple[int, ...], ...]:
        return _rows(self.output_ids, self.batch_size, self.sequence_length)


def encode_example(example: Example, codec: TokenCodec) -> EncodedExample:
    if not isinstance(example, Example):
        raise TypeError("example must be an Example")
    if not isinstance(codec, TokenCodec):
        raise TypeError("codec must be a TokenCodec")
    encoded = EncodedExample(
        source_ids=codec.encode(example.source),
        output_ids=codec.encode(example.target),
        input_ids=codec.encode(example.inputs),
        target_ids=codec.encode(example.targets),
        reversed=example.reversed,
    )
    if encoded.input_ids[encoded.sequence_length] != codec.delimiter_id:
        raise ValueError("teacher-forced input is missing its delimiter")
    return encoded


def make_batch(
    examples: Sequence[Example],
    codec: TokenCodec,
    *,
    example_indices: Sequence[int] | None = None,
) -> Batch:
    """Encode a nonempty sequence while retaining original dataset indices."""

    if not isinstance(codec, TokenCodec):
        raise TypeError("codec must be a TokenCodec")
    if not examples:
        raise ValueError("examples must not be empty")
    encoded = tuple(encode_example(example, codec) for example in examples)
    sequence_length = encoded[0].sequence_length
    if any(item.sequence_length != sequence_length for item in encoded):
        raise ValueError("all examples in a batch must have the same length")

    if example_indices is None:
        indices = tuple(range(len(encoded)))
    else:
        indices = tuple(example_indices)
        if len(indices) != len(encoded):
            raise ValueError("example_indices must match examples")

    return Batch(
        example_indices=indices,
        input_ids=tuple(
            token_id for item in encoded for token_id in item.input_ids
        ),
        target_ids=tuple(
            token_id for item in encoded for token_id in item.target_ids
        ),
        output_ids=tuple(
            token_id for item in encoded for token_id in item.output_ids
        ),
        reversed=tuple(item.reversed for item in encoded),
        batch_size=len(encoded),
        sequence_length=sequence_length,
    )


def iter_batches(
    examples: Sequence[Example],
    codec: TokenCodec,
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
    epoch: int = 0,
    drop_last: bool = False,
) -> Iterator[Batch]:
    """Iterate deterministic batches, with epoch-specific seeded shuffling."""

    if not isinstance(codec, TokenCodec):
        raise TypeError("codec must be a TokenCodec")
    _require_positive_int("batch_size", batch_size)
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be a bool")
    if not isinstance(drop_last, bool):
        raise TypeError("drop_last must be a bool")
    _require_int("seed", seed)
    _require_int("epoch", epoch)
    if epoch < 0:
        raise ValueError("epoch must be nonnegative")

    order = list(range(len(examples)))
    if shuffle:
        random.Random(_mixed_epoch_seed(seed, epoch)).shuffle(order)

    def generate() -> Iterator[Batch]:
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            if drop_last and len(indices) != batch_size:
                break
            if not indices:
                continue
            yield make_batch(
                tuple(examples[index] for index in indices),
                codec,
                example_indices=indices,
            )

    return generate()


def _mixed_epoch_seed(seed: int, epoch: int) -> int:
    mask = (1 << 64) - 1
    golden_ratio = 0x9E3779B97F4A7C15
    return (seed & mask) ^ (((epoch + 1) * golden_ratio) & mask)


def _rows(
    values: tuple[int, ...],
    row_count: int,
    row_width: int,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        values[row * row_width : (row + 1) * row_width]
        for row in range(row_count)
    )


def _require_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")


def _require_positive_int(name: str, value: object) -> None:
    _require_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


__all__ = [
    "Batch",
    "EncodedExample",
    "TokenCodec",
    "encode_example",
    "iter_batches",
    "make_batch",
]
