"""Dependency-free autoregressive text generation utilities.

Native models use request-local incremental decode sessions. Protocol-style
models retain the original full-forward fallback used by simple test doubles
and alternate implementations. Both paths keep the same rolling context,
sampling policy, and byte-safe result contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import operator
import random
from typing import Protocol, Sequence, runtime_checkable

from ..native import DecoderOnlyTransformer, Tokenizer


def _integer(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validated_logits(logits: Sequence[float]) -> tuple[float, ...]:
    try:
        values = tuple(logits)
    except TypeError as error:
        raise TypeError("logits must be a sequence of real numbers") from error
    if not values:
        raise ValueError("logits must not be empty")
    return tuple(
        _finite_real(value, f"logits[{index}]")
        for index, value in enumerate(values)
    )


@runtime_checkable
class SamplingStrategy(Protocol):
    """Policy that selects one token ID from a vector of logits."""

    def sample(self, logits: Sequence[float]) -> int:
        """Return an index in ``[0, len(logits))``."""


class GreedySampler:
    """Select the highest-logit token, preferring the lowest ID on ties."""

    __slots__ = ()

    def sample(self, logits: Sequence[float]) -> int:
        values = _validated_logits(logits)
        return max(range(len(values)), key=values.__getitem__)


class TemperatureSampler:
    """Sample from a temperature-scaled softmax with a private seeded RNG."""

    __slots__ = ("_random", "_seed", "_temperature", "_top_k")

    def __init__(
        self,
        temperature: float = 1.0,
        *,
        top_k: int | None = None,
        seed: int = 0,
    ) -> None:
        configured_temperature = _finite_real(
            temperature,
            "temperature",
        )
        if configured_temperature <= 0.0:
            raise ValueError("temperature must be greater than zero")

        configured_top_k = (
            None
            if top_k is None
            else _integer(top_k, "top_k", 1)
        )
        configured_seed = _integer(seed, "seed", 0)
        if configured_seed > (1 << 64) - 1:
            raise ValueError("seed must be at most 2**64 - 1")

        self._temperature = configured_temperature
        self._top_k = configured_top_k
        self._seed = configured_seed
        self._random = random.Random(configured_seed)

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def top_k(self) -> int | None:
        return self._top_k

    @property
    def seed(self) -> int:
        return self._seed

    def sample(self, logits: Sequence[float]) -> int:
        values = _validated_logits(logits)
        if self._top_k is not None and self._top_k > len(values):
            raise ValueError(
                "top_k must not exceed the number of logits"
            )

        ranked_ids = sorted(
            range(len(values)),
            key=lambda token: (-values[token], token),
        )
        candidate_ids = ranked_ids[
            : self._top_k if self._top_k is not None else len(ranked_ids)
        ]

        largest = values[candidate_ids[0]]
        weights = [
            math.exp(
                (values[token] - largest) / self._temperature
            )
            for token in candidate_ids
        ]
        total = math.fsum(weights)
        threshold = self._random.random() * total
        cumulative = 0.0
        for token, weight in zip(candidate_ids, weights):
            cumulative += weight
            if threshold < cumulative:
                return token

        # Floating-point rounding can leave threshold infinitesimally above
        # the accumulated sum. The final candidate still owns that tail.
        return candidate_ids[-1]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Tokens and byte-safe decoded output from one generation request."""

    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    raw_bytes: bytes
    text: str

    @property
    def prompt_tokens(self) -> tuple[int, ...]:
        """Alias for ``prompt_token_ids``."""

        return self.prompt_token_ids

    @property
    def generated_tokens(self) -> tuple[int, ...]:
        """Alias for ``generated_token_ids``."""

        return self.generated_token_ids

    @property
    def tokens(self) -> tuple[int, ...]:
        """Alias for all ``token_ids``."""

        return self.token_ids


class TextGenerator:
    """Generate text with a model, tokenizer, and swappable sampler."""

    __slots__ = (
        "_kv_cache",
        "_kv_cache_block_size",
        "_model",
        "_sampler",
        "_tokenizer",
    )

    def __init__(
        self,
        model: DecoderOnlyTransformer,
        tokenizer: Tokenizer,
        sampler: SamplingStrategy | None = None,
        *,
        kv_cache: str = "paged",
        kv_cache_block_size: int = 16,
    ) -> None:
        configured_sampler = (
            GreedySampler() if sampler is None else sampler
        )
        if not callable(getattr(configured_sampler, "sample", None)):
            raise TypeError("sampler must provide a callable sample() method")

        config = getattr(model, "config", None)
        if config is None:
            raise TypeError("model must expose a config")
        _integer(
            getattr(config, "maximum_context", None),
            "model.config.maximum_context",
            1,
        )
        _integer(
            getattr(config, "vocabulary_size", None),
            "model.config.vocabulary_size",
            1,
        )
        if not callable(getattr(tokenizer, "encode", None)):
            raise TypeError(
                "tokenizer must provide a callable encode() method"
            )
        if not callable(getattr(tokenizer, "decode_bytes", None)):
            raise TypeError(
                "tokenizer must provide a callable decode_bytes() method"
            )
        if not isinstance(kv_cache, str):
            raise TypeError("kv_cache must be 'contiguous' or 'paged'")
        if kv_cache not in {"contiguous", "paged"}:
            raise ValueError(
                "kv_cache must be 'contiguous' or 'paged'"
            )
        configured_block_size = _integer(
            kv_cache_block_size,
            "kv_cache_block_size",
            1,
        )

        self._kv_cache = kv_cache
        self._kv_cache_block_size = configured_block_size
        self._model = model
        self._tokenizer = tokenizer
        self._sampler = configured_sampler

    @property
    def model(self) -> DecoderOnlyTransformer:
        return self._model

    @property
    def tokenizer(self) -> Tokenizer:
        return self._tokenizer

    @property
    def sampler(self) -> SamplingStrategy:
        return self._sampler

    @property
    def kv_cache(self) -> str:
        return self._kv_cache

    @property
    def kv_cache_block_size(self) -> int:
        return self._kv_cache_block_size

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
    ) -> GenerationResult:
        """Generate exactly ``max_new_tokens`` after a nonempty prompt."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a str")
        token_count = _integer(
            max_new_tokens,
            "max_new_tokens",
            0,
        )
        maximum_context = _integer(
            self._model.config.maximum_context,
            "model.config.maximum_context",
            1,
        )
        vocabulary_size = _integer(
            self._model.config.vocabulary_size,
            "model.config.vocabulary_size",
            1,
        )

        encoded_prompt = self._tokenizer.encode(prompt)
        try:
            raw_prompt_tokens = tuple(encoded_prompt)
        except TypeError as error:
            raise TypeError(
                "tokenizer.encode() must return an iterable of token IDs"
            ) from error
        if not raw_prompt_tokens:
            raise ValueError("prompt must encode to at least one token")
        prompt_tokens = tuple(
            self._token_id(token, vocabulary_size, f"prompt token {index}")
            for index, token in enumerate(raw_prompt_tokens)
        )

        all_tokens = list(prompt_tokens)
        generated_tokens: list[int] = []
        if token_count != 0 and isinstance(
            self._model,
            DecoderOnlyTransformer,
        ):
            self._generate_with_decode_session(
                all_tokens,
                generated_tokens,
                token_count=token_count,
                maximum_context=maximum_context,
                vocabulary_size=vocabulary_size,
            )
        else:
            self._generate_with_full_forward(
                all_tokens,
                generated_tokens,
                token_count=token_count,
                maximum_context=maximum_context,
                vocabulary_size=vocabulary_size,
            )

        decoded = self._tokenizer.decode_bytes(all_tokens)
        if not isinstance(decoded, bytes):
            raise TypeError("tokenizer.decode_bytes() must return bytes")
        return GenerationResult(
            prompt_token_ids=prompt_tokens,
            generated_token_ids=tuple(generated_tokens),
            token_ids=tuple(all_tokens),
            raw_bytes=decoded,
            text=decoded.decode("utf-8", errors="replace"),
        )

    def _generate_with_full_forward(
        self,
        all_tokens: list[int],
        generated_tokens: list[int],
        *,
        token_count: int,
        maximum_context: int,
        vocabulary_size: int,
    ) -> None:
        for _ in range(token_count):
            context_tokens = all_tokens[-maximum_context:]
            logits = self._model([context_tokens])
            try:
                final_logits = self._final_position_logits(
                    logits,
                    sequence_length=len(context_tokens),
                    vocabulary_size=vocabulary_size,
                )
                selected = self._sampler.sample(final_logits)
                next_token = self._token_id(
                    selected,
                    vocabulary_size,
                    "sampler result",
                )
            finally:
                logits.close()
            all_tokens.append(next_token)
            generated_tokens.append(next_token)

    def _generate_with_decode_session(
        self,
        all_tokens: list[int],
        generated_tokens: list[int],
        *,
        token_count: int,
        maximum_context: int,
        vocabulary_size: int,
    ) -> None:
        with self._model.decode_session(
            cache=self._kv_cache,
            block_size=self._kv_cache_block_size,
        ) as session:
            if session.capacity != maximum_context:
                raise RuntimeError(
                    "decode-session capacity does not match model context"
                )

            context_tokens = all_tokens[-maximum_context:]
            current_logits: tuple[float, ...] = ()
            for token in context_tokens:
                current_logits = session.step(token)

            for index in range(token_count):
                next_token = self._token_id(
                    self._sampler.sample(
                        _validated_logits(current_logits)
                    ),
                    vocabulary_size,
                    "sampler result",
                )
                all_tokens.append(next_token)
                generated_tokens.append(next_token)

                if index + 1 == token_count:
                    continue
                if session.size == session.capacity:
                    session.reset()
                    context_tokens = all_tokens[-maximum_context:]
                    for token in context_tokens:
                        current_logits = session.step(token)
                else:
                    current_logits = session.step(next_token)

    @staticmethod
    def _token_id(value: object, vocabulary_size: int, name: str) -> int:
        token = _integer(value, name, 0)
        if token >= vocabulary_size:
            raise ValueError(
                f"{name} must be less than vocabulary size "
                f"{vocabulary_size}"
            )
        return token

    @staticmethod
    def _final_position_logits(
        logits: object,
        *,
        sequence_length: int,
        vocabulary_size: int,
    ) -> tuple[float, ...]:
        try:
            shape = tuple(logits.shape)
        except (AttributeError, TypeError) as error:
            raise TypeError("model logits must expose a shape") from error
        expected_shape = (1, sequence_length, vocabulary_size)
        if shape != expected_shape:
            raise ValueError(
                f"model logits shape must be {expected_shape}, got {shape}"
            )

        try:
            flattened = tuple(logits.tolist())
        except (AttributeError, TypeError) as error:
            raise TypeError(
                "model logits must provide tolist()"
            ) from error
        expected_values = sequence_length * vocabulary_size
        if len(flattened) != expected_values:
            raise ValueError(
                "model logits data length does not match its shape"
            )
        final_start = (sequence_length - 1) * vocabulary_size
        return _validated_logits(
            flattened[final_start : final_start + vocabulary_size]
        )


__all__ = [
    "GenerationResult",
    "GreedySampler",
    "SamplingStrategy",
    "TemperatureSampler",
    "TextGenerator",
]
