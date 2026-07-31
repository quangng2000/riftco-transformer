from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from transformer_lab import (  # noqa: E402
    DecoderOnlyTransformer,
    Tokenizer,
    TransformerConfig,
    backend_available,
)
from transformer_lab.generation import (  # noqa: E402
    GreedySampler,
    TemperatureSampler,
    TextGenerator,
)


@dataclass(frozen=True)
class FakeConfig:
    vocabulary_size: int
    maximum_context: int


class FakeLogits:
    def __init__(
        self,
        shape: tuple[int, ...],
        values: list[float],
    ) -> None:
        self.shape = shape
        self._values = values
        self.closed = False

    def tolist(self) -> list[float]:
        return list(self._values)

    def close(self) -> None:
        self.closed = True


class FakeModel:
    def __init__(
        self,
        *,
        vocabulary_size: int,
        maximum_context: int,
        next_tokens: list[int],
    ) -> None:
        self.config = FakeConfig(vocabulary_size, maximum_context)
        self._next_tokens = iter(next_tokens)
        self.contexts: list[tuple[int, ...]] = []
        self.logits: list[FakeLogits] = []

    def __call__(self, batch: list[list[int]]) -> FakeLogits:
        context = tuple(batch[0])
        self.contexts.append(context)
        next_token = next(self._next_tokens)
        values = [0.0] * (
            len(context) * self.config.vocabulary_size
        )
        # A misleading maximum in an earlier position catches generation
        # code that does not slice the final position.
        if len(context) > 1:
            values[0] = 100.0
        final_start = (
            len(context) - 1
        ) * self.config.vocabulary_size
        values[final_start + next_token] = 10.0
        logits = FakeLogits(
            (
                1,
                len(context),
                self.config.vocabulary_size,
            ),
            values,
        )
        self.logits.append(logits)
        return logits


class ForwardOnlyModel:
    """Expose only the legacy full-forward protocol of a native model."""

    def __init__(self, model: DecoderOnlyTransformer) -> None:
        self.config = model.config
        self._model = model
        self.contexts: list[tuple[int, ...]] = []

    def __call__(self, batch: list[list[int]]) -> object:
        self.contexts.append(tuple(batch[0]))
        return self._model(batch)


class DigitTokenizer:
    def encode(self, text: str) -> list[int]:
        return [int(character) for character in text]

    def decode_bytes(self, tokens: list[int]) -> bytes:
        return "".join(str(token) for token in tokens).encode("ascii")


class RaisingSampler:
    def sample(self, _logits: list[float]) -> int:
        raise RuntimeError("sampling failed")


class GenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not backend_available("cpu"):
            raise AssertionError("the CPU backend must be available")

    def test_greedy_selects_first_maximum_and_rejects_bad_logits(
        self,
    ) -> None:
        sampler = GreedySampler()
        self.assertEqual(sampler.sample([-2.0, 3.0, 3.0]), 1)
        for logits, error_type in (
            ([], ValueError),
            ([0.0, math.nan], ValueError),
            ([0.0, math.inf], ValueError),
            ([False, 1.0], TypeError),
        ):
            with self.subTest(logits=logits):
                with self.assertRaises(error_type):
                    sampler.sample(logits)

    def test_temperature_sampling_is_seeded_and_top_k_is_respected(
        self,
    ) -> None:
        first = TemperatureSampler(0.75, seed=17)
        second = TemperatureSampler(0.75, seed=17)
        first_tokens = [
            first.sample([0.0, 0.2, 0.4, 0.8])
            for _ in range(20)
        ]
        second_tokens = [
            second.sample([0.0, 0.2, 0.4, 0.8])
            for _ in range(20)
        ]
        self.assertEqual(first_tokens, second_tokens)
        self.assertEqual(
            [
                TemperatureSampler(
                    100.0,
                    top_k=1,
                    seed=seed,
                ).sample([-5.0, 7.0, 6.0])
                for seed in range(5)
            ],
            [1] * 5,
        )

    def test_temperature_sampler_validates_configuration(self) -> None:
        for temperature in (
            True,
            0.0,
            -1.0,
            math.inf,
            math.nan,
        ):
            with self.subTest(temperature=temperature):
                with self.assertRaises((TypeError, ValueError)):
                    TemperatureSampler(temperature)
        for top_k in (False, 0, -1, 1.5):
            with self.subTest(top_k=top_k):
                with self.assertRaises((TypeError, ValueError)):
                    TemperatureSampler(top_k=top_k)
        for seed in (False, -1, 1 << 64):
            with self.subTest(seed=seed):
                with self.assertRaises((TypeError, ValueError)):
                    TemperatureSampler(seed=seed)
        with self.assertRaisesRegex(ValueError, "number of logits"):
            TemperatureSampler(top_k=4).sample([1.0, 2.0, 3.0])

    def test_generation_crops_context_and_uses_final_position(self) -> None:
        model = FakeModel(
            vocabulary_size=10,
            maximum_context=3,
            next_tokens=[5, 6, 7],
        )
        generator = TextGenerator(
            model,
            DigitTokenizer(),
            GreedySampler(),
        )

        result = generator.generate("1234", max_new_tokens=3)

        self.assertEqual(
            model.contexts,
            [(2, 3, 4), (3, 4, 5), (4, 5, 6)],
        )
        self.assertEqual(result.prompt_token_ids, (1, 2, 3, 4))
        self.assertEqual(result.generated_token_ids, (5, 6, 7))
        self.assertEqual(result.token_ids, (1, 2, 3, 4, 5, 6, 7))
        self.assertEqual(result.raw_bytes, b"1234567")
        self.assertEqual(result.text, "1234567")
        self.assertTrue(all(logits.closed for logits in model.logits))

    def test_generation_closes_logits_when_sampling_fails(self) -> None:
        model = FakeModel(
            vocabulary_size=10,
            maximum_context=3,
            next_tokens=[5],
        )
        generator = TextGenerator(
            model,
            DigitTokenizer(),
            RaisingSampler(),
        )

        with self.assertRaisesRegex(RuntimeError, "sampling failed"):
            generator.generate("123", max_new_tokens=1)

        self.assertEqual(len(model.logits), 1)
        self.assertTrue(model.logits[0].closed)

    def test_zero_tokens_skips_model_and_empty_prompt_is_rejected(
        self,
    ) -> None:
        model = FakeModel(
            vocabulary_size=10,
            maximum_context=3,
            next_tokens=[],
        )
        generator = TextGenerator(model, DigitTokenizer())

        result = generator.generate("12", max_new_tokens=0)

        self.assertEqual(result.generated_tokens, ())
        self.assertEqual(result.tokens, (1, 2))
        self.assertEqual(model.contexts, [])
        with self.assertRaisesRegex(ValueError, "at least one token"):
            generator.generate("", max_new_tokens=0)
        for value in (False, -1, 1.5):
            with self.subTest(max_new_tokens=value):
                with self.assertRaises((TypeError, ValueError)):
                    generator.generate("1", max_new_tokens=value)

    def test_result_preserves_invalid_utf8_as_bytes(self) -> None:
        with Tokenizer(b"a\xff") as tokenizer:
            model = FakeModel(
                vocabulary_size=tokenizer.vocab_size,
                maximum_context=2,
                next_tokens=[1],
            )
            result = TextGenerator(model, tokenizer).generate(
                "a",
                max_new_tokens=1,
            )

        self.assertEqual(result.raw_bytes, b"a\xff")
        self.assertEqual(result.text, "a\ufffd")

    def test_real_cpu_model_and_tokenizer_generate_requested_count(
        self,
    ) -> None:
        with Tokenizer("abc abc") as tokenizer:
            config = TransformerConfig(
                vocabulary_size=tokenizer.vocab_size,
                maximum_context=3,
                model_width=4,
                head_count=2,
                block_count=1,
                feed_forward_width=8,
                random_seed=71,
            )
            with DecoderOnlyTransformer(config).to("cpu") as model:
                result = TextGenerator(model, tokenizer).generate(
                    "ab",
                    max_new_tokens=2,
                )

        self.assertEqual(len(result.prompt_token_ids), 2)
        self.assertEqual(len(result.generated_token_ids), 2)
        self.assertEqual(len(result.token_ids), 4)
        self.assertEqual(
            result.text,
            result.raw_bytes.decode("utf-8", errors="replace"),
        )

    def test_native_generation_uses_decode_session_with_rolling_parity(
        self,
    ) -> None:
        with Tokenizer("0123456789") as tokenizer:
            config = TransformerConfig(
                vocabulary_size=tokenizer.vocab_size,
                maximum_context=3,
                model_width=4,
                head_count=2,
                block_count=1,
                feed_forward_width=8,
                random_seed=79,
            )
            with DecoderOnlyTransformer(config) as model:
                legacy_model = ForwardOnlyModel(model)
                legacy = TextGenerator(
                    legacy_model,
                    tokenizer,
                ).generate("1234", max_new_tokens=4)

                generator = TextGenerator(
                    model,
                    tokenizer,
                    kv_cache="paged",
                    kv_cache_block_size=2,
                )
                self.assertEqual(generator.kv_cache, "paged")
                self.assertEqual(generator.kv_cache_block_size, 2)
                with mock.patch.object(
                    DecoderOnlyTransformer,
                    "__call__",
                    side_effect=AssertionError(
                        "native generation used full forward"
                    ),
                ):
                    cached = generator.generate(
                        "1234",
                        max_new_tokens=4,
                    )

        self.assertEqual(
            legacy_model.contexts,
            [
                tuple(legacy.token_ids[index : index + 3])
                for index in range(1, 5)
            ],
        )
        self.assertEqual(cached.prompt_token_ids, legacy.prompt_token_ids)
        self.assertEqual(
            cached.generated_token_ids,
            legacy.generated_token_ids,
        )
        self.assertEqual(cached.token_ids, legacy.token_ids)
        self.assertEqual(cached.raw_bytes, legacy.raw_bytes)
        self.assertEqual(cached.text, legacy.text)

    def test_native_generation_closes_session_when_sampling_fails(
        self,
    ) -> None:
        with Tokenizer("0123") as tokenizer:
            config = TransformerConfig(
                vocabulary_size=tokenizer.vocab_size,
                maximum_context=3,
                model_width=4,
                head_count=2,
                block_count=1,
                feed_forward_width=8,
                random_seed=83,
            )
            with DecoderOnlyTransformer(config) as model:
                generator = TextGenerator(
                    model,
                    tokenizer,
                    RaisingSampler(),
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "sampling failed",
                ):
                    generator.generate("12", max_new_tokens=1)

                # A leaked session would keep the model-mutation guard active.
                self.assertIs(model.to("cpu"), model)

    def test_generation_validates_kv_cache_configuration(self) -> None:
        model = FakeModel(
            vocabulary_size=10,
            maximum_context=3,
            next_tokens=[],
        )
        with self.assertRaises(TypeError):
            TextGenerator(
                model,
                DigitTokenizer(),
                kv_cache=1,
            )
        with self.assertRaises(ValueError):
            TextGenerator(
                model,
                DigitTokenizer(),
                kv_cache="unknown",
            )
        for block_size in (False, 0, -1, 1.5):
            with self.subTest(block_size=block_size):
                with self.assertRaises((TypeError, ValueError)):
                    TextGenerator(
                        model,
                        DigitTokenizer(),
                        kv_cache_block_size=block_size,
                    )


if __name__ == "__main__":
    unittest.main()
