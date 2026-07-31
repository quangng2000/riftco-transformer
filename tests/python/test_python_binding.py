from __future__ import annotations

import copy
import gc
import math
import sys
import threading
import unittest
import weakref
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from transformer_lab import (  # noqa: E402
    ACTIVATION_CHECKPOINTING_DISABLED,
    ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK,
    Adam,
    DecodeSession,
    FULL_SEQUENCE_ATTENTION_FLASH,
    FULL_SEQUENCE_ATTENTION_MATERIALIZED,
    LORA_TARGET_ATTENTION_QUERY,
    LORA_TARGET_ATTENTION_VALUE,
    LORA_TARGET_DEFAULT,
    LORA_TARGET_NAMES,
    LoraConfig,
    STATUS_BACKEND_UNAVAILABLE,
    STATUS_INVALID_ARGUMENT,
    STATUS_OUT_OF_RANGE,
    Context,
    DecoderOnlyTransformer,
    Tensor,
    TensorLabError,
    Tokenizer,
    TransformerConfig,
    backend_available,
    cross_entropy,
)
from transformer_lab import _abi_version_is_compatible  # noqa: E402


class PythonBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not backend_available("cpu"):
            raise AssertionError("the CPU backend must be available")

    def test_abi_compatibility_policy(self) -> None:
        self.assertFalse(_abi_version_is_compatible(0x00010002))
        self.assertFalse(_abi_version_is_compatible(0x00010003))
        self.assertFalse(_abi_version_is_compatible(0x00010004))
        self.assertFalse(_abi_version_is_compatible(0x00010005))
        self.assertFalse(_abi_version_is_compatible(0x00010006))
        self.assertFalse(_abi_version_is_compatible(0x00010007))
        self.assertTrue(_abi_version_is_compatible(0x00010008))
        self.assertTrue(_abi_version_is_compatible(0x00010009))
        self.assertFalse(_abi_version_is_compatible(0x00010001))
        self.assertFalse(_abi_version_is_compatible(0x00010000))
        self.assertFalse(_abi_version_is_compatible(0x000000FF))
        self.assertFalse(_abi_version_is_compatible(0x00020000))

    def test_tokenizer_deterministic_vocabulary(self) -> None:
        with Tokenizer(b"cab\ncab") as tokenizer:
            self.assertEqual(tokenizer.method, "byte")
            self.assertEqual(tokenizer.vocab_size, 4)
            self.assertEqual(tokenizer.vocabulary_bytes, b"\nabc")
            self.assertEqual(
                tokenizer.vocabulary,
                (b"\n", b"a", b"b", b"c"),
            )
            self.assertEqual(
                tokenizer.encode_bytes(b"cab\ncab"),
                [3, 1, 2, 0, 3, 1, 2],
            )

        with Tokenizer(bytearray(b"\nbacc")) as reordered:
            self.assertEqual(reordered.vocabulary_bytes, b"\nabc")

        with Tokenizer(b"cab\ncab", method="byte") as explicit:
            self.assertEqual(explicit.method, "byte")
            self.assertEqual(explicit.vocabulary_bytes, b"\nabc")
            self.assertEqual(
                explicit.encode_bytes(b"cab\ncab"),
                [3, 1, 2, 0, 3, 1, 2],
            )

    def test_bpe_compression_vocabulary_and_unseen_bytes(self) -> None:
        with Tokenizer(
            b"abababab",
            method="bpe",
            vocabulary_size=257,
            minimum_pair_frequency=2,
        ) as tokenizer:
            self.assertEqual(tokenizer.method, "bpe")
            self.assertEqual(tokenizer.vocab_size, 257)
            self.assertEqual(tokenizer.vocabulary[0], b"\0")
            self.assertEqual(tokenizer.vocabulary[97], b"a")
            self.assertEqual(tokenizer.vocabulary[255], b"\xff")
            self.assertEqual(tokenizer.vocabulary[256], b"ab")

            encoded = tokenizer.encode_bytes(b"abababab")
            self.assertEqual(encoded, [256, 256, 256, 256])
            self.assertLess(len(encoded), len(b"abababab"))
            self.assertEqual(
                tokenizer.decode_bytes(encoded),
                b"abababab",
            )

            unseen = b"\xff\0z"
            self.assertEqual(
                tokenizer.decode_bytes(tokenizer.encode_bytes(unseen)),
                unseen,
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "use vocabulary",
            ):
                _ = tokenizer.vocabulary_bytes

    def test_bpe_tie_breaking_is_deterministic(self) -> None:
        corpus = b"ababacac"
        with Tokenizer(
            corpus,
            method="bpe",
            vocabulary_size=257,
        ) as first:
            first_vocabulary = first.vocabulary
            first_encoding = first.encode_bytes(corpus)
        with Tokenizer(
            corpus,
            method="bpe",
            vocabulary_size=257,
        ) as second:
            self.assertEqual(second.vocabulary, first_vocabulary)
            self.assertEqual(second.encode_bytes(corpus), first_encoding)
        self.assertEqual(first_vocabulary[256], b"ab")

    def test_tokenizer_state_round_trip(self) -> None:
        corpus = b"abababab"
        with Tokenizer(
            corpus,
            method="bpe",
            vocabulary_size=258,
        ) as original:
            rules = original.merge_rules
            expected_vocabulary = original.vocabulary
            expected_tokens = original.encode_bytes(corpus)

        self.assertEqual(
            rules,
            ((97, 98, 256), (256, 256, 257)),
        )
        with Tokenizer.from_state(
            method="bpe",
            merge_rules=rules,
        ) as restored:
            self.assertEqual(restored.method, "bpe")
            self.assertEqual(restored.merge_rules, rules)
            self.assertEqual(restored.vocabulary, expected_vocabulary)
            self.assertEqual(restored.encode_bytes(corpus), expected_tokens)
            self.assertEqual(restored.decode_bytes(expected_tokens), corpus)

        with Tokenizer.from_state(
            method="byte",
            byte_vocabulary=(ord("z"), ord("a")),
        ) as restored_byte:
            self.assertEqual(restored_byte.vocabulary_bytes, b"za")
            self.assertEqual(restored_byte.encode("az"), [1, 0])
            with self.assertRaisesRegex(RuntimeError, "BPE"):
                _ = restored_byte.merge_rules

        with self.assertRaises(TensorLabError):
            Tokenizer.from_state(
                method="bpe",
                merge_rules=((97, 98, 257),),
            )
        with self.assertRaises(ValueError):
            Tokenizer.from_state(
                method="byte",
                byte_vocabulary=(97,),
                merge_rules=((97, 97, 256),),
            )

    def test_tokenizer_utf8_and_embedded_nul_round_trip(self) -> None:
        text = "café 🙂"
        with Tokenizer(text) as tokenizer:
            tokens = tokenizer.encode(text)
            self.assertEqual(len(tokens), len(text.encode("utf-8")))
            self.assertEqual(tokenizer.decode(tokens), text)

        corpus = memoryview(bytearray(b"\0abc"))
        with Tokenizer(corpus) as tokenizer:
            source = bytearray(b"a\0c")
            tokens = tokenizer.encode_bytes(memoryview(source))
            source[:] = b"bbb"
            self.assertEqual(tokenizer.decode_bytes(tokens), b"a\0c")

        with Tokenizer(b"\xff") as tokenizer:
            self.assertEqual(tokenizer.decode_bytes([0]), b"\xff")
            with self.assertRaises(UnicodeDecodeError):
                tokenizer.decode([0])

    def test_tokenizer_empty_inputs_and_native_errors(self) -> None:
        with Tokenizer(b"abc") as tokenizer:
            self.assertEqual(tokenizer.encode_bytes(b""), [])
            self.assertEqual(tokenizer.decode_bytes([]), b"")
            self.assertEqual(tokenizer.encode(""), [])
            self.assertEqual(tokenizer.decode(iter(())), "")

            with self.assertRaises(TensorLabError) as encode_error:
                tokenizer.encode_bytes(b"d")
            self.assertEqual(
                encode_error.exception.status,
                STATUS_INVALID_ARGUMENT,
            )
            self.assertIn("absent", encode_error.exception.detail)

            with self.assertRaises(TensorLabError) as decode_error:
                tokenizer.decode_bytes([tokenizer.vocab_size])
            self.assertEqual(
                decode_error.exception.status,
                STATUS_OUT_OF_RANGE,
            )
            self.assertIn("outside", decode_error.exception.detail)

        with self.assertRaises(TensorLabError) as corpus_error:
            Tokenizer(b"")
        self.assertEqual(
            corpus_error.exception.status,
            STATUS_INVALID_ARGUMENT,
        )
        with self.assertRaises(TensorLabError) as bpe_corpus_error:
            Tokenizer(b"", method="bpe")
        self.assertEqual(
            bpe_corpus_error.exception.status,
            STATUS_INVALID_ARGUMENT,
        )

    def test_tokenizer_python_validation(self) -> None:
        with self.assertRaises(TypeError):
            Tokenizer(123)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            Tokenizer(b"abc", method=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Tokenizer(b"abc", method="wordpiece")
        with self.assertRaises(ValueError):
            Tokenizer(b"abc", vocabulary_size=256)
        with self.assertRaises(ValueError):
            Tokenizer(b"abc", minimum_pair_frequency=1)

        with self.assertRaises(TypeError):
            Tokenizer(b"abc", method="bpe", vocabulary_size=True)
        with self.assertRaises(TypeError):
            Tokenizer(b"abc", method="bpe", vocabulary_size=256.0)
        with self.assertRaises(ValueError):
            Tokenizer(b"abc", method="bpe", vocabulary_size=255)
        with self.assertRaises(OverflowError):
            Tokenizer(b"abc", method="bpe", vocabulary_size=1 << 32)
        with self.assertRaises(TypeError):
            Tokenizer(
                b"abc",
                method="bpe",
                minimum_pair_frequency=True,
            )
        with self.assertRaises(TypeError):
            Tokenizer(
                b"abc",
                method="bpe",
                minimum_pair_frequency=2.0,
            )
        with self.assertRaises(ValueError):
            Tokenizer(
                b"abc",
                method="bpe",
                minimum_pair_frequency=0,
            )
        with Tokenizer(
            b"abc",
            method="bpe",
            vocabulary_size=256,
            minimum_pair_frequency=1,
        ) as tokenizer:
            self.assertEqual(tokenizer.method, "bpe")

        with Tokenizer("abc") as tokenizer:
            with self.assertRaises(TypeError):
                tokenizer.encode_bytes("abc")
            with self.assertRaises(TypeError):
                tokenizer.encode_bytes(3)
            with self.assertRaises(TypeError):
                tokenizer.encode(b"abc")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                tokenizer.decode_bytes(True)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                tokenizer.decode_bytes([True])
            with self.assertRaises(TypeError):
                tokenizer.decode_bytes([1.5])
            with self.assertRaises(ValueError):
                tokenizer.decode_bytes([-1])
            with self.assertRaises(OverflowError):
                tokenizer.decode_bytes([1 << 32])

    def test_tokenizer_lifecycle_and_closed_handle(self) -> None:
        tokenizer = Tokenizer(
            b"abcabc",
            method="bpe",
            vocabulary_size=257,
        )
        with self.assertRaises(TypeError):
            copy.copy(tokenizer)
        with self.assertRaises(TypeError):
            copy.deepcopy(tokenizer)

        with tokenizer as active:
            self.assertIs(active, tokenizer)
            self.assertFalse(tokenizer.closed)
        tokenizer.close()
        self.assertTrue(tokenizer.closed)

        operations = (
            lambda: tokenizer.method,
            lambda: tokenizer.vocab_size,
            lambda: tokenizer.vocabulary,
            lambda: tokenizer.vocabulary_bytes,
            lambda: tokenizer.encode_bytes(b"a"),
            lambda: tokenizer.decode_bytes([0]),
            lambda: tokenizer.encode("a"),
            lambda: tokenizer.decode([0]),
            tokenizer.__enter__,
        )
        for operation in operations:
            with self.assertRaises(RuntimeError):
                operation()

    def test_cpu_tensor_creation_zeros_and_scalar(self) -> None:
        with Context("cpu") as context:
            self.assertEqual(context.backend, "cpu")

            source = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            with Tensor.from_data(context, (2, 3), source) as tensor:
                source[0] = 999.0
                self.assertEqual(tensor.backend, "cpu")
                self.assertEqual(tensor.rank, 2)
                self.assertEqual(tensor.shape, (2, 3))
                self.assertEqual(tensor.numel, 6)
                self.assertEqual(
                    tensor.tolist(),
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                )

            with Tensor.zeros(context, (2, 2)) as zeros:
                self.assertEqual(zeros.shape, (2, 2))
                self.assertEqual(zeros.tolist(), [0.0, 0.0, 0.0, 0.0])

            with Tensor.from_data(context, (), [3.5]) as scalar:
                self.assertEqual(scalar.rank, 0)
                self.assertEqual(scalar.shape, ())
                self.assertEqual(scalar.numel, 1)
                self.assertEqual(scalar.tolist(), [3.5])

    def test_cpu_matmul(self) -> None:
        with Context() as context:
            with Tensor.from_data(
                context,
                (2, 3),
                [1, 2, 3, 4, 5, 6],
            ) as left:
                with Tensor.from_data(
                    context,
                    (3, 2),
                    [7, 8, 9, 10, 11, 12],
                ) as right:
                    with left @ right as product:
                        self.assertEqual(product.shape, (2, 2))
                        self.assertEqual(
                            product.tolist(),
                            [58.0, 64.0, 139.0, 154.0],
                        )
                        self.assertIs(product.context, context)

    def test_native_errors_become_python_exceptions(self) -> None:
        with Context("cpu") as context:
            with self.assertRaises(TensorLabError) as creation_error:
                Tensor.from_data(context, (2, 2), [1.0])
            self.assertEqual(
                creation_error.exception.status,
                STATUS_INVALID_ARGUMENT,
            )
            self.assertIn("value count", creation_error.exception.detail)

            with Tensor.zeros(context, (2, 3)) as left:
                with Tensor.zeros(context, (4, 2)) as right:
                    with self.assertRaises(TensorLabError) as matmul_error:
                        left @ right
            self.assertEqual(
                matmul_error.exception.status,
                STATUS_INVALID_ARGUMENT,
            )
            self.assertIn("inner dimensions", matmul_error.exception.detail)

        with self.assertRaises(ValueError):
            backend_available("not-a-backend")

    def test_close_is_idempotent_and_context_is_retained(self) -> None:
        context = Context("cpu")
        context_reference = weakref.ref(context)
        tensor = Tensor.zeros(context, (2,))

        del context
        gc.collect()
        self.assertIsNotNone(context_reference())
        self.assertIs(tensor.context, context_reference())

        tensor.close()
        tensor.close()
        self.assertTrue(tensor.closed)
        with self.assertRaises(RuntimeError):
            tensor.tolist()
        with self.assertRaises(RuntimeError):
            _ = tensor.context

        gc.collect()
        self.assertIsNone(context_reference())
        del tensor

        context = Context("cpu")
        context.close()
        context.close()
        self.assertTrue(context.closed)
        with self.assertRaises(RuntimeError):
            Tensor.zeros(context, (1,))

    def test_tensor_survives_public_context_close(self) -> None:
        context = Context("cpu")
        tensor = Tensor.from_data(
            context,
            (1, 2),
            [4.0, 5.0],
        )
        other = Tensor.from_data(
            context,
            (2, 1),
            [2.0, 3.0],
        )
        context.close()

        self.assertEqual(tensor.backend, "cpu")
        self.assertEqual(tensor.tolist(), [4.0, 5.0])
        with tensor @ other as product:
            self.assertEqual(product.backend, "cpu")
            self.assertEqual(product.tolist(), [23.0])
            self.assertIs(product.context, context)

        other.close()
        tensor.close()

    def test_native_handles_cannot_be_copied(self) -> None:
        with Context("cpu") as context:
            with self.assertRaises(TypeError):
                copy.copy(context)
            with self.assertRaises(TypeError):
                copy.deepcopy(context)

            with Tensor.from_data(context, (2,), [4.0, 5.0]) as tensor:
                with self.assertRaises(TypeError):
                    copy.copy(tensor)
                with self.assertRaises(TypeError):
                    copy.deepcopy(tensor)

                self.assertEqual(context.backend, "cpu")
                self.assertEqual(tensor.tolist(), [4.0, 5.0])

    def test_concurrent_close_waits_for_in_flight_matmul(self) -> None:
        size = 128
        context = Context("cpu")
        left = Tensor.from_data(
            context,
            (size, size),
            [1.0] * (size * size),
        )
        right = Tensor.from_data(
            context,
            (size, size),
            [1.0] * (size * size),
        )
        started = threading.Event()
        products: list[Tensor] = []
        failures: list[BaseException] = []

        def multiply() -> None:
            try:
                # Signal only after entering the same reentrant handle lock
                # used by matmul and close, making the race deterministic.
                with left._lock:
                    started.set()
                    products.append(left @ right)
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=multiply)
        worker.start()
        self.assertTrue(started.wait(timeout=2.0))

        # close() must wait for the native matmul to release the handle lock.
        left.close()
        worker.join(timeout=10.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(len(products), 1)
        self.assertEqual(products[0].shape, (size, size))

        products[0].close()
        right.close()
        context.close()

    def test_metal_parity_or_unavailable_error(self) -> None:
        if not backend_available("metal"):
            with self.assertRaises(TensorLabError) as error:
                Context("metal")
            self.assertEqual(
                error.exception.status,
                STATUS_BACKEND_UNAVAILABLE,
            )
            return

        with Context("cpu") as cpu_context:
            with Context("metal") as metal_context:
                values_left = [1, 2, 3, 4, 5, 6]
                values_right = [7, 8, 9, 10, 11, 12]
                with Tensor.from_data(
                    cpu_context,
                    (2, 3),
                    values_left,
                ) as cpu_left:
                    with Tensor.from_data(
                        cpu_context,
                        (3, 2),
                        values_right,
                    ) as cpu_right:
                        with cpu_left @ cpu_right as cpu_product:
                            expected = cpu_product.tolist()

                    with Tensor.from_data(
                        metal_context,
                        (2, 3),
                        values_left,
                    ) as metal_left:
                        with Tensor.from_data(
                            metal_context,
                            (3, 2),
                            values_right,
                        ) as metal_right:
                            # Contexts choose construction backends; tensor
                            # storage retains the resources needed afterward.
                            metal_context.close()
                            self.assertEqual(metal_left.backend, "metal")
                            self.assertEqual(metal_right.backend, "metal")
                            with metal_left @ metal_right as metal_product:
                                actual = metal_product.tolist()
                                self.assertEqual(
                                    metal_product.backend,
                                    "metal",
                                )
                                self.assertIs(
                                    metal_product.context,
                                    metal_context,
                                )

                        with self.assertRaises(TensorLabError) as mixed_error:
                            cpu_left @ metal_left
                        self.assertEqual(
                            mixed_error.exception.status,
                            STATUS_INVALID_ARGUMENT,
                        )

        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(
                actual_value,
                expected_value,
                places=5,
            )

    def test_end_to_end_bpe_model_training_surface(self) -> None:
        corpus = "abababab"
        with Tokenizer(
            corpus,
            method="bpe",
            vocabulary_size=257,
        ) as tokenizer:
            corpus_tokens = tokenizer.encode(corpus)
            self.assertEqual(tokenizer.decode(corpus_tokens), corpus)
            tokens = [corpus_tokens[0:2]]
            targets = [corpus_tokens[1:3]]
            vocabulary_size = tokenizer.vocab_size

        config = TransformerConfig(
            vocabulary_size=vocabulary_size,
            maximum_context=4,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            random_seed=137,
        )
        model = DecoderOnlyTransformer(config).to("cpu")
        parameters = model.parameters()
        optimizer = Adam(parameters, learning_rate=1.0e-2)
        logits = model(tokens)
        loss = cross_entropy(logits, targets)

        self.assertEqual(model.backend, "cpu")
        self.assertEqual(parameters.backend, "cpu")
        self.assertEqual(len(parameters), 22)
        self.assertEqual(
            parameters.names[0],
            "token_embedding.weight",
        )
        self.assertEqual(logits.backend, "cpu")
        self.assertEqual(logits.shape, (1, 2, vocabulary_size))
        self.assertEqual(loss.shape, ())
        self.assertEqual(loss.numel, 1)
        self.assertTrue(math.isfinite(loss.item()))
        self.assertGreater(loss.item(), 0.0)
        self.assertEqual(optimizer.backend, "cpu")
        self.assertEqual(optimizer.parameter_count, 22)
        self.assertEqual(optimizer.step_count, 0)

        with self.assertRaises(TensorLabError) as transfer_error:
            model.to("cpu")
        self.assertEqual(
            transfer_error.exception.status,
            STATUS_INVALID_ARGUMENT,
        )
        self.assertIn("while", transfer_error.exception.detail)

        loss.backward()
        stats = optimizer.step()
        self.assertEqual(stats.step, 1)
        self.assertTrue(math.isfinite(stats.gradient_norm))
        self.assertGreaterEqual(stats.gradient_norm, 0.0)
        self.assertGreater(stats.clip_scale, 0.0)
        self.assertLessEqual(stats.clip_scale, 1.0)
        self.assertEqual(optimizer.step_count, 1)
        optimizer.zero_grad()

        with self.assertRaises(TensorLabError) as stale_error:
            loss.backward()
        self.assertIn(
            "after backward",
            stale_error.exception.detail,
        )

        loss.close()
        logits.close()
        stale_logits = model(tokens)
        self.assertEqual(optimizer.step().step, 2)
        with self.assertRaises(TensorLabError) as epoch_error:
            cross_entropy(stale_logits, targets)
        self.assertIn(
            "after an optimizer step",
            epoch_error.exception.detail,
        )
        stale_logits.close()
        optimizer.close()
        self.assertIs(model.to("cpu"), model)

        parameters.close()
        model.close()

    def test_full_sequence_attention_selector_and_flash_backward(self) -> None:
        self.assertEqual(FULL_SEQUENCE_ATTENTION_MATERIALIZED, 0)
        self.assertEqual(FULL_SEQUENCE_ATTENTION_FLASH, 1)
        config = TransformerConfig(
            vocabulary_size=5,
            maximum_context=4,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            random_seed=271,
        )

        with self.assertRaisesRegex(ValueError, "materialized.*flash"):
            DecoderOnlyTransformer(config, attention="unknown")
        with self.assertRaises(TypeError):
            DecoderOnlyTransformer(config, attention=True)
        with self.assertRaises(TypeError):
            DecoderOnlyTransformer(config, "flash")

        with (
            DecoderOnlyTransformer(config) as materialized,
            DecoderOnlyTransformer(config, attention="flash") as flash,
        ):
            self.assertEqual(
                materialized.full_sequence_attention,
                "materialized",
            )
            self.assertEqual(flash.full_sequence_attention, "flash")
            self.assertIs(
                materialized.set_full_sequence_attention(
                    FULL_SEQUENCE_ATTENTION_FLASH
                ),
                materialized,
            )
            self.assertEqual(
                materialized.full_sequence_attention,
                "flash",
            )
            self.assertIs(
                materialized.set_full_sequence_attention("materialized"),
                materialized,
            )
            with self.assertRaises(ValueError):
                materialized.set_full_sequence_attention(99)
            self.assertEqual(
                materialized.full_sequence_attention,
                "materialized",
            )

            materialized_logits = materialized([[0, 1, 2]])
            flash_logits = flash([[0, 1, 2]])
            materialized_loss = cross_entropy(
                materialized_logits,
                [[1, 2, 3]],
            )
            flash_loss = cross_entropy(
                flash_logits,
                [[1, 2, 3]],
            )
            try:
                self.assertEqual(
                    materialized_logits.shape,
                    flash_logits.shape,
                )
                for actual, expected in zip(
                    flash_logits.tolist(),
                    materialized_logits.tolist(),
                ):
                    self.assertAlmostEqual(actual, expected, places=5)
                self.assertAlmostEqual(
                    flash_loss.item(),
                    materialized_loss.item(),
                    places=5,
                )
                materialized_loss.backward()
                flash_loss.backward()
            finally:
                flash_loss.close()
                materialized_loss.close()
                flash_logits.close()
                materialized_logits.close()

    def test_activation_checkpointing_selector_and_backward(self) -> None:
        self.assertEqual(ACTIVATION_CHECKPOINTING_DISABLED, 0)
        self.assertEqual(
            ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK,
            1,
        )
        config = TransformerConfig(
            vocabulary_size=5,
            maximum_context=4,
            model_width=4,
            head_count=2,
            block_count=2,
            feed_forward_width=8,
            random_seed=277,
        )

        with self.assertRaisesRegex(ValueError, "disabled.*block"):
            DecoderOnlyTransformer(
                config,
                activation_checkpointing="unknown",
            )
        with self.assertRaises(TypeError):
            DecoderOnlyTransformer(
                config,
                activation_checkpointing=True,
            )

        with (
            DecoderOnlyTransformer(
                config,
                attention="flash",
            ) as regular,
            DecoderOnlyTransformer(
                config,
                attention="flash",
                activation_checkpointing="block",
            ) as checkpointed,
        ):
            self.assertEqual(
                regular.activation_checkpointing,
                "disabled",
            )
            self.assertEqual(
                checkpointed.activation_checkpointing,
                "block",
            )
            self.assertIs(
                regular.set_activation_checkpointing(
                    ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK
                ),
                regular,
            )
            self.assertEqual(
                regular.activation_checkpointing,
                "block",
            )
            self.assertIs(
                regular.set_activation_checkpointing("disabled"),
                regular,
            )
            with self.assertRaises(ValueError):
                regular.set_activation_checkpointing(99)
            self.assertEqual(
                regular.activation_checkpointing,
                "disabled",
            )

            regular_logits = regular([[0, 1, 2]])
            checkpointed_logits = checkpointed([[0, 1, 2]])
            regular_loss = cross_entropy(
                regular_logits,
                [[1, 2, 3]],
            )
            checkpointed_loss = cross_entropy(
                checkpointed_logits,
                [[1, 2, 3]],
            )
            try:
                for actual, expected in zip(
                    checkpointed_logits.tolist(),
                    regular_logits.tolist(),
                ):
                    self.assertAlmostEqual(actual, expected, places=5)
                self.assertAlmostEqual(
                    checkpointed_loss.item(),
                    regular_loss.item(),
                    places=5,
                )
                regular_loss.backward()
                checkpointed.set_full_sequence_attention("materialized")
                checkpointed_loss.backward()
            finally:
                checkpointed_loss.close()
                regular_loss.close()
                checkpointed_logits.close()
                regular_logits.close()

        retained_model = DecoderOnlyTransformer(
            config,
            activation_checkpointing="block",
        )
        retained_logits = retained_model([[0, 1, 2]])
        retained_loss = cross_entropy(
            retained_logits,
            [[1, 2, 3]],
        )
        retained_model.close()
        try:
            retained_loss.backward()
        finally:
            retained_loss.close()
            retained_logits.close()

    def test_parameter_state_metadata_and_transactional_load(self) -> None:
        config = TransformerConfig(
            vocabulary_size=5,
            maximum_context=3,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            random_seed=41,
        )
        with DecoderOnlyTransformer(config).to("cpu") as model:
            with model.parameters() as parameters:
                self.assertEqual(len(parameters.shapes), len(parameters))
                self.assertEqual(
                    parameters.shapes[0],
                    (config.vocabulary_size, config.model_width),
                )
                values = parameters.flat_values()
                self.assertEqual(len(values), parameters.total_numel)
                self.assertEqual(
                    parameters.total_numel,
                    sum(
                        math.prod(shape)
                        for shape in parameters.shapes
                    ),
                )

                replacement = list(values)
                replacement[0] += 0.125
                parameters.load_flat_values(replacement)
                self.assertEqual(
                    parameters.flat_values(),
                    tuple(replacement),
                )

                logits = model([[0, 1]])
                try:
                    with self.assertRaisesRegex(
                        TensorLabError,
                        "variable graphs",
                    ):
                        parameters.load_flat_values(values)
                finally:
                    logits.close()

                with Adam(parameters) as optimizer:
                    with self.assertRaisesRegex(
                        TensorLabError,
                        "optimizers",
                    ):
                        parameters.load_flat_values(values)

                parameters.load_flat_values(values)
                self.assertEqual(parameters.flat_values(), values)
                with self.assertRaisesRegex(ValueError, "exactly"):
                    parameters.load_flat_values(values[:-1])
                with self.assertRaisesRegex(ValueError, "finite"):
                    invalid = list(values)
                    invalid[0] = math.nan
                    parameters.load_flat_values(invalid)

    def test_lora_config_validation_and_public_targets(self) -> None:
        self.assertEqual(
            LORA_TARGET_DEFAULT,
            LORA_TARGET_ATTENTION_QUERY |
            LORA_TARGET_ATTENTION_VALUE,
        )
        self.assertEqual(
            LORA_TARGET_NAMES,
            (
                "attention.query",
                "attention.key",
                "attention.value",
                "attention.output",
                "feed_forward.expand",
                "feed_forward.project",
                "language_model_head",
            ),
        )
        self.assertEqual(
            LoraConfig(
                rank=2,
                alpha=4,
                targets=["attention.query"],
                random_seed=17,
            ),
            LoraConfig(
                rank=2,
                alpha=4.0,
                targets=("attention.query",),
                random_seed=17,
            ),
        )
        self.assertEqual(
            LoraConfig(
                targets=(
                    "attention.value",
                    "attention.query",
                )
            ).targets,
            (
                "attention.query",
                "attention.value",
            ),
        )

        with self.assertRaises(TypeError):
            LoraConfig(rank=True)
        with self.assertRaises(ValueError):
            LoraConfig(rank=0)
        with self.assertRaises(TypeError):
            LoraConfig(alpha=True)
        with self.assertRaises(ValueError):
            LoraConfig(alpha=math.inf)
        with self.assertRaises(ValueError):
            LoraConfig(alpha=1.0e100)
        with self.assertRaises(ValueError):
            LoraConfig(alpha=1.0e100)
        with self.assertRaises(TypeError):
            LoraConfig(targets="attention.query")
        with self.assertRaises(ValueError):
            LoraConfig(targets=("unknown",))
        with self.assertRaises(ValueError):
            LoraConfig(
                targets=("attention.query", "attention.query")
            )
        with self.assertRaises(ValueError):
            LoraConfig(random_seed=-1)

    def test_lora_adapter_training_merge_and_lifetimes(self) -> None:
        config = TransformerConfig(
            vocabulary_size=5,
            maximum_context=3,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            random_seed=419,
        )
        tokens = [[0, 1]]
        targets = [[1, 2]]

        with DecoderOnlyTransformer(config).to("cpu") as model:
            self.assertFalse(model.lora_attached)
            self.assertIsNone(model.lora_config)
            with self.assertRaisesRegex(
                TensorLabError,
                "no attached LoRA",
            ):
                model.adapter_parameters()

            with model.parameters() as base_parameters:
                base_names = base_parameters.names
                base_values = base_parameters.flat_values()
                with model(tokens) as logits:
                    output_before_attachment = logits.tolist()

                lora = LoraConfig(
                    rank=2,
                    alpha=4.0,
                    targets=(
                        "attention.query",
                        "attention.value",
                    ),
                    random_seed=421,
                )
                self.assertIs(model.attach_lora(lora), model)
                self.assertTrue(model.lora_attached)
                self.assertEqual(model.lora_config, lora)
                with model.parameters() as attached_base_parameters:
                    self.assertEqual(
                        attached_base_parameters.names,
                        base_names,
                    )
                    self.assertEqual(
                        attached_base_parameters.flat_values(),
                        base_values,
                    )
                with self.assertRaisesRegex(
                    TensorLabError,
                    "already has",
                ):
                    model.attach_lora(lora)

                with model(tokens) as logits:
                    output_after_attachment = logits.tolist()
                self.assertEqual(
                    len(output_after_attachment),
                    len(output_before_attachment),
                )
                for actual, expected in zip(
                    output_after_attachment,
                    output_before_attachment,
                ):
                    self.assertAlmostEqual(actual, expected, places=6)

                adapters = model.adapter_parameters()
                self.assertEqual(
                    adapters.names,
                    (
                        "blocks.0.attention.query.lora_a.weight",
                        "blocks.0.attention.query.lora_b.weight",
                        "blocks.0.attention.value.lora_a.weight",
                        "blocks.0.attention.value.lora_b.weight",
                    ),
                )
                self.assertEqual(
                    adapters.shapes,
                    (
                        (2, 4),
                        (4, 2),
                        (2, 4),
                        (4, 2),
                    ),
                )
                self.assertEqual(adapters.total_numel, 32)
                adapter_values_before = adapters.flat_values()

                live_logits = model(tokens)
                try:
                    with self.assertRaisesRegex(
                        TensorLabError,
                        "variable graphs",
                    ):
                        model.merge_lora()
                finally:
                    live_logits.close()

                optimizer = Adam(
                    adapters,
                    learning_rate=1.0e-2,
                )
                with model(tokens) as logits:
                    with cross_entropy(logits, targets) as loss:
                        loss.backward()
                        statistics = optimizer.step()
                self.assertEqual(statistics.step, 1)
                self.assertEqual(
                    base_parameters.flat_values(),
                    base_values,
                )
                adapter_values_after = adapters.flat_values()
                self.assertTrue(
                    any(
                        before != after
                        for before, after in zip(
                            adapter_values_before,
                            adapter_values_after,
                        )
                    )
                )

                with self.assertRaisesRegex(
                    TensorLabError,
                    "optimizers",
                ):
                    model.merge_lora()
                optimizer.close()

                with model(tokens) as logits:
                    output_before_merge = logits.tolist()
                with self.assertRaisesRegex(
                    TensorLabError,
                    "adapter parameter lists",
                ):
                    model.merge_lora()
                adapters.close()

                self.assertIs(model.merge_lora(), model)
                self.assertFalse(model.lora_attached)
                self.assertIsNone(model.lora_config)
                with self.assertRaisesRegex(
                    TensorLabError,
                    "no attached LoRA",
                ):
                    model.adapter_parameters()
                with self.assertRaisesRegex(
                    TensorLabError,
                    "no attached LoRA",
                ):
                    model.merge_lora()

                with model(tokens) as logits:
                    output_after_merge = logits.tolist()
                for actual, expected in zip(
                    output_after_merge,
                    output_before_merge,
                ):
                    self.assertAlmostEqual(actual, expected, places=4)
                self.assertEqual(
                    base_parameters.names,
                    base_names,
                )
                self.assertNotEqual(
                    base_parameters.flat_values(),
                    base_values,
                )

    def test_decode_sessions_match_full_forward_and_own_model(
        self,
    ) -> None:
        config = TransformerConfig(
            vocabulary_size=5,
            maximum_context=4,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            random_seed=1009,
        )
        model = DecoderOnlyTransformer(config)

        with self.assertRaises(TypeError):
            DecodeSession()
        with self.assertRaises(ValueError):
            model.decode_session(cache="unknown")
        with self.assertRaises(ValueError):
            model.decode_session(block_size=0)
        with self.assertRaises(TypeError):
            model.decode_session(block_size=True)

        prefix = [1, 2, 3, 4]
        for cache, expected_block_size in (
            ("contiguous", config.maximum_context),
            ("paged", 2),
        ):
            with self.subTest(cache=cache):
                session = model.decode_session(
                    cache=cache,
                    block_size=2,
                )
                self.assertIsInstance(session, DecodeSession)
                self.assertEqual(session.cache, cache)
                self.assertEqual(session.cache_kind, cache)
                self.assertEqual(
                    session.block_size,
                    expected_block_size,
                )
                self.assertEqual(session.capacity, config.maximum_context)
                self.assertEqual(session.size, 0)
                with self.assertRaises(TypeError):
                    copy.copy(session)
                with self.assertRaises(TypeError):
                    copy.deepcopy(session)

                for index, token in enumerate(prefix):
                    with model([prefix[: index + 1]]) as full_logits:
                        expected = full_logits.tolist()[
                            -config.vocabulary_size :
                        ]
                    actual = session.step(token)
                    self.assertEqual(
                        len(actual),
                        config.vocabulary_size,
                    )
                    for actual_value, expected_value in zip(
                        actual,
                        expected,
                    ):
                        self.assertAlmostEqual(
                            actual_value,
                            expected_value,
                            places=5,
                        )
                    self.assertEqual(session.size, index + 1)

                with self.assertRaises(TensorLabError) as full_error:
                    session.step(0)
                self.assertEqual(
                    full_error.exception.status,
                    STATUS_OUT_OF_RANGE,
                )
                self.assertEqual(session.size, session.capacity)
                session.reset()
                self.assertEqual(session.size, 0)
                with self.assertRaises(TensorLabError) as token_error:
                    session.step(config.vocabulary_size)
                self.assertEqual(
                    token_error.exception.status,
                    STATUS_OUT_OF_RANGE,
                )
                self.assertEqual(session.size, 0)
                session.close()
                session.close()
                self.assertTrue(session.closed)
                with self.assertRaises(RuntimeError):
                    session.step(0)

        parameters = model.parameters()
        values = parameters.flat_values()
        guarded = model.decode_session()
        with self.assertRaisesRegex(TensorLabError, "decode sessions"):
            model.to("cpu")
        with self.assertRaisesRegex(TensorLabError, "decode sessions"):
            parameters.load_flat_values(values)
        guarded.close()
        parameters.load_flat_values(values)
        parameters.close()

        model.attach_lora()
        merge_guard = model.decode_session()
        with self.assertRaisesRegex(TensorLabError, "decode sessions"):
            model.merge_lora()
        merge_guard.close()
        model.merge_lora()

        retained = model.decode_session()
        model.close()
        self.assertEqual(
            len(retained.step(0)),
            config.vocabulary_size,
        )
        retained.close()

    def test_paged_decode_session_on_metal_when_available(self) -> None:
        if not backend_available("metal"):
            self.skipTest("Metal is unavailable")

        config = TransformerConfig(
            vocabulary_size=5,
            maximum_context=3,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            random_seed=1019,
        )
        with DecoderOnlyTransformer(config).to("metal") as model:
            with model.decode_session(
                cache="paged",
                block_size=2,
            ) as session:
                self.assertEqual(session.cache, "paged")
                self.assertEqual(session.block_size, 2)
                self.assertEqual(session.size, 0)
                logits = session.step(1)
                self.assertEqual(len(logits), config.vocabulary_size)
                self.assertTrue(all(math.isfinite(x) for x in logits))
                self.assertEqual(session.size, 1)
                session.reset()
                self.assertEqual(session.size, 0)
                self.assertEqual(
                    len(session.step(2)),
                    config.vocabulary_size,
                )

    def test_training_handles_retain_native_model_state(self) -> None:
        config = TransformerConfig(
            vocabulary_size=5,
            maximum_context=3,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            random_seed=211,
        )
        model = DecoderOnlyTransformer(config)
        parameters = model.parameters()
        optimizer = Adam(parameters)
        logits = model([0, 1])
        loss = cross_entropy(logits, [1, 2])

        model.close()
        parameters.close()
        logits.close()

        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        stats = optimizer.step()
        self.assertEqual(stats.step, 1)
        self.assertEqual(optimizer.step_count, 1)

        loss.close()
        optimizer.close()

    def test_token_batch_validation_and_native_errors(self) -> None:
        config = TransformerConfig(
            vocabulary_size=5,
            maximum_context=4,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
        )
        with DecoderOnlyTransformer(config) as model:
            with self.assertRaises(ValueError):
                model([])
            with self.assertRaises(ValueError):
                model([[0, 1], [2]])
            with self.assertRaises(ValueError):
                model([-1])
            with self.assertRaises(OverflowError):
                model([1 << 32])
            with self.assertRaises(TypeError):
                model([1.5])

            with model([[0, 1]]) as logits:
                with self.assertRaises(TensorLabError) as target_error:
                    cross_entropy(logits, [1])
                self.assertEqual(
                    target_error.exception.status,
                    STATUS_INVALID_ARGUMENT,
                )

            parameters = model.parameters()
            with self.assertRaises(TensorLabError) as adam_error:
                Adam(parameters, learning_rate=-1.0)
            self.assertEqual(
                adam_error.exception.status,
                STATUS_INVALID_ARGUMENT,
            )
            # A failed optimizer construction must not leave a native
            # lifetime guard behind.
            self.assertIs(model.to("cpu"), model)
            parameters.close()

    def test_full_model_metal_or_unavailable_error(self) -> None:
        config = TransformerConfig(
            vocabulary_size=5,
            maximum_context=3,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            random_seed=307,
        )
        model = DecoderOnlyTransformer(config)
        if not backend_available("metal"):
            with self.assertRaises(TensorLabError) as error:
                model.to("metal")
            self.assertEqual(
                error.exception.status,
                STATUS_BACKEND_UNAVAILABLE,
            )
            self.assertEqual(model.backend, "cpu")
            model.close()
            return

        self.assertIs(model.to("metal"), model)
        self.assertEqual(model.backend, "metal")
        optimizer = Adam(model.parameters())
        logits = model([[0, 1]])
        loss = cross_entropy(logits, [[1, 2]])
        self.assertEqual(logits.backend, "metal")
        self.assertEqual(loss.backend, "metal")
        self.assertTrue(math.isfinite(loss.item()))
        loss.backward()
        stats = optimizer.step()
        self.assertEqual(stats.step, 1)
        self.assertTrue(math.isfinite(stats.gradient_norm))

        loss.close()
        logits.close()
        optimizer.close()
        model.close()


if __name__ == "__main__":
    unittest.main()
