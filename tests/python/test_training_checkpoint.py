from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import random
import tempfile
import unittest
import zipfile

from riftco_transformer import (
    Adam,
    AdamState,
    DecoderOnlyTransformer,
    LoraConfig,
    RiftcoTransformerError,
    TransformerConfig,
    cross_entropy,
)
from riftco_transformer.checkpoints import (
    MANIFEST_NAME,
    FORMAT_VERSION,
    TrainingCheckpoint,
)
from riftco_transformer.training import (
    BatchSourceState,
    CausalLanguageModelTrainer,
    RandomWindowBatchSource,
    TrainingLoopConfig,
)


TOKENS = tuple(range(7)) * 8


def tiny_config(*, random_seed: int = 901) -> TransformerConfig:
    return TransformerConfig(
        vocabulary_size=7,
        maximum_context=3,
        model_width=4,
        head_count=2,
        block_count=1,
        feed_forward_width=8,
        random_seed=random_seed,
    )


def source(tokens: tuple[int, ...] = TOKENS) -> RandomWindowBatchSource:
    return RandomWindowBatchSource(
        tokens,
        batch_size=2,
        context_size=3,
        random_seed=37,
    )


def run_steps(
    model: DecoderOnlyTransformer,
    optimizer: Adam,
    batches: RandomWindowBatchSource,
    count: int,
) -> None:
    CausalLanguageModelTrainer(model, optimizer).run(
        batches,
        TrainingLoopConfig(
            steps=count,
            evaluation_interval=count,
            loss_average_window=count,
        ),
    )


def rewrite_archive(
    source_path: Path,
    destination: Path,
    replacements: dict[str, bytes],
) -> None:
    with zipfile.ZipFile(source_path, mode="r") as archive:
        entries = tuple(
            (info.filename, archive.read(info))
            for info in archive.infolist()
        )
    with zipfile.ZipFile(
        destination,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        for name, payload in entries:
            archive.writestr(name, replacements.get(name, payload))


class TrainingCheckpointTests(unittest.TestCase):
    def test_dense_v1_archive_remains_loadable(self) -> None:
        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current_path = root / "current.riftckpt"
            legacy_path = root / "legacy.riftckpt"
            with DecoderOnlyTransformer(config) as model:
                with model.parameters() as parameters:
                    with Adam(parameters) as optimizer:
                        TrainingCheckpoint.capture(model, optimizer).save(
                            current_path
                        )
            with zipfile.ZipFile(current_path, mode="r") as archive:
                manifest = json.loads(archive.read(MANIFEST_NAME))
            manifest["format_version"] = 1
            del manifest["model"]["packed_state_file"]
            del manifest["checkpoint_id"]
            canonical = json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            manifest["checkpoint_id"] = hashlib.sha256(canonical).hexdigest()
            legacy_manifest = json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            rewrite_archive(
                current_path,
                legacy_path,
                {MANIFEST_NAME: legacy_manifest},
            )

            loaded = TrainingCheckpoint.load(legacy_path)
            self.assertFalse(loaded.quantized)
            with DecoderOnlyTransformer(config) as model:
                with model.parameters() as parameters:
                    with Adam(parameters) as optimizer:
                        loaded.restore(model, optimizer)

    def test_packed_qlora_payload_round_trip_stays_resident(self) -> None:
        config = tiny_config()
        lora = LoraConfig(rank=2, alpha=4.0, random_seed=71)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packed.riftckpt"
            with DecoderOnlyTransformer(config) as model:
                model.quantize_nf4(
                    block_size=32,
                    double_quantization=True,
                    scale_block_size=32,
                )
                model.attach_lora(lora)
                batches = source()
                with model.adapter_parameters() as parameters:
                    with Adam(parameters, learning_rate=0.01) as optimizer:
                        run_steps(model, optimizer, batches, 2)
                        expected_packed = model.packed_quantized_state
                        expected_memory = model.quantized_memory
                        expected_optimizer = optimizer.state()
                        with model.parameters() as base_parameters:
                            expected_base = base_parameters.flat_values()
                        TrainingCheckpoint.capture(
                            model, optimizer, source=batches
                        ).save(path)

            with zipfile.ZipFile(path, mode="r") as archive:
                manifest = json.loads(archive.read(MANIFEST_NAME))
                packed_name = manifest["model"]["packed_state_file"]
                self.assertEqual(manifest["format_version"], FORMAT_VERSION)
                self.assertTrue(manifest["model"]["quantized"])
                self.assertEqual(
                    manifest["members"][packed_name]["encoding"],
                    "riftco-packed-nf4-v1",
                )
                self.assertEqual(archive.read(packed_name), expected_packed)

            checkpoint = TrainingCheckpoint.load(path)
            self.assertTrue(checkpoint.quantized)
            self.assertEqual(checkpoint.packed_model_state, expected_packed)
            with DecoderOnlyTransformer(
                tiny_config(random_seed=1901)
            ) as restored_model:
                # Deliberately construct a different packed scale encoding;
                # restore must replace it transactionally with archive bytes.
                restored_model.quantize_nf4(
                    block_size=64,
                    double_quantization=False,
                )
                restored_model.attach_lora(
                    LoraConfig(rank=2, alpha=4.0, random_seed=1701)
                )
                restored_source = source()
                with restored_model.adapter_parameters() as parameters:
                    with Adam(parameters, learning_rate=0.01) as optimizer:
                        checkpoint.restore(
                            restored_model, optimizer, restored_source
                        )
                        self.assertTrue(
                            restored_model.quantized_linear_weights
                        )
                        self.assertEqual(
                            restored_model.packed_quantized_state,
                            expected_packed,
                        )
                        self.assertEqual(
                            restored_model.quantized_memory,
                            expected_memory,
                        )
                        self.assertEqual(optimizer.state(), expected_optimizer)
                        with restored_model.parameters() as base_parameters:
                            self.assertEqual(
                                base_parameters.flat_values(), expected_base
                            )

    def test_packed_qlora_exact_resume_matches_uninterrupted(self) -> None:
        config = tiny_config()
        lora = LoraConfig(rank=2, alpha=4.0, random_seed=73)
        total_steps = 4
        split_steps = 2

        with DecoderOnlyTransformer(config) as uninterrupted_model:
            uninterrupted_model.quantize_nf4(32, scale_block_size=32)
            uninterrupted_model.attach_lora(lora)
            uninterrupted_source = source()
            with uninterrupted_model.adapter_parameters() as parameters:
                with Adam(parameters, learning_rate=0.01) as optimizer:
                    run_steps(
                        uninterrupted_model,
                        optimizer,
                        uninterrupted_source,
                        total_steps,
                    )
                    uninterrupted_state = optimizer.state()
                    uninterrupted_packed = (
                        uninterrupted_model.packed_quantized_state
                    )
                    uninterrupted_source_state = (
                        uninterrupted_source.checkpoint_state()
                    )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qlora-resume.riftckpt"
            with DecoderOnlyTransformer(config) as split_model:
                split_model.quantize_nf4(32, scale_block_size=32)
                split_model.attach_lora(lora)
                split_source = source()
                with split_model.adapter_parameters() as parameters:
                    with Adam(parameters, learning_rate=0.01) as optimizer:
                        run_steps(
                            split_model,
                            optimizer,
                            split_source,
                            split_steps,
                        )
                        TrainingCheckpoint.capture(
                            split_model,
                            optimizer,
                            source=split_source,
                        ).save(path)

            checkpoint = TrainingCheckpoint.load(path)
            with DecoderOnlyTransformer(
                tiny_config(random_seed=1903)
            ) as resumed_model:
                resumed_model.quantize_nf4(64, double_quantization=False)
                resumed_model.attach_lora(
                    LoraConfig(rank=2, alpha=4.0, random_seed=1703)
                )
                resumed_source = source()
                with resumed_model.adapter_parameters() as parameters:
                    with Adam(
                        parameters,
                        learning_rate=0.01,
                        state_storage="paged",
                        page_size=7,
                    ) as optimizer:
                        checkpoint.restore(
                            resumed_model, optimizer, resumed_source
                        )
                        self.assertEqual(
                            resumed_model.packed_quantized_state,
                            uninterrupted_packed,
                        )
                        run_steps(
                            resumed_model,
                            optimizer,
                            resumed_source,
                            total_steps - split_steps,
                        )
                        resumed_state = optimizer.state()
                        resumed_packed = resumed_model.packed_quantized_state

        self.assertEqual(resumed_state, uninterrupted_state)
        self.assertEqual(resumed_packed, uninterrupted_packed)
        self.assertEqual(
            resumed_source.checkpoint_state(),
            uninterrupted_source_state,
        )

    def test_packed_state_native_import_is_transactional(self) -> None:
        config = tiny_config()
        with DecoderOnlyTransformer(config) as model:
            model.quantize_nf4(32, scale_block_size=32)
            before = model.packed_quantized_state
            with self.assertRaisesRegex(
                RiftcoTransformerError, "truncated packed model state"
            ):
                model.load_packed_quantized_state(before[:-1])
            self.assertEqual(model.packed_quantized_state, before)

    def test_packed_state_rejects_noncanonical_inactive_negative_zero(
        self,
    ) -> None:
        config = tiny_config()
        with DecoderOnlyTransformer(config) as model:
            model.quantize_nf4(32, double_quantization=False)
            before = model.packed_quantized_state
            mutated = bytearray(before)
            # Header/count (16 bytes), seven uint64 metadata fields (56),
            # then the four-byte inactive scale offset. Flip only its sign.
            mutated[75] ^= 0x80
            with self.assertRaisesRegex(
                RiftcoTransformerError,
                "inconsistent NF4 scale metadata",
            ):
                model.load_packed_quantized_state(mutated)
            self.assertEqual(model.packed_quantized_state, before)

    def test_qlora_restore_rolls_back_frozen_base_after_packed_failure(
        self,
    ) -> None:
        config = tiny_config()
        lora = LoraConfig(rank=2, alpha=4.0, random_seed=79)
        with DecoderOnlyTransformer(config) as source_model:
            source_model.quantize_nf4(32, scale_block_size=32)
            source_model.attach_lora(lora)
            source_batches = source()
            with source_model.adapter_parameters() as parameters:
                with Adam(parameters, learning_rate=0.01) as optimizer:
                    run_steps(source_model, optimizer, source_batches, 1)
                    checkpoint = TrainingCheckpoint.capture(
                        source_model,
                        optimizer,
                        source=source_batches,
                    )

        assert checkpoint.packed_model_state is not None
        checkpoint._packed_model_state = checkpoint.packed_model_state[:-1]
        with DecoderOnlyTransformer(
            tiny_config(random_seed=1911)
        ) as target_model:
            target_model.quantize_nf4(64, double_quantization=False)
            target_model.attach_lora(lora)
            target_batches = source()
            with target_model.adapter_parameters() as parameters:
                with Adam(parameters, learning_rate=0.01) as optimizer:
                    with target_model.parameters() as base_parameters:
                        base_before = base_parameters.flat_values()
                    packed_before = target_model.packed_quantized_state
                    optimizer_before = optimizer.state()
                    source_before = target_batches.checkpoint_state()
                    random.seed(1931)
                    random_before = random.getstate()

                    with self.assertRaisesRegex(
                        RiftcoTransformerError,
                        "truncated packed model state",
                    ):
                        checkpoint.restore(
                            target_model,
                            optimizer,
                            target_batches,
                        )

                    with target_model.parameters() as base_parameters:
                        self.assertEqual(
                            base_parameters.flat_values(), base_before
                        )
                    self.assertEqual(
                        target_model.packed_quantized_state, packed_before
                    )
                    self.assertEqual(optimizer.state(), optimizer_before)
                    self.assertEqual(
                        target_batches.checkpoint_state(), source_before
                    )
                    self.assertEqual(random.getstate(), random_before)

    def test_checkpoint_rejects_epsilon_that_cannot_remain_positive_f32(
        self,
    ) -> None:
        config = tiny_config()
        with DecoderOnlyTransformer(config) as model:
            with model.parameters() as parameters:
                with Adam(parameters) as optimizer:
                    checkpoint = TrainingCheckpoint.capture(model, optimizer)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = checkpoint.save(root / "valid.riftckpt")
            with zipfile.ZipFile(valid_path, mode="r") as archive:
                manifest = json.loads(archive.read(MANIFEST_NAME))

            for description, invalid in (
                ("overflow", 1.0e300),
                ("underflow", 1.0e-50),
            ):
                with self.subTest(description=description):
                    invalid_manifest = dict(manifest)
                    invalid_manifest["model"] = dict(manifest["model"])
                    invalid_manifest["model"]["config"] = dict(
                        manifest["model"]["config"]
                    )
                    invalid_manifest["model"]["config"][
                        "layer_norm_epsilon"
                    ] = invalid
                    invalid_path = root / f"{description}.riftckpt"
                    rewrite_archive(
                        valid_path,
                        invalid_path,
                        {
                            MANIFEST_NAME: json.dumps(
                                invalid_manifest,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8")
                        },
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "finite, strictly positive float32",
                    ):
                        TrainingCheckpoint.load(invalid_path)

                    object.__setattr__(
                        config,
                        "layer_norm_epsilon",
                        invalid,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "finite, strictly positive float32",
                    ):
                        checkpoint.save(
                            root / f"{description}-export.riftckpt"
                        )
                    object.__setattr__(
                        config,
                        "layer_norm_epsilon",
                        1.0e-5,
                    )

    def test_capture_keeps_optimizer_identity_after_parameter_list_close(
        self,
    ) -> None:
        config = tiny_config()
        with DecoderOnlyTransformer(config) as model:
            parameters = model.parameters()
            optimizer = Adam(parameters, learning_rate=0.01)
            expected_names = parameters.names
            expected_shapes = parameters.shapes
            parameters.close()
            with optimizer:
                self.assertEqual(optimizer.parameter_names, expected_names)
                self.assertEqual(optimizer.parameter_shapes, expected_shapes)
                checkpoint = TrainingCheckpoint.capture(model, optimizer)
                self.assertEqual(checkpoint.optimizer_step, 0)

    def test_optimizer_must_belong_to_the_exact_model(self) -> None:
        config = tiny_config()
        with DecoderOnlyTransformer(config) as checkpoint_model:
            with checkpoint_model.parameters() as parameters:
                with Adam(parameters) as checkpoint_optimizer:
                    checkpoint = TrainingCheckpoint.capture(
                        checkpoint_model, checkpoint_optimizer
                    )

        with DecoderOnlyTransformer(config) as target_model:
            with DecoderOnlyTransformer(config) as optimizer_model:
                with optimizer_model.parameters() as parameters:
                    with Adam(parameters) as optimizer:
                        self.assertFalse(
                            optimizer.owns_parameters_of(target_model)
                        )
                        self.assertTrue(
                            optimizer.owns_parameters_of(optimizer_model)
                        )
                        with self.assertRaisesRegex(
                            ValueError, "this exact model"
                        ):
                            CausalLanguageModelTrainer(
                                target_model, optimizer
                            )
                        before = optimizer.state()
                        with self.assertRaisesRegex(
                            ValueError, "this exact model"
                        ):
                            checkpoint.restore(target_model, optimizer)
                        self.assertEqual(optimizer.state(), before)

    def test_execution_policies_are_persisted_and_validated(self) -> None:
        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.riftckpt"
            with DecoderOnlyTransformer(
                config,
                attention="flash",
                activation_checkpointing="block",
            ) as model:
                with model.parameters() as parameters:
                    with Adam(parameters) as optimizer:
                        TrainingCheckpoint.capture(
                            model, optimizer
                        ).save(path)
            checkpoint = TrainingCheckpoint.load(path)

        self.assertEqual(checkpoint.full_sequence_attention, "flash")
        self.assertEqual(checkpoint.activation_checkpointing, "block")
        with DecoderOnlyTransformer(config) as model:
            with model.parameters() as parameters:
                with Adam(parameters) as optimizer:
                    with self.assertRaisesRegex(
                        ValueError, "full-sequence attention"
                    ):
                        checkpoint.validate(model, optimizer)
        with DecoderOnlyTransformer(
            config, attention="flash"
        ) as model:
            with model.parameters() as parameters:
                with Adam(parameters) as optimizer:
                    with self.assertRaisesRegex(
                        ValueError, "activation checkpointing"
                    ):
                        checkpoint.validate(model, optimizer)

    def test_long_running_subnormal_beta_powers_round_trip(self) -> None:
        config = tiny_config()
        step_count = 715_308
        with DecoderOnlyTransformer(config) as model:
            with model.parameters() as parameters:
                with Adam(parameters) as optimizer:
                    initial = optimizer.state()
                    beta1_power = 1.0
                    beta2_power = 1.0
                    for _ in range(step_count):
                        beta1_power *= optimizer.options.beta1
                        beta2_power *= optimizer.options.beta2
                    optimizer.load_state(
                        AdamState(
                            step_count=step_count,
                            beta1_power=beta1_power,
                            beta2_power=beta2_power,
                            parameter_values=initial.parameter_values,
                            first_moments=initial.first_moments,
                            second_moments=initial.second_moments,
                        )
                    )
                    checkpoint = TrainingCheckpoint.capture(model, optimizer)
                    self.assertEqual(checkpoint.optimizer_step, step_count)

    def test_oversized_manifest_is_rejected_before_save(self) -> None:
        config = tiny_config()
        with DecoderOnlyTransformer(config) as model:
            with model.parameters() as parameters:
                with Adam(parameters) as optimizer:
                    with self.assertRaisesRegex(
                        ValueError, "manifest is too large"
                    ):
                        TrainingCheckpoint.capture(
                            model,
                            optimizer,
                            metadata={"padding": "x" * (1 << 20)},
                        )

    def test_exactly_zero_backward_is_still_a_pending_step(self) -> None:
        config = TransformerConfig(
            vocabulary_size=1,
            maximum_context=2,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            random_seed=907,
        )
        with DecoderOnlyTransformer(config) as model:
            with model.parameters() as parameters:
                with Adam(parameters) as optimizer:
                    with model([[0, 0]]) as logits:
                        with cross_entropy(logits, [[0, 0]]) as loss:
                            self.assertEqual(loss.item(), 0.0)
                            loss.backward()
                    with self.assertRaisesRegex(
                        RiftcoTransformerError,
                        "clean post-step boundary",
                    ):
                        optimizer.state()
                    with self.assertRaisesRegex(
                        RiftcoTransformerError,
                        "clean post-step boundary",
                    ):
                        TrainingCheckpoint.capture(model, optimizer)
                    optimizer.zero_gradients()
                    optimizer.state()

    def test_exact_resume_matches_uninterrupted_training(self) -> None:
        config = tiny_config()
        total_steps = 5
        split_steps = 2

        with DecoderOnlyTransformer(config) as uninterrupted_model:
            uninterrupted_source = source()
            with uninterrupted_model.parameters() as parameters:
                with Adam(parameters, learning_rate=0.01) as optimizer:
                    run_steps(
                        uninterrupted_model,
                        optimizer,
                        uninterrupted_source,
                        total_steps,
                    )
                    uninterrupted_state = optimizer.state()
                    uninterrupted_source_state = (
                        uninterrupted_source.checkpoint_state()
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = root / "split.riftckpt"
            second_path = root / "round-trip.riftckpt"
            with DecoderOnlyTransformer(config) as split_model:
                split_source = source()
                with split_model.parameters() as parameters:
                    with Adam(parameters, learning_rate=0.01) as optimizer:
                        run_steps(
                            split_model,
                            optimizer,
                            split_source,
                            split_steps,
                        )
                        random.seed(8128)
                        expected_python_random_state = random.getstate()
                        checkpoint = TrainingCheckpoint.capture(
                            split_model,
                            optimizer,
                            source=split_source,
                            global_step=12,
                            metadata={"purpose": "exact-resume-test"},
                        )
                        checkpoint.save(first_path)

            loaded = TrainingCheckpoint.load(first_path)
            loaded.save(second_path)
            self.assertEqual(
                first_path.read_bytes(), second_path.read_bytes()
            )
            self.assertEqual(loaded.global_step, 12)
            self.assertEqual(loaded.optimizer_step, split_steps)
            self.assertEqual(
                loaded.metadata, {"purpose": "exact-resume-test"}
            )

            random.seed(99)
            with DecoderOnlyTransformer(config) as resumed_model:
                resumed_source = source()
                with resumed_model.parameters() as parameters:
                    with Adam(
                        parameters,
                        learning_rate=0.01,
                        state_storage="paged",
                        page_size=7,
                    ) as optimizer:
                        restored = loaded.restore(
                            resumed_model,
                            optimizer,
                            resumed_source,
                        )
                        self.assertEqual(restored.global_step, 12)
                        self.assertEqual(
                            random.getstate(), expected_python_random_state
                        )
                        run_steps(
                            resumed_model,
                            optimizer,
                            resumed_source,
                            total_steps - split_steps,
                        )
                        resumed_state = optimizer.state()

        self.assertEqual(
            resumed_state.parameter_values,
            uninterrupted_state.parameter_values,
        )
        self.assertEqual(
            resumed_state.first_moments,
            uninterrupted_state.first_moments,
        )
        self.assertEqual(
            resumed_state.second_moments,
            uninterrupted_state.second_moments,
        )
        self.assertEqual(
            resumed_state.beta1_power,
            uninterrupted_state.beta1_power,
        )
        self.assertEqual(
            resumed_state.beta2_power,
            uninterrupted_state.beta2_power,
        )
        self.assertEqual(resumed_state.step_count, total_steps)
        self.assertEqual(
            resumed_source.checkpoint_state(),
            uninterrupted_source_state,
        )

    def test_validation_and_corruption_fail_before_mutation(self) -> None:
        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "valid.riftckpt"
            with DecoderOnlyTransformer(config) as trained_model:
                batches = source()
                with trained_model.parameters() as parameters:
                    with Adam(parameters, learning_rate=0.01) as optimizer:
                        run_steps(trained_model, optimizer, batches, 1)
                        TrainingCheckpoint.capture(
                            trained_model,
                            optimizer,
                            source=batches,
                        ).save(valid_path)

            with zipfile.ZipFile(valid_path, mode="r") as archive:
                manifest = archive.read(MANIFEST_NAME)
                manifest_value = json.loads(manifest)
                first_name = manifest_value["optimizer"][
                    "first_moments_file"
                ]
                first_moments = bytearray(archive.read(first_name))
            first_moments[-1] ^= 0x01
            corrupt_path = root / "corrupt.riftckpt"
            rewrite_archive(
                valid_path,
                corrupt_path,
                {first_name: bytes(first_moments)},
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                TrainingCheckpoint.load(corrupt_path)

            duplicate_manifest = (
                b'{"format":"duplicate",' + manifest[1:]
            )
            duplicate_path = root / "duplicate.riftckpt"
            rewrite_archive(
                valid_path,
                duplicate_path,
                {MANIFEST_NAME: duplicate_manifest},
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON"):
                TrainingCheckpoint.load(duplicate_path)

            loaded = TrainingCheckpoint.load(valid_path)
            wrong_tokens = list(TOKENS)
            wrong_tokens[-1] = 6 if wrong_tokens[-1] != 6 else 5
            mismatched_source = source(tuple(wrong_tokens))
            with DecoderOnlyTransformer(config) as target_model:
                with target_model.parameters() as parameters:
                    with Adam(
                        parameters, learning_rate=0.01
                    ) as optimizer:
                        before = optimizer.state()
                        random.seed(144)
                        random_before = random.getstate()
                        with self.assertRaisesRegex(
                            ValueError, "fingerprint"
                        ):
                            loaded.restore(
                                target_model,
                                optimizer,
                                mismatched_source,
                            )
                        self.assertEqual(optimizer.state(), before)
                        self.assertEqual(random.getstate(), random_before)

    def test_capture_requires_clean_boundary_and_explicit_source_policy(
        self,
    ) -> None:
        config = tiny_config()
        with DecoderOnlyTransformer(config) as model:
            with model.parameters() as parameters:
                with Adam(parameters) as optimizer:
                    logits = model([[0, 1, 2]])
                    loss = cross_entropy(logits, [[1, 2, 3]])
                    loss.backward()
                    loss.close()
                    logits.close()
                    with self.assertRaisesRegex(
                        RiftcoTransformerError,
                        "clean post-step boundary",
                    ):
                        TrainingCheckpoint.capture(model, optimizer)
                    optimizer.zero_gradients()

                    class StatelessSource:
                        def next_batch(self) -> object:
                            raise AssertionError

                    with self.assertRaisesRegex(
                        TypeError, "pass source=None"
                    ):
                        TrainingCheckpoint.capture(
                            model,
                            optimizer,
                            source=StatelessSource(),
                        )

    def test_lora_adapter_state_round_trip(self) -> None:
        config = tiny_config()
        lora = LoraConfig(rank=2, alpha=4.0, random_seed=71)
        with DecoderOnlyTransformer(config) as trained_model:
            trained_model.attach_lora(lora)
            batches = source()
            with trained_model.adapter_parameters() as parameters:
                with Adam(parameters, learning_rate=0.01) as optimizer:
                    run_steps(trained_model, optimizer, batches, 2)
                    checkpoint = TrainingCheckpoint.capture(
                        trained_model,
                        optimizer,
                        source=batches,
                    )
                    expected = optimizer.state()
                    with trained_model.parameters() as base_parameters:
                        expected_base = base_parameters.flat_values()

        with DecoderOnlyTransformer(
            tiny_config(random_seed=1907)
        ) as restored_model:
            restored_model.attach_lora(
                LoraConfig(rank=2, alpha=4.0, random_seed=1707)
            )
            restored_source = source()
            with restored_model.adapter_parameters() as parameters:
                with Adam(parameters, learning_rate=0.01) as optimizer:
                    checkpoint.restore(
                        restored_model, optimizer, restored_source
                    )
                    actual = optimizer.state()
                    with restored_model.parameters() as base_parameters:
                        actual_base = base_parameters.flat_values()
        self.assertEqual(actual, expected)
        self.assertEqual(actual_base, expected_base)
        self.assertTrue(
            all(math.isfinite(value) for value in actual.parameter_values)
        )

    def test_source_callback_failure_rolls_back_every_live_component(
        self,
    ) -> None:
        random_state = random.Random(19).getstate()
        checkpoint_source_state = BatchSourceState(
            source_type="test_source",
            fingerprint="a" * 64,
            batches_emitted=7,
            random_state=random_state,
        )
        previous_source_state = BatchSourceState(
            source_type="test_source",
            fingerprint="a" * 64,
            batches_emitted=2,
            random_state=random.Random(23).getstate(),
        )

        class Source:
            def __init__(
                self,
                state: BatchSourceState,
                *,
                fail_target_once: bool = False,
                fail_every_restore: bool = False,
            ) -> None:
                self.state = state
                self.fail_target_once = fail_target_once
                self.fail_every_restore = fail_every_restore
                self.target_failed = False

            def checkpoint_state(self) -> BatchSourceState:
                return self.state

            def restore_checkpoint_state(
                self, state: BatchSourceState
            ) -> None:
                self.state = state
                should_fail_target = (
                    self.fail_target_once
                    and not self.target_failed
                    and state.batches_emitted
                    == checkpoint_source_state.batches_emitted
                )
                if should_fail_target:
                    self.target_failed = True
                if self.fail_every_restore or should_fail_target:
                    raise RuntimeError("source restore failed")

        config = tiny_config()
        with DecoderOnlyTransformer(config) as trained_model:
            with trained_model.parameters() as parameters:
                with Adam(parameters, learning_rate=0.01) as optimizer:
                    run_steps(trained_model, optimizer, source(), 1)
                    checkpoint = TrainingCheckpoint.capture(
                        trained_model,
                        optimizer,
                        source=Source(checkpoint_source_state),
                    )

        with DecoderOnlyTransformer(config) as target_model:
            with target_model.parameters() as parameters:
                with Adam(parameters, learning_rate=0.01) as optimizer:
                    target_source = Source(
                        previous_source_state,
                        fail_target_once=True,
                    )
                    optimizer_before = optimizer.state()
                    random.seed(151)
                    random_before = random.getstate()
                    with self.assertRaisesRegex(
                        RuntimeError, "source restore failed"
                    ):
                        checkpoint.restore(
                            target_model, optimizer, target_source
                        )
                    self.assertEqual(optimizer.state(), optimizer_before)
                    self.assertEqual(
                        target_source.checkpoint_state(),
                        previous_source_state,
                    )
                    self.assertEqual(random.getstate(), random_before)

                    irrecoverable_source = Source(
                        previous_source_state,
                        fail_every_restore=True,
                    )
                    with self.assertRaisesRegex(
                        RuntimeError, "state may be indeterminate"
                    ):
                        checkpoint.restore(
                            target_model,
                            optimizer,
                            irrecoverable_source,
                        )
                    self.assertEqual(optimizer.state(), optimizer_before)
                    self.assertEqual(random.getstate(), random_before)


if __name__ == "__main__":
    unittest.main()
