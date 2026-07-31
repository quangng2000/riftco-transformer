from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

from transformer_lab import (
    DecoderOnlyTransformer,
    LoraConfig,
    Tokenizer,
    TransformerConfig,
)
from transformer_lab.artifacts import (
    MANIFEST_NAME,
    WEIGHTS_NAME,
    ModelBundle,
)
from transformer_lab.post_training import (
    FULL_SEQUENCE_OBJECTIVE,
    InstructionExample,
    PlainChatFormatter,
    PostTrainingConfig,
    post_train,
)
from transformer_lab.pretraining import (
    PretrainingConfig,
    pretrain_files,
    pretrain_splits,
    pretrain_text,
)
from transformer_lab.serving import (
    ServingConfig,
    create_http_server,
)
from transformer_lab.training import (
    ExampleWindowBatchSource,
    RandomWindowBatchSource,
    SequenceWindowBatchSource,
    TrainingBatch,
    fixed_batches,
)


CORPUS = "abcdabcdabcdabcd"


def make_tiny_bundle(
    *,
    stage: str = "pretraining",
    parent_artifact_id: str | None = None,
) -> ModelBundle:
    with Tokenizer(CORPUS, method="byte") as tokenizer:
        config = TransformerConfig(
            vocabulary_size=tokenizer.vocab_size,
            maximum_context=4,
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            random_seed=101,
        )
        with DecoderOnlyTransformer(config).to("cpu") as model:
            return ModelBundle.capture(
                model,
                tokenizer,
                stage=stage,
                parent_artifact_id=parent_artifact_id,
                metadata={
                    "purpose": "stage-stack-test",
                    "nested": {"deterministic": True},
                },
            )


def write_bundle_parts(
    path: Path,
    manifest: bytes,
    weights: bytes,
) -> None:
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_STORED,
    ) as archive:
        archive.writestr(MANIFEST_NAME, manifest)
        archive.writestr(WEIGHTS_NAME, weights)


def parameter_values(
    bundle: ModelBundle,
) -> dict[str, tuple[float, ...]]:
    result: dict[str, tuple[float, ...]] = {}
    offset = 0
    for parameter in bundle.parameters:
        next_offset = offset + parameter.value_count
        result[parameter.name] = bundle.weights[offset:next_offset]
        offset = next_offset
    return result


class CompactInstructionFormatter:
    def format(self, example: InstructionExample) -> str:
        return (
            example.prompt
            + example.response
            + example.prompt
            + example.response
        )


class ArtifactTests(unittest.TestCase):
    def test_capture_rejects_active_lora_and_accepts_merged_model(
        self,
    ) -> None:
        with Tokenizer(CORPUS, method="byte") as tokenizer:
            config = TransformerConfig(
                vocabulary_size=tokenizer.vocab_size,
                maximum_context=4,
                model_width=4,
                head_count=2,
                block_count=1,
                feed_forward_width=8,
                random_seed=101,
            )
            with DecoderOnlyTransformer(config).to("cpu") as model:
                model.attach_lora(
                    LoraConfig(rank=2, alpha=4.0, random_seed=103)
                )
                self.assertTrue(model.lora_attached)
                with self.assertRaisesRegex(RuntimeError, "active LoRA"):
                    ModelBundle.capture(
                        model,
                        tokenizer,
                        stage="post_training",
                    )

                model.merge_lora()
                self.assertFalse(model.lora_attached)
                bundle = ModelBundle.capture(
                    model,
                    tokenizer,
                    stage="post_training",
                )

        self.assertEqual(bundle.stage, "post_training")

    def test_artifact_handoff_to_fresh_process_generates_json(self) -> None:
        bundle = make_tiny_bundle()
        environment = os.environ.copy()
        self.assertIn("TRANSFORMER_LAB_LIBRARY", environment)
        self.assertIn("PYTHONPATH", environment)
        child_program = """
import json
from pathlib import Path
import sys

from transformer_lab.artifacts import ModelBundle
from transformer_lab.serving import ModelService, ServingConfig

bundle = ModelBundle.load(Path(sys.argv[1]))
with ModelService(
    bundle,
    ServingConfig(backend="cpu", maximum_new_tokens=1),
) as service:
    generated = service.generate(
        "ab",
        max_new_tokens=1,
        temperature=0.0,
    )
    print(json.dumps(
        {
            "artifact_id": service.artifact_id,
            "stage": service.stage,
            "backend": service.backend,
            "prompt_token_ids": list(generated.prompt_token_ids),
            "generated_token_ids": list(generated.generated_token_ids),
            "token_ids": list(generated.token_ids),
            "text": generated.text,
        },
        sort_keys=True,
        separators=(",", ":"),
    ))
"""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = bundle.save(root / "handoff.tlab")
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    child_program,
                    str(artifact_path),
                ],
                cwd=root,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=15,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        output = json.loads(completed.stdout)
        self.assertEqual(output["artifact_id"], bundle.artifact_id)
        self.assertEqual(output["stage"], "pretraining")
        self.assertEqual(output["backend"], "cpu")
        self.assertEqual(output["prompt_token_ids"], [0, 1])
        self.assertEqual(len(output["generated_token_ids"]), 1)
        self.assertEqual(len(output["token_ids"]), 3)
        self.assertIsInstance(output["text"], str)

    def test_bpe_tokenizer_state_survives_round_trip(self) -> None:
        corpus = "abababab"
        with Tokenizer(
            corpus,
            method="bpe",
            vocabulary_size=257,
            minimum_pair_frequency=2,
        ) as tokenizer:
            encoded = tokenizer.encode(corpus)
            config = TransformerConfig(
                vocabulary_size=tokenizer.vocab_size,
                maximum_context=4,
                model_width=4,
                head_count=2,
                block_count=1,
                feed_forward_width=8,
                random_seed=103,
            )
            with DecoderOnlyTransformer(config).to("cpu") as model:
                bundle = ModelBundle.capture(
                    model,
                    tokenizer,
                    stage="pretraining",
                )

        self.assertEqual(bundle.tokenizer.method, "bpe")
        self.assertEqual(len(bundle.tokenizer.merge_rules), 1)
        with tempfile.TemporaryDirectory() as directory:
            artifact_path = bundle.save(Path(directory) / "bpe.tlab")
            loaded = ModelBundle.load(artifact_path)
        with loaded.instantiate("cpu") as runtime:
            self.assertEqual(runtime.tokenizer.method, "bpe")
            self.assertEqual(
                runtime.tokenizer.merge_rules,
                bundle.tokenizer.merge_rules,
            )
            self.assertEqual(runtime.tokenizer.encode(corpus), encoded)
            self.assertEqual(runtime.tokenizer.decode(encoded), corpus)

    def test_round_trip_is_deterministic_and_restores_runtime(self) -> None:
        bundle = make_tiny_bundle()

        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.tlab"
            second_path = Path(directory) / "second.tlab"
            bundle.save(first_path)
            loaded = ModelBundle.load(first_path)
            loaded.save(second_path)

            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())
            self.assertEqual(loaded.artifact_id, bundle.artifact_id)
            self.assertEqual(loaded.config, bundle.config)
            self.assertEqual(loaded.tokenizer, bundle.tokenizer)
            self.assertEqual(loaded.parameters, bundle.parameters)
            self.assertEqual(loaded.weights, bundle.weights)
            self.assertEqual(loaded.stage, "pretraining")
            self.assertIsNone(loaded.parent_artifact_id)
            self.assertEqual(loaded.metadata, bundle.metadata)

        with loaded.instantiate("cpu") as runtime:
            self.assertEqual(runtime.model.backend, "cpu")
            self.assertEqual(runtime.tokenizer.encode("ab"), [0, 1])
            self.assertEqual(runtime.tokenizer.decode([0, 1]), "ab")
            with runtime.model.parameters() as parameters:
                self.assertEqual(parameters.names, tuple(
                    parameter.name for parameter in loaded.parameters
                ))
                self.assertEqual(parameters.shapes, tuple(
                    parameter.shape for parameter in loaded.parameters
                ))
                self.assertEqual(parameters.flat_values(), loaded.weights)
            with runtime.model([[0, 1]]) as logits:
                self.assertEqual(
                    logits.shape,
                    (1, 2, loaded.config.vocabulary_size),
                )
                self.assertTrue(
                    all(math.isfinite(value) for value in logits.tolist())
                )

    def test_corrupted_weights_and_manifest_are_rejected(self) -> None:
        bundle = make_tiny_bundle()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = bundle.save(root / "valid.tlab")
            with zipfile.ZipFile(valid_path, mode="r") as archive:
                manifest = archive.read(MANIFEST_NAME)
                weights = archive.read(WEIGHTS_NAME)

            corrupted_weights = bytearray(weights)
            corrupted_weights[-1] ^= 0x01
            bad_weights_path = root / "bad-weights.tlab"
            write_bundle_parts(
                bad_weights_path,
                manifest,
                bytes(corrupted_weights),
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                ModelBundle.load(bad_weights_path)

            manifest_value = json.loads(manifest)
            manifest_value["artifact_id"] = "0" * 64
            bad_manifest_path = root / "bad-manifest.tlab"
            write_bundle_parts(
                bad_manifest_path,
                json.dumps(
                    manifest_value,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8"),
                weights,
            )
            with self.assertRaisesRegex(ValueError, "artifact ID"):
                ModelBundle.load(bad_manifest_path)

            invalid_zip_path = root / "not-a-bundle.tlab"
            invalid_zip_path.write_bytes(b"not a zip artifact")
            with self.assertRaisesRegex(ValueError, "valid ZIP"):
                ModelBundle.load(invalid_zip_path)


class TrainingPrimitiveTests(unittest.TestCase):
    def test_batch_sources_are_deterministic_and_preserve_boundaries(
        self,
    ) -> None:
        first = RandomWindowBatchSource(
            tuple(range(10)),
            batch_size=3,
            context_size=3,
            random_seed=17,
        )
        second = RandomWindowBatchSource(
            tuple(range(10)),
            batch_size=3,
            context_size=3,
            random_seed=17,
        )
        first_batches = fixed_batches(first, 3)
        second_batches = fixed_batches(second, 3)
        self.assertEqual(first_batches, second_batches)
        for batch in first_batches:
            self.assertEqual(batch.batch_size, 3)
            self.assertEqual(batch.context_size, 3)
            for inputs, targets in zip(batch.inputs, batch.targets):
                self.assertEqual(inputs[1:], targets[:-1])
                self.assertEqual(targets[-1], inputs[-1] + 1)

        bounded = SequenceWindowBatchSource(
            (
                (0, 1, 2, 3),
                (10, 11, 12, 13),
            ),
            batch_size=16,
            context_size=2,
            random_seed=23,
        ).next_batch()
        for inputs, targets in zip(bounded.inputs, bounded.targets):
            all_values = inputs + targets
            self.assertTrue(
                all(value < 10 for value in all_values)
                or all(value >= 10 for value in all_values)
            )
            self.assertEqual(inputs[1:], targets[:-1])

        example_uniform = ExampleWindowBatchSource(
            (
                (0, 1, 2, 3),
                tuple(range(100, 112)),
            ),
            batch_size=1_000,
            context_size=2,
            random_seed=31,
        ).next_batch()
        short_example_count = sum(
            row[0] < 100 for row in example_uniform.inputs
        )
        self.assertGreater(short_example_count, 400)
        self.assertLess(short_example_count, 600)

    def test_training_batch_rejects_non_rectangular_data(self) -> None:
        with self.assertRaisesRegex(ValueError, "rectangular"):
            TrainingBatch(
                inputs=((0, 1), (2,)),
                targets=((1, 2), (3, 4)),
            )
        with self.assertRaisesRegex(ValueError, "same row count"):
            TrainingBatch(
                inputs=((0, 1),),
                targets=((1, 2), (2, 3)),
            )


class PipelineStageTests(unittest.TestCase):
    def test_plain_chat_training_and_inference_prefix_match(self) -> None:
        formatter = PlainChatFormatter()
        prefix = formatter.format_prompt("  Explain attention.  ")
        self.assertEqual(
            prefix,
            "### User:\nExplain attention.\n### Assistant:\n",
        )
        self.assertEqual(
            formatter.format(
                InstructionExample(
                    prompt="  Explain attention.  ",
                    response="  It mixes relevant values.  ",
                )
            ),
            prefix + "It mixes relevant values.\n",
        )

    def test_explicit_pretraining_splits_preserve_validation_boundary(
        self,
    ) -> None:
        config = PretrainingConfig(
            steps=1,
            context_size=2,
            batch_size=1,
            validation_batch_count=1,
            evaluation_interval=1,
            loss_average_window=1,
            tokenizer_method="byte",
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            learning_rate=1.0e-2,
            random_seed=7,
            validation_random_seed=17,
            backend="cpu",
        )
        direct = pretrain_splits("abcdabcd", "dcba", config)
        self.assertEqual(direct.training_token_count, 8)
        self.assertEqual(direct.validation_token_count, 4)
        self.assertEqual(
            direct.bundle.metadata["validation_source"],
            "explicit",
        )
        self.assertIsNone(
            direct.bundle.metadata["applied_validation_fraction"]
        )
        self.assertEqual(
            direct.bundle.metadata["training_text_sha256"],
            hashlib.sha256(b"abcdabcd").hexdigest(),
        )
        self.assertEqual(
            direct.bundle.metadata["validation_text_sha256"],
            hashlib.sha256(b"dcba").hexdigest(),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "train.txt"
            validation_path = root / "validation.txt"
            training_path.write_text("abcdabcd", encoding="utf-8")
            validation_path.write_text("dcba", encoding="utf-8")
            from_files = pretrain_files(
                training_path,
                validation_path,
                config,
            )

        self.assertEqual(
            from_files.bundle.artifact_id,
            direct.bundle.artifact_id,
        )

    def test_explicit_pretraining_rejects_validation_leakage(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct held-out corpora"):
            pretrain_splits("abcd", "abcd")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path = root / "train.txt"
            copied_path = root / "copied-validation.txt"
            training_path.write_text("abcd", encoding="utf-8")
            copied_path.write_text("abcd", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "distinct held-out files",
            ):
                pretrain_files(training_path, training_path)
            with self.assertRaisesRegex(
                ValueError,
                "distinct held-out corpora",
            ):
                pretrain_files(training_path, copied_path)

    def test_post_training_config_validates_fine_tuning_method(self) -> None:
        self.assertEqual(
            PostTrainingConfig().fine_tuning_method,
            "full",
        )
        with self.assertRaisesRegex(ValueError, "fine_tuning_method"):
            PostTrainingConfig(fine_tuning_method="unknown")
        with self.assertRaisesRegex(TypeError, "LoraConfig"):
            PostTrainingConfig(lora=object())  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "sampling_strategy"):
            PostTrainingConfig(sampling_strategy="unknown")

    def test_training_configs_validate_execution_policies(self) -> None:
        self.assertEqual(PretrainingConfig().attention, "materialized")
        self.assertEqual(PostTrainingConfig().attention, "materialized")
        self.assertEqual(
            PretrainingConfig().activation_checkpointing,
            "disabled",
        )
        self.assertEqual(
            PostTrainingConfig().activation_checkpointing,
            "disabled",
        )
        legacy_lora = LoraConfig(rank=2, alpha=4.0)
        legacy_positional = PostTrainingConfig(
            1,
            2,
            1,
            1,
            1,
            1.0e-3,
            29,
            "cpu",
            "lora",
            "window_uniform",
            legacy_lora,
        )
        self.assertEqual(legacy_positional.fine_tuning_method, "lora")
        self.assertEqual(
            legacy_positional.sampling_strategy,
            "window_uniform",
        )
        self.assertIs(legacy_positional.lora, legacy_lora)
        self.assertEqual(legacy_positional.attention, "materialized")
        self.assertEqual(
            legacy_positional.activation_checkpointing,
            "disabled",
        )
        with self.assertRaisesRegex(ValueError, "attention"):
            PretrainingConfig(attention="unknown")
        with self.assertRaisesRegex(ValueError, "attention"):
            PostTrainingConfig(attention="unknown")
        with self.assertRaisesRegex(
            ValueError,
            "activation_checkpointing",
        ):
            PretrainingConfig(activation_checkpointing="unknown")
        with self.assertRaisesRegex(
            ValueError,
            "activation_checkpointing",
        ):
            PostTrainingConfig(activation_checkpointing="unknown")

    def test_pretraining_to_post_training_records_lineage(self) -> None:
        published_metrics = []
        pretraining_config = PretrainingConfig(
            steps=1,
            context_size=2,
            batch_size=1,
            validation_fraction=0.25,
            validation_batch_count=1,
            evaluation_interval=1,
            loss_average_window=1,
            tokenizer_method="byte",
            model_width=4,
            head_count=2,
            block_count=1,
            feed_forward_width=8,
            learning_rate=1.0e-2,
            random_seed=7,
            validation_random_seed=17,
            backend="cpu",
            attention="flash",
            activation_checkpointing="block",
        )
        self.assertEqual(pretraining_config.attention, "flash")
        self.assertEqual(
            pretraining_config.activation_checkpointing,
            "block",
        )
        pretraining = pretrain_text(
            CORPUS,
            pretraining_config,
            metric_sink=published_metrics.append,
        )

        self.assertEqual(pretraining.bundle.stage, "pretraining")
        self.assertIsNone(pretraining.bundle.parent_artifact_id)
        self.assertEqual(pretraining.training_token_count, 12)
        self.assertEqual(pretraining.validation_token_count, 4)
        self.assertEqual(tuple(published_metrics), pretraining.metrics)
        self.assertEqual(
            [(metric.phase, metric.step) for metric in pretraining.metrics],
            [("validation", 0), ("train", 1), ("validation", 1)],
        )
        self.assertEqual(
            pretraining.bundle.metadata["objective"],
            "next_token_prediction",
        )
        self.assertEqual(
            pretraining.bundle.metadata["training_config"][
                "activation_checkpointing"
            ],
            "block",
        )
        self.assertEqual(
            pretraining.bundle.metadata["applied_validation_fraction"],
            0.25,
        )
        base_weights = pretraining.bundle.weights

        post_training_config = PostTrainingConfig(
            steps=1,
            context_size=2,
            batch_size=1,
            evaluation_interval=1,
            loss_average_window=1,
            learning_rate=1.0e-3,
            random_seed=29,
            backend="cpu",
            attention="flash",
            activation_checkpointing="block",
        )
        self.assertEqual(post_training_config.attention, "flash")
        self.assertEqual(
            post_training_config.activation_checkpointing,
            "block",
        )
        post_training = post_train(
            pretraining.bundle,
            (InstructionExample(prompt="ab", response="cd"),),
            post_training_config,
            formatter=CompactInstructionFormatter(),
        )

        self.assertEqual(post_training.bundle.stage, "post_training")
        self.assertEqual(
            post_training.bundle.parent_artifact_id,
            pretraining.bundle.artifact_id,
        )
        self.assertNotEqual(
            post_training.bundle.artifact_id,
            pretraining.bundle.artifact_id,
        )
        self.assertEqual(pretraining.bundle.weights, base_weights)
        self.assertEqual(post_training.example_count, 1)
        self.assertEqual(post_training.fine_tuning_method, "full")
        self.assertEqual(post_training.objective, FULL_SEQUENCE_OBJECTIVE)
        self.assertEqual(
            [(metric.phase, metric.step) for metric in post_training.metrics],
            [("train", 1)],
        )
        self.assertEqual(
            post_training.bundle.metadata["objective"],
            FULL_SEQUENCE_OBJECTIVE,
        )
        self.assertEqual(
            post_training.bundle.metadata["training_config"][
                "activation_checkpointing"
            ],
            "block",
        )
        self.assertFalse(
            post_training.bundle.metadata["response_only_loss"]
        )
        self.assertEqual(
            post_training.bundle.metadata["fine_tuning_method"],
            "full",
        )
        self.assertEqual(
            post_training.bundle.metadata["optimizer_parameter_scope"],
            "all_model_parameters",
        )
        self.assertFalse(post_training.bundle.metadata["lora_merged"])

        with tempfile.TemporaryDirectory() as directory:
            artifact_path = post_training.bundle.save(
                Path(directory) / "assistant.tlab"
            )
            reloaded = ModelBundle.load(artifact_path)
        self.assertEqual(
            reloaded.parent_artifact_id,
            pretraining.bundle.artifact_id,
        )
        self.assertEqual(reloaded.artifact_id, post_training.bundle.artifact_id)

    def test_lora_post_training_merges_a_serving_ready_child_bundle(
        self,
    ) -> None:
        base = make_tiny_bundle()
        base_weights = base.weights
        base_values = parameter_values(base)
        result = post_train(
            base,
            (InstructionExample(prompt="ab", response="cd"),),
            PostTrainingConfig(
                steps=2,
                context_size=2,
                batch_size=1,
                evaluation_interval=2,
                loss_average_window=1,
                learning_rate=1.0e-2,
                random_seed=31,
                backend="cpu",
                fine_tuning_method="lora",
                activation_checkpointing="block",
                lora=LoraConfig(
                    rank=2,
                    alpha=4.0,
                    random_seed=37,
                ),
            ),
            formatter=CompactInstructionFormatter(),
        )

        self.assertEqual(result.fine_tuning_method, "lora")
        self.assertEqual(result.bundle.stage, "post_training")
        self.assertEqual(
            result.bundle.parent_artifact_id,
            base.artifact_id,
        )
        self.assertEqual(result.bundle.parameters, base.parameters)
        self.assertEqual(base.weights, base_weights)
        self.assertEqual(
            result.bundle.metadata["fine_tuning_method"],
            "lora",
        )
        self.assertEqual(
            result.bundle.metadata["optimizer_parameter_scope"],
            "lora_adapters",
        )
        self.assertTrue(result.bundle.metadata["lora_merged"])
        self.assertEqual(
            result.bundle.metadata["training_config"][
                "activation_checkpointing"
            ],
            "block",
        )
        self.assertFalse(result.bundle.metadata["adapter_state_included"])

        merged_values = parameter_values(result.bundle)
        selected_names = {
            "blocks.0.attention.query.weight",
            "blocks.0.attention.value.weight",
        }
        for name, values in base_values.items():
            if name not in selected_names:
                self.assertEqual(
                    merged_values[name],
                    values,
                    f"LoRA unexpectedly changed {name}",
                )
        self.assertTrue(
            any(
                merged_values[name] != base_values[name]
                for name in selected_names
            ),
            "LoRA should change at least one selected projection",
        )

        with result.bundle.instantiate("cpu") as runtime:
            self.assertFalse(runtime.model.lora_attached)
            with runtime.model([[0, 1]]) as logits:
                self.assertEqual(
                    logits.shape,
                    (1, 2, result.bundle.config.vocabulary_size),
                )

    def test_post_training_reports_too_short_formatted_examples(self) -> None:
        class TooShortFormatter:
            def format(self, _example: InstructionExample) -> str:
                return "ab"

        with self.assertRaisesRegex(ValueError, "too-short example indices"):
            post_train(
                make_tiny_bundle(),
                (InstructionExample(prompt="ab", response="cd"),),
                PostTrainingConfig(
                    steps=1,
                    context_size=2,
                    batch_size=1,
                    evaluation_interval=1,
                    loss_average_window=1,
                    backend="cpu",
                ),
                formatter=TooShortFormatter(),
            )


class ServingTests(unittest.TestCase):
    def test_serving_config_validates_kv_cache_options(self) -> None:
        config = ServingConfig(
            kv_cache="contiguous",
            kv_cache_block_size=8,
        )
        self.assertEqual(config.kv_cache, "contiguous")
        self.assertEqual(config.kv_cache_block_size, 8)
        with self.assertRaises(TypeError):
            ServingConfig(kv_cache=1)
        with self.assertRaises(ValueError):
            ServingConfig(kv_cache="unknown")
        for block_size in (False, 0, -1, 1.5):
            with self.subTest(block_size=block_size):
                with self.assertRaises((TypeError, ValueError)):
                    ServingConfig(kv_cache_block_size=block_size)

    def test_http_health_and_generation(self) -> None:
        bundle = make_tiny_bundle(
            stage="post_training",
            parent_artifact_id="a" * 64,
        )
        server = create_http_server(
            bundle,
            host="127.0.0.1",
            port=0,
            config=ServingConfig(
                backend="cpu",
                maximum_new_tokens=2,
                maximum_request_bytes=1024,
                kv_cache="paged",
                kv_cache_block_size=2,
            ),
        )
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        thread.start()
        port = int(server.server_address[1])
        base_url = f"http://127.0.0.1:{port}"

        try:
            with urlopen(f"{base_url}/", timeout=5.0) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.headers.get_content_type(),
                    "text/html",
                )
                self.assertEqual(
                    response.headers["X-Content-Type-Options"],
                    "nosniff",
                )
                self.assertEqual(
                    response.headers["X-Frame-Options"],
                    "DENY",
                )
                self.assertEqual(
                    response.headers["Cache-Control"],
                    "no-store",
                )
                self.assertIn(
                    "frame-ancestors 'none'",
                    response.headers["Content-Security-Policy"],
                )
                chat_html = response.read().decode("utf-8")
            self.assertIn("Riftco Transformer", chat_html)
            self.assertIn('fetch("/v1/generate"', chat_html)
            self.assertIn(
                r"### User:\n${message}\n### Assistant:\n",
                chat_html,
            )
            self.assertNotIn("innerHTML", chat_html)

            rejected_host = Request(
                f"{base_url}/health",
                headers={"Host": "attacker.example"},
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(rejected_host, timeout=5.0)
            self.assertEqual(error.exception.code, 421)
            error.exception.close()

            with urlopen(f"{base_url}/health", timeout=5.0) as response:
                self.assertEqual(response.status, 200)
                health = json.load(response)
            self.assertEqual(
                health,
                {
                    "status": "ok",
                    "artifact_id": bundle.artifact_id,
                    "artifact_stage": "post_training",
                    "backend": "cpu",
                    "vocabulary_size": 4,
                    "maximum_context": 4,
                    "maximum_new_tokens": 2,
                    "kv_cache": "paged",
                    "kv_cache_block_size": 2,
                },
            )

            wrong_content_type = Request(
                f"{base_url}/v1/generate",
                data=b'{"prompt":"ab"}',
                headers={"Content-Type": "text/plain"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(wrong_content_type, timeout=5.0)
            self.assertEqual(error.exception.code, 415)
            try:
                error_body = json.loads(error.exception.read())
            finally:
                error.exception.close()
            self.assertIn("application/json", error_body["detail"])

            request_body = json.dumps(
                {
                    "prompt": "ab",
                    "max_new_tokens": 1,
                    "temperature": 0.0,
                }
            ).encode("utf-8")
            request = Request(
                f"{base_url}/v1/generate",
                data=request_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=5.0) as response:
                self.assertEqual(response.status, 200)
                generated = json.load(response)
            self.assertEqual(generated["artifact_id"], bundle.artifact_id)
            self.assertEqual(generated["prompt"], "ab")
            self.assertEqual(generated["prompt_token_ids"], [0, 1])
            self.assertEqual(len(generated["generated_token_ids"]), 1)
            self.assertEqual(len(generated["token_ids"]), 3)
            self.assertIsInstance(generated["generated_text"], str)

            oversized_request = Request(
                f"{base_url}/v1/generate",
                data=json.dumps(
                    {
                        "prompt": "ab",
                        "max_new_tokens": 3,
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as error:
                urlopen(oversized_request, timeout=5.0)
            self.assertEqual(error.exception.code, 400)
            try:
                error_body = json.loads(error.exception.read())
            finally:
                error.exception.close()
            self.assertIn("serving limit", error_body["detail"])
        finally:
            server.shutdown()
            thread.join(timeout=5.0)
            server.server_close()

        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
