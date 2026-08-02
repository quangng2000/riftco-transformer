from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT))

from riftco_transformer import (  # noqa: E402
    DecoderOnlyTransformer,
    Tokenizer,
    TransformerConfig,
)
from riftco_transformer.artifacts import ModelBundle  # noqa: E402
from riftco_transformer.data import PreparedDataset, PreparedFile  # noqa: E402
from labs.lora_rank import (  # noqa: E402
    IdentityPromptFormatter,
    InstructionSplits,
    LoraRankExperimentConfig,
    compare_lora_ranks,
    evaluate_instruction_examples,
    load_instruction_splits,
    load_prepared_instruction_splits,
)
import labs.lora_rank.protocol as lora_rank_module  # noqa: E402
from riftco_transformer.post_training import (  # noqa: E402
    InstructionExample,
    PlainChatFormatter,
)
from labs.lora_rank import run as rank_cli  # noqa: E402


CORPUS = "abcdabcdabcdabcd"


class CompactFormatter:
    def format(self, example: InstructionExample) -> str:
        return (example.prompt + example.response) * 2


class CountingFormatter(CompactFormatter):
    def __init__(self, events: list[tuple[str, object]]) -> None:
        self.calls: list[tuple[str, str]] = []
        self.events = events

    def format(self, example: InstructionExample) -> str:
        key = (example.prompt, example.response)
        self.calls.append(key)
        self.events.append(("format", key))
        return super().format(example)


class PromptOnlyFormatter:
    def format(self, example: InstructionExample) -> str:
        return example.prompt


def make_tiny_bundle() -> ModelBundle:
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
                stage="pretraining",
                metadata={"purpose": "lora-rank-test"},
            )


def write_jsonl(
    path: Path,
    examples: tuple[InstructionExample, ...],
    *,
    category: str | None = None,
) -> None:
    with path.open("w", encoding="utf-8") as output:
        for example in examples:
            record = {
                "prompt": example.prompt,
                "response": example.response,
            }
            if category is not None:
                record["category"] = category
            json.dump(
                record,
                output,
                ensure_ascii=False,
            )
            output.write("\n")


def prepared_dataset_view(root: Path) -> PreparedDataset:
    manifest = {"format": "riftco-transformer.prepared-dataset.v1"}
    manifest_bytes = (
        json.dumps(manifest, sort_keys=True) + "\n"
    ).encode("utf-8")
    files: dict[str, PreparedFile] = {}
    for partition in ("train", "validation", "test"):
        path = root / f"{partition}.jsonl"
        content = path.read_bytes()
        files[partition] = PreparedFile(
            path=path,
            media_type="application/x-ndjson",
            records=sum(
                1 for line in content.splitlines() if line.strip()
            ),
            bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )
    return PreparedDataset(
        directory=root,
        manifest_path=root / "manifest.json",
        files=MappingProxyType(files),
        manifest=MappingProxyType(manifest),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


class LoraRankExperimentTests(unittest.TestCase):
    def test_cli_stages_and_exclusively_publishes_complete_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "rank-output"
            with rank_cli.staged_output_directory(destination) as staging:
                staged_path = staging
                (staging / "comparison.json").write_text(
                    "{}\n",
                    encoding="utf-8",
                )
                self.assertFalse(destination.exists())

            self.assertFalse(staged_path.exists())
            self.assertEqual(
                (destination / "comparison.json").read_text(
                    encoding="utf-8"
                ),
                "{}\n",
            )
            with self.assertRaises(FileExistsError):
                with rank_cli.staged_output_directory(destination):
                    pass

            empty_destination = root / "existing-empty-output"
            empty_destination.mkdir()
            with self.assertRaises(FileExistsError):
                with rank_cli.staged_output_directory(empty_destination):
                    pass
            self.assertTrue(empty_destination.is_dir())
            self.assertEqual(tuple(empty_destination.iterdir()), ())

            failed_destination = root / "failed-output"
            with self.assertRaisesRegex(RuntimeError, "training failed"):
                with rank_cli.staged_output_directory(
                    failed_destination
                ) as failed_staging:
                    (failed_staging / "partial.rift").write_bytes(b"partial")
                    raise RuntimeError("training failed")
            self.assertFalse(failed_destination.exists())
            self.assertFalse(failed_staging.exists())

            raced_destination = root / "raced-output"
            with self.assertRaises(FileExistsError):
                with rank_cli.staged_output_directory(
                    raced_destination
                ) as raced_staging:
                    raced_destination.mkdir()
            self.assertTrue(raced_destination.is_dir())
            self.assertFalse(raced_staging.exists())

    def test_cli_provenance_uses_verified_manifest_snapshot(self) -> None:
        manifest = {
            "format": "riftco-transformer.prepared-dataset.v1",
            "source": {
                "revision": "revision-123",
                "license": "cc-by-sa-3.0",
                "selection": {
                    "row_ranges": [{"offset": 100, "length": 20}],
                },
            },
            "files": {
                partition: {
                    "sha256": partition * 8,
                }
                for partition in ("train", "validation", "test")
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "manifest.json"
            manifest_bytes = (
                json.dumps(manifest, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8")
            manifest_path.write_bytes(manifest_bytes)
            files = MappingProxyType(
                {
                    partition: PreparedFile(
                        path=root / f"{partition}.jsonl",
                        media_type="application/x-ndjson",
                        records=1,
                        bytes=1,
                        sha256=partition * 8,
                    )
                    for partition in ("train", "validation", "test")
                }
            )
            prepared = PreparedDataset(
                directory=root,
                manifest_path=manifest_path,
                files=files,
                manifest=MappingProxyType(manifest),
                manifest_sha256=hashlib.sha256(
                    manifest_bytes
                ).hexdigest(),
            )

            provenance = rank_cli.prepared_dataset_provenance(prepared)
            self.assertEqual(provenance["manifest"], manifest)
            self.assertEqual(
                provenance["manifest_sha256"],
                hashlib.sha256(manifest_bytes).hexdigest(),
            )
            manifest_path.write_text("{}\n", encoding="utf-8")
            self.assertEqual(
                rank_cli.prepared_dataset_provenance(prepared),
                provenance,
            )

    def test_prepared_split_loader_is_stable_and_rejects_leakage(
        self,
    ) -> None:
        train = (InstructionExample("ab", "cd"),)
        validation = (InstructionExample("ac", "bd"),)
        test = (InstructionExample("ad", "bc"),)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "train.jsonl",
                train,
                category="open_qa",
            )
            write_jsonl(root / "validation.jsonl", validation)
            write_jsonl(root / "test.jsonl", test)
            first = load_instruction_splits(
                root / "train.jsonl",
                root / "validation.jsonl",
                root / "test.jsonl",
            )
            second = load_instruction_splits(
                root / "train.jsonl",
                root / "validation.jsonl",
                root / "test.jsonl",
            )
            prepared = prepared_dataset_view(root)
            with mock.patch.object(
                lora_rank_module,
                "verify_prepared_dataset",
                return_value=prepared,
            ) as verify:
                verified = load_prepared_instruction_splits(root)
            verify.assert_called_once_with(root)
            self.assertEqual(verified, first)
            self.assertEqual(
                load_prepared_instruction_splits(prepared),
                first,
            )

            write_jsonl(root / "test.jsonl", train)
            with self.assertRaisesRegex(ValueError, "splits overlap"):
                load_instruction_splits(
                    root / "train.jsonl",
                    root / "validation.jsonl",
                    root / "test.jsonl",
                )

        self.assertEqual(first, second)
        self.assertEqual(first.fingerprints, second.fingerprints)
        self.assertEqual(len(first.fingerprints.train), 64)

    def test_prepared_loader_rejects_split_changed_after_verification(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(
                root / "train.jsonl",
                (InstructionExample("ab", "cd"),),
            )
            write_jsonl(
                root / "validation.jsonl",
                (InstructionExample("ac", "bd"),),
            )
            write_jsonl(
                root / "test.jsonl",
                (InstructionExample("ad", "bc"),),
            )
            prepared = prepared_dataset_view(root)
            train_path = root / "train.jsonl"

            def verify_then_mutate(_source: object) -> PreparedDataset:
                verified_content = train_path.read_bytes()
                changed_content = verified_content.replace(
                    b'"ab"',
                    b'"zz"',
                    1,
                )
                self.assertEqual(
                    len(changed_content),
                    len(verified_content),
                )
                self.assertNotEqual(changed_content, verified_content)
                train_path.write_bytes(changed_content)
                return prepared

            with mock.patch.object(
                lora_rank_module,
                "verify_prepared_dataset",
                side_effect=verify_then_mutate,
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "train.jsonl SHA-256 does not match",
                ):
                    load_prepared_instruction_splits(root)

    def test_held_out_evaluation_is_read_only_and_scores_each_target(
        self,
    ) -> None:
        bundle = make_tiny_bundle()
        examples = (InstructionExample("ab", "cd"),)
        original_examples = tuple(examples)

        with bundle.instantiate("cpu") as runtime:
            with runtime.model.parameters() as parameters:
                weights_before = parameters.flat_values()
            evaluation = evaluate_instruction_examples(
                runtime.model,
                runtime.tokenizer,
                examples,
                context_size=3,
                formatter=CompactFormatter(),
            )
            with runtime.model.parameters() as parameters:
                weights_after = parameters.flat_values()

        self.assertEqual(examples, original_examples)
        self.assertEqual(weights_before, weights_after)
        self.assertEqual(evaluation.example_count, 1)
        self.assertEqual(evaluation.usable_example_count, 1)
        self.assertEqual(evaluation.skipped_example_count, 0)
        self.assertEqual(evaluation.target_token_count, 7)
        self.assertEqual(evaluation.chunk_count, 3)
        self.assertEqual(evaluation.forward_batch_count, 3)
        self.assertTrue(math.isfinite(evaluation.loss))
        self.assertAlmostEqual(
            evaluation.perplexity,
            math.exp(evaluation.loss),
        )
        self.assertGreaterEqual(evaluation.elapsed_seconds, 0.0)

        with bundle.instantiate("cpu") as runtime:
            partial = evaluate_instruction_examples(
                runtime.model,
                runtime.tokenizer,
                (
                    InstructionExample("a", "b"),
                    InstructionExample("ab", "c"),
                ),
                context_size=3,
                formatter=PromptOnlyFormatter(),
            )
        self.assertEqual(partial.example_count, 2)
        self.assertEqual(partial.usable_example_count, 1)
        self.assertEqual(partial.skipped_example_count, 1)
        self.assertEqual(partial.target_token_count, 1)
        self.assertEqual(partial.forward_batch_count, 1)

    def test_rank_sweep_shares_base_data_seeds_and_scale(self) -> None:
        bundle = make_tiny_bundle()
        base_weights = bundle.weights
        splits = InstructionSplits(
            train=(InstructionExample("ab", "cd"),),
            validation=(InstructionExample("ac", "bd"),),
            test=(InstructionExample("ad", "bc"),),
        )
        split_snapshot = (
            splits.train,
            splits.validation,
            splits.test,
        )
        events: list[tuple[str, object]] = []
        formatter = CountingFormatter(events)
        config = LoraRankExperimentConfig(
            ranks=(1, 2),
            alpha_over_rank=2.0,
            adapter_random_seed=37,
            training_random_seed=31,
            steps=1,
            context_size=2,
            batch_size=1,
            evaluation_interval=1,
            loss_average_window=1,
            learning_rate=1.0e-2,
            backend="cpu",
            evaluation_context_size=3,
            inference_max_new_tokens=1,
            kv_cache_block_size=2,
        )

        original_evaluate = (
            lora_rank_module.evaluate_instruction_examples
        )

        def tracked_evaluate(*args: object, **kwargs: object) -> object:
            events.append(("evaluate", args[2]))
            return original_evaluate(*args, **kwargs)

        with mock.patch.object(
            lora_rank_module,
            "evaluate_instruction_examples",
            side_effect=tracked_evaluate,
        ):
            comparison = compare_lora_ranks(
                bundle,
                splits,
                config,
                formatter=formatter,
                inference_prompts=("ab",),
                inference_prompt_formatter=IdentityPromptFormatter(),
            )

        self.assertEqual(bundle.weights, base_weights)
        self.assertEqual(
            (splits.train, splits.validation, splits.test),
            split_snapshot,
        )
        self.assertEqual(
            formatter.calls,
            [("ab", "cd"), ("ac", "bd"), ("ad", "bc")],
        )
        self.assertEqual(comparison.base_artifact_id, bundle.artifact_id)
        self.assertEqual(comparison.resolved_backend, "cpu")
        self.assertEqual(len(comparison.trials), 2)
        self.assertIn(comparison.best_rank, (1, 2))
        self.assertEqual(
            comparison.best_trial,
            comparison.ranked_by_validation[0],
        )
        self.assertEqual(
            comparison.baseline_validation.target_token_count,
            7,
        )
        self.assertEqual(comparison.baseline_test.target_token_count, 7)
        self.assertEqual(comparison.selected_test.target_token_count, 7)

        evaluated_splits = [
            value
            for event, value in events
            if event == "evaluate"
        ]
        self.assertEqual(
            evaluated_splits,
            [
                splits.validation,
                splits.validation,
                splits.validation,
                splits.test,
                splits.test,
            ],
        )
        first_evaluation = next(
            index
            for index, event in enumerate(events)
            if event[0] == "evaluate"
        )
        self.assertLess(
            events.index(("format", ("ad", "bc"))),
            first_evaluation,
        )

        for trial, expected_rank in zip(comparison.trials, (1, 2)):
            with self.subTest(rank=expected_rank):
                self.assertEqual(trial.rank, expected_rank)
                self.assertEqual(trial.alpha, 2.0 * expected_rank)
                self.assertEqual(trial.alpha_over_rank, 2.0)
                self.assertEqual(
                    trial.adapter_parameter_count,
                    16 * expected_rank,
                )
                self.assertEqual(
                    trial.starting_artifact_id,
                    bundle.artifact_id,
                )
                self.assertEqual(
                    trial.bundle.parent_artifact_id,
                    bundle.artifact_id,
                )
                self.assertEqual(len(trial.training_metrics), 1)
                self.assertEqual(
                    trial.validation.target_token_count,
                    7,
                )
                self.assertEqual(
                    trial.validation.usable_example_count,
                    1,
                )
                self.assertEqual(
                    trial.validation.skipped_example_count,
                    0,
                )
                self.assertAlmostEqual(
                    trial.validation_loss_delta_from_base,
                    (
                        trial.validation.loss
                        - comparison.baseline_validation.loss
                    ),
                )
                if trial.rank == comparison.best_rank:
                    self.assertIsNotNone(trial.test)
                    assert trial.test is not None
                    self.assertEqual(trial.test.target_token_count, 7)
                    self.assertTrue(math.isfinite(trial.test.loss))
                    self.assertAlmostEqual(
                        trial.test_loss_delta_from_base,
                        (
                            trial.test.loss
                            - comparison.baseline_test.loss
                        ),
                    )
                else:
                    self.assertIsNone(trial.test)
                    self.assertIsNone(trial.test_loss_delta_from_base)
                self.assertGreaterEqual(trial.training_seconds, 0.0)
                self.assertEqual(len(trial.inference_samples), 1)
                sample = trial.inference_samples[0]
                self.assertEqual(sample.prompt, "ab")
                self.assertEqual(sample.formatted_prompt, "ab")
                self.assertEqual(len(sample.generated_token_ids), 1)
                self.assertGreaterEqual(sample.elapsed_seconds, 0.0)
                self.assertGreater(sample.tokens_per_second, 0.0)

                metadata = trial.bundle.metadata["training_config"]
                self.assertEqual(metadata["random_seed"], 31)
                self.assertEqual(metadata["steps"], 1)
                self.assertEqual(
                    metadata["sampling_strategy"],
                    "example_uniform",
                )
                self.assertEqual(metadata["attention"], "materialized")
                self.assertEqual(
                    metadata["activation_checkpointing"],
                    "disabled",
                )
                self.assertEqual(metadata["lora"]["random_seed"], 37)
                self.assertEqual(
                    metadata["lora"]["rank"],
                    expected_rank,
                )

    def test_accelerator_configuration_and_cli_parsing(self) -> None:
        for backend in ("cuda", "tpu"):
            with self.subTest(backend=backend):
                config = LoraRankExperimentConfig(
                    ranks=(1,),
                    backend=backend,
                )
                self.assertEqual(config.backend, backend)
                arguments = rank_cli.build_parser().parse_args(
                    [
                        "--base",
                        "base.rift",
                        "--data",
                        "prepared",
                        "--backend",
                        backend,
                    ]
                )
                self.assertEqual(arguments.backend, backend)

    def test_configuration_and_overlap_validation(self) -> None:
        for ranks, error_type in (
            ((), ValueError),
            ((1, 1), ValueError),
            ((0,), ValueError),
            ((True,), TypeError),
        ):
            with self.subTest(ranks=ranks):
                with self.assertRaises(error_type):
                    LoraRankExperimentConfig(ranks=ranks)

        with self.assertRaisesRegex(ValueError, "attention"):
            LoraRankExperimentConfig(attention="unknown")
        with self.assertRaisesRegex(ValueError, "activation_checkpointing"):
            LoraRankExperimentConfig(
                activation_checkpointing="unknown"
            )

        positional = LoraRankExperimentConfig(
            (1,),
            2.0,
            ("attention.query", "attention.value"),
            11,
            13,
            1,
            2,
            1,
            1,
            1,
            1.0e-3,
            "cpu",
            "example_uniform",
            2,
            True,
            3,
            "paged",
            4,
        )
        self.assertEqual(positional.evaluation_context_size, 2)
        self.assertEqual(positional.kv_cache_block_size, 4)
        self.assertEqual(positional.attention, "materialized")
        self.assertEqual(positional.activation_checkpointing, "disabled")

        unicode_space = "\N{NO-BREAK SPACE}"
        unicode_example = InstructionExample(
            f"{unicode_space}prompt{unicode_space}",
            f"{unicode_space}response{unicode_space}",
        )
        self.assertEqual(
            PlainChatFormatter().format(unicode_example),
            "### User:\n"
            f"{unicode_space}prompt{unicode_space}\n"
            "### Assistant:\n"
            f"{unicode_space}response{unicode_space}\n",
        )

        overlapping = InstructionSplits(
            train=(InstructionExample("ab", "cd"),),
            validation=(InstructionExample("ab", "cd"),),
            test=(InstructionExample("ac", "bd"),),
        )
        with self.assertRaisesRegex(ValueError, "splits overlap"):
            overlapping.validate_disjoint()

        whitespace_variant = InstructionSplits(
            train=(InstructionExample("ab ", "cd"),),
            validation=(InstructionExample("ac", "bd"),),
            test=(InstructionExample("ab", "cd "),),
        )
        with self.assertRaisesRegex(
            ValueError,
            "overlap after instruction formatting",
        ):
            compare_lora_ranks(
                make_tiny_bundle(),
                whitespace_variant,
                LoraRankExperimentConfig(
                    ranks=(1,),
                    steps=1,
                    context_size=2,
                    batch_size=1,
                    evaluation_interval=1,
                    loss_average_window=1,
                    backend="cpu",
                ),
                formatter=PlainChatFormatter(),
            )

        with self.assertRaisesRegex(ValueError, "would not be low-rank"):
            compare_lora_ranks(
                make_tiny_bundle(),
                InstructionSplits(
                    train=(InstructionExample("ab", "cd"),),
                    validation=(InstructionExample("ac", "bd"),),
                    test=(InstructionExample("ad", "bc"),),
                ),
                LoraRankExperimentConfig(
                    ranks=(5,),
                    steps=1,
                    context_size=2,
                    batch_size=1,
                    evaluation_interval=1,
                    loss_average_window=1,
                    backend="cpu",
                ),
                formatter=CompactFormatter(),
            )


if __name__ == "__main__":
    unittest.main()
