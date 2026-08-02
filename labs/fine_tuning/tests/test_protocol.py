from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT))

from riftco_transformer import (  # noqa: E402
    DecoderOnlyTransformer,
    LoraConfig,
    Tokenizer,
    TransformerConfig,
    backend_available,
)
from riftco_transformer.artifacts import ModelBundle  # noqa: E402
from labs.fine_tuning import (  # noqa: E402
    FineTuningCandidate,
    FineTuningExperimentConfig,
    compare_fine_tuning,
)
import labs.fine_tuning.protocol as fine_module  # noqa: E402
from labs._support.reporting import json_safe  # noqa: E402
import riftco_transformer.post_training.evaluation as evaluation_module  # noqa: E402
from riftco_transformer.post_training import (  # noqa: E402
    InstructionExample,
    InstructionSplits,
    PostTrainingConfig,
    evaluate_instruction_examples,
)
from labs.fine_tuning import run as fine_cli  # noqa: E402


CORPUS = "abcdabcdabcdabcd"


class CompactFormatter:
    def format(self, example: InstructionExample) -> str:
        return (
            example.prompt
            + example.response
            + example.prompt
            + example.response
        )


class CountingFormatter(CompactFormatter):
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def format(self, example: InstructionExample) -> str:
        self.calls.append((example.prompt, example.response))
        return super().format(example)


class NonFiniteLoss:
    def __enter__(self) -> NonFiniteLoss:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def item(self) -> float:
        return float("nan")


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
                metadata={"purpose": "fine-tuning-comparison-test"},
            )


def make_splits() -> InstructionSplits:
    return InstructionSplits(
        train=(
            InstructionExample("ab", "c"),
            InstructionExample("abc", "d"),
        ),
        validation=(
            InstructionExample("ac", "b"),
            InstructionExample("acd", "b"),
        ),
        test=(
            InstructionExample("ad", "c"),
            InstructionExample("abd", "c"),
        ),
    )


def training_config(
    method: str,
    *,
    rank: int = 1,
) -> PostTrainingConfig:
    return PostTrainingConfig(
        steps=1,
        context_size=2,
        batch_size=1,
        evaluation_interval=1,
        loss_average_window=1,
        learning_rate=1.0e-2,
        random_seed=31,
        backend="cpu",
        fine_tuning_method=method,
        lora=LoraConfig(
            rank=rank,
            alpha=2.0 * rank,
            random_seed=37,
        ),
    )


class FineTuningExperimentTests(unittest.TestCase):
    def test_full_and_lora_use_one_selection_safe_metric(self) -> None:
        bundle = make_tiny_bundle()
        base_weights = bundle.weights
        splits = make_splits()
        split_snapshot = (splits.train, splits.validation, splits.test)
        formatter = CountingFormatter()
        candidates = (
            FineTuningCandidate(
                "full",
                training_config("full"),
            ),
            FineTuningCandidate(
                "lora-rank-1",
                training_config("lora", rank=1),
            ),
            FineTuningCandidate(
                "lora-rank-2",
                training_config("lora", rank=2),
            ),
        )
        config = FineTuningExperimentConfig(
            candidates=candidates,
            evaluation_context_size=3,
            evaluation_batch_size=2,
        )
        evaluated_splits: list[object] = []
        original_evaluate = fine_module.evaluate_instruction_examples

        def tracked_evaluate(*args: object, **kwargs: object) -> object:
            evaluated_splits.append(args[2])
            return original_evaluate(*args, **kwargs)

        with mock.patch.object(
            fine_module,
            "evaluate_instruction_examples",
            side_effect=tracked_evaluate,
        ):
            comparison = compare_fine_tuning(
                bundle,
                splits,
                config,
                formatter=formatter,
            )

        self.assertEqual(bundle.weights, base_weights)
        self.assertEqual(
            (splits.train, splits.validation, splits.test),
            split_snapshot,
        )
        self.assertEqual(
            formatter.calls,
            [
                (example.prompt, example.response)
                for examples in split_snapshot
                for example in examples
            ],
        )
        self.assertEqual(
            evaluated_splits,
            [
                splits.train,
                splits.validation,
                splits.train,
                splits.validation,
                splits.train,
                splits.validation,
                splits.train,
                splits.validation,
                splits.test,
                splits.test,
                splits.test,
            ],
        )
        self.assertEqual(comparison.resolved_backend, "cpu")
        self.assertEqual(comparison.evaluation_context_size, 3)
        self.assertEqual(len(comparison.fingerprints.train), 64)
        self.assertEqual(len(comparison.trials), 3)
        self.assertEqual(len(comparison.selected_trials), 2)
        self.assertIn("full", comparison.selected_candidate_names)
        selected_lora = tuple(
            name
            for name in comparison.selected_candidate_names
            if name.startswith("lora-")
        )
        self.assertEqual(len(selected_lora), 1)

        base_parameter_count = sum(
            parameter.value_count for parameter in bundle.parameters
        )
        full_trial = comparison.trials[0]
        self.assertEqual(
            full_trial.trainable_parameter_count,
            base_parameter_count,
        )
        self.assertEqual(full_trial.trainable_fraction, 1.0)
        self.assertIsNotNone(full_trial.test)
        self.assertTrue(full_trial.selected_for_test)

        lora_trials = comparison.trials[1:]
        self.assertLess(
            lora_trials[0].trainable_parameter_count,
            base_parameter_count,
        )
        self.assertEqual(
            lora_trials[1].trainable_parameter_count,
            2 * lora_trials[0].trainable_parameter_count,
        )
        self.assertEqual(
            sum(trial.test is not None for trial in lora_trials),
            1,
        )

        summary = fine_cli.comparison_summary(
            comparison,
            {"manifest_sha256": "a" * 64, "manifest": {}},
            Path("results/final-comparison"),
        )
        self.assertEqual(summary["format"], fine_cli.REPORT_FORMAT)
        self.assertEqual(
            summary["selected_candidate_names"],
            list(comparison.selected_candidate_names),
        )
        self.assertTrue(
            summary["selection_policy"][
                "test_split_must_be_retired_after_report"
            ]
        )
        self.assertEqual(len(summary["trials"]), 3)
        json.dumps(
            json_safe(summary),
            allow_nan=False,
        )

        for trial in comparison.trials:
            with self.subTest(candidate=trial.name):
                self.assertEqual(
                    trial.bundle.parent_artifact_id,
                    bundle.artifact_id,
                )
                self.assertAlmostEqual(
                    trial.validation_generalization_gap,
                    trial.validation.loss - trial.train.loss,
                )
                self.assertAlmostEqual(
                    trial.validation_loss_delta_from_base,
                    (
                        trial.validation.loss
                        - comparison.baseline_validation.loss
                    ),
                )
                self.assertTrue(math.isfinite(trial.train.perplexity))
                self.assertTrue(math.isfinite(trial.validation.perplexity))
                if trial.test is None:
                    self.assertIsNone(trial.test_generalization_gap)
                    self.assertIsNone(trial.test_loss_delta_from_base)
                else:
                    self.assertAlmostEqual(
                        trial.test_generalization_gap,
                        trial.test.loss - trial.train.loss,
                    )
                    self.assertAlmostEqual(
                        trial.test_loss_delta_from_base,
                        trial.test.loss - comparison.baseline_test.loss,
                    )

    def test_qlora_candidate_runs_and_exports_an_fp32_bundle(self) -> None:
        comparison = compare_fine_tuning(
            make_tiny_bundle(),
            make_splits(),
            FineTuningExperimentConfig(
                candidates=(
                    FineTuningCandidate(
                        "qlora-rank-1",
                        training_config("qlora", rank=1),
                    ),
                ),
                evaluation_context_size=3,
                evaluation_batch_size=2,
            ),
            formatter=CompactFormatter(),
        )

        self.assertEqual(comparison.resolved_backend, "cpu")
        self.assertEqual(len(comparison.trials), 1)
        trial = comparison.trials[0]
        self.assertEqual(trial.fine_tuning_method, "qlora")
        self.assertTrue(trial.selected_for_test)
        self.assertIsNotNone(trial.test)
        quantization = trial.bundle.metadata["quantization"]
        self.assertIsInstance(quantization, dict)
        assert isinstance(quantization, dict)
        self.assertEqual(quantization["format"], "nf4")
        self.assertTrue(quantization["export_materialized_to_fp32"])

    def test_batched_evaluation_preserves_token_weighted_loss(self) -> None:
        bundle = make_tiny_bundle()
        examples = make_splits().validation
        with bundle.instantiate("cpu") as runtime:
            unbatched = evaluate_instruction_examples(
                runtime.model,
                runtime.tokenizer,
                examples,
                context_size=3,
                batch_size=1,
                formatter=CompactFormatter(),
            )
            batched = evaluate_instruction_examples(
                runtime.model,
                runtime.tokenizer,
                examples,
                context_size=3,
                batch_size=2,
                formatter=CompactFormatter(),
            )

        self.assertEqual(
            unbatched.target_token_count,
            batched.target_token_count,
        )
        self.assertEqual(unbatched.chunk_count, batched.chunk_count)
        self.assertLessEqual(
            batched.forward_batch_count,
            unbatched.forward_batch_count,
        )
        self.assertAlmostEqual(unbatched.loss, batched.loss, places=6)
        self.assertAlmostEqual(
            unbatched.perplexity,
            batched.perplexity,
            places=5,
        )

    def test_evaluation_rejects_nonfinite_loss_before_selection(self) -> None:
        bundle = make_tiny_bundle()
        with bundle.instantiate("cpu") as runtime:
            with mock.patch.object(
                evaluation_module,
                "cross_entropy",
                return_value=NonFiniteLoss(),
            ):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    evaluate_instruction_examples(
                        runtime.model,
                        runtime.tokenizer,
                        make_splits().validation,
                        context_size=3,
                        formatter=CompactFormatter(),
                    )

    @unittest.skipUnless(
        backend_available("metal"),
        "Metal backend is unavailable in this build",
    )
    def test_full_and_lora_generalization_smoke_on_metal(self) -> None:
        candidates = (
            FineTuningCandidate(
                "full",
                replace(training_config("full"), backend="metal"),
            ),
            FineTuningCandidate(
                "lora-rank-1",
                replace(
                    training_config("lora", rank=1),
                    backend="metal",
                ),
            ),
        )
        comparison = compare_fine_tuning(
            make_tiny_bundle(),
            make_splits(),
            FineTuningExperimentConfig(
                candidates=candidates,
                evaluation_context_size=3,
                evaluation_batch_size=2,
            ),
            formatter=CompactFormatter(),
        )

        self.assertEqual(comparison.resolved_backend, "metal")
        self.assertEqual(len(comparison.selected_trials), 2)
        self.assertTrue(
            all(trial.test is not None for trial in comparison.trials)
        )

    @unittest.skipUnless(
        backend_available("cuda"),
        "CUDA backend is unavailable in this build",
    )
    def test_full_and_lora_generalization_smoke_on_cuda(self) -> None:
        candidates = (
            FineTuningCandidate(
                "full",
                replace(training_config("full"), backend="cuda"),
            ),
            FineTuningCandidate(
                "lora-rank-1",
                replace(
                    training_config("lora", rank=1),
                    backend="cuda",
                ),
            ),
        )
        comparison = compare_fine_tuning(
            make_tiny_bundle(),
            make_splits(),
            FineTuningExperimentConfig(
                candidates=candidates,
                evaluation_context_size=3,
                evaluation_batch_size=2,
            ),
            formatter=CompactFormatter(),
        )

        self.assertEqual(comparison.resolved_backend, "cuda")
        self.assertEqual(len(comparison.selected_trials), 2)
        self.assertTrue(
            all(trial.test is not None for trial in comparison.trials)
        )

    @unittest.skipUnless(
        backend_available("tpu"),
        "TPU backend is unavailable in this build",
    )
    def test_full_and_lora_generalization_smoke_on_tpu(self) -> None:
        candidates = (
            FineTuningCandidate(
                "full",
                replace(training_config("full"), backend="tpu"),
            ),
            FineTuningCandidate(
                "lora-rank-1",
                replace(
                    training_config("lora", rank=1),
                    backend="tpu",
                ),
            ),
        )
        comparison = compare_fine_tuning(
            make_tiny_bundle(),
            make_splits(),
            FineTuningExperimentConfig(
                candidates=candidates,
                evaluation_context_size=3,
                evaluation_batch_size=2,
            ),
            formatter=CompactFormatter(),
        )

        self.assertEqual(comparison.resolved_backend, "tpu")
        self.assertEqual(len(comparison.selected_trials), 2)
        self.assertTrue(
            all(trial.test is not None for trial in comparison.trials)
        )

    def test_candidate_and_comparison_configuration_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "name"):
            FineTuningCandidate(" ", training_config("full"))
        duplicate = FineTuningCandidate(
            "same",
            training_config("full"),
        )
        with self.assertRaisesRegex(ValueError, "unique"):
            FineTuningExperimentConfig(candidates=(duplicate, duplicate))
        with self.assertRaisesRegex(ValueError, "evaluation_attention"):
            FineTuningExperimentConfig(evaluation_attention="unknown")

    def test_cli_builds_separate_full_and_lora_recipes(self) -> None:
        arguments = fine_cli.build_parser().parse_args(
            [
                "--base",
                "base.rift",
                "--data",
                "prepared",
                "--methods",
                "full,lora",
                "--lora-ranks",
                "2,1",
                "--steps",
                "3",
                "--context",
                "4",
                "--batch-size",
                "2",
                "--full-learning-rate",
                "0.001",
                "--lora-learning-rate",
                "0.01",
                "--attention",
                "flash",
                "--activation-checkpointing",
                "block",
                "--backend",
                "tpu",
            ]
        )
        candidates = fine_cli.build_candidates(arguments)

        self.assertEqual(
            tuple(candidate.name for candidate in candidates),
            ("full", "lora-rank-1", "lora-rank-2"),
        )
        self.assertEqual(
            tuple(candidate.selection_group for candidate in candidates),
            ("full", "lora", "lora"),
        )
        self.assertEqual(candidates[0].config.learning_rate, 0.001)
        self.assertEqual(candidates[1].config.learning_rate, 0.01)
        self.assertEqual(candidates[1].config.lora.rank, 1)
        self.assertEqual(candidates[2].config.lora.rank, 2)
        for candidate in candidates:
            self.assertEqual(candidate.config.backend, "tpu")
            self.assertEqual(candidate.config.attention, "flash")
            self.assertEqual(
                candidate.config.activation_checkpointing,
                "block",
            )


if __name__ == "__main__":
    unittest.main()
