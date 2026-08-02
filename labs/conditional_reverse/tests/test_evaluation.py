from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT))

from riftco_transformer.programmed import Representation  # noqa: E402
from labs.conditional_reverse.config import AnalysisConfig  # noqa: E402
from labs.conditional_reverse.data import TokenCodec, make_batch  # noqa: E402
from labs.conditional_reverse.evaluation import (  # noqa: E402
    EvaluationMetrics,
    EvaluationResult,
    balanced_examples,
    evaluate_model,
    fit_program_output_pca,
    run_paired_ablations,
    score_hypotheses,
)
from labs.conditional_reverse.protocol import (  # noqa: E402
    ProtocolConfig,
    make_example,
)
import labs.conditional_reverse.evaluation as evaluation_module  # noqa: E402


class _Context:
    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        return None


class _FakeLoss(_Context):
    def item(self) -> float:
        return 0.25


class _FakeLogits(_Context):
    def __init__(self, shape, values) -> None:
        self.shape = shape
        self._values = values

    def tolist(self):
        return list(self._values)


class _FakeTrace:
    def __init__(self, entries) -> None:
        self._entries = {entry.name: entry for entry in entries}

    def at(self, name: str):
        return self._entries[name]


class _FakeModel:
    def __init__(self, shape, values, trace, expected_rows) -> None:
        self.shape = shape
        self.values = values
        self.trace = trace
        self.expected_rows = expected_rows
        self.options = []

    def forward(self, rows, options):
        if tuple(tuple(row) for row in rows) != self.expected_rows:
            raise AssertionError("forward did not receive rectangular input rows")
        self.options.append(options)
        return SimpleNamespace(
            logits=_FakeLogits(self.shape, self.values),
            representations=self.trace,
        )


def _empty_metrics() -> EvaluationMetrics:
    return EvaluationMetrics(
        loss=0.0,
        example_count=2,
        target_token_count=6,
        correct_target_token_count=0,
        correct_sequence_count=0,
        target_token_accuracy=0.0,
        exact_sequence_accuracy=0.0,
        reverse_example_count=1,
        reverse_target_token_accuracy=0.0,
        reverse_exact_sequence_accuracy=0.0,
        copy_example_count=1,
        copy_target_token_accuracy=0.0,
        copy_exact_sequence_accuracy=0.0,
    )


def _result(token_scores=(0.0, 0.0), exact_scores=(0.0, 0.0)):
    return EvaluationResult(
        metrics=_empty_metrics(),
        predictions=((0, 0, 0), (0, 0, 0)),
        per_example_token_accuracy=tuple(token_scores),
        per_example_exact_accuracy=tuple(exact_scores),
        prediction_fingerprint="0" * 64,
    )


class ConditionalReverseEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = ProtocolConfig(
            sequence_length=3,
            alphabet="abcd",
            reverse_when_first_is="a",
        )
        self.codec = TokenCodec.from_protocol(self.protocol)
        self.examples = (
            make_example("abc", self.protocol),
            make_example("bca", self.protocol),
        )

    def test_evaluation_uses_rectangular_rows_and_target_half(self) -> None:
        batch = make_batch(self.examples, self.codec)
        vocabulary = self.codec.vocabulary_size
        shape = (2, 6, vocabulary)
        logits = [0.0] * (2 * 6 * vocabulary)
        for row, targets in enumerate(batch.output_rows):
            for position, target in enumerate(targets, start=3):
                logits[(row * 6 + position) * vocabulary + target] = 10.0
        raw = Representation(
            name="program.output.raw",
            shape=(2, 3, 2),
            values=tuple(float(index) for index in range(12)),
        )
        model = _FakeModel(
            shape,
            logits,
            _FakeTrace((raw,)),
            batch.input_rows,
        )
        runtime = SimpleNamespace(protocol=self.protocol, model=model)
        captured = []

        def fake_loss(_logits, targets, offset, count):
            self.assertEqual(tuple(tuple(row) for row in targets), batch.target_rows)
            self.assertEqual((offset, count), (3, 3))
            return _FakeLoss()

        with patch.object(
            evaluation_module,
            "cross_entropy_time_range",
            side_effect=fake_loss,
        ):
            result = evaluate_model(
                runtime,
                self.examples,
                self.codec,
                2,
                capture_name="program.output.raw",
                capture_sink=captured.append,
            )
        self.assertEqual(result.metrics.loss, 0.25)
        self.assertEqual(result.metrics.target_token_accuracy, 1.0)
        self.assertEqual(result.metrics.exact_sequence_accuracy, 1.0)
        self.assertEqual(result.metrics.reverse_exact_sequence_accuracy, 1.0)
        self.assertEqual(result.metrics.copy_exact_sequence_accuracy, 1.0)
        self.assertEqual(result.predictions, batch.output_rows)
        self.assertEqual(captured, [raw])
        self.assertTrue(model.options[0].capture_representations)
        hypotheses = score_hypotheses(self.examples, result.predictions, self.codec)
        self.assertEqual(hypotheses.conditional.exact_sequence_accuracy, 1.0)

    def test_ablation_suite_keeps_pairing_and_all_program_conditions(self) -> None:
        baseline = _result((1.0, 0.5), (1.0, 0.0))
        learned = _result((0.5, 0.5), (0.0, 0.0))
        program = _result((0.75, 0.25), (1.0, 0.0))
        combined = _result((0.0, 0.0), (0.0, 0.0))
        runtime = SimpleNamespace(has_program=True)
        with patch.object(
            evaluation_module,
            "evaluate_model",
            side_effect=(baseline, learned, program, combined),
        ) as evaluate:
            suite = run_paired_ablations(
                runtime,
                self.examples,
                self.codec,
                2,
            )
        self.assertEqual(
            tuple(condition.name for condition in suite.conditions),
            ("learned_attention", "program_output", "combined"),
        )
        self.assertEqual(evaluate.call_count, 4)
        self.assertEqual(
            suite.conditions[0].token_accuracy_effect.deltas,
            (-0.5, 0.0),
        )

    def test_probe_pca_streams_only_raw_program_rows(self) -> None:
        raw = Representation(
            name="program.output.raw",
            shape=(2, 3, 2),
            values=(
                -2.0,
                -2.0,
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                2.0,
                2.0,
                3.0,
                3.0,
            ),
        )
        runtime = SimpleNamespace(
            has_program=True,
            config=SimpleNamespace(program_width=2),
        )

        def fake_evaluate(*_args, **kwargs):
            self.assertEqual(kwargs["capture_name"], "program.output.raw")
            self.assertFalse(kwargs["retain_per_example"])
            kwargs["capture_sink"](raw)
            return _result()

        with patch.object(
            evaluation_module,
            "evaluate_model",
            side_effect=fake_evaluate,
        ):
            result = fit_program_output_pca(
                runtime,
                self.examples,
                self.codec,
                2,
                AnalysisConfig(pca_components=2),
            )
        self.assertEqual(result.representation_name, "program.output.raw")
        self.assertEqual(result.pca.observation_count, 6)
        self.assertAlmostEqual(result.pca.explained_variance_ratio[0], 1.0)

    def test_balanced_examples_interleave_equal_branches(self) -> None:
        examples = self.examples + (
            make_example("acd", self.protocol),
            make_example("cba", self.protocol),
        )
        balanced = balanced_examples(examples)
        self.assertEqual(len(balanced), 4)
        self.assertEqual(
            tuple(item.reversed for item in balanced),
            (True, False, True, False),
        )


if __name__ == "__main__":
    unittest.main()
