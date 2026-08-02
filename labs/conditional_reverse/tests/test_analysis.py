from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.conditional_reverse.analysis import (  # noqa: E402
    PcaAccumulator,
    SelectorTarget,
    compare_paired_conditions,
    fit_pca,
    make_selector_steering,
    paired_effect,
    project_rows,
    steering_specs_for_variant,
)
from labs.conditional_reverse.config import Variant  # noqa: E402


class ConditionalReverseAnalysisTests(unittest.TestCase):
    def test_pca_recovers_axis_and_projects_centered_rows(self) -> None:
        result = fit_pca(((-2.0, 0.0), (0.0, 0.0), (2.0, 0.0)), 2)
        self.assertTrue(result.converged)
        self.assertEqual(result.observation_count, 3)
        self.assertEqual(result.feature_count, 2)
        self.assertEqual(result.mean, (0.0, 0.0))
        self.assertAlmostEqual(result.eigenvalues[0], 4.0)
        self.assertAlmostEqual(result.eigenvalues[1], 0.0)
        self.assertAlmostEqual(result.explained_variance_ratio[0], 1.0)
        self.assertAlmostEqual(result.components[0][0], 1.0)
        self.assertAlmostEqual(result.components[0][1], 0.0)
        projected = project_rows(((1.5, 2.0),), result)
        self.assertAlmostEqual(projected[0][0], 1.5)
        self.assertAlmostEqual(abs(projected[0][1]), 2.0)

    def test_pca_is_dependency_free_and_handles_constant_rows(self) -> None:
        result = fit_pca(((3.0, -1.0), (3.0, -1.0), (3.0, -1.0)), 2)
        self.assertEqual(result.mean, (3.0, -1.0))
        self.assertEqual(result.eigenvalues, (0.0, 0.0))
        self.assertEqual(result.explained_variance_ratio, (0.0, 0.0))
        self.assertEqual(project_rows((), result), ())

    def test_online_pca_matches_batch_fit_across_capture_batches(self) -> None:
        rows = ((-2.0, -2.0), (-1.0, -1.0), (1.0, 1.0), (2.0, 2.0))
        expected = fit_pca(rows, 2)
        accumulator = PcaAccumulator()
        accumulator.update(rows[:2])
        accumulator.update(rows[2:])
        actual = accumulator.finish(2)
        self.assertEqual(actual.observation_count, 4)
        self.assertEqual(actual.feature_count, 2)
        for left, right in zip(actual.eigenvalues, expected.eigenvalues):
            self.assertAlmostEqual(left, right)
        for left, right in zip(actual.components[0], expected.components[0]):
            self.assertAlmostEqual(left, right)

    def test_pca_rejects_ragged_nonfinite_or_undersized_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            fit_pca(((1.0, 2.0),), 1)
        with self.assertRaisesRegex(ValueError, "rectangular"):
            fit_pca(((1.0, 2.0), (3.0,)), 1)
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_pca(((1.0, 2.0), (3.0, math.inf)), 1)
        with self.assertRaisesRegex(ValueError, "component_count"):
            fit_pca(((1.0,), (2.0,)), 2)

    def test_paired_effect_retains_per_example_alignment(self) -> None:
        result = paired_effect(
            (0.5, 0.5, 0.5),
            (0.75, 0.25, 0.5),
        )
        self.assertEqual(result.deltas, (0.25, -0.25, 0.0))
        self.assertEqual(result.mean_delta, 0.0)
        self.assertEqual(result.mean_effect, 0.0)
        self.assertEqual(result.improved_count, 1)
        self.assertEqual(result.worsened_count, 1)
        self.assertEqual(result.tied_count, 1)
        self.assertGreater(result.standard_error, 0.0)

        loss_result = paired_effect(
            (1.0, 2.0),
            (0.5, 3.0),
            higher_is_better=False,
        )
        self.assertEqual(loss_result.deltas, (-0.5, 1.0))
        self.assertEqual(loss_result.improved_count, 1)
        self.assertEqual(loss_result.worsened_count, 1)

    def test_named_paired_conditions_share_one_baseline(self) -> None:
        results = compare_paired_conditions(
            (1.0, 0.5),
            {
                "learned_attention_roll": (0.5, 0.25),
                "program_output_roll": (0.75, 0.0),
            },
        )
        self.assertEqual(
            tuple(results),
            ("learned_attention_roll", "program_output_roll"),
        )
        self.assertEqual(results["learned_attention_roll"].example_count, 2)
        with self.assertRaisesRegex(ValueError, "equal length"):
            paired_effect((1.0,), (1.0, 2.0))

    def test_selector_steering_is_semantic_only_for_f(self) -> None:
        reverse = make_selector_steering(
            SelectorTarget.REVERSE,
            4,
            strength=3.0,
        )
        copy = make_selector_steering(
            SelectorTarget.COPY,
            4,
            strength=3.0,
        )
        self.assertEqual(reverse.label, "only_reverse_basis")
        self.assertEqual(copy.label, "only_copy_basis")
        self.assertEqual(reverse.input_index, 0)
        self.assertEqual(reverse.positions, (0,))
        self.assertEqual(reverse.scales, (3.0, 0.0, 0.0, 0.0))
        self.assertEqual(copy.scales, (0.0, 3.0, 3.0, 3.0))
        self.assertEqual(
            steering_specs_for_variant(Variant.F, 4, strength=3.0),
            (reverse, copy),
        )
        for variant in (Variant.P, Variant.T, Variant.I):
            with self.subTest(variant=variant):
                self.assertEqual(steering_specs_for_variant(variant, 4), ())


if __name__ == "__main__":
    unittest.main()
