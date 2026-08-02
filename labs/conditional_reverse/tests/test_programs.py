from __future__ import annotations

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.conditional_reverse.config import Variant  # noqa: E402
from labs.conditional_reverse.programs import (  # noqa: E402
    CoefficientInitialization,
    build_program_spec,
    f_coefficient_index,
    iter_f_coefficient_indices,
    iter_p_coefficient_indices,
    make_f_reference_map,
    make_p_reference_map,
    output_major_flat_index,
    p_coefficient_index,
)


class ConditionalReverseProgramTests(unittest.TestCase):
    def test_output_major_flattening_formula(self) -> None:
        self.assertEqual(
            output_major_flat_index(2, (1, 3), 5, (4, 7)),
            (2 * 4 + 1) * 7 + 3,
        )
        with self.assertRaisesRegex(ValueError, "match"):
            output_major_flat_index(0, (1,), 2, (2, 2))

    def test_p_indices_are_the_exact_reversal_matrix(self) -> None:
        # Shape [D,D], D=4. Each row selects the same symbol at the
        # opposite sequence position.
        expected = (2, 7, 8, 13)
        self.assertEqual(tuple(iter_p_coefficient_indices(2, 2)), expected)
        self.assertEqual(p_coefficient_index(0, 0, 2, 2), 2)
        self.assertEqual(p_coefficient_index(1, 1, 2, 2), 13)
        reference = make_p_reference_map(2, 2)
        self.assertEqual(reference.logical_shape, (4, 4))
        self.assertEqual(reference.logical_coefficient_count, 16)
        self.assertEqual(reference.nonzero_count, 4)
        self.assertEqual(reference.coefficient_at(0, (2,)), 1.0)
        self.assertEqual(reference.coefficient_at(0, (0,)), 0.0)

    def test_f_indices_encode_reverse_zero_and_copy_otherwise(self) -> None:
        # Shape [D,D,D], D=4. The first input coordinate is confined to
        # position zero. Selector coordinate 0 reverses, coordinate 1 copies.
        expected = (2, 4, 19, 21, 32, 38, 49, 55)
        self.assertEqual(tuple(iter_f_coefficient_indices(2, 2)), expected)
        self.assertEqual(f_coefficient_index(0, 0, 0, 2, 2), 2)
        self.assertEqual(f_coefficient_index(0, 0, 1, 2, 2), 4)
        reference = make_f_reference_map(2, 2)
        self.assertEqual(reference.logical_shape, (4, 4, 4))
        self.assertEqual(reference.logical_coefficient_count, 64)
        self.assertEqual(reference.nonzero_count, 8)
        self.assertEqual(reference.coefficient_at(0, (0, 2)), 1.0)
        self.assertEqual(reference.coefficient_at(0, (1, 0)), 1.0)
        self.assertEqual(reference.coefficient_at(0, (0, 0)), 0.0)

    def test_variant_specs_keep_all_policy_in_the_lab(self) -> None:
        f_spec = build_program_spec(Variant.F, 2, 3, seed=11)
        p_spec = build_program_spec(Variant.P, 2, 3, seed=11)
        t_spec = build_program_spec(Variant.T, 2, 3, seed=11)
        self.assertIsNotNone(f_spec)
        self.assertIsNotNone(p_spec)
        self.assertIsNotNone(t_spec)
        assert f_spec is not None
        assert p_spec is not None
        assert t_spec is not None

        dimension = 6
        self.assertEqual(f_spec.reference_map.input_dimensions, (dimension, dimension))
        self.assertEqual(f_spec.reference_map.nonzero_count, 3 * dimension)
        self.assertEqual(f_spec.input_projection_groups, (0, 0))
        self.assertTrue(f_spec.shares_input_projection)
        self.assertEqual(f_spec.initialization, CoefficientInitialization.COMPILED)
        self.assertFalse(f_spec.trainable)
        self.assertEqual(f_spec.attention_query_axis, 1)

        self.assertEqual(p_spec.reference_map.input_dimensions, (dimension,))
        self.assertEqual(p_spec.reference_map.nonzero_count, dimension)
        self.assertEqual(p_spec.input_projection_groups, (0,))
        self.assertFalse(p_spec.shares_input_projection)
        self.assertFalse(p_spec.trainable)

        self.assertEqual(t_spec.reference_map, f_spec.reference_map)
        self.assertEqual(
            t_spec.initialization,
            CoefficientInitialization.RANDOM_UNIFORM,
        )
        self.assertTrue(t_spec.trainable)
        self.assertEqual(t_spec.random_seed, 11)
        self.assertEqual(t_spec.input_projection_groups, (0, 0))
        self.assertIsNone(build_program_spec(Variant.I, 2, 3))

    def test_default_f_and_t_specs_remain_sparse_in_python(self) -> None:
        # Paper dimensions are L=15 and K=10, so D=150. Python retains 1,500
        # exact reference indices rather than allocating 3,375,000 floats.
        f_spec = build_program_spec(Variant.F, 15, 10)
        t_spec = build_program_spec(Variant.T, 15, 10)
        assert f_spec is not None
        assert t_spec is not None
        self.assertEqual(f_spec.dimension, 150)
        self.assertEqual(f_spec.reference_map.nonzero_count, 1_500)
        self.assertEqual(
            f_spec.reference_map.logical_coefficient_count,
            3_375_000,
        )
        self.assertEqual(len(t_spec.reference_map.nonzero_flat_indices), 1_500)

    def test_index_validation_rejects_invalid_coordinates(self) -> None:
        with self.assertRaisesRegex(ValueError, "position"):
            p_coefficient_index(2, 0, 2, 2)
        with self.assertRaisesRegex(ValueError, "symbol"):
            f_coefficient_index(0, 0, 2, 2, 2)
        with self.assertRaisesRegex(TypeError, "variant"):
            build_program_spec("F", 2, 2)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
