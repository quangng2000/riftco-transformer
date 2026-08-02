from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT))

from labs.conditional_reverse.config import (  # noqa: E402
    ModelConfig,
    Variant,
)
from labs.conditional_reverse.model import (  # noqa: E402
    build_model,
    resolve_backend,
    translate_program_spec,
)
from labs.conditional_reverse.programs import build_program_spec  # noqa: E402
from labs.conditional_reverse.protocol import ProtocolConfig  # noqa: E402
import labs.conditional_reverse.model as model_module  # noqa: E402


class _FakeMap:
    calls: list[tuple[tuple[int, ...], int, tuple[int, ...], tuple[float, ...]]] = []

    @classmethod
    def from_sparse(cls, dimensions, output, indices, values):
        instance = cls()
        instance.closed = False
        cls.calls.append(
            (tuple(dimensions), output, tuple(indices), tuple(values))
        )
        return instance

    def close(self) -> None:
        self.closed = True


class _KeywordRecord:
    def __init__(self, **values) -> None:
        self.__dict__.update(values)


class _FakeModel:
    def __init__(self, config, program=None) -> None:
        self.config = config
        self.program = program
        self.backend = "cpu"
        self.closed = False
        self.has_program = program is not None

    def to(self, backend: str):
        self.backend = backend
        return self

    def close(self) -> None:
        self.closed = True


class ConditionalReverseModelTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeMap.calls.clear()

    def test_sparse_specs_translate_to_public_generic_objects(self) -> None:
        with patch.multiple(
            model_module,
            MultilinearMap=_FakeMap,
            NeuralLoweringConfig=_KeywordRecord,
            NativeProgramInputLayout=_KeywordRecord,
            ProgramBranch=_KeywordRecord,
        ):
            for variant, strategy, initialization, trainable, groups in (
                (Variant.F, "linear_attention", "compiled", False, (0, 0)),
                (Variant.P, "linear", "compiled", False, (0,)),
                (Variant.T, "linear_attention", "random_uniform", True, (0, 0)),
            ):
                with self.subTest(variant=variant):
                    spec = build_program_spec(variant, 2, 3, seed=19)
                    assert spec is not None
                    native = translate_program_spec(spec)
                    call = _FakeMap.calls[-1]
                    self.assertEqual(call[0], spec.reference_map.input_dimensions)
                    self.assertEqual(call[1], spec.reference_map.output_dimension)
                    self.assertEqual(call[2], spec.reference_map.nonzero_flat_indices)
                    self.assertEqual(call[3], (1.0,) * spec.reference_map.nonzero_count)
                    self.assertEqual(native.branch.lowering.strategy, strategy)
                    self.assertEqual(
                        native.branch.lowering.initialization,
                        initialization,
                    )
                    self.assertEqual(native.branch.lowering.trainable, trainable)
                    self.assertEqual(
                        tuple(item.projection_group for item in native.branch.inputs),
                        groups,
                    )
                    self.assertFalse(native.branch.input_projection_bias)
                    self.assertFalse(native.branch.merge_bias)
                    native.close()
                    self.assertTrue(native.map.closed)

    def test_i_builds_no_map_or_program_branch(self) -> None:
        protocol = ProtocolConfig(
            sequence_length=3,
            alphabet="abcd",
            reverse_when_first_is="a",
        )
        config = ModelConfig(
            variant=Variant.I,
            model_width=8,
            head_count=2,
            feed_forward_width=32,
        )
        with patch.multiple(
            model_module,
            ProgramAugmentedModelConfig=_KeywordRecord,
            ProgramAugmentedModel=_FakeModel,
        ), patch.object(
            model_module,
            "translate_program_spec",
            side_effect=AssertionError("I must not translate a map"),
        ):
            runtime = build_model(protocol, config)
            self.assertIsNone(runtime.program)
            self.assertIsNone(runtime.program_spec)
            self.assertFalse(runtime.has_program)
            self.assertIsNone(runtime.model.program)
            runtime.close()
            self.assertTrue(runtime.model.closed)

    def test_backend_resolution_is_explicit(self) -> None:
        with patch.object(
            model_module,
            "backend_available",
            side_effect=lambda name: name in {"cpu", "metal"},
        ):
            self.assertEqual(resolve_backend("auto"), "metal")
            self.assertEqual(resolve_backend("cpu"), "cpu")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                resolve_backend("cuda")


if __name__ == "__main__":
    unittest.main()
