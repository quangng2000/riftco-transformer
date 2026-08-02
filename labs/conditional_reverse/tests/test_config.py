from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.conditional_reverse.config import (  # noqa: E402
    ModelConfig,
    Profile,
    TrainingConfig,
    Variant,
    make_profile,
)


class ConditionalReverseConfigTests(unittest.TestCase):
    def test_variants_are_lab_owned_and_program_presence_is_explicit(self) -> None:
        self.assertEqual(
            tuple(variant.value for variant in Variant),
            ("F", "P", "T", "I"),
        )
        self.assertTrue(Variant.F.has_program)
        self.assertTrue(Variant.P.has_program)
        self.assertTrue(Variant.T.has_program)
        self.assertFalse(Variant.I.has_program)

    def test_paper_profile_preserves_historical_dimensions(self) -> None:
        config = make_profile(
            Profile.PAPER,
            variant=Variant.T,
            seed=17,
            backend="METAL",
        )
        self.assertEqual(config.profile, Profile.PAPER)
        self.assertEqual(config.protocol.sequence_length, 15)
        self.assertEqual(config.split_sizes.train, 10_000)
        self.assertEqual(config.split_sizes.probe, 5_000)
        self.assertEqual(config.model.variant, Variant.T)
        self.assertEqual(config.model.model_width, 20)
        self.assertEqual(config.model.head_count, 2)
        self.assertEqual(config.model.parallel_attention_count, 2)
        self.assertEqual(config.model.feed_forward_width, 80)
        self.assertEqual(config.model.program_width, 10)
        self.assertEqual(config.model.backend, "metal")
        self.assertEqual(config.model.seed, 17)
        self.assertEqual(config.training.epochs, 10)
        self.assertEqual(config.training.batch_size, 128)
        self.assertEqual(config.training.learning_rate, 0.01)
        self.assertEqual(
            config.training.maximum_gradient_norm,
            3.4028234663852886e38,
        )

    def test_quick_profile_is_bounded_and_reproducible(self) -> None:
        left = make_profile("quick", variant=Variant.I, seed=9)
        right = make_profile(Profile.QUICK, variant=Variant.I, seed=9)
        self.assertEqual(left, right)
        self.assertEqual(left.protocol.sequence_length, 3)
        self.assertEqual(left.protocol.alphabet, "abcdefghij")
        self.assertEqual(left.protocol.reverse_when_first_is, "ae")
        self.assertEqual(left.split_sizes.total, 320)
        self.assertEqual(left.split_sizes.validation, 64)
        self.assertEqual(left.split_sizes.test, 64)
        self.assertEqual(left.training.epochs, 8)
        self.assertEqual(left.training.maximum_steps, 64)
        self.assertEqual(left.training.evaluation_batch_size, 64)
        self.assertEqual(left.analysis.pca_components, 2)
        self.assertEqual(left.model.program_width, 4)
        self.assertEqual(left.model.feed_forward_width, 24)

    def test_model_and_training_validation_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(model_width=7, head_count=2)
        with self.assertRaisesRegex(ValueError, "backend"):
            ModelConfig(backend="browser")
        with self.assertRaisesRegex(TypeError, "variant"):
            ModelConfig(variant="F")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "maximum_steps"):
            TrainingConfig(maximum_steps=0)
        with self.assertRaisesRegex(ValueError, "beta1"):
            TrainingConfig(beta1=1.0)
        with self.assertRaises(FrozenInstanceError):
            ModelConfig().model_width = 40  # type: ignore[misc]

    def test_importing_lab_does_not_load_framework_or_native_bindings(self) -> None:
        program = """
import json
import sys
import labs.conditional_reverse
print(json.dumps(sorted(
    name for name in sys.modules
    if name == "riftco_transformer" or name.startswith("riftco_transformer.")
)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[]")


if __name__ == "__main__":
    unittest.main()
