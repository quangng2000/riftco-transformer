from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_SCRIPT = PROJECT_ROOT / "examples" / "python" / "train_tiny.py"

script_specification = importlib.util.spec_from_file_location(
    "transformer_lab_train_tiny",
    TRAINING_SCRIPT,
)
if script_specification is None or script_specification.loader is None:
    raise RuntimeError(f"could not load {TRAINING_SCRIPT}")
train_tiny = importlib.util.module_from_spec(script_specification)
script_specification.loader.exec_module(train_tiny)


def fields(line: str) -> dict[str, str]:
    return dict(
        field.split("=", maxsplit=1)
        for field in line.split()
        if "=" in field
    )


class TrainTinyTests(unittest.TestCase):
    def test_split_reserves_a_disjoint_contiguous_tail(self) -> None:
        corpus = list(range(20))

        training, validation = train_tiny.split_token_corpus(
            corpus,
            context_size=3,
            validation_fraction=0.25,
        )

        self.assertEqual(training, list(range(15)))
        self.assertEqual(validation, list(range(15, 20)))
        self.assertEqual(training + validation, corpus)

    def test_split_enforces_valid_arguments_and_region_sizes(self) -> None:
        for invalid_fraction in (
            -0.1,
            0.0,
            1.0,
            math.inf,
            math.nan,
        ):
            with self.subTest(validation_fraction=invalid_fraction):
                with self.assertRaisesRegex(
                    ValueError,
                    "validation fraction",
                ):
                    train_tiny.split_token_corpus(
                        list(range(20)),
                        context_size=3,
                        validation_fraction=invalid_fraction,
                    )

        with self.assertRaisesRegex(ValueError, "context size"):
            train_tiny.split_token_corpus(
                list(range(20)),
                context_size=0,
                validation_fraction=0.2,
            )

        with self.assertRaisesRegex(
            ValueError,
            "training and validation",
        ):
            train_tiny.split_token_corpus(
                list(range(7)),
                context_size=3,
                validation_fraction=0.25,
            )

        training, validation = train_tiny.split_token_corpus(
            list(range(8)),
            context_size=3,
            validation_fraction=0.25,
        )
        self.assertEqual(len(training), 4)
        self.assertEqual(len(validation), 4)

        with self.assertRaisesRegex(
            ValueError,
            "training and validation",
        ):
            train_tiny.split_token_corpus(
                list(range(20)),
                context_size=3,
                validation_fraction=0.9,
            )

    def test_validation_batches_are_fixed_and_shifted(self) -> None:
        validation_tokens = list(range(20))

        first = train_tiny.fixed_validation_batches(
            validation_tokens,
            batch_size=2,
            context_size=3,
            batch_count=3,
        )
        second = train_tiny.fixed_validation_batches(
            validation_tokens,
            batch_size=2,
            context_size=3,
            batch_count=3,
        )

        self.assertEqual(first, second)
        for inputs, targets in first:
            self.assertEqual(len(inputs), 2)
            self.assertEqual(len(targets), 2)
            for input_row, target_row in zip(inputs, targets):
                self.assertEqual(input_row[1:], target_row[:-1])

        with self.assertRaisesRegex(
            ValueError,
            "validation batch count",
        ):
            train_tiny.fixed_validation_batches(
                validation_tokens,
                batch_size=2,
                context_size=3,
                batch_count=0,
            )

    def test_cli_reports_moving_average_and_fixed_validation_loss(
        self,
    ) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "python")
        corpus = (
            "one fish two fish red fish blue fish.\n"
            "small models learn one token at a time.\n"
        ) * 8
        result = subprocess.run(
            [
                sys.executable,
                str(TRAINING_SCRIPT),
                "--backend",
                "cpu",
                "--steps",
                "3",
                "--context",
                "4",
                "--batch-size",
                "2",
                "--validation-fraction",
                "0.2",
                "--validation-batches",
                "2",
                "--eval-every",
                "2",
                "--loss-average-window",
                "2",
                "--tokenizer",
                "byte",
                "--text",
                corpus,
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        lines = result.stdout.splitlines()
        self.assertIn("training_tokens=", lines[0])
        self.assertIn("validation_tokens=", lines[0])

        training_lines = [
            line for line in lines if line.startswith("step=")
        ]
        validation_lines = [
            line for line in lines if line.startswith("validation ")
        ]
        self.assertEqual(len(training_lines), 3)
        self.assertEqual(len(validation_lines), 3)
        self.assertEqual(
            [fields(line)["step"] for line in validation_lines],
            ["0", "2", "3"],
        )

        first_training = fields(training_lines[0])
        second_training = fields(training_lines[1])
        first_loss = float(first_training["train_loss"])
        second_loss = float(second_training["train_loss"])
        first_average = float(
            first_training["train_loss_average"]
        )
        second_average = float(
            second_training["train_loss_average"]
        )
        third_training = fields(training_lines[2])
        third_loss = float(third_training["train_loss"])
        third_average = float(
            third_training["train_loss_average"]
        )
        self.assertAlmostEqual(first_average, first_loss, places=5)
        self.assertAlmostEqual(
            second_average,
            (first_loss + second_loss) / 2.0,
            places=5,
        )
        self.assertAlmostEqual(
            third_average,
            (second_loss + third_loss) / 2.0,
            places=5,
        )

        for line in validation_lines:
            values = fields(line)
            self.assertTrue(
                math.isfinite(float(values["validation_loss"]))
            )
            self.assertTrue(math.isfinite(float(values["perplexity"])))

    def test_cli_rejects_invalid_reporting_options(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "python")
        for option in ("--eval-every", "--loss-average-window"):
            with self.subTest(option=option):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(TRAINING_SCRIPT),
                        option,
                        "0",
                    ],
                    cwd=PROJECT_ROOT,
                    env=environment,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"{option} must be greater than zero",
                    result.stderr,
                )

    def test_cli_default_bpe_path_reports_validation(self) -> None:
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "python")
        corpus = (
            "hello transformer! tiny models learn from text. "
            "hello tokenizer! validation checks unseen "
            "model-training text."
        )
        result = subprocess.run(
            [
                sys.executable,
                str(TRAINING_SCRIPT),
                "--backend",
                "cpu",
                "--steps",
                "1",
                "--context",
                "4",
                "--text",
                corpus,
            ],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )

        lines = result.stdout.splitlines()
        self.assertIn("tokenizer=bpe", lines[0])
        validation_lines = [
            line for line in lines if line.startswith("validation ")
        ]
        self.assertEqual(
            [fields(line)["step"] for line in validation_lines],
            ["0", "1"],
        )


if __name__ == "__main__":
    unittest.main()
