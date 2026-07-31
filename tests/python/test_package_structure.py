from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

import transformer_lab
import transformer_lab.artifacts
import transformer_lab.data
import transformer_lab.experiments
import transformer_lab.native
import transformer_lab.post_training
import transformer_lab.pretraining
import transformer_lab.serving
import transformer_lab.training
from transformer_lab import Tensor
from transformer_lab.artifact import ModelBundle as LegacyModelBundle
from transformer_lab.artifacts import ModelBundle
from transformer_lab.data import HuggingFaceDatasetClient
from transformer_lab.experiments import LoraRankExperimentConfig
from transformer_lab.generation import TextGenerator as LegacyTextGenerator
from transformer_lab.native import Tensor as NativeTensor
from transformer_lab.post_training import PostTrainingConfig
from transformer_lab.pretraining import PretrainingConfig
from transformer_lab.serving import (
    ModelService,
    TextGenerator,
    create_http_server,
)
from transformer_lab.training import CausalLanguageModelTrainer


class PackageStructureTests(unittest.TestCase):
    def test_public_types_are_owned_by_responsibility_packages(self) -> None:
        expected_modules = {
            ModelBundle: "transformer_lab.artifacts.bundle",
            HuggingFaceDatasetClient: "transformer_lab.data.client",
            LoraRankExperimentConfig: (
                "transformer_lab.experiments.lora_rank"
            ),
            CausalLanguageModelTrainer: "transformer_lab.training.engine",
            PretrainingConfig: "transformer_lab.pretraining.pipeline",
            PostTrainingConfig: "transformer_lab.post_training.pipeline",
            TextGenerator: "transformer_lab.serving.generation",
            ModelService: "transformer_lab.serving.service",
            create_http_server: "transformer_lab.serving.http",
            NativeTensor: "transformer_lab.native.bindings",
        }
        for value, expected_module in expected_modules.items():
            with self.subTest(value=value.__name__):
                self.assertEqual(value.__module__, expected_module)

    def test_compatibility_facades_reexport_identical_objects(self) -> None:
        self.assertIs(Tensor, NativeTensor)
        self.assertIs(LegacyModelBundle, ModelBundle)
        self.assertIs(LegacyTextGenerator, TextGenerator)

    def test_stage_names_are_physical_packages(self) -> None:
        packages = (
            transformer_lab.native,
            transformer_lab.artifacts,
            transformer_lab.data,
            transformer_lab.experiments,
            transformer_lab.training,
            transformer_lab.pretraining,
            transformer_lab.post_training,
            transformer_lab.serving,
        )
        for package in packages:
            with self.subTest(package=package.__name__):
                package_paths = tuple(Path(path) for path in package.__path__)
                self.assertEqual(len(package_paths), 1)
                self.assertTrue(package_paths[0].is_dir())
                self.assertTrue((package_paths[0] / "__init__.py").is_file())

    def test_serving_import_does_not_load_training_stages(self) -> None:
        program = """
import json
import sys
import transformer_lab.serving

print(json.dumps(sorted(
    name
    for name in sys.modules
    if name in {
        "transformer_lab.training",
        "transformer_lab.pretraining",
        "transformer_lab.post_training",
    }
)))
"""
        completed = subprocess.run(
            [sys.executable, "-c", program],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), [])


if __name__ == "__main__":
    unittest.main()
