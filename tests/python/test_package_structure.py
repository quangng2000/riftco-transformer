from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

import riftco_transformer
import riftco_transformer.artifacts
import riftco_transformer.data
import riftco_transformer.experiments
import riftco_transformer.native
import riftco_transformer.post_training
import riftco_transformer.pretraining
import riftco_transformer.serving
import riftco_transformer.training
from riftco_transformer import Tensor
from riftco_transformer.artifacts import ModelBundle
from riftco_transformer.data import HuggingFaceDatasetClient
from riftco_transformer.experiments import LoraRankExperimentConfig
from riftco_transformer.native import Tensor as NativeTensor
from riftco_transformer.post_training import PostTrainingConfig
from riftco_transformer.pretraining import PretrainingConfig
from riftco_transformer.serving import (
    ModelService,
    TextGenerator,
    create_http_server,
)
from riftco_transformer.training import CausalLanguageModelTrainer


class PackageStructureTests(unittest.TestCase):
    def test_public_types_are_owned_by_responsibility_packages(self) -> None:
        expected_modules = {
            ModelBundle: "riftco_transformer.artifacts.bundle",
            HuggingFaceDatasetClient: "riftco_transformer.data.client",
            LoraRankExperimentConfig: (
                "riftco_transformer.experiments.lora_rank"
            ),
            CausalLanguageModelTrainer: "riftco_transformer.training.engine",
            PretrainingConfig: "riftco_transformer.pretraining.pipeline",
            PostTrainingConfig: "riftco_transformer.post_training.pipeline",
            TextGenerator: "riftco_transformer.serving.generation",
            ModelService: "riftco_transformer.serving.service",
            create_http_server: "riftco_transformer.serving.http",
            NativeTensor: "riftco_transformer.native.bindings",
        }
        for value, expected_module in expected_modules.items():
            with self.subTest(value=value.__name__):
                self.assertEqual(value.__module__, expected_module)

    def test_root_reexports_native_objects(self) -> None:
        self.assertIs(Tensor, NativeTensor)

    def test_stage_names_are_physical_packages(self) -> None:
        packages = (
            riftco_transformer.native,
            riftco_transformer.artifacts,
            riftco_transformer.data,
            riftco_transformer.experiments,
            riftco_transformer.training,
            riftco_transformer.pretraining,
            riftco_transformer.post_training,
            riftco_transformer.serving,
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
import riftco_transformer.serving

print(json.dumps(sorted(
    name
    for name in sys.modules
    if name in {
        "riftco_transformer.training",
        "riftco_transformer.pretraining",
        "riftco_transformer.post_training",
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
