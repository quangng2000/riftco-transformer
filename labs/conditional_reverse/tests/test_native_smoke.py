from __future__ import annotations

from dataclasses import replace
import math
import os
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "python"))
sys.path.insert(0, str(PROJECT_ROOT))


@unittest.skipUnless(
    os.environ.get("RIFTCO_TRANSFORMER_LIBRARY"),
    "native quick smoke requires RIFTCO_TRANSFORMER_LIBRARY",
)
class ConditionalReverseNativeSmokeTests(unittest.TestCase):
    def test_quick_f_runs_all_64_adam_steps(self) -> None:
        from labs.conditional_reverse.config import Variant, make_profile
        from labs.conditional_reverse.model import resolve_backend
        from labs.conditional_reverse.protocol import generate_disjoint_splits
        from labs.conditional_reverse.run import execute_variant

        backend = resolve_backend("auto")
        config = make_profile(
            "quick",
            variant=Variant.F,
            seed=42,
            backend=backend,
        )
        config = replace(config, model=replace(config.model, backend=backend))
        splits = generate_disjoint_splits(config.protocol, config.split_sizes)
        result = execute_variant(config, splits)
        self.assertEqual(result.training.final_step, 64)
        self.assertEqual(result.training.epochs[-1].epoch, 8)
        self.assertTrue(math.isfinite(result.test.metrics.loss))
        self.assertEqual(result.probe_pca.representation_name, "program.output.raw")
        self.assertEqual(
            result.probe_pca.pca.observation_count,
            config.split_sizes.probe * config.protocol.sequence_length,
        )


if __name__ == "__main__":
    unittest.main()
