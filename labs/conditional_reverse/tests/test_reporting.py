from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.conditional_reverse.config import Variant, make_profile  # noqa: E402
from labs.conditional_reverse.protocol import generate_disjoint_splits  # noqa: E402
from labs.conditional_reverse.reporting import (  # noqa: E402
    LEARNED_REPORT_FORMAT,
    VariantReport,
    build_learned_report,
    collect_provenance,
    complete_provenance,
    write_new_json,
)


@dataclass(frozen=True)
class _History:
    final_step: int


class ConditionalReverseReportingTests(unittest.TestCase):
    def test_learned_report_serializes_explicit_final_step_and_provenance(self) -> None:
        config = make_profile("quick", variant=Variant.F, seed=7)
        splits = generate_disjoint_splits(config.protocol, config.split_sizes)
        variant = VariantReport(
            variant="F",
            backend="cpu",
            parameters={"total_numel": 10},
            training=_History(final_step=64),
            validation={"loss": 1.0},
            test={"loss": 0.5},
            test_hypotheses={"conditional": 1.0},
            probe_pca={"observation_count": 12},
            ablations={"conditions": []},
            steering=None,
            applicability={"selector_steering": True},
        )
        report = build_learned_report(
            config,
            splits,
            (variant,),
            {"git_commit": "abc", "framework_abi": {"major": 2}},
        )
        self.assertEqual(report["format"], LEARNED_REPORT_FORMAT)
        self.assertEqual(report["config"]["requested_variants"], ["F"])
        self.assertNotIn("variant", report["config"]["model"])
        self.assertEqual(report["variants"]["F"]["training"]["final_step"], 64)
        self.assertEqual(report["provenance"]["git_commit"], "abc")
        json.dumps(report, allow_nan=False)

    def test_writer_validates_before_create_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with self.assertRaisesRegex(ValueError, "non-finite"):
                write_new_json(output, {"bad": float("nan")})
            self.assertFalse(output.exists())
            write_new_json(output, {"ok": True})
            self.assertEqual(json.loads(output.read_text()), {"ok": True})
            with self.assertRaises(FileExistsError):
                write_new_json(output, {"ok": False})
            self.assertEqual(json.loads(output.read_text()), {"ok": True})

    def test_native_library_hash_and_run_stability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "native-library"
            payload = b"exact native artifact"
            library.write_bytes(payload)
            with patch.dict(
                "os.environ",
                {"RIFTCO_TRANSFORMER_LIBRARY": str(library)},
            ):
                started = collect_provenance(
                    repository_root=PROJECT_ROOT,
                    command=("conditional-reverse",),
                    framework_abi={"major": 2, "minor": 5},
                )
                completed = collect_provenance(
                    repository_root=PROJECT_ROOT,
                    command=("conditional-reverse",),
                    framework_abi={"major": 2, "minor": 5},
                )
        artifact = started["native_library"]
        self.assertEqual(artifact["size_bytes"], len(payload))
        self.assertEqual(artifact["sha256"], hashlib.sha256(payload).hexdigest())
        stable = complete_provenance(started, completed)
        self.assertEqual(stable["completed_at_utc"], completed["created_at_utc"])

        changed = dict(completed)
        changed["git_dirty"] = not bool(completed["git_dirty"])
        with self.assertRaisesRegex(RuntimeError, "changed during the run"):
            complete_provenance(started, changed)


if __name__ == "__main__":
    unittest.main()
