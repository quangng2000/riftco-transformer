from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.conditional_reverse.config import Variant  # noqa: E402
from labs.conditional_reverse.run import main, parse_variants  # noqa: E402
import labs.conditional_reverse.run as run_module  # noqa: E402


class ConditionalReverseRunTests(unittest.TestCase):
    def test_variant_parser_accepts_all_or_unique_subsets(self) -> None:
        self.assertEqual(parse_variants("all"), tuple(Variant))
        self.assertEqual(parse_variants("f,t"), (Variant.F, Variant.T))
        with self.assertRaisesRegex(Exception, "unique"):
            parse_variants("F,F")
        with self.assertRaisesRegex(Exception, "subset"):
            parse_variants("F,X")

    def test_protocol_only_cli_does_not_import_framework(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "protocol.json"
            program = """
import json
import sys
from labs.conditional_reverse.run import main
result = main([
    "--protocol-only",
    "--profile", "quick",
    "--output", sys.argv[1],
])
print("RESULT=" + str(result))
print("NATIVE=" + json.dumps(sorted(
    name for name in sys.modules
    if name == "riftco_transformer" or name.startswith("riftco_transformer.")
)))
"""
            completed = subprocess.run(
                [sys.executable, "-c", program, str(output)],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("RESULT=0", completed.stdout)
            self.assertIn("NATIVE=[]", completed.stdout)
            report = json.loads(output.read_text())
            self.assertEqual(report["split_sizes"]["train"], 128)
            self.assertEqual(report["split_sizes"]["test"], 64)

    def test_existing_output_fails_before_variant_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.json"
            output.write_text("original\n")
            with patch.object(
                run_module,
                "execute_variant",
                side_effect=AssertionError("training must not start"),
            ) as execute:
                result = main(("--profile", "quick", "--output", str(output)))
            self.assertEqual(result, 1)
            execute.assert_not_called()
            self.assertEqual(output.read_text(), "original\n")


if __name__ == "__main__":
    unittest.main()
