from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from labs.conditional_reverse.protocol import (  # noqa: E402
    ProtocolConfig,
    SplitSizes,
    evaluate,
    generate_disjoint_splits,
    make_example,
    predict_copy,
    predict_oracle,
    verify_disjoint,
)
from labs.conditional_reverse.run import (  # noqa: E402
    REPORT_FORMAT,
    build_report,
    write_new_json,
)


class ConditionalReverseProtocolTests(unittest.TestCase):
    def test_reverse_and_copy_semantics(self) -> None:
        config = ProtocolConfig(
            sequence_length=3,
            alphabet="abcd",
            reverse_when_first_is="a",
        )
        reverse = make_example("abc", config)
        copy = make_example("bca", config)
        self.assertEqual(reverse.target, "cba")
        self.assertTrue(reverse.reversed)
        self.assertEqual(copy.target, "bca")
        self.assertFalse(copy.reversed)
        self.assertEqual(reverse.tokens, "abc|cba")
        self.assertEqual(reverse.inputs, reverse.tokens[:-1])
        self.assertEqual(reverse.targets, reverse.tokens[1:])

    def test_splits_are_deterministic_and_source_disjoint(self) -> None:
        config = ProtocolConfig(
            sequence_length=4,
            alphabet="abcd",
            reverse_when_first_is="a",
            seed=7,
        )
        sizes = SplitSizes(train=20, probe=10, validation=8, test=8)
        left = generate_disjoint_splits(config, sizes)
        right = generate_disjoint_splits(config, sizes)
        self.assertEqual(left, right)
        verify_disjoint(left)
        sources = [
            example.source
            for examples in left.values()
            for example in examples
        ]
        self.assertEqual(len(sources), len(set(sources)))

    def test_split_request_cannot_exceed_source_space(self) -> None:
        config = ProtocolConfig(
            sequence_length=2,
            alphabet="ab",
            reverse_when_first_is="a",
        )
        with self.assertRaisesRegex(ValueError, "source space"):
            generate_disjoint_splits(
                config,
                SplitSizes(train=2, probe=1, validation=1, test=1),
            )

    def test_branch_metrics_expose_copy_shortcut(self) -> None:
        config = ProtocolConfig(
            sequence_length=3,
            alphabet="abcd",
            reverse_when_first_is="a",
        )
        examples = (
            make_example("abc", config),
            make_example("bca", config),
        )
        oracle = evaluate(examples, predict_oracle)
        copy = evaluate(examples, predict_copy)
        self.assertEqual(oracle.exact_sequence_accuracy, 1.0)
        self.assertEqual(oracle.reverse_target_token_accuracy, 1.0)
        self.assertEqual(oracle.copy_target_token_accuracy, 1.0)
        self.assertEqual(copy.copy_target_token_accuracy, 1.0)
        self.assertLess(copy.reverse_target_token_accuracy, 1.0)

    def test_protocol_report_points_to_the_separate_learned_path(self) -> None:
        report = build_report(
            ProtocolConfig(
                sequence_length=4,
                alphabet="abcd",
                reverse_when_first_is="a",
            ),
            SplitSizes(train=20, probe=10, validation=8, test=8),
        )
        self.assertEqual(report["format"], REPORT_FORMAT)
        self.assertTrue(report["source_disjoint"])
        self.assertEqual(
            report["ownership"]["learned_fpti_status"],
            "available_via_learned_cli",
        )

    def test_report_writer_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "report.json"
            write_new_json(output, {"ok": True})
            self.assertEqual(json.loads(output.read_text()), {"ok": True})
            with self.assertRaises(FileExistsError):
                write_new_json(output, {"ok": False})


if __name__ == "__main__":
    unittest.main()
