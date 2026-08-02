"""Generate and audit the Python-owned conditional-reversal protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Sequence, cast

from .protocol import (
    Example,
    ProtocolConfig,
    SplitSizes,
    evaluate,
    generate_disjoint_splits,
    predict_copy,
    predict_oracle,
    predict_reverse,
    split_fingerprint,
    verify_disjoint,
)


REPORT_FORMAT = "riftco-transformer.conditional-reverse-protocol.v1"


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate source-disjoint conditional-reversal splits and audit "
            "the oracle, copy-only, and reverse-only controls."
        )
    )
    parser.add_argument("--sequence-length", type=positive_integer, default=15)
    parser.add_argument("--alphabet", default="abcdefghijklmnopqrstuvwxyz")
    parser.add_argument("--reverse-when-first-is", default="aeiou")
    parser.add_argument("--delimiter", default="|")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=positive_integer, default=10_000)
    parser.add_argument("--probe", type=positive_integer, default=5_000)
    parser.add_argument("--validation", type=positive_integer, default=1_000)
    parser.add_argument("--test", type=positive_integer, default=1_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/conditional-reverse/protocol.json"),
        help="New JSON report path; an existing file is never overwritten.",
    )
    return parser


def build_report(
    config: ProtocolConfig,
    sizes: SplitSizes,
) -> dict[str, object]:
    splits = generate_disjoint_splits(config, sizes)
    verify_disjoint(splits)
    return {
        "format": REPORT_FORMAT,
        "ownership": {
            "protocol": "python_lab",
            "installed_framework_api": False,
            "learned_fpti_status": "not_implemented_in_current_python_lab",
        },
        "config": asdict(config),
        "split_sizes": asdict(sizes),
        "source_disjoint": True,
        "fingerprints": {
            name: split_fingerprint(examples)
            for name, examples in splits.items()
        },
        "validation_controls": _control_metrics(splits["validation"]),
        "test_controls": _control_metrics(splits["test"]),
    }


def _control_metrics(examples: Sequence[Example]) -> dict[str, object]:
    return {
        "conditional_oracle": asdict(evaluate(examples, predict_oracle)),
        "copy_only": asdict(evaluate(examples, predict_copy)),
        "reverse_only": asdict(evaluate(examples, predict_reverse)),
    }


def write_new_json(path: Path, report: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, allow_nan=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def main() -> int:
    arguments = build_parser().parse_args()
    config = ProtocolConfig(
        sequence_length=arguments.sequence_length,
        alphabet=arguments.alphabet,
        reverse_when_first_is=arguments.reverse_when_first_is,
        delimiter=arguments.delimiter,
        seed=arguments.seed,
    )
    sizes = SplitSizes(
        train=arguments.train,
        probe=arguments.probe,
        validation=arguments.validation,
        test=arguments.test,
    )
    try:
        report = build_report(config, sizes)
        write_new_json(arguments.output, report)
    except Exception as error:
        print(f"conditional-reversal protocol failed: {error}")
        return 1
    test_controls = cast(dict[str, object], report["test_controls"])
    oracle = cast(dict[str, object], test_controls["conditional_oracle"])
    print(f"report: {arguments.output}")
    print(
        "test conditional-oracle exact accuracy: "
        f"{100.0 * float(oracle['exact_sequence_accuracy']):.2f}%"
    )
    print(
        "F/P/T/I learned execution is intentionally absent until a public "
        "generic program-augmented model composition exists."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
