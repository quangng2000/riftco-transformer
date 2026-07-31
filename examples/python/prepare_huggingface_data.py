"""Prepare a small, reproducible Hugging Face dataset without dependencies."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from transformer_lab.data import (
    DATASET_PRESETS,
    HuggingFaceDatasetClient,
    SplitFractions,
    StableHashSplitter,
    prepare_huggingface_dataset,
)


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must not be negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download a bounded, revision-recorded dataset sample through "
            "the official Hugging Face datasets-server API."
        )
    )
    parser.add_argument(
        "--preset",
        choices=tuple(DATASET_PRESETS),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=positive_integer,
        required=True,
        help="number of source rows to sample",
    )
    parser.add_argument(
        "--offset",
        type=nonnegative_integer,
        default=0,
        help="start of the eligible source range (default: 0)",
    )
    parser.add_argument(
        "--source-split",
        help=(
            "override the preset source split, for example validation for "
            "TinyStories; the value is checked against /splits"
        ),
    )
    parser.add_argument(
        "--selection",
        choices=("seeded_pages", "sequential"),
        default="seeded_pages",
        help=(
            "sample efficient seeded pages across the eligible source range "
            "or read rows sequentially (default: seeded_pages)"
        ),
    )
    parser.add_argument(
        "--page-size",
        type=positive_integer,
        default=100,
        help="rows per API request; maximum 100 (default: 100)",
    )
    parser.add_argument("--seed", default="7")
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=nonnegative_integer, default=2)
    parser.add_argument("--retry-backoff", type=float, default=0.25)
    return parser


def main() -> int:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        fractions = SplitFractions(
            train=arguments.train_fraction,
            validation=arguments.validation_fraction,
            test=arguments.test_fraction,
        )
        splitter = StableHashSplitter(fractions, seed=arguments.seed)
        client = HuggingFaceDatasetClient(
            timeout_seconds=arguments.timeout,
            maximum_retries=arguments.retries,
            retry_backoff_seconds=arguments.retry_backoff,
            token=os.environ.get("HF_TOKEN"),
        )
        prepared = prepare_huggingface_dataset(
            DATASET_PRESETS[arguments.preset],
            arguments.output,
            client=client,
            splitter=splitter,
            offset=arguments.offset,
            limit=arguments.limit,
            page_size=arguments.page_size,
            selection=arguments.selection,
            source_split=arguments.source_split,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

    counts = prepared.manifest["counts"]
    print(f"Prepared {arguments.preset!r} at {prepared.directory}")
    print(
        "Records: "
        f"train={counts['train']}, "
        f"validation={counts['validation']}, "
        f"test={counts['test']}, "
        f"duplicates_removed={counts['duplicates_removed']}"
    )
    print(f"Manifest: {prepared.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
