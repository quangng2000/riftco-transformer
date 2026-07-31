"""Stage 1: pretrain a tiny base model and save a portable artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from riftco_transformer.pretraining import (
    PretrainingConfig,
    pretrain_file,
    pretrain_files,
)
from riftco_transformer.training import TrainingMetric


DEFAULT_CORPUS = Path("data/pretraining/tiny_corpus.txt")
DEFAULT_OUTPUT = Path("results/stages/tiny_pretrained.rift")


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def fraction(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if not 0.0 < result < 1.0:
        raise argparse.ArgumentTypeError("must be between zero and one")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 1: self-supervised next-token pretraining from UTF-8 text."
        )
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_CORPUS,
        help=f"UTF-8 training corpus (default: {DEFAULT_CORPUS}).",
    )
    parser.add_argument(
        "--validation-file",
        type=Path,
        default=None,
        help=(
            "Optional explicit UTF-8 validation corpus. When omitted, "
            "--validation-fraction is held out from --file."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination .rift artifact (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "metal"),
        default="auto",
        help="Execution backend; auto prefers Metal when available.",
    )
    parser.add_argument(
        "--attention",
        choices=("materialized", "flash"),
        default="materialized",
        help="Full-sequence attention implementation (default: materialized).",
    )
    parser.add_argument(
        "--activation-checkpointing",
        choices=("disabled", "block"),
        default="disabled",
        help=(
            "Activation retention policy; block recomputes transformer "
            "blocks during backward (default: disabled)."
        ),
    )
    parser.add_argument(
        "--steps",
        type=positive_integer,
        default=10,
        help="Optimizer steps (default: 10).",
    )
    parser.add_argument(
        "--context",
        type=positive_integer,
        default=8,
        help="Tokens per causal training window (default: 8).",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=2,
        help="Random windows per optimizer step (default: 2).",
    )
    parser.add_argument(
        "--tokenizer",
        choices=("byte", "bpe"),
        default="bpe",
        help="Tokenizer fitted on the training split (default: bpe).",
    )
    parser.add_argument(
        "--vocab-size",
        type=positive_integer,
        default=272,
        help="Maximum BPE vocabulary size (default: 272).",
    )
    parser.add_argument(
        "--min-pair-frequency",
        type=positive_integer,
        default=2,
        help="Minimum frequency for a BPE merge (default: 2).",
    )
    parser.add_argument(
        "--model-width",
        type=positive_integer,
        default=16,
        help="Transformer hidden width (default: 16).",
    )
    parser.add_argument(
        "--heads",
        type=positive_integer,
        default=4,
        help="Attention head count (default: 4).",
    )
    parser.add_argument(
        "--blocks",
        type=positive_integer,
        default=1,
        help="Transformer block count (default: 1).",
    )
    parser.add_argument(
        "--feed-forward-width",
        type=positive_integer,
        default=32,
        help="Feed-forward hidden width (default: 32).",
    )
    parser.add_argument(
        "--validation-fraction",
        type=fraction,
        default=0.1,
        help="Raw-text fraction held out for validation (default: 0.1).",
    )
    parser.add_argument(
        "--validation-batches",
        type=positive_integer,
        default=2,
        help="Fixed validation batches per evaluation (default: 2).",
    )
    parser.add_argument(
        "--eval-every",
        type=positive_integer,
        default=5,
        help="Evaluate every N optimizer steps (default: 5).",
    )
    return parser


def print_metric(metric: TrainingMetric) -> None:
    if metric.phase == "train":
        print(
            "[pretraining] "
            f"step={metric.step} "
            f"loss={metric.loss:.6f} "
            f"loss_average={metric.loss_average:.6f} "
            f"gradient_norm={metric.gradient_norm:.6f}"
        )
        return
    print(
        "[pretraining] "
        f"step={metric.step} "
        f"validation_loss={metric.loss:.6f} "
        f"perplexity={metric.perplexity:.6f}"
    )


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        config = PretrainingConfig(
            steps=arguments.steps,
            context_size=arguments.context,
            batch_size=arguments.batch_size,
            validation_fraction=arguments.validation_fraction,
            validation_batch_count=arguments.validation_batches,
            evaluation_interval=arguments.eval_every,
            loss_average_window=min(10, arguments.steps),
            tokenizer_method=arguments.tokenizer,
            vocabulary_size=arguments.vocab_size,
            minimum_pair_frequency=arguments.min_pair_frequency,
            model_width=arguments.model_width,
            head_count=arguments.heads,
            block_count=arguments.blocks,
            feed_forward_width=arguments.feed_forward_width,
            backend=arguments.backend,
            attention=arguments.attention,
            activation_checkpointing=arguments.activation_checkpointing,
        )
        if arguments.validation_file is None:
            result = pretrain_file(
                arguments.file,
                config,
                metric_sink=print_metric,
            )
        else:
            result = pretrain_files(
                arguments.file,
                arguments.validation_file,
                config,
                metric_sink=print_metric,
            )
        destination = result.bundle.save(arguments.output)
    except Exception as error:
        print(f"pretraining failed: {error}", file=sys.stderr)
        return 1

    print(
        "[pretraining] "
        f"saved={destination} "
        f"stage={result.bundle.stage} "
        f"artifact_id={result.bundle.artifact_id} "
        f"training_tokens={result.training_token_count} "
        f"validation_tokens={result.validation_token_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
