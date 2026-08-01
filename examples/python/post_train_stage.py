"""Stage 2: post-train a base artifact on prompt/response examples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from riftco_transformer import LoraConfig
from riftco_transformer.artifacts import ModelBundle
from riftco_transformer.post_training import (
    PostTrainingConfig,
    post_train_jsonl,
)
from riftco_transformer.training import TrainingMetric


DEFAULT_BASE = Path("results/stages/tiny_pretrained.rift")
DEFAULT_INSTRUCTIONS = Path(
    "data/post_training/tiny_instructions.jsonl"
)
DEFAULT_OUTPUT = Path("results/stages/tiny_post_trained.rift")


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def positive_float(value: str) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a number") from error
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: supervised post-training from instruction JSONL."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=DEFAULT_BASE,
        help=f"Pretrained .rift artifact (default: {DEFAULT_BASE}).",
    )
    parser.add_argument(
        "--instructions",
        type=Path,
        default=DEFAULT_INSTRUCTIONS,
        help=(
            "UTF-8 prompt/response JSONL "
            f"(default: {DEFAULT_INSTRUCTIONS})."
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
        choices=("auto", "cpu", "metal", "cuda", "tpu"),
        default="auto",
        help=(
            "Execution backend; auto prefers TPU, then CUDA, Metal, and CPU."
        ),
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
        default=5,
        help="Supervised optimizer steps (default: 5).",
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
        help="Instruction windows per optimizer step (default: 2).",
    )
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=1.0e-3,
        help="Adam learning rate (default: 0.001).",
    )
    parser.add_argument(
        "--fine-tuning-method",
        choices=("full", "lora"),
        default="full",
        help=(
            "Optimize every base parameter or LoRA adapters only "
            "(default: full)."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("example_uniform", "window_uniform"),
        default="example_uniform",
        help=(
            "Sample instruction examples uniformly or weight them by their "
            "number of token windows (default: example_uniform)."
        ),
    )
    parser.add_argument(
        "--lora-rank",
        type=positive_integer,
        default=4,
        help="LoRA rank when --fine-tuning-method=lora (default: 4).",
    )
    parser.add_argument(
        "--lora-alpha",
        type=positive_float,
        default=8.0,
        help="LoRA scaling alpha when LoRA is selected (default: 8).",
    )
    parser.add_argument(
        "--eval-every",
        type=positive_integer,
        default=5,
        help="Metric reporting interval in steps (default: 5).",
    )
    return parser


def print_metric(metric: TrainingMetric) -> None:
    print(
        "[post-training] "
        f"step={metric.step} "
        f"loss={metric.loss:.6f} "
        f"loss_average={metric.loss_average:.6f} "
        f"gradient_norm={metric.gradient_norm:.6f}"
    )


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        base_bundle = ModelBundle.load(arguments.base)
        print(
            "[post-training] "
            f"loaded={arguments.base} "
            f"stage={base_bundle.stage} "
            f"artifact_id={base_bundle.artifact_id}"
        )
        result = post_train_jsonl(
            base_bundle,
            arguments.instructions,
            PostTrainingConfig(
                steps=arguments.steps,
                context_size=arguments.context,
                batch_size=arguments.batch_size,
                evaluation_interval=arguments.eval_every,
                loss_average_window=min(10, arguments.steps),
                learning_rate=arguments.learning_rate,
                backend=arguments.backend,
                attention=arguments.attention,
                activation_checkpointing=(
                    arguments.activation_checkpointing
                ),
                fine_tuning_method=arguments.fine_tuning_method,
                sampling_strategy=arguments.sampling_strategy,
                lora=LoraConfig(
                    rank=arguments.lora_rank,
                    alpha=arguments.lora_alpha,
                ),
            ),
            metric_sink=print_metric,
        )
        destination = result.bundle.save(arguments.output)
    except Exception as error:
        print(f"post-training failed: {error}", file=sys.stderr)
        return 1

    print(
        "[post-training] "
        f"saved={destination} "
        f"stage={result.bundle.stage} "
        f"artifact_id={result.bundle.artifact_id} "
        f"parent_artifact_id={result.bundle.parent_artifact_id} "
        f"objective={result.objective} "
        f"fine_tuning_method={result.fine_tuning_method} "
        f"examples={result.example_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
