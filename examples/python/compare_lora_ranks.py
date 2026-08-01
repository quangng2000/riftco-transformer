"""Compare LoRA ranks on prepared instruction train/validation/test data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from riftco_transformer.artifacts import ModelBundle
from riftco_transformer.data import verify_prepared_dataset
from riftco_transformer.experiments import (
    LoraRankExperimentConfig,
    compare_lora_ranks,
    load_prepared_instruction_splits,
)
from riftco_transformer.experiments._reporting import (
    json_safe,
    prepared_dataset_provenance,
    staged_output_directory,
    write_json,
)


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


def comma_separated_ranks(value: str) -> tuple[int, ...]:
    try:
        ranks = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be comma-separated integers"
        ) from error
    if not ranks or any(rank <= 0 for rank in ranks):
        raise argparse.ArgumentTypeError(
            "ranks must be greater than zero"
        )
    if len(set(ranks)) != len(ranks):
        raise argparse.ArgumentTypeError("ranks must not contain duplicates")
    return ranks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train each LoRA rank from the same model artifact, select by "
            "validation loss, and report held-out test loss."
        )
    )
    parser.add_argument(
        "--base",
        type=Path,
        required=True,
        help="Pretrained .rift model artifact.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help=(
            "Prepared data directory containing train.jsonl, "
            "validation.jsonl, and test.jsonl."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiments/lora-ranks"),
        help=(
            "Directory for tuned artifacts and comparison.json "
            "(default: results/experiments/lora-ranks)."
        ),
    )
    parser.add_argument(
        "--ranks",
        type=comma_separated_ranks,
        default=(1, 2, 4, 8),
        help="Comma-separated LoRA ranks (default: 1,2,4,8).",
    )
    parser.add_argument(
        "--alpha-over-rank",
        type=positive_float,
        default=2.0,
        help="Shared alpha/rank scale (default: 2).",
    )
    parser.add_argument(
        "--steps",
        type=positive_integer,
        default=20,
        help="Optimizer steps for every rank (default: 20).",
    )
    parser.add_argument(
        "--context",
        type=positive_integer,
        default=16,
        help="Training and evaluation context length (default: 16).",
    )
    parser.add_argument(
        "--batch-size",
        type=positive_integer,
        default=2,
        help="Training examples per step (default: 2).",
    )
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=1.0e-3,
        help="Adam learning rate shared by all ranks (default: 0.001).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=29,
        help="Shared deterministic training seed (default: 29).",
    )
    parser.add_argument(
        "--adapter-seed",
        type=int,
        default=5489,
        help="Shared deterministic LoRA initialization seed (default: 5489).",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "metal", "cuda", "tpu"),
        default="auto",
        help=(
            "Execution backend; auto prefers TPU, then CUDA, Metal, and CPU, "
            "and resolves once for the complete sweep."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("example_uniform", "window_uniform"),
        default="example_uniform",
        help=(
            "Training sampler shared by every rank; example_uniform avoids "
            "overweighting long instructions (default: example_uniform)."
        ),
    )
    parser.add_argument(
        "--attention",
        choices=("materialized", "flash"),
        default="materialized",
        help="Training and evaluation attention implementation.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        choices=("disabled", "block"),
        default="disabled",
        help="Training activation policy (default: disabled).",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help=(
            "Optional greedy inference prompt; repeat for multiple prompts."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=positive_integer,
        default=16,
        help="Tokens generated per inference prompt (default: 16).",
    )
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        with staged_output_directory(arguments.output) as staging:
            bundle = ModelBundle.load(arguments.base)
            prepared = verify_prepared_dataset(arguments.data)
            splits = load_prepared_instruction_splits(prepared)
            provenance = prepared_dataset_provenance(prepared)
            comparison = compare_lora_ranks(
                bundle,
                splits,
                LoraRankExperimentConfig(
                    ranks=arguments.ranks,
                    alpha_over_rank=arguments.alpha_over_rank,
                    adapter_random_seed=arguments.adapter_seed,
                    training_random_seed=arguments.seed,
                    steps=arguments.steps,
                    context_size=arguments.context,
                    batch_size=arguments.batch_size,
                    evaluation_interval=min(10, arguments.steps),
                    loss_average_window=min(10, arguments.steps),
                    learning_rate=arguments.learning_rate,
                    backend=arguments.backend,
                    sampling_strategy=arguments.sampling_strategy,
                    attention=arguments.attention,
                    activation_checkpointing=(
                        arguments.activation_checkpointing
                    ),
                    evaluation_context_size=arguments.context,
                    inference_max_new_tokens=arguments.max_new_tokens,
                ),
                inference_prompts=arguments.prompt,
            )
            trial_entries: list[dict[str, object]] = []
            for trial in comparison.trials:
                artifact_name = f"lora-rank-{trial.rank}.rift"
                trial.bundle.save(staging / artifact_name)
                trial_entries.append(
                    {
                        "rank": trial.rank,
                        "alpha": trial.alpha,
                        "alpha_over_rank": trial.alpha_over_rank,
                        "adapter_parameter_count": (
                            trial.adapter_parameter_count
                        ),
                        "artifact_id": trial.bundle.artifact_id,
                        "artifact_path": str(
                            arguments.output / artifact_name
                        ),
                        "training_seconds": trial.training_seconds,
                        "validation_loss_delta_from_base": (
                            trial.validation_loss_delta_from_base
                        ),
                        "validation": asdict(trial.validation),
                        "test_loss_delta_from_base": (
                            trial.test_loss_delta_from_base
                        ),
                        "test": (
                            None
                            if trial.test is None
                            else asdict(trial.test)
                        ),
                        "inference_samples": [
                            asdict(sample)
                            for sample in trial.inference_samples
                        ],
                    }
                )
            summary = {
                "base_artifact_id": comparison.base_artifact_id,
                "resolved_backend": comparison.resolved_backend,
                "dataset_provenance": provenance,
                "fingerprints": asdict(comparison.fingerprints),
                "best_rank": comparison.best_rank,
                "config": asdict(comparison.config),
                "baseline_validation": asdict(
                    comparison.baseline_validation
                ),
                "baseline_test": asdict(comparison.baseline_test),
                "trials": trial_entries,
            }
            write_json(staging / "comparison.json", summary)
        summary_path = arguments.output / "comparison.json"
    except Exception as error:
        print(f"LoRA rank comparison failed: {error}", file=sys.stderr)
        return 1

    print("rank  adapter_params  validation_loss  test_loss  train_seconds")
    for trial in comparison.ranked_by_validation:
        test_loss = (
            "-"
            if trial.test is None
            else f"{trial.test.loss:.6f}"
        )
        print(
            f"{trial.rank:>4}  "
            f"{trial.adapter_parameter_count:>14}  "
            f"{trial.validation.loss:>15.6f}  "
            f"{test_loss:>9}  "
            f"{trial.training_seconds:>13.6f}"
        )
    print(f"best rank by validation loss: {comparison.best_rank}")
    print(
        "held-out test loss for selected rank: "
        f"{comparison.selected_test.loss:.6f}"
    )
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
