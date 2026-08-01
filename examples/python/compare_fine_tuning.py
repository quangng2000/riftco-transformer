"""Compare full fine-tuning and LoRA on verified held-out splits."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

from riftco_transformer import LoraConfig
from riftco_transformer.artifacts import ModelBundle
from riftco_transformer.data import verify_prepared_dataset
from riftco_transformer.experiments import (
    FineTuningCandidate,
    FineTuningComparison,
    FineTuningExperimentConfig,
    compare_fine_tuning,
    load_prepared_instruction_splits,
)
from riftco_transformer.experiments._reporting import (
    prepared_dataset_provenance,
    staged_output_directory,
    write_json,
)
from riftco_transformer.post_training import PostTrainingConfig
from riftco_transformer.training import TrainingMetric


REPORT_FORMAT = "riftco-transformer.fine-tuning-generalization.v1"


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


def comma_separated_methods(value: str) -> tuple[str, ...]:
    methods = tuple(part.strip().lower() for part in value.split(","))
    if not methods or any(
        method not in {"full", "lora"} for method in methods
    ):
        raise argparse.ArgumentTypeError(
            "must contain only comma-separated 'full' and 'lora'"
        )
    if len(set(methods)) != len(methods):
        raise argparse.ArgumentTypeError("methods must not contain duplicates")
    return methods


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
    return tuple(sorted(ranks))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train full and LoRA recipes from one base artifact, select only "
            "on validation, and report final held-out generalization."
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
            "Verified prepared-data directory containing train.jsonl, "
            "validation.jsonl, test.jsonl, and manifest.json."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiments/full-vs-lora"),
        help=(
            "New directory for artifacts and comparison.json "
            "(default: results/experiments/full-vs-lora)."
        ),
    )
    parser.add_argument(
        "--methods",
        type=comma_separated_methods,
        default=("full", "lora"),
        help="Methods to evaluate (default: full,lora).",
    )
    parser.add_argument(
        "--lora-ranks",
        type=comma_separated_ranks,
        default=(1, 2, 4, 8),
        help="Validation-selected LoRA ranks (default: 1,2,4,8).",
    )
    parser.add_argument(
        "--alpha-over-rank",
        type=positive_float,
        default=2.0,
        help="Shared LoRA alpha/rank scale (default: 2).",
    )
    parser.add_argument(
        "--steps",
        type=positive_integer,
        default=20,
        help="Optimizer steps per candidate (default: 20).",
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
        help="Training and evaluation batch size (default: 2).",
    )
    parser.add_argument(
        "--full-learning-rate",
        type=positive_float,
        default=1.0e-3,
        help="Adam learning rate for full fine-tuning (default: 0.001).",
    )
    parser.add_argument(
        "--lora-learning-rate",
        type=positive_float,
        default=1.0e-3,
        help="Adam learning rate for LoRA candidates (default: 0.001).",
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
            "and resolves once for every candidate."
        ),
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=("example_uniform", "window_uniform"),
        default="example_uniform",
        help="Shared training sampler (default: example_uniform).",
    )
    parser.add_argument(
        "--attention",
        choices=("materialized", "flash"),
        default="materialized",
        help="Shared training and evaluation attention implementation.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        choices=("disabled", "block"),
        default="disabled",
        help="Shared training activation policy (default: disabled).",
    )
    return parser


def build_candidates(
    arguments: argparse.Namespace,
) -> tuple[FineTuningCandidate, ...]:
    common = {
        "steps": arguments.steps,
        "context_size": arguments.context,
        "batch_size": arguments.batch_size,
        "evaluation_interval": min(10, arguments.steps),
        "loss_average_window": min(10, arguments.steps),
        "random_seed": arguments.seed,
        "backend": arguments.backend,
        "sampling_strategy": arguments.sampling_strategy,
        "attention": arguments.attention,
        "activation_checkpointing": arguments.activation_checkpointing,
    }
    candidates: list[FineTuningCandidate] = []
    if "full" in arguments.methods:
        candidates.append(
            FineTuningCandidate(
                "full",
                PostTrainingConfig(
                    **common,
                    learning_rate=arguments.full_learning_rate,
                    fine_tuning_method="full",
                ),
            )
        )
    if "lora" in arguments.methods:
        for rank in arguments.lora_ranks:
            candidates.append(
                FineTuningCandidate(
                    f"lora-rank-{rank}",
                    PostTrainingConfig(
                        **common,
                        learning_rate=arguments.lora_learning_rate,
                        fine_tuning_method="lora",
                        lora=LoraConfig(
                            rank=rank,
                            alpha=arguments.alpha_over_rank * rank,
                            random_seed=arguments.adapter_seed,
                        ),
                    ),
                )
            )
    return tuple(candidates)


def print_metric(candidate: str, metric: TrainingMetric) -> None:
    print(
        f"[{candidate}] "
        f"step={metric.step} "
        f"training_loss={metric.loss:.6f} "
        f"loss_average={metric.loss_average:.6f}"
    )


def comparison_summary(
    comparison: FineTuningComparison,
    provenance: dict[str, object],
    output: Path,
) -> dict[str, object]:
    # Deliberately avoid serializing ModelBundle bytes into the JSON report.
    trials: list[dict[str, object]] = []
    for trial in comparison.trials:
        artifact_name = f"{trial.name}.rift"
        trials.append(
            {
                "name": trial.name,
                "fine_tuning_method": trial.fine_tuning_method,
                "selection_group": trial.selection_group,
                "selected_for_test": trial.selected_for_test,
                "training_config": asdict(trial.candidate.config),
                "trainable_parameter_count": (
                    trial.trainable_parameter_count
                ),
                "base_parameter_count": trial.base_parameter_count,
                "trainable_fraction": trial.trainable_fraction,
                "artifact_id": trial.bundle.artifact_id,
                "artifact_path": str(output / artifact_name),
                "training_seconds": trial.training_seconds,
                "training_metrics": [
                    asdict(metric) for metric in trial.training_metrics
                ],
                "train": asdict(trial.train),
                "validation": asdict(trial.validation),
                "test": None if trial.test is None else asdict(trial.test),
                "train_loss_delta_from_base": (
                    trial.train_loss_delta_from_base
                ),
                "validation_loss_delta_from_base": (
                    trial.validation_loss_delta_from_base
                ),
                "test_loss_delta_from_base": trial.test_loss_delta_from_base,
                "validation_generalization_gap": (
                    trial.validation_generalization_gap
                ),
                "test_generalization_gap": (
                    trial.test_generalization_gap
                ),
            }
        )
    return {
        "format": REPORT_FORMAT,
        "objective": "full_sequence_causal_sft",
        "base_artifact_id": comparison.base_artifact_id,
        "resolved_backend": comparison.resolved_backend,
        "evaluation_context_size": comparison.evaluation_context_size,
        "dataset_provenance": provenance,
        "fingerprints": asdict(comparison.fingerprints),
        "selection_policy": {
            "metric": "validation_loss",
            "scope": "one winner per selection_group",
            "tie_break": "candidate_declaration_order",
            "test_access": "after_all_validation_selection",
            "test_split_must_be_retired_after_report": True,
        },
        "selected_candidate_names": list(
            comparison.selected_candidate_names
        ),
        "baseline": {
            "train": asdict(comparison.baseline_train),
            "validation": asdict(comparison.baseline_validation),
            "test": asdict(comparison.baseline_test),
            "validation_generalization_gap": (
                comparison.baseline_validation_generalization_gap
            ),
            "test_generalization_gap": (
                comparison.baseline_test_generalization_gap
            ),
        },
        "config": asdict(comparison.config),
        "trials": trials,
    }


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        with staged_output_directory(arguments.output) as staging:
            bundle = ModelBundle.load(arguments.base)
            prepared = verify_prepared_dataset(arguments.data)
            splits = load_prepared_instruction_splits(prepared)
            comparison = compare_fine_tuning(
                bundle,
                splits,
                FineTuningExperimentConfig(
                    candidates=build_candidates(arguments),
                    evaluation_context_size=arguments.context,
                    evaluation_batch_size=arguments.batch_size,
                    evaluation_attention=arguments.attention,
                ),
                metric_sink=print_metric,
            )
            for trial in comparison.trials:
                trial.bundle.save(staging / f"{trial.name}.rift")
            summary = comparison_summary(
                comparison,
                prepared_dataset_provenance(prepared),
                arguments.output,
            )
            write_json(staging / "comparison.json", summary)
        summary_path = arguments.output / "comparison.json"
    except Exception as error:
        print(f"fine-tuning comparison failed: {error}", file=sys.stderr)
        return 1

    print(
        "candidate       trainable  train_loss  validation_loss  "
        "validation_gap  test_loss  test_perplexity"
    )
    for trial in comparison.trials:
        test_loss = "-" if trial.test is None else f"{trial.test.loss:.6f}"
        test_perplexity = (
            "-" if trial.test is None else f"{trial.test.perplexity:.6f}"
        )
        print(
            f"{trial.name:<15} "
            f"{trial.trainable_parameter_count:>9}  "
            f"{trial.train.loss:>10.6f}  "
            f"{trial.validation.loss:>15.6f}  "
            f"{trial.validation_generalization_gap:>14.6f}  "
            f"{test_loss:>9}  "
            f"{test_perplexity:>15}"
        )
    print(
        "selected for final test: "
        + ", ".join(comparison.selected_candidate_names)
    )
    print(f"summary: {summary_path}")
    print(
        "The test split was consumed for this final method comparison; "
        "retire it before further tuning."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
