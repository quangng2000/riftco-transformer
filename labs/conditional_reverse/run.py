"""Run protocol-only audits or the executable learned F/P/T/I study."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

from .config import ExperimentConfig, Profile, Variant, make_profile
from .protocol import (
    Example,
    ProtocolConfig,
    SplitSizes,
    generate_disjoint_splits,
    verify_disjoint,
)
from .reporting import (
    PROTOCOL_REPORT_FORMAT,
    VariantReport,
    build_learned_report,
    build_protocol_report,
    collect_provenance,
    complete_provenance,
    write_new_json,
)


# Backward-compatible protocol helpers remain useful to downstream audits.
REPORT_FORMAT = PROTOCOL_REPORT_FORMAT
build_report = build_protocol_report


def positive_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if result <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return result


def parse_variants(value: str) -> tuple[Variant, ...]:
    normalized = value.strip().upper()
    if normalized == "ALL":
        return tuple(Variant)
    if not normalized:
        raise argparse.ArgumentTypeError("variants must not be empty")
    result: list[Variant] = []
    for item in normalized.split(","):
        try:
            variant = Variant(item.strip())
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "variants must be all or a comma-separated subset of F,P,T,I"
            ) from error
        if variant in result:
            raise argparse.ArgumentTypeError("variants must be unique")
        result.append(variant)
    return tuple(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train and analyze the Python-owned conditional-reverse F/P/T/I "
            "study, or run its dependency-free protocol audit."
        )
    )
    parser.add_argument(
        "--protocol-only",
        action="store_true",
        help="Generate only data/control evidence; no framework import or training.",
    )
    parser.add_argument(
        "--profile",
        choices=tuple(profile.value for profile in Profile),
        default=Profile.QUICK.value,
    )
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=parse_variants("all"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "metal", "cuda", "tpu"),
        default="auto",
    )
    parser.add_argument(
        "--steps",
        type=positive_integer,
        help="Override the profile's maximum Adam steps.",
    )
    parser.add_argument("--sequence-length", type=positive_integer)
    parser.add_argument("--alphabet")
    parser.add_argument("--reverse-when-first-is")
    parser.add_argument("--delimiter")
    parser.add_argument("--train", type=positive_integer)
    parser.add_argument("--probe", type=positive_integer)
    parser.add_argument("--validation", type=positive_integer)
    parser.add_argument("--test", type=positive_integer)
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON path. Existing files are never overwritten.",
    )
    return parser


def execute_variant(
    config: ExperimentConfig,
    splits: Mapping[str, Sequence[Example]],
) -> VariantReport:
    """Execute one variant; imports of the framework remain on this path."""

    from .data import TokenCodec
    from .evaluation import (
        evaluate_model,
        fit_program_output_pca,
        run_f_steering,
        run_paired_ablations,
        score_hypotheses,
    )
    from .model import build_model
    from .training import train_model

    codec = TokenCodec.from_protocol(config.protocol)
    with build_model(config.protocol, config.model) as runtime:
        parameters = runtime.parameter_manifest()
        history = train_model(
            runtime,
            splits["train"],
            splits["validation"],
            codec,
            config.training,
        )
        validation = evaluate_model(
            runtime,
            splits["validation"],
            codec,
            config.training.evaluation_batch_size,
            retain_per_example=False,
        )
        test = evaluate_model(
            runtime,
            splits["test"],
            codec,
            config.training.evaluation_batch_size,
        )
        hypotheses = score_hypotheses(
            splits["test"],
            test.predictions,
            codec,
        )
        ablations = run_paired_ablations(
            runtime,
            splits["test"],
            codec,
            config.training.evaluation_batch_size,
            shift=config.analysis.ablation_shift,
        )
        pca = (
            fit_program_output_pca(
                runtime,
                splits["probe"],
                codec,
                config.training.evaluation_batch_size,
                config.analysis,
            )
            if runtime.has_program
            else None
        )
        steering = (
            run_f_steering(
                runtime,
                splits["test"],
                codec,
                config.training.evaluation_batch_size,
                config.analysis,
            )
            if runtime.variant is Variant.F
            else None
        )
        return VariantReport(
            variant=runtime.variant.value,
            backend=runtime.backend,
            parameters=parameters,
            training=history,
            validation=validation,
            test=test,
            test_hypotheses=hypotheses,
            probe_pca=pca,
            ablations=ablations,
            steering=steering,
            applicability={
                "program_output_pca": runtime.has_program,
                "program_output_ablation": runtime.has_program,
                "selector_steering": runtime.variant is Variant.F,
                "selector_steering_reason": (
                    "F has the compiled conditional selector basis"
                    if runtime.variant is Variant.F
                    else "semantic selector coordinates are not defined"
                ),
            },
        )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    variants = arguments.variants
    profile = Profile(arguments.profile)
    base = make_profile(
        profile,
        variant=variants[0],
        seed=arguments.seed,
        backend="cpu",
    )
    base = _apply_data_overrides(base, arguments)
    output = arguments.output or _default_output(
        profile,
        arguments.seed,
        protocol_only=arguments.protocol_only,
    )
    if output.exists():
        print(
            f"conditional-reverse run failed: output already exists: {output}",
            file=sys.stderr,
        )
        return 1

    try:
        if arguments.protocol_only:
            report = build_protocol_report(base.protocol, base.split_sizes)
            write_new_json(output, report)
            print(f"protocol report: {output}")
            return 0

        # The native package is intentionally imported only after the
        # protocol-only branch has returned.
        from .model import framework_abi, resolve_backend

        backend = resolve_backend(arguments.backend)
        base = replace(base, model=replace(base.model, backend=backend))
        base = _apply_step_override(base, arguments.steps)
        splits = generate_disjoint_splits(base.protocol, base.split_sizes)
        verify_disjoint(splits)
        command = (
            tuple(sys.argv)
            if argv is None
            else (sys.argv[0],) + tuple(argv)
        )
        abi = framework_abi()
        started_provenance = collect_provenance(
            repository_root=Path(__file__).resolve().parents[2],
            command=command,
            framework_abi=abi,
        )
        variant_reports: list[VariantReport] = []
        for variant in variants:
            variant_config = replace(
                base,
                model=base.model.with_variant(variant),
            )
            print(f"running {variant.value} on {backend} ...", flush=True)
            result = execute_variant(variant_config, splits)
            variant_reports.append(result)
            test_result = result.test
            test_metrics = getattr(test_result, "metrics")
            print(
                f"{variant.value}: step "
                f"{getattr(result.training, 'final_step')}, test exact "
                f"{100.0 * test_metrics.exact_sequence_accuracy:.2f}%",
                flush=True,
            )
        completed_provenance = collect_provenance(
            repository_root=Path(__file__).resolve().parents[2],
            command=command,
            framework_abi=abi,
        )
        provenance = complete_provenance(
            started_provenance,
            completed_provenance,
        )
        report = build_learned_report(
            base,
            splits,
            variant_reports,
            provenance,
        )
        write_new_json(output, report)
        print(f"learned report: {output}")
        return 0
    except Exception as error:
        print(f"conditional-reverse run failed: {error}", file=sys.stderr)
        return 1


def _apply_data_overrides(
    config: ExperimentConfig,
    arguments: argparse.Namespace,
) -> ExperimentConfig:
    protocol = ProtocolConfig(
        sequence_length=(
            config.protocol.sequence_length
            if arguments.sequence_length is None
            else arguments.sequence_length
        ),
        alphabet=(
            config.protocol.alphabet
            if arguments.alphabet is None
            else arguments.alphabet
        ),
        reverse_when_first_is=(
            config.protocol.reverse_when_first_is
            if arguments.reverse_when_first_is is None
            else arguments.reverse_when_first_is
        ),
        delimiter=(
            config.protocol.delimiter
            if arguments.delimiter is None
            else arguments.delimiter
        ),
        seed=config.protocol.seed,
    )
    sizes = SplitSizes(
        train=config.split_sizes.train if arguments.train is None else arguments.train,
        probe=config.split_sizes.probe if arguments.probe is None else arguments.probe,
        validation=(
            config.split_sizes.validation
            if arguments.validation is None
            else arguments.validation
        ),
        test=config.split_sizes.test if arguments.test is None else arguments.test,
    )
    return replace(config, protocol=protocol, split_sizes=sizes)


def _apply_step_override(
    config: ExperimentConfig,
    steps: int | None,
) -> ExperimentConfig:
    if steps is None:
        return config
    steps_per_epoch = math.ceil(
        config.split_sizes.train / config.training.batch_size
    )
    epochs = max(config.training.epochs, math.ceil(steps / steps_per_epoch))
    return replace(
        config,
        training=replace(
            config.training,
            epochs=epochs,
            maximum_steps=steps,
        ),
    )


def _default_output(
    profile: Profile,
    seed: int,
    *,
    protocol_only: bool,
) -> Path:
    kind = "protocol" if protocol_only else "learned"
    return Path("runs/conditional-reverse") / (
        f"{profile.value}-{kind}-seed-{seed}.json"
    )


if __name__ == "__main__":
    raise SystemExit(main())
