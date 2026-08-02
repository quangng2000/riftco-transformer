"""Strict, provenance-rich JSON reports for the Python-owned lab."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Mapping, Sequence

from .config import ExperimentConfig
from .protocol import (
    Example,
    ProtocolConfig,
    SplitSizes,
    evaluate,
    predict_copy,
    predict_oracle,
    predict_reverse,
    split_fingerprint,
    verify_disjoint,
)


PROTOCOL_REPORT_FORMAT = "riftco-transformer.conditional-reverse-protocol.v1"
LEARNED_REPORT_FORMAT = "riftco-transformer.conditional-reverse.learned.v2"


@dataclass(frozen=True, slots=True)
class VariantReport:
    variant: str
    backend: str
    parameters: object
    training: object
    validation: object
    test: object
    test_hypotheses: object
    probe_pca: object | None
    ablations: object
    steering: object | None
    applicability: Mapping[str, object]


def build_protocol_report(
    config: ProtocolConfig,
    sizes: SplitSizes,
) -> dict[str, object]:
    from .protocol import generate_disjoint_splits

    splits = generate_disjoint_splits(config, sizes)
    verify_disjoint(splits)
    return {
        "format": PROTOCOL_REPORT_FORMAT,
        "ownership": {
            "protocol": "python_lab",
            "installed_framework_api": False,
            "learned_fpti_status": "available_via_learned_cli",
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


def build_learned_report(
    config: ExperimentConfig,
    splits: Mapping[str, Sequence[Example]],
    variant_reports: Sequence[VariantReport],
    provenance: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    verify_disjoint(splits)
    expected_splits = {"train", "probe", "validation", "test"}
    if set(splits) != expected_splits:
        raise ValueError("splits must contain train, probe, validation, and test")
    if not variant_reports:
        raise ValueError("variant_reports must not be empty")
    variants: dict[str, object] = {}
    for report in variant_reports:
        if not isinstance(report, VariantReport):
            raise TypeError("variant_reports must contain VariantReport values")
        if report.variant in variants:
            raise ValueError(f"duplicate variant report: {report.variant}")
        variants[report.variant] = _jsonable(report)
    configuration = _jsonable(config)
    if not isinstance(configuration, dict):
        raise AssertionError(
            "experiment configuration must serialize as an object"
        )
    model_configuration = configuration.get("model")
    if isinstance(model_configuration, dict):
        model_configuration.pop("variant", None)
    configuration["requested_variants"] = [
        report.variant for report in variant_reports
    ]
    return {
        "format": LEARNED_REPORT_FORMAT,
        "ownership": {
            "task_and_data": "python_lab",
            "variant_policy": "python_lab",
            "training_loop": "python_lab",
            "evaluation_and_analysis": "python_lab",
            "tensor_autograd_optimizer_and_program_execution": (
                "installed_framework_public_api"
            ),
        },
        "provenance": _jsonable(provenance),
        "config": configuration,
        "splits": {
            "source_disjoint": True,
            "sizes": {name: len(examples) for name, examples in splits.items()},
            "fingerprints": {
                name: split_fingerprint(examples)
                for name, examples in splits.items()
            },
        },
        "controls": {
            "validation": _control_metrics(splits["validation"]),
            "test": _control_metrics(splits["test"]),
        },
        "variants": variants,
    }


def collect_provenance(
    *,
    repository_root: Path,
    command: Sequence[str],
    framework_abi: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Collect stable environment facts without importing native bindings."""

    root = repository_root.resolve()
    commit = _git_output(root, ("rev-parse", "HEAD"))
    dirty_output = _git_output(root, ("status", "--porcelain"))
    try:
        package_version = importlib.metadata.version("riftco-transformer")
    except importlib.metadata.PackageNotFoundError:
        package_version = "source-tree"
    result: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit or "unknown",
        "git_dirty": bool(dirty_output),
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "package_version": package_version,
        "command": tuple(command),
    }
    if framework_abi is not None:
        result["framework_abi"] = dict(framework_abi)
    configured_library = os.environ.get("RIFTCO_TRANSFORMER_LIBRARY")
    if configured_library:
        library = Path(configured_library).expanduser().resolve()
        if library.is_file():
            result["native_library"] = {
                "path": str(library),
                "size_bytes": library.stat().st_size,
                "sha256": _file_sha256(library),
            }
    return result


def complete_provenance(
    started: Mapping[str, object],
    completed: Mapping[str, object],
) -> dict[str, object]:
    """Reject source/runtime drift and attach the run completion time."""

    if not isinstance(started, Mapping) or not isinstance(completed, Mapping):
        raise TypeError("started and completed provenance must be mappings")
    started_identity = {
        key: value for key, value in started.items() if key != "created_at_utc"
    }
    completed_identity = {
        key: value for key, value in completed.items() if key != "created_at_utc"
    }
    if started_identity != completed_identity:
        raise RuntimeError("source or runtime provenance changed during the run")
    completed_at = completed.get("created_at_utc")
    if not isinstance(completed_at, str) or not completed_at:
        raise ValueError("completed provenance needs created_at_utc")
    result = dict(started)
    result["completed_at_utc"] = completed_at
    return result


def write_new_json(path: Path, report: object) -> None:
    """Serialize finite JSON and create the destination without replacement."""

    if not isinstance(path, Path):
        raise TypeError("path must be a Path")
    payload = json.dumps(
        _jsonable(report),
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())


def _control_metrics(examples: Sequence[Example]) -> dict[str, object]:
    return {
        "conditional_oracle": asdict(evaluate(examples, predict_oracle)),
        "copy_only": asdict(evaluate(examples, predict_copy)),
        "reverse_only": asdict(evaluate(examples, predict_reverse)),
    }


def _git_output(root: Path, arguments: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            ("git",) + tuple(arguments),
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError("reports cannot contain non-finite floats")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("report mapping keys must be strings")
            result[key] = _jsonable(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"value is not JSON-reportable: {type(value).__name__}")


__all__ = [
    "LEARNED_REPORT_FORMAT",
    "PROTOCOL_REPORT_FORMAT",
    "VariantReport",
    "build_learned_report",
    "build_protocol_report",
    "collect_provenance",
    "complete_provenance",
    "write_new_json",
]
