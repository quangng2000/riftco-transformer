"""Crash-safe filesystem helpers shared by repository-owned labs."""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

from riftco_transformer.data import PreparedDataset


def _publish_directory_no_replace(source: Path, destination: Path) -> None:
    """Publish a staging directory without importing framework internals."""

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"lab output already exists: {destination}"
        )
    # The destination check above gives the lab a clear failure mode. The
    # staging directory and destination share a parent, so the rename itself
    # remains atomic for readers on supported local filesystems.
    source.rename(destination)


@contextmanager
def staged_output_directory(destination: Path) -> Iterator[Path]:
    """Publish a complete new output directory with one rename."""

    if not isinstance(destination, Path):
        raise TypeError("destination must be a Path")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"experiment output already exists: {destination}"
        )
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging: Path | None = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=parent,
        )
    )
    try:
        yield staging
        fsync_directory(staging)
        _publish_directory_no_replace(staging, destination)
        staging = None
        fsync_directory(parent)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def write_json(path: Path, value: object) -> None:
    """Durably write strict, human-readable JSON inside a staging tree."""

    with path.open("w", encoding="utf-8") as output:
        json.dump(
            json_safe(value),
            output,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def prepared_dataset_provenance(
    prepared: PreparedDataset,
) -> dict[str, object]:
    """Capture provenance from the same verified manifest snapshot."""

    if not isinstance(prepared, PreparedDataset):
        raise TypeError("prepared must be a PreparedDataset")
    return {
        "manifest_sha256": prepared.manifest_sha256,
        "manifest": dict(prepared.manifest),
    }


def fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {
            str(key): json_safe(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


__all__ = [
    "json_safe",
    "prepared_dataset_provenance",
    "staged_output_directory",
    "write_json",
]
