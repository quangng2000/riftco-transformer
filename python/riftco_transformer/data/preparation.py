"""End-to-end Hugging Face dataset preparation with provenance."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import MappingProxyType
from typing import BinaryIO, Mapping
from urllib.parse import urlencode

from .._atomic_publish import publish_directory_no_replace
from .adapters import DatasetPreset
from .client import (
    MAX_ROWS_PER_REQUEST,
    DatasetRowsPage,
    DatasetSchemaError,
    HuggingFaceDatasetClient,
)
from .serialization import RecordSerializer
from .splitting import (
    PARTITION_NAMES,
    StableHashSplitter,
    canonical_record_bytes,
)


MANIFEST_FILE = "manifest.json"
PREPARED_DATASET_FORMAT = "riftco-transformer.prepared-dataset.v1"


@dataclass(frozen=True, slots=True)
class PreparedFile:
    """Verified metadata for one prepared partition."""

    path: Path
    media_type: str
    records: int
    bytes: int
    sha256: str

    def read_verified_bytes(self) -> bytes:
        """Read one immutable buffer and verify it against this metadata."""

        try:
            content = self.path.read_bytes()
        except OSError as error:
            raise ValueError(
                f"cannot read prepared dataset file: {self.path}"
            ) from error
        if len(content) != self.bytes:
            raise ValueError(
                f"{self.path.name} byte count does not match"
            )
        if hashlib.sha256(content).hexdigest() != self.sha256:
            raise ValueError(
                f"{self.path.name} SHA-256 does not match"
            )
        return content


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """A verified, immutable view of a prepared dataset directory."""

    directory: Path
    manifest_path: Path
    files: Mapping[str, PreparedFile]
    manifest: Mapping[str, object]
    manifest_sha256: str

    def file(self, partition: str) -> Path:
        if partition not in self.files:
            raise ValueError(f"unknown dataset partition: {partition!r}")
        return self.files[partition].path


def prepare_huggingface_dataset(
    preset: DatasetPreset,
    destination: str | Path,
    *,
    client: HuggingFaceDatasetClient | None = None,
    splitter: StableHashSplitter | None = None,
    offset: int = 0,
    limit: int | None = None,
    page_size: int = MAX_ROWS_PER_REQUEST,
    selection: str = "seeded_pages",
    source_split: str | None = None,
) -> PreparedDataset:
    """Download, validate, adapt, split, and atomically publish a dataset.

    ``destination`` must not already exist. Refusing replacement keeps the
    directory-level publish operation atomic and prevents accidental loss of
    an earlier experiment's data.
    """

    if not isinstance(preset, DatasetPreset):
        raise TypeError("preset must be a DatasetPreset")
    configured_client = (
        HuggingFaceDatasetClient() if client is None else client
    )
    if not isinstance(configured_client, HuggingFaceDatasetClient):
        raise TypeError("client must be a HuggingFaceDatasetClient")
    configured_splitter = (
        StableHashSplitter() if splitter is None else splitter
    )
    if not isinstance(configured_splitter, StableHashSplitter):
        raise TypeError("splitter must be a StableHashSplitter")
    source_offset = _nonnegative_integer(offset, "offset")
    source_limit = (
        None if limit is None else _positive_integer(limit, "limit")
    )
    request_page_size = _positive_integer(page_size, "page_size")
    if request_page_size > MAX_ROWS_PER_REQUEST:
        raise ValueError(
            f"page_size must be at most {MAX_ROWS_PER_REQUEST}"
        )
    if selection not in {"seeded_pages", "sequential"}:
        raise ValueError(
            "selection must be 'seeded_pages' or 'sequential'"
        )
    selected_source_split = (
        preset.split
        if source_split is None
        else _nonblank_string(source_split, "source_split")
    )
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(
            "prepared dataset destination already exists: "
            f"{destination_path}"
        )

    catalog = configured_client.get_split_catalog(preset.dataset)
    requested_split = (
        preset.dataset,
        preset.config,
        selected_source_split,
    )
    available = {
        (entry.dataset, entry.config, entry.split)
        for entry in catalog.splits
    }
    if requested_split not in available:
        raise DatasetSchemaError(
            "requested source split is not present in the dataset catalog: "
            f"{preset.dataset}/{preset.config}/{selected_source_split}"
        )

    probe = configured_client.get_rows_page(
        preset.dataset,
        preset.config,
        selected_source_split,
        offset=source_offset,
        length=1,
    )
    if not probe.rows:
        raise DatasetSchemaError(
            "the requested dataset range did not contain any rows"
        )
    if (
        catalog.revision is not None
        and probe.revision is not None
        and catalog.revision != probe.revision
    ):
        raise DatasetSchemaError(
            "dataset revision changed between split discovery and row download"
        )
    selected_ranges = _select_source_ranges(
        total_rows=probe.total_rows,
        offset=source_offset,
        limit=source_limit,
        page_size=request_page_size,
        strategy=selection,
        seed_material=(
            f"{configured_splitter.seed}\0{preset.dataset}\0"
            f"{preset.config}\0{selected_source_split}"
        ),
    )

    writer = _AtomicDatasetWriter(destination_path, preset.serializer)
    downloaded_rows = 0
    duplicate_rows = 0
    seen_records: set[bytes] = set()
    strata: dict[str, dict[str, int]] = {
        "all": {},
        **{partition: {} for partition in PARTITION_NAMES},
    }
    with writer:
        for range_offset, range_length in selected_ranges:
            page = configured_client.get_rows_page(
                preset.dataset,
                preset.config,
                selected_source_split,
                offset=range_offset,
                length=range_length,
            )
            _validate_page_consistency(page, probe)
            if len(page.rows) != range_length:
                raise DatasetSchemaError(
                    "the rows endpoint returned a short selected page"
                )
            for row in page.rows:
                downloaded_rows += 1
                record = preset.adapter.adapt(row)
                identity_record = preset.identity_record(record)
                identity = hashlib.sha256(
                    canonical_record_bytes(identity_record)
                ).digest()
                if identity in seen_records:
                    duplicate_rows += 1
                    continue
                seen_records.add(identity)
                partition = configured_splitter.assign(identity_record)
                writer.append(partition, record)
                if preset.stratification_field is not None:
                    stratum = record.get(preset.stratification_field)
                    if not isinstance(stratum, str) or not stratum:
                        raise DatasetSchemaError(
                            "adapted record is missing its configured "
                            "stratification field"
                        )
                    _increment(strata["all"], stratum)
                    _increment(strata[partition], stratum)

        if downloaded_rows == 0:
            raise DatasetSchemaError(
                "the requested dataset range did not contain any rows"
            )
        resolved_revision = probe.revision or catalog.revision
        resolved_selection = (
            "sequential_full_range"
            if source_limit is None
            or source_limit >= probe.total_rows - source_offset
            else selection
        )
        source_parameters = {
            "dataset": preset.dataset,
            "config": preset.config,
            "split": selected_source_split,
        }
        source = {
            "provider": "huggingface.datasets-server",
            "dataset": preset.dataset,
            "config": preset.config,
            "split": selected_source_split,
            "revision": resolved_revision,
            "dataset_url": preset.source_url,
            "revision_url": (
                None
                if resolved_revision is None
                else f"{preset.source_url}/tree/{resolved_revision}"
            ),
            "rows_api_url": (
                f"{configured_client.base_url}/rows?"
                f"{urlencode(source_parameters)}"
            ),
            "license": preset.license_id,
            "record_kind": preset.record_kind,
            "lab_training_support": preset.lab_training_support,
            "usage_note": preset.usage_note,
            "total_rows_reported": probe.total_rows,
            "offset": source_offset,
            "limit": source_limit,
            "page_size": request_page_size,
            "selection": {
                "strategy": resolved_selection,
                "seed": (
                    configured_splitter.seed
                    if resolved_selection == "seeded_pages"
                    else None
                ),
                "row_ranges": [
                    {"offset": start, "length": length}
                    for start, length in selected_ranges
                ],
            },
        }
        partitioning = {
            "strategy": "seeded_stable_sha256_v1",
            "seed": configured_splitter.seed,
            "fractions": configured_splitter.fractions.as_dict(),
            "identity_fields": list(preset.identity_fields),
            "leakage_policy": (
                "records with identical identity fields cannot cross partitions"
            ),
        }
        return writer.publish(
            preset_name=preset.name,
            adapter_id=preset.adapter.adapter_id,
            source=source,
            partitioning=partitioning,
            source_row_count=downloaded_rows,
            duplicate_row_count=duplicate_rows,
            stratification_field=preset.stratification_field,
            strata=strata if preset.stratification_field else None,
        )


def _select_source_ranges(
    *,
    total_rows: int,
    offset: int,
    limit: int | None,
    page_size: int,
    strategy: str,
    seed_material: str,
) -> tuple[tuple[int, int], ...]:
    available_rows = total_rows - offset
    if available_rows <= 0:
        return ()
    selected_rows = (
        available_rows if limit is None else min(limit, available_rows)
    )
    if (
        limit is None
        or strategy == "sequential"
        or selected_rows == available_rows
    ):
        return _chunked_range(offset, selected_rows, page_size)

    block_count = math.ceil(available_rows / page_size)
    seed_digest = hashlib.sha256(
        (
            "riftco_transformer.seeded_page_selection.v1\0"
            f"{seed_material}\0{offset}\0{total_rows}\0{page_size}"
        ).encode("utf-8")
    ).digest()
    start = int.from_bytes(seed_digest[:16], "big") % block_count
    if block_count == 1:
        stride = 1
    else:
        stride = (
            int.from_bytes(seed_digest[16:], "big") % (block_count - 1)
        ) + 1
        while math.gcd(stride, block_count) != 1:
            stride = (stride % (block_count - 1)) + 1

    chosen_blocks: list[int] = []
    capacity = 0
    for position in range(block_count):
        block = (start + position * stride) % block_count
        chosen_blocks.append(block)
        block_offset = offset + block * page_size
        capacity += min(page_size, total_rows - block_offset)
        if capacity >= selected_rows:
            break
    chosen_blocks.sort()

    result: list[tuple[int, int]] = []
    remaining = selected_rows
    for block in chosen_blocks:
        block_offset = offset + block * page_size
        block_length = min(
            page_size,
            total_rows - block_offset,
            remaining,
        )
        if block_length > 0:
            result.append((block_offset, block_length))
            remaining -= block_length
        if remaining == 0:
            break
    if remaining != 0:
        raise RuntimeError("seeded page selection did not satisfy the limit")
    return tuple(result)


def _chunked_range(
    offset: int,
    length: int,
    page_size: int,
) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    next_offset = offset
    remaining = length
    while remaining > 0:
        page_length = min(page_size, remaining)
        ranges.append((next_offset, page_length))
        next_offset += page_length
        remaining -= page_length
    return tuple(ranges)


def _validate_page_consistency(
    page: DatasetRowsPage,
    reference: DatasetRowsPage,
) -> None:
    if page.total_rows != reference.total_rows:
        raise DatasetSchemaError(
            "num_rows_total changed during selected-page download"
        )
    if page.feature_names != reference.feature_names:
        raise DatasetSchemaError(
            "dataset features changed during selected-page download"
        )
    if page.revision != reference.revision:
        raise DatasetSchemaError(
            "dataset revision changed during selected-page download"
        )


def _increment(counts: dict[str, int], value: str) -> None:
    counts[value] = counts.get(value, 0) + 1


def verify_prepared_dataset(
    directory: str | Path,
) -> PreparedDataset:
    """Validate manifest structure and every declared output digest."""

    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"prepared dataset directory does not exist: {root}")
    manifest_path = root / MANIFEST_FILE
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"cannot read prepared dataset manifest: {manifest_path}"
        ) from error
    try:
        raw_manifest = _strict_json_loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(
            f"prepared dataset manifest is invalid JSON: {manifest_path}"
        ) from error
    manifest = _manifest_object(raw_manifest, "manifest")
    if manifest.get("format") != PREPARED_DATASET_FORMAT:
        raise ValueError("prepared dataset manifest has an unsupported format")
    source = _manifest_object(
        _required(manifest, "source", "manifest"),
        "source",
    )
    for source_field in (
        "dataset",
        "config",
        "split",
        "dataset_url",
        "rows_api_url",
        "license",
    ):
        _manifest_string(source, source_field, "source")
    partitioning = _manifest_object(
        _required(manifest, "partitioning", "manifest"),
        "partitioning",
    )
    identity_fields = _required(
        partitioning,
        "identity_fields",
        "partitioning",
    )
    if (
        not isinstance(identity_fields, list)
        or not identity_fields
        or any(
            not isinstance(name, str) or not name
            for name in identity_fields
        )
        or len(set(identity_fields)) != len(identity_fields)
    ):
        raise ValueError(
            "manifest partitioning.identity_fields must be unique strings"
        )
    serialization = _manifest_object(
        _required(manifest, "serialization", "manifest"),
        "serialization",
    )
    extension = _manifest_string(
        serialization,
        "file_extension",
        "serialization",
    )
    if not extension.startswith(".") or Path(extension).name != extension:
        raise ValueError("manifest serialization.file_extension is invalid")
    counts = _manifest_object(
        _required(manifest, "counts", "manifest"),
        "counts",
    )
    source_rows = _manifest_nonnegative_integer(
        counts,
        "source_rows",
        "counts",
    )
    duplicate_rows = _manifest_nonnegative_integer(
        counts,
        "duplicates_removed",
        "counts",
    )
    adapted_records = _manifest_nonnegative_integer(
        counts,
        "adapted_records",
        "counts",
    )
    if source_rows != adapted_records + duplicate_rows:
        raise ValueError(
            "manifest source_rows must equal adapted records plus duplicates"
        )
    partition_counts = {
        name: _manifest_nonnegative_integer(counts, name, "counts")
        for name in PARTITION_NAMES
    }
    if sum(partition_counts.values()) != adapted_records:
        raise ValueError("manifest partition counts do not sum to records")
    deduplication = _manifest_object(
        _required(manifest, "deduplication", "manifest"),
        "deduplication",
    )
    declared_duplicates = _manifest_nonnegative_integer(
        deduplication,
        "duplicates_removed",
        "deduplication",
    )
    if declared_duplicates != duplicate_rows:
        raise ValueError(
            "manifest deduplication count does not match counts"
        )
    selection = _manifest_object(
        _required(source, "selection", "source"),
        "source.selection",
    )
    raw_ranges = _required(selection, "row_ranges", "source.selection")
    if not isinstance(raw_ranges, list):
        raise ValueError("source.selection.row_ranges must be an array")
    selected_count = 0
    previous_end = -1
    for index, raw_range in enumerate(raw_ranges):
        context = f"source.selection.row_ranges[{index}]"
        selected_range = _manifest_object(raw_range, context)
        range_offset = _manifest_nonnegative_integer(
            selected_range,
            "offset",
            context,
        )
        range_length = _manifest_nonnegative_integer(
            selected_range,
            "length",
            context,
        )
        if range_length == 0:
            raise ValueError(f"{context}.length must be positive")
        if range_offset < previous_end:
            raise ValueError(
                "source selection row ranges must be ordered and disjoint"
            )
        previous_end = range_offset + range_length
        selected_count += range_length
    if selected_count != source_rows:
        raise ValueError(
            "source selection ranges do not sum to source_rows"
        )

    raw_files = _manifest_object(
        _required(manifest, "files", "manifest"),
        "files",
    )
    if set(raw_files) != set(PARTITION_NAMES):
        raise ValueError(
            "manifest files must contain train, validation, and test"
        )
    prepared_files: dict[str, PreparedFile] = {}
    for partition in PARTITION_NAMES:
        context = f"files.{partition}"
        file_entry = _manifest_object(raw_files[partition], context)
        relative_name = _manifest_string(file_entry, "path", context)
        expected_name = f"{partition}{extension}"
        if relative_name != expected_name:
            raise ValueError(
                f"{context}.path must be {expected_name!r}"
            )
        media_type = _manifest_string(file_entry, "media_type", context)
        if media_type != _manifest_string(
            serialization,
            "media_type",
            "serialization",
        ):
            raise ValueError(
                f"{context}.media_type does not match serialization"
            )
        record_count = _manifest_nonnegative_integer(
            file_entry,
            "records",
            context,
        )
        byte_count = _manifest_nonnegative_integer(
            file_entry,
            "bytes",
            context,
        )
        expected_digest = _manifest_string(file_entry, "sha256", context)
        if (
            len(expected_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_digest
            )
        ):
            raise ValueError(f"{context}.sha256 must be lowercase SHA-256")
        if record_count != partition_counts[partition]:
            raise ValueError(
                f"{context}.records does not match counts.{partition}"
            )
        file_path = root / relative_name
        actual_bytes, actual_digest = _hash_file(file_path)
        if actual_bytes != byte_count:
            raise ValueError(f"{relative_name} byte count does not match")
        if actual_digest != expected_digest:
            raise ValueError(f"{relative_name} SHA-256 does not match")
        if extension == ".jsonl":
            _verify_jsonl(file_path, record_count)
        prepared_files[partition] = PreparedFile(
            path=file_path,
            media_type=media_type,
            records=record_count,
            bytes=byte_count,
            sha256=expected_digest,
        )

    return PreparedDataset(
        directory=root,
        manifest_path=manifest_path,
        files=MappingProxyType(prepared_files),
        manifest=MappingProxyType(dict(manifest)),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


@dataclass(slots=True)
class _PartitionFile:
    relative_path: str
    media_type: str
    stream: BinaryIO
    digest: object
    records: int = 0
    bytes: int = 0


class _AtomicDatasetWriter:
    """Stage a complete directory and publish it with one rename."""

    __slots__ = (
        "_destination",
        "_files",
        "_published",
        "_serializer",
        "_staging",
    )

    def __init__(
        self,
        destination: Path,
        serializer: RecordSerializer,
    ) -> None:
        if not isinstance(destination, Path):
            raise TypeError("destination must be a Path")
        self._destination = destination
        self._serializer = serializer
        self._files: dict[str, _PartitionFile] = {}
        self._staging: Path | None = None
        self._published = False

    def __enter__(self) -> "_AtomicDatasetWriter":
        if self._destination.exists():
            raise FileExistsError(
                "prepared dataset destination already exists: "
                f"{self._destination}"
            )
        parent = self._destination.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{self._destination.name}.staging-",
                dir=parent,
            )
        )
        self._staging = staging
        try:
            for partition in PARTITION_NAMES:
                relative_path = (
                    f"{partition}{self._serializer.file_extension}"
                )
                stream = (staging / relative_path).open("wb")
                self._files[partition] = _PartitionFile(
                    relative_path=relative_path,
                    media_type=self._serializer.media_type,
                    stream=stream,
                    digest=hashlib.sha256(),
                )
        except BaseException:
            self._cleanup()
            raise
        return self

    def __exit__(
        self,
        _error_type: object,
        _error: object,
        _traceback: object,
    ) -> None:
        if not self._published:
            self._cleanup()

    def append(
        self,
        partition: str,
        record: Mapping[str, str],
    ) -> None:
        if self._staging is None or self._published:
            raise RuntimeError("dataset writer is not active")
        if partition not in self._files:
            raise ValueError(f"unknown dataset partition: {partition!r}")
        encoded = self._serializer.serialize(record)
        if not isinstance(encoded, bytes) or not encoded:
            raise TypeError("serializer must return nonempty bytes")
        output = self._files[partition]
        output.stream.write(encoded)
        output.digest.update(encoded)
        output.records += 1
        output.bytes += len(encoded)

    def publish(
        self,
        *,
        preset_name: str,
        adapter_id: str,
        source: Mapping[str, object],
        partitioning: Mapping[str, object],
        source_row_count: int,
        duplicate_row_count: int,
        stratification_field: str | None,
        strata: Mapping[str, Mapping[str, int]] | None,
    ) -> PreparedDataset:
        if self._staging is None or self._published:
            raise RuntimeError("dataset writer is not active")
        for output in self._files.values():
            output.stream.flush()
            os.fsync(output.stream.fileno())
            output.stream.close()
        counts = {
            "source_rows": source_row_count,
            "duplicates_removed": duplicate_row_count,
            "adapted_records": sum(
                output.records for output in self._files.values()
            ),
            **{
                partition: self._files[partition].records
                for partition in PARTITION_NAMES
            },
        }
        files = {
            partition: {
                "path": output.relative_path,
                "media_type": output.media_type,
                "records": output.records,
                "bytes": output.bytes,
                "sha256": output.digest.hexdigest(),
            }
            for partition, output in self._files.items()
        }
        manifest = {
            "format": PREPARED_DATASET_FORMAT,
            "preset": preset_name,
            "adapter": adapter_id,
            "serialization": {
                "serializer": self._serializer.serializer_id,
                "file_extension": self._serializer.file_extension,
                "media_type": self._serializer.media_type,
            },
            "source": dict(source),
            "partitioning": dict(partitioning),
            "deduplication": {
                "strategy": "canonical_identity_fields_sha256_v1",
                "duplicates_removed": duplicate_row_count,
            },
            "stratification": (
                None
                if stratification_field is None
                else {
                    "field": stratification_field,
                    "assignment": (
                        "stable content hash; counts are recorded per stratum"
                    ),
                    "counts": {
                        scope: dict(sorted(values.items()))
                        for scope, values in (strata or {}).items()
                    },
                }
            ),
            "counts": counts,
            "files": files,
        }
        manifest_bytes = (
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        manifest_path = self._staging / MANIFEST_FILE
        with manifest_path.open("wb") as manifest_file:
            manifest_file.write(manifest_bytes)
            manifest_file.flush()
            os.fsync(manifest_file.fileno())

        _fsync_directory(self._staging)
        publish_directory_no_replace(
            self._staging,
            self._destination,
        )
        self._staging = None
        self._published = True
        _fsync_directory(self._destination.parent)
        return verify_prepared_dataset(self._destination)

    def _cleanup(self) -> None:
        for output in self._files.values():
            if not output.stream.closed:
                output.stream.close()
        self._files.clear()
        if self._staging is not None:
            shutil.rmtree(self._staging, ignore_errors=True)
            self._staging = None


def _verify_jsonl(path: Path, expected_records: int) -> None:
    records = 0
    try:
        with path.open("rb") as input_file:
            for line_number, raw_line in enumerate(input_file, start=1):
                if not raw_line.strip():
                    raise ValueError(
                        f"{path.name}:{line_number} must not be blank"
                    )
                try:
                    value = _strict_json_loads(raw_line)
                except (
                    UnicodeDecodeError,
                    json.JSONDecodeError,
                    ValueError,
                ) as error:
                    raise ValueError(
                        f"{path.name}:{line_number} is invalid JSON"
                    ) from error
                record = _manifest_object(
                    value,
                    f"{path.name}:{line_number}",
                )
                if not record or any(
                    not isinstance(name, str)
                    or not name
                    or not isinstance(field, str)
                    for name, field in record.items()
                ):
                    raise ValueError(
                        f"{path.name}:{line_number} must contain string fields"
                    )
                records += 1
    except OSError as error:
        raise ValueError(f"cannot read prepared file: {path}") from error
    if records != expected_records:
        raise ValueError(f"{path.name} record count does not match")


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as input_file:
            while True:
                chunk = input_file.read(1024 * 1024)
                if not chunk:
                    break
                byte_count += len(chunk)
                digest.update(chunk)
    except OSError as error:
        raise ValueError(f"cannot read prepared file: {path}") from error
    return byte_count, digest.hexdigest()


def _strict_json_loads(body: bytes) -> object:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for name, value in pairs:
            if name in result:
                raise ValueError(f"duplicate JSON key: {name}")
            result[name] = value
        return result

    return json.loads(
        body.decode("utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=unique_object,
    )


def _manifest_object(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    return value


def _required(
    value: Mapping[str, object],
    name: str,
    context: str,
) -> object:
    if name not in value:
        raise ValueError(f"{context} requires {name}")
    return value[name]


def _manifest_string(
    value: Mapping[str, object],
    name: str,
    context: str,
) -> str:
    result = _required(value, name, context)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{context}.{name} must be a nonempty string")
    return result


def _manifest_nonnegative_integer(
    value: Mapping[str, object],
    name: str,
    context: str,
) -> int:
    result = _required(value, name, context)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError(
            f"{context}.{name} must be a nonnegative integer"
        )
    return result


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _nonblank_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def _fsync_directory(path: Path) -> None:
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


__all__ = [
    "MANIFEST_FILE",
    "PREPARED_DATASET_FORMAT",
    "PreparedDataset",
    "PreparedFile",
    "prepare_huggingface_dataset",
    "verify_prepared_dataset",
]
