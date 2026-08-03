"""Dependency-free ONNX interchange for ``riftco_decoder_v1``.

This module contains a small, purpose-built Protocol Buffers encoder rather
than a replacement for the ONNX Python package.  It exports a real standard-
operator inference graph and imports only the exact canonical graph emitted by
this module.  Arbitrary or edited ONNX graphs remain outside the supported
interoperability boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tempfile
from types import MappingProxyType
from typing import Iterable, Mapping, NoReturn, Sequence

from ..artifacts import ModelBundle, TokenizerSpec
from ..native import TransformerConfig
from .contracts import (
    Float32Tensor,
    build_bundle_from_tensors,
    bundle_tensors,
    current_decoder_parameter_specs,
)


ONNX_IR_VERSION = 8
ONNX_OPSET_VERSION = 18
RIFTCO_ARCHITECTURE_ID = "riftco_decoder_v1"
ONNX_CANONICAL_GRAPH_FORMAT = "riftco_decoder_v1_onnx"
ONNX_CANONICAL_GRAPH_VERSION = 1
ONNX_PRODUCER_VERSION = "0.6.0"
ONNX_INPUT_CONTRACT = (
    "batch>=1;sequence>=1;0<=input_ids<vocabulary_size;"
    "sequence<=maximum_context"
)
ONNX_SIDECAR_FORMAT = "riftco-onnx-sidecar"
ONNX_SIDECAR_VERSION = 1
ONNX_SIDECAR_SUFFIX = ".riftco.json"
MAXIMUM_ONNX_SIDECAR_BYTES = 2 << 20
# The dependency-free importer eagerly materializes canonical FP32 weights.
# Keep its supported wire size explicit instead of letting an attacker-provided
# length consume arbitrary process memory. This still permits roughly 250M F32
# parameters before graph overhead.
MAXIMUM_ONNX_FILE_BYTES = 1 << 30
MAXIMUM_ONNX_PROTOBUF_FIELDS = 1 << 16
MAXIMUM_ONNX_NODES = 1 << 14
MAXIMUM_ONNX_INITIALIZERS = 1 << 13

_TENSOR_FLOAT = 1
_TENSOR_INT64 = 7
_ATTRIBUTE_FLOAT = 1
_ATTRIBUTE_INT = 2
_ATTRIBUTE_INTS = 7
_MAXIMUM_PROTO_DEPTH = 64
_MAXIMUM_ONNX_STRING_BYTES = 4 << 20
_MAXIMUM_ONNX_METADATA_ENTRIES = 64
_MAXIMUM_ONNX_OPSET_IMPORTS = 16
_MAXIMUM_ONNX_GRAPH_VALUES = 16
_MAXIMUM_ONNX_NODE_VALUES = 16
_MAXIMUM_ONNX_NODE_ATTRIBUTES = 32
_MAXIMUM_ONNX_TENSOR_RANK = 32
_MAXIMUM_ONNX_ATTRIBUTE_INTS = 64
_MAXIMUM_PROTOBUF_FIELD_NUMBER = (1 << 29) - 1
_ONNX_BASE_PROTOBUF_OVERHEAD_BYTES = (
    MAXIMUM_ONNX_SIDECAR_BYTES + (4 << 20)
)
_ONNX_PROTOBUF_OVERHEAD_PER_BLOCK_BYTES = 64 << 10
_SEMANTIC_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
)


class UnsupportedONNXImportError(NotImplementedError):
    """An ONNX graph cannot be proven equivalent to the native decoder."""


@dataclass(frozen=True, slots=True)
class ONNXNodeInfo:
    """One standard-operator node in topological graph order."""

    name: str
    op_type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ONNXTensorInfo:
    """One graph initializer without materializing its values."""

    name: str
    shape: tuple[int, ...]
    data_type: int
    byte_count: int | None


@dataclass(frozen=True, slots=True)
class ONNXValueInfo:
    """One typed graph input or output with static or symbolic dimensions."""

    name: str
    data_type: int
    shape: tuple[int | str, ...]


@dataclass(frozen=True, slots=True)
class ONNXInspection:
    """Structural summary decoded without protobuf or ONNX dependencies."""

    ir_version: int
    opset_version: int
    producer_name: str
    producer_version: str
    domain: str
    model_version: int
    graph_name: str
    nodes: tuple[ONNXNodeInfo, ...]
    initializers: tuple[ONNXTensorInfo, ...]
    inputs: tuple[ONNXValueInfo, ...]
    outputs: tuple[ONNXValueInfo, ...]
    metadata: Mapping[str, str]
    file_size: int


@dataclass(frozen=True, slots=True)
class _Attribute:
    name: str
    kind: int
    value: float | int | tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Node:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    name: str
    op_type: str
    attributes: tuple[_Attribute, ...] = ()


@dataclass(frozen=True, slots=True)
class _Initializer:
    name: str
    shape: tuple[int, ...]
    data_type: int
    raw_data: bytes


class _GraphBuilder:
    __slots__ = ("initializers", "nodes", "_names")

    def __init__(self) -> None:
        self.initializers: list[_Initializer] = []
        self.nodes: list[_Node] = []
        self._names: set[str] = set()

    def initializer(
        self,
        name: str,
        shape: Sequence[int],
        data_type: int,
        raw_data: bytes,
    ) -> str:
        self._claim(name)
        self.initializers.append(
            _Initializer(name, tuple(shape), data_type, raw_data)
        )
        return name

    def node(
        self,
        op_type: str,
        inputs: Sequence[str],
        output: str,
        *,
        name: str,
        attributes: Sequence[_Attribute] = (),
    ) -> str:
        self._claim(output)
        self.nodes.append(
            _Node(
                inputs=tuple(inputs),
                outputs=(output,),
                name=name,
                op_type=op_type,
                attributes=tuple(attributes),
            )
        )
        return output

    def _claim(self, name: str) -> None:
        if not name or name in self._names:
            raise RuntimeError(f"duplicate internal ONNX value name: {name!r}")
        self._names.add(name)


def export_onnx(
    bundle: ModelBundle,
    path: str | os.PathLike[str],
) -> Path:
    """Export a deterministic ONNX opset-18 decoder inference graph.

    The graph accepts nonempty ``input_ids`` with symbolic
    ``[batch, sequence]`` INT64 shape and returns FP32 ``logits`` with shape
    ``[batch, sequence, vocabulary]``. Invalid token IDs and sequence lengths
    above the learned position table fail at an ONNX ``Gather`` boundary.
    """

    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle")
    maximum_file_bytes = _maximum_onnx_file_bytes(bundle.config)
    tensors = bundle_tensors(bundle)
    graph = _build_graph(bundle, tensors)
    encoded = _encode_model(bundle, graph)
    if len(encoded) > maximum_file_bytes:
        raise ValueError(
            "canonical ONNX export exceeds the derived eager-import size limit"
        )
    sidecar = _encode_sidecar(bundle, encoded)

    destination = Path(path)
    _publish_onnx_pair(
        destination,
        encoded,
        onnx_sidecar_path(destination),
        sidecar,
    )
    return destination


def onnx_sidecar_path(path: str | os.PathLike[str]) -> Path:
    """Return the required adjacent tokenizer/artifact sidecar path."""

    source = Path(path)
    return source.with_name(source.name + ONNX_SIDECAR_SUFFIX)


def inspect_onnx(path: str | os.PathLike[str]) -> ONNXInspection:
    """Decode the ModelProto and graph structure without third-party modules."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    data, _ = _read_onnx_snapshot(source, MAXIMUM_ONNX_FILE_BYTES)
    return _inspect_onnx_bytes(data)


def _inspect_onnx_bytes(data: bytes) -> ONNXInspection:
    """Inspect one already-bounded immutable ONNX byte snapshot."""

    fields = tuple(_protobuf_fields(memoryview(data), depth=0))
    ir_version = _required_single_varint(fields, 1, "ModelProto.ir_version")
    producer_name = _optional_single_string(fields, 2)
    producer_version = _optional_single_string(fields, 3)
    domain = _optional_single_string(fields, 4)
    model_version = _optional_single_varint(fields, 5, default=0)
    graph_bytes = _required_single_bytes(fields, 7, "ModelProto.graph")

    standard_opset: int | None = None
    opset_payloads = _all_bytes(fields, 8)
    if len(opset_payloads) > _MAXIMUM_ONNX_OPSET_IMPORTS:
        raise ValueError("ONNX model contains too many operator-set imports")
    for payload in opset_payloads:
        opset_fields = tuple(_protobuf_fields(payload, depth=1))
        opset_domain = _optional_single_string(opset_fields, 1)
        opset_version = _required_single_varint(
            opset_fields, 2, "OperatorSetIdProto.version"
        )
        if opset_domain == "":
            standard_opset = opset_version
    if standard_opset is None:
        raise ValueError("ONNX model does not import the standard operator set")

    metadata: dict[str, str] = {}
    metadata_payloads = _all_bytes(fields, 14)
    if len(metadata_payloads) > _MAXIMUM_ONNX_METADATA_ENTRIES:
        raise ValueError("ONNX model contains too many metadata entries")
    for payload in metadata_payloads:
        entry_fields = tuple(_protobuf_fields(payload, depth=1))
        key = _required_single_string(entry_fields, 1, "metadata key")
        value = _required_single_string(entry_fields, 2, "metadata value")
        if key in metadata:
            raise ValueError(f"duplicate ONNX metadata key: {key!r}")
        metadata[key] = value

    graph_fields = tuple(_protobuf_fields(graph_bytes, depth=1))
    graph_name = _required_single_string(graph_fields, 2, "GraphProto.name")
    node_payloads = _all_bytes(graph_fields, 1)
    if len(node_payloads) > MAXIMUM_ONNX_NODES:
        raise ValueError("ONNX graph contains too many nodes")
    nodes = tuple(
        _inspect_node(payload)
        for payload in node_payloads
    )
    initializer_payloads = _all_bytes(graph_fields, 5)
    if len(initializer_payloads) > MAXIMUM_ONNX_INITIALIZERS:
        raise ValueError("ONNX graph contains too many initializers")
    initializers = tuple(
        _inspect_tensor(payload)
        for payload in initializer_payloads
    )
    initializer_names = [tensor.name for tensor in initializers]
    if len(set(initializer_names)) != len(initializer_names):
        raise ValueError("ONNX graph contains duplicate initializer names")
    input_payloads = _all_bytes(graph_fields, 11)
    output_payloads = _all_bytes(graph_fields, 12)
    if (
        len(input_payloads) > _MAXIMUM_ONNX_GRAPH_VALUES
        or len(output_payloads) > _MAXIMUM_ONNX_GRAPH_VALUES
    ):
        raise ValueError("ONNX graph contains too many inputs or outputs")
    inputs = tuple(
        _inspect_value_info(payload)
        for payload in input_payloads
    )
    outputs = tuple(
        _inspect_value_info(payload)
        for payload in output_payloads
    )
    if not inputs or not outputs:
        raise ValueError("ONNX graph must have at least one input and output")

    return ONNXInspection(
        ir_version=ir_version,
        opset_version=standard_opset,
        producer_name=producer_name,
        producer_version=producer_version,
        domain=domain,
        model_version=model_version,
        graph_name=graph_name,
        nodes=nodes,
        initializers=initializers,
        inputs=inputs,
        outputs=outputs,
        metadata=MappingProxyType(metadata),
        file_size=len(data),
    )


def load_onnx(
    path: str | os.PathLike[str],
    *,
    sidecar_path: str | os.PathLike[str] | None = None,
) -> ModelBundle:
    """Import only an unchanged canonical Riftco decoder ONNX export.

    ONNX does not define tokenizer semantics, so the adjacent
    ``<model>.onnx.riftco.json`` sidecar written by :func:`export_onnx` is
    mandatory.  The loader reconstructs the complete native bundle, regenerates
    the expected graph, and accepts it only when topology, attributes,
    initializers, metadata, and protobuf bytes all match the canonical exporter.
    This is deliberately not a general ONNX-to-Riftco compiler.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    companion = (
        onnx_sidecar_path(source)
        if sidecar_path is None
        else Path(sidecar_path)
    )
    sidecar = _read_sidecar(companion)
    config = _parse_sidecar_config(sidecar["config"])
    try:
        maximum_file_bytes = _maximum_onnx_file_bytes(config)
        data, actual_digest = _read_onnx_snapshot(
            source,
            maximum_file_bytes,
        )
    except ValueError as error:
        raise UnsupportedONNXImportError(
            "ONNX file exceeds the bounded canonical size permitted by its "
            "Riftco sidecar config"
        ) from error
    expected_digest = sidecar["onnx_sha256"]
    if actual_digest != expected_digest:
        raise UnsupportedONNXImportError(
            "ONNX bytes do not match the required Riftco sidecar checksum; "
            "the graph or sidecar was edited or paired with the wrong file"
        )

    try:
        inspection = _inspect_onnx_bytes(data)
    except ValueError as error:
        raise UnsupportedONNXImportError(
            "ONNX protobuf or graph structure is malformed or unsupported"
        ) from error
    _validate_model_header(inspection)
    _validate_graph_io(inspection, config)
    _validate_graph_metadata(inspection.metadata, config, sidecar)
    tokenizer = _parse_sidecar_tokenizer(sidecar["tokenizer"], config)

    try:
        model_fields = tuple(_protobuf_fields(memoryview(data), depth=0))
        graph_bytes = _required_single_bytes(
            model_fields,
            7,
            "ModelProto.graph",
        )
        graph_fields = tuple(_protobuf_fields(graph_bytes, depth=1))
        decoded_nodes = tuple(
            _decode_node(payload)
            for payload in _all_bytes(graph_fields, 1)
        )
        decoded_initializers = tuple(
            _decode_initializer(payload)
            for payload in _all_bytes(graph_fields, 5)
        )
    except ValueError as error:
        raise UnsupportedONNXImportError(
            "ONNX nodes or initializers use a malformed or unsupported "
            "encoding"
        ) from error
    initializer_by_name = {
        initializer.name: initializer
        for initializer in decoded_initializers
    }
    if len(initializer_by_name) != len(decoded_initializers):
        raise UnsupportedONNXImportError(
            "ONNX graph contains duplicate initializer names"
        )

    tensors: dict[str, Float32Tensor] = {}
    for spec in current_decoder_parameter_specs(config):
        initializer = initializer_by_name.get(spec.name)
        if initializer is None:
            raise UnsupportedONNXImportError(
                f"ONNX graph is missing native initializer {spec.name!r}"
            )
        if initializer.data_type != _TENSOR_FLOAT:
            raise UnsupportedONNXImportError(
                f"ONNX initializer {spec.name!r} must use FLOAT data"
            )
        if initializer.shape != spec.shape:
            raise UnsupportedONNXImportError(
                f"ONNX initializer {spec.name!r} has shape "
                f"{initializer.shape}; expected {spec.shape}"
            )
        try:
            tensors[spec.name] = Float32Tensor.from_little_endian_bytes(
                spec.shape,
                initializer.raw_data,
            )
        except (TypeError, ValueError) as error:
            raise UnsupportedONNXImportError(
                f"ONNX initializer {spec.name!r} is not canonical finite F32"
            ) from error

    try:
        bundle = build_bundle_from_tensors(
            config=config,
            tokenizer=tokenizer,
            tensors=tensors,
            stage=sidecar["stage"],
            parent_artifact_id=sidecar["parent_artifact_id"],
            metadata=sidecar["metadata"],
        )
    except (TypeError, ValueError) as error:
        raise UnsupportedONNXImportError(
            "ONNX sidecar cannot reconstruct a valid Riftco model bundle"
        ) from error
    if bundle.artifact_id != sidecar["source_artifact_id"]:
        raise UnsupportedONNXImportError(
            "ONNX sidecar source_artifact_id does not match the reconstructed "
            "config, tokenizer, weights, and artifact metadata"
        )

    expected_graph = _build_graph(bundle, tensors)
    if decoded_nodes != tuple(expected_graph.nodes):
        raise UnsupportedONNXImportError(
            "ONNX node topology or operator attributes do not exactly match "
            "the canonical Riftco decoder graph"
        )
    if decoded_initializers != tuple(expected_graph.initializers):
        raise UnsupportedONNXImportError(
            "ONNX initializer names, order, shapes, dtypes, or values do not "
            "exactly match the canonical Riftco decoder graph"
        )
    canonical = _encode_model(
        bundle,
        expected_graph,
        producer_version=inspection.producer_version,
    )
    if data != canonical:
        raise UnsupportedONNXImportError(
            "ONNX protobuf is not the exact canonical Riftco decoder export; "
            "unknown fields, metadata edits, or noncanonical encodings are "
            "not importable"
        )
    return bundle


def _encode_sidecar(bundle: ModelBundle, onnx_bytes: bytes) -> bytes:
    config = bundle.config
    tokenizer: dict[str, object] = {"method": bundle.tokenizer.method}
    if bundle.tokenizer.method == "byte":
        tokenizer["byte_vocabulary"] = list(
            bundle.tokenizer.byte_vocabulary
        )
    else:
        tokenizer["merge_rules"] = [
            list(rule) for rule in bundle.tokenizer.merge_rules
        ]
    value: dict[str, object] = {
        "config": {
            "block_count": config.block_count,
            "feed_forward_width": config.feed_forward_width,
            "head_count": config.head_count,
            "layer_norm_epsilon": config.layer_norm_epsilon,
            "maximum_context": config.maximum_context,
            "model_width": config.model_width,
            "random_seed": config.random_seed,
            "vocabulary_size": config.vocabulary_size,
        },
        "format": ONNX_SIDECAR_FORMAT,
        "format_version": ONNX_SIDECAR_VERSION,
        "graph_format": ONNX_CANONICAL_GRAPH_FORMAT,
        "graph_format_version": ONNX_CANONICAL_GRAPH_VERSION,
        "metadata": bundle.metadata,
        "onnx_sha256": hashlib.sha256(onnx_bytes).hexdigest(),
        "parent_artifact_id": bundle.parent_artifact_id,
        "source_artifact_id": bundle.artifact_id,
        "stage": bundle.stage,
        "tokenizer": tokenizer,
    }
    try:
        encoded = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as error:
        raise TypeError("ONNX sidecar is not finite JSON") from error
    if len(encoded) > MAXIMUM_ONNX_SIDECAR_BYTES:
        raise ValueError("ONNX sidecar exceeds the size limit")
    return encoded


def _publish_onnx_pair(
    destination: Path,
    onnx_bytes: bytes,
    companion: Path,
    sidecar_bytes: bytes,
) -> None:
    """Publish both fixed-name files with rollback on ordinary failures.

    Portable filesystems cannot atomically replace two directory entries as a
    unit. Both payloads are therefore staged and fsynced first, existing files
    are backed up without removing them, and any failed replace restores the
    complete previous pair before the exception escapes.
    """

    if destination.parent != companion.parent:
        raise ValueError("ONNX model and sidecar must share one directory")
    for path in (destination, companion):
        if path.exists() and path.is_dir():
            raise IsADirectoryError(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    staged_model: Path | None = None
    staged_sidecar: Path | None = None
    backup_model: Path | None = None
    backup_sidecar: Path | None = None
    try:
        staged_model = _stage_file(destination, onnx_bytes)
        staged_sidecar = _stage_file(companion, sidecar_bytes)
        backup_model = _backup_file(destination)
        backup_sidecar = _backup_file(companion)
        try:
            os.replace(staged_model, destination)
            staged_model = None
            os.replace(staged_sidecar, companion)
            staged_sidecar = None
            _fsync_directory(destination.parent)
        except BaseException as error:
            rollback_errors: list[OSError] = []
            for target, backup in (
                (destination, backup_model),
                (companion, backup_sidecar),
            ):
                try:
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        os.replace(backup, target)
                except OSError as rollback_error:
                    rollback_errors.append(rollback_error)
            _fsync_directory(destination.parent)
            if rollback_errors:
                preserved = tuple(
                    str(path)
                    for path in (backup_model, backup_sidecar)
                    if path is not None and path.exists()
                )
                backup_model = None
                backup_sidecar = None
                raise RuntimeError(
                    "ONNX publication failed and the previous model/sidecar "
                    "pair could not be completely restored; preserved rollback "
                    f"files: {preserved}"
                ) from error
            raise
    finally:
        for temporary in (
            staged_model,
            staged_sidecar,
            backup_model,
            backup_sidecar,
        ):
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _stage_file(destination: Path, data: bytes) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        return temporary_path
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _backup_file(source: Path) -> Path | None:
    if not source.exists() and not source.is_symlink():
        return None
    descriptor, backup_name = tempfile.mkstemp(
        prefix=f".{source.name}.",
        suffix=".rollback",
        dir=source.parent,
    )
    os.close(descriptor)
    backup = Path(backup_name)
    backup.unlink()
    try:
        try:
            os.link(source, backup, follow_symlinks=False)
        except (NotImplementedError, OSError):
            shutil.copy2(source, backup, follow_symlinks=False)
            if not backup.is_symlink():
                with backup.open("rb") as input_file:
                    os.fsync(input_file.fileno())
        return backup
    except BaseException:
        backup.unlink(missing_ok=True)
        raise


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Directory fsync is unsupported on some otherwise supported hosts.
        pass
    finally:
        os.close(descriptor)


def _maximum_onnx_file_bytes(config: TransformerConfig) -> int:
    specs = current_decoder_parameter_specs(config)
    parameter_bytes = sum(spec.value_count for spec in specs) * 4
    overhead = (
        _ONNX_BASE_PROTOBUF_OVERHEAD_BYTES
        + config.block_count * _ONNX_PROTOBUF_OVERHEAD_PER_BLOCK_BYTES
    )
    if parameter_bytes > MAXIMUM_ONNX_FILE_BYTES - overhead:
        raise ValueError(
            "decoder parameters exceed the dependency-free ONNX eager-import "
            f"limit of {MAXIMUM_ONNX_FILE_BYTES} bytes"
        )
    return parameter_bytes + overhead


def _read_onnx_snapshot(
    source: Path,
    maximum_bytes: int,
) -> tuple[bytes, str]:
    with source.open("rb") as input_file:
        file_size = os.fstat(input_file.fileno()).st_size
        if file_size > maximum_bytes:
            raise ValueError(
                f"ONNX file size {file_size} exceeds limit {maximum_bytes}"
            )
        data = input_file.read(maximum_bytes + 1)
    if len(data) > maximum_bytes:
        raise ValueError(
            f"ONNX file size exceeds limit {maximum_bytes} while reading"
        )
    return data, hashlib.sha256(data).hexdigest()


def _read_sidecar(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise UnsupportedONNXImportError(
            "ONNX import requires the adjacent Riftco tokenizer/artifact "
            f"sidecar {path.name!r}; ONNX alone does not define a tokenizer"
        )
    with path.open("rb") as input_file:
        encoded = input_file.read(MAXIMUM_ONNX_SIDECAR_BYTES + 1)
    if len(encoded) > MAXIMUM_ONNX_SIDECAR_BYTES:
        raise UnsupportedONNXImportError("ONNX sidecar exceeds the size limit")
    try:
        decoded = encoded.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKey,
        RecursionError,
        ValueError,
    ) as error:
        raise UnsupportedONNXImportError(
            "ONNX sidecar is not valid strict JSON"
        ) from error
    if not isinstance(value, dict):
        raise UnsupportedONNXImportError(
            "ONNX sidecar must contain a JSON object"
        )
    expected_keys = {
        "config",
        "format",
        "format_version",
        "graph_format",
        "graph_format_version",
        "metadata",
        "onnx_sha256",
        "parent_artifact_id",
        "source_artifact_id",
        "stage",
        "tokenizer",
    }
    if set(value) != expected_keys:
        raise UnsupportedONNXImportError(
            "ONNX sidecar fields do not match the supported version"
        )
    if value.get("format") != ONNX_SIDECAR_FORMAT:
        raise UnsupportedONNXImportError("unknown ONNX sidecar format")
    if (
        type(value.get("format_version")) is not int
        or value["format_version"] != ONNX_SIDECAR_VERSION
    ):
        raise UnsupportedONNXImportError(
            "unsupported ONNX sidecar format version"
        )
    if value.get("graph_format") != ONNX_CANONICAL_GRAPH_FORMAT:
        raise UnsupportedONNXImportError(
            "unsupported ONNX canonical graph format"
        )
    if (
        type(value.get("graph_format_version")) is not int
        or value["graph_format_version"] != ONNX_CANONICAL_GRAPH_VERSION
    ):
        raise UnsupportedONNXImportError(
            "unsupported ONNX canonical graph format version"
        )
    _require_sha256(value.get("onnx_sha256"), "onnx_sha256")
    _require_sha256(value.get("source_artifact_id"), "source_artifact_id")
    parent = value.get("parent_artifact_id")
    if parent is not None:
        _require_sha256(parent, "parent_artifact_id")
    stage = value.get("stage")
    if not isinstance(stage, str) or not stage.strip():
        raise UnsupportedONNXImportError(
            "ONNX sidecar stage must be a nonempty string"
        )
    for name in ("config", "tokenizer", "metadata"):
        if not isinstance(value.get(name), dict):
            raise UnsupportedONNXImportError(
                f"ONNX sidecar {name} must be an object"
            )
    return value


def _validate_model_header(inspection: ONNXInspection) -> None:
    expected = (
        ("IR version", inspection.ir_version, ONNX_IR_VERSION),
        ("opset version", inspection.opset_version, ONNX_OPSET_VERSION),
        ("producer", inspection.producer_name, "riftco-transformer"),
        ("domain", inspection.domain, "ai.riftco"),
        (
            "model version",
            inspection.model_version,
            ONNX_CANONICAL_GRAPH_VERSION,
        ),
        (
            "graph name",
            inspection.graph_name,
            "riftco_decoder_v1_inference",
        ),
    )
    for label, actual, required in expected:
        if actual != required or type(actual) is not type(required):
            raise UnsupportedONNXImportError(
                f"ONNX {label} is {actual!r}; expected {required!r}"
            )
    if _SEMANTIC_VERSION.fullmatch(inspection.producer_version) is None:
        raise UnsupportedONNXImportError(
            "ONNX producer version must use MAJOR.MINOR.PATCH form"
        )


def _parse_sidecar_config(value: object) -> TransformerConfig:
    if not isinstance(value, dict):
        raise UnsupportedONNXImportError("ONNX sidecar config must be an object")
    expected_keys = {
        "block_count",
        "feed_forward_width",
        "head_count",
        "layer_norm_epsilon",
        "maximum_context",
        "model_width",
        "random_seed",
        "vocabulary_size",
    }
    if set(value) != expected_keys:
        raise UnsupportedONNXImportError(
            "ONNX sidecar config fields do not match TransformerConfig"
        )
    epsilon = value.get("layer_norm_epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise UnsupportedONNXImportError(
            "ONNX sidecar layer_norm_epsilon must be a number"
        )
    try:
        config = TransformerConfig(
            vocabulary_size=_json_integer(value, "vocabulary_size"),
            maximum_context=_json_integer(value, "maximum_context"),
            model_width=_json_integer(value, "model_width"),
            head_count=_json_integer(value, "head_count"),
            block_count=_json_integer(value, "block_count"),
            feed_forward_width=_json_integer(value, "feed_forward_width"),
            random_seed=_json_integer(value, "random_seed"),
            layer_norm_epsilon=epsilon,
        )
        current_decoder_parameter_specs(config)
    except (TypeError, ValueError) as error:
        raise UnsupportedONNXImportError(
            "ONNX sidecar contains an invalid TransformerConfig"
        ) from error
    return config


def _validate_graph_io(
    inspection: ONNXInspection,
    config: TransformerConfig,
) -> None:
    expected_inputs = (
        ONNXValueInfo(
            "input_ids",
            _TENSOR_INT64,
            ("batch", "sequence"),
        ),
    )
    expected_outputs = (
        ONNXValueInfo(
            "logits",
            _TENSOR_FLOAT,
            ("batch", "sequence", config.vocabulary_size),
        ),
    )
    if inspection.inputs != expected_inputs:
        raise UnsupportedONNXImportError(
            "ONNX inputs do not match dynamic INT64 input_ids[batch, sequence]"
        )
    if inspection.outputs != expected_outputs:
        raise UnsupportedONNXImportError(
            "ONNX outputs do not match FP32 logits[batch, sequence, vocabulary]"
        )


def _parse_sidecar_tokenizer(
    value: object,
    config: TransformerConfig,
) -> TokenizerSpec:
    if not isinstance(value, dict):
        raise UnsupportedONNXImportError(
            "ONNX sidecar tokenizer must be an object"
        )
    method = value.get("method")
    try:
        if method == "byte":
            if set(value) != {"method", "byte_vocabulary"}:
                raise ValueError("byte tokenizer fields are not exact")
            raw_vocabulary = value.get("byte_vocabulary")
            if not isinstance(raw_vocabulary, list):
                raise ValueError("byte_vocabulary must be an array")
            tokenizer = TokenizerSpec(
                method="byte",
                byte_vocabulary=tuple(
                    _plain_json_integer(item, "byte_vocabulary")
                    for item in raw_vocabulary
                ),
            )
        elif method == "bpe":
            if set(value) != {"method", "merge_rules"}:
                raise ValueError("BPE tokenizer fields are not exact")
            raw_rules = value.get("merge_rules")
            if not isinstance(raw_rules, list):
                raise ValueError("merge_rules must be an array")
            rules: list[tuple[int, int, int]] = []
            for index, raw_rule in enumerate(raw_rules):
                if not isinstance(raw_rule, list) or len(raw_rule) != 3:
                    raise ValueError(
                        f"merge_rules[{index}] must have three integers"
                    )
                rules.append(
                    tuple(
                        _plain_json_integer(item, f"merge_rules[{index}]")
                        for item in raw_rule
                    )
                )
            tokenizer = TokenizerSpec(method="bpe", merge_rules=tuple(rules))
        else:
            raise ValueError("tokenizer method must be 'byte' or 'bpe'")
    except (TypeError, ValueError) as error:
        raise UnsupportedONNXImportError(
            "ONNX sidecar contains an invalid Riftco tokenizer"
        ) from error
    if tokenizer.vocabulary_size != config.vocabulary_size:
        raise UnsupportedONNXImportError(
            "ONNX sidecar tokenizer vocabulary does not match the graph config"
        )
    return tokenizer


def _validate_graph_metadata(
    metadata: Mapping[str, str],
    config: TransformerConfig,
    sidecar: Mapping[str, object],
) -> None:
    expected_keys = {
        "ai.riftco.architecture",
        "ai.riftco.artifact_id",
        "ai.riftco.input_contract",
        "ai.riftco.maximum_context",
        "ai.riftco.onnx_format",
        "ai.riftco.onnx_format_version",
        "ai.riftco.stage",
        "ai.riftco.topology",
        "ai.riftco.transformer_config",
    }
    if set(metadata) != expected_keys:
        raise UnsupportedONNXImportError(
            "ONNX metadata keys do not match the canonical Riftco export"
        )
    expected_strings = {
        "ai.riftco.architecture": RIFTCO_ARCHITECTURE_ID,
        "ai.riftco.artifact_id": sidecar["source_artifact_id"],
        "ai.riftco.input_contract": ONNX_INPUT_CONTRACT,
        "ai.riftco.maximum_context": str(config.maximum_context),
        "ai.riftco.onnx_format": sidecar["graph_format"],
        "ai.riftco.onnx_format_version": str(
            sidecar["graph_format_version"]
        ),
        "ai.riftco.stage": sidecar["stage"],
        "ai.riftco.topology": (
            "learned_absolute_positions;pre_layer_norm;"
            "separate_qkvo;gelu;untied_lm_head"
        ),
    }
    for name, expected in expected_strings.items():
        if metadata.get(name) != expected:
            raise UnsupportedONNXImportError(
                f"ONNX metadata {name!r} does not match the Riftco sidecar"
            )
    raw_config = metadata["ai.riftco.transformer_config"]
    try:
        graph_config = json.loads(
            raw_config,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (
        json.JSONDecodeError,
        _DuplicateJsonKey,
        RecursionError,
        ValueError,
    ) as error:
        raise UnsupportedONNXImportError(
            "ONNX transformer_config metadata is not strict JSON"
        ) from error
    if not isinstance(graph_config, dict):
        raise UnsupportedONNXImportError(
            "ONNX transformer_config metadata must be an object"
        )
    expected_config = {
        "block_count": config.block_count,
        "feed_forward_width": config.feed_forward_width,
        "head_count": config.head_count,
        "layer_norm_epsilon": config.layer_norm_epsilon,
        "maximum_context": config.maximum_context,
        "model_width": config.model_width,
        "vocabulary_size": config.vocabulary_size,
    }
    if graph_config != expected_config:
        raise UnsupportedONNXImportError(
            "ONNX transformer_config metadata does not match the sidecar"
        )


def _require_sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UnsupportedONNXImportError(
            f"ONNX sidecar {name} must be a lowercase SHA-256 digest"
        )


def _json_integer(mapping: Mapping[str, object], name: str) -> int:
    return _plain_json_integer(mapping.get(name), name)


def _plain_json_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


class _DuplicateJsonKey(ValueError):
    pass


def _unique_json_object(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise _DuplicateJsonKey(name)
        result[name] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _build_graph(
    bundle: ModelBundle,
    tensors: Mapping[str, Float32Tensor],
) -> _GraphBuilder:
    config = bundle.config
    width = config.model_width
    heads = config.head_count
    head_width = width // heads
    builder = _GraphBuilder()

    for name, tensor in tensors.items():
        builder.initializer(
            name,
            tensor.shape,
            _TENSOR_FLOAT,
            tensor.to_little_endian_bytes(),
        )

    _int64_initializer(builder, "_riftco.position_start", (), (0,))
    _int64_initializer(builder, "_riftco.position_delta", (), (1,))
    _int64_initializer(builder, "_riftco.shape_index_one", (), (1,))
    _int64_initializer(
        builder,
        "_riftco.vocabulary_size",
        (),
        (config.vocabulary_size,),
    )
    _int64_initializer(builder, "_riftco.model_width", (1,), (width,))
    _int64_initializer(
        builder,
        "_riftco.head_shape_tail",
        (2,),
        (heads, head_width),
    )
    _int64_initializer(builder, "_riftco.axis_zero", (1,), (0,))
    _int64_initializer(builder, "_riftco.axis_one", (1,), (1,))
    _float_initializer(builder, "_riftco.zero", (), (0.0,))
    _float_initializer(builder, "_riftco.negative_infinity", (), (-math.inf,))
    _float_initializer(
        builder,
        "_riftco.attention_scale",
        (),
        (1.0 / math.sqrt(float(head_width)),),
    )
    _float_initializer(
        builder,
        "_riftco.inverse_sqrt_two",
        (),
        (0.70710678118654752440,),
    )
    _float_initializer(builder, "_riftco.one", (), (1.0,))
    _float_initializer(builder, "_riftco.half", (), (0.5,))

    input_shape = builder.node(
        "Shape",
        ("input_ids",),
        "_riftco.input_shape",
        name="riftco.input_shape",
    )
    sequence_length = builder.node(
        "Gather",
        (input_shape, "_riftco.shape_index_one"),
        "_riftco.sequence_length",
        name="riftco.sequence_length",
        attributes=(_int_attribute("axis", 0),),
    )
    input_element_count = builder.node(
        "Size",
        ("input_ids",),
        "_riftco.input_element_count",
        name="riftco.input_element_count",
    )
    input_nonempty = builder.node(
        "Greater",
        (input_element_count, "_riftco.position_start"),
        "_riftco.input_nonempty",
        name="riftco.input_nonempty",
    )
    sequence_guard_index = builder.node(
        "Where",
        (
            input_nonempty,
            "_riftco.position_start",
            "_riftco.position_delta",
        ),
        "_riftco.sequence_guard_index",
        name="riftco.sequence_guard_index",
    )
    sequence_vector = builder.node(
        "Unsqueeze",
        (sequence_length, "_riftco.axis_zero"),
        "_riftco.sequence_vector",
        name="riftco.sequence_vector",
    )
    checked_sequence_length = builder.node(
        "Gather",
        (sequence_vector, sequence_guard_index),
        "_riftco.checked_sequence_length",
        name="riftco.checked_sequence_length",
        attributes=(_int_attribute("axis", 0),),
    )
    positions = builder.node(
        "Range",
        (
            "_riftco.position_start",
            checked_sequence_length,
            "_riftco.position_delta",
        ),
        "_riftco.positions",
        name="riftco.positions",
    )
    model_shape = builder.node(
        "Concat",
        (input_shape, "_riftco.model_width"),
        "_riftco.model_shape",
        name="riftco.model_shape",
        attributes=(_int_attribute("axis", 0),),
    )
    head_shape = builder.node(
        "Concat",
        (input_shape, "_riftco.head_shape_tail"),
        "_riftco.head_shape",
        name="riftco.head_shape",
        attributes=(_int_attribute("axis", 0),),
    )
    position_rows = builder.node(
        "Unsqueeze",
        (positions, "_riftco.axis_one"),
        "_riftco.position_rows",
        name="riftco.position_rows",
    )
    position_columns = builder.node(
        "Unsqueeze",
        (positions, "_riftco.axis_zero"),
        "_riftco.position_columns",
        name="riftco.position_columns",
    )
    causal_allowed = builder.node(
        "GreaterOrEqual",
        (position_rows, position_columns),
        "_riftco.causal_allowed",
        name="riftco.causal_allowed",
    )
    causal_mask = builder.node(
        "Where",
        (
            causal_allowed,
            "_riftco.zero",
            "_riftco.negative_infinity",
        ),
        "_riftco.causal_mask",
        name="riftco.causal_mask",
    )
    token_ids_nonnegative = builder.node(
        "GreaterOrEqual",
        ("input_ids", "_riftco.position_start"),
        "_riftco.token_ids_nonnegative",
        name="riftco.token_ids_nonnegative",
    )
    token_ids_below_vocabulary = builder.node(
        "Less",
        ("input_ids", "_riftco.vocabulary_size"),
        "_riftco.token_ids_below_vocabulary",
        name="riftco.token_ids_below_vocabulary",
    )
    token_ids_valid = builder.node(
        "And",
        (token_ids_nonnegative, token_ids_below_vocabulary),
        "_riftco.token_ids_valid",
        name="riftco.token_ids_valid",
    )
    checked_token_ids = builder.node(
        "Where",
        (
            token_ids_valid,
            "input_ids",
            "_riftco.vocabulary_size",
        ),
        "_riftco.checked_token_ids",
        name="riftco.checked_token_ids",
    )
    token_embedding = builder.node(
        "Gather",
        ("token_embedding.weight", checked_token_ids),
        "embedding.token",
        name="embedding.token",
        attributes=(_int_attribute("axis", 0),),
    )
    position_embedding = builder.node(
        "Gather",
        ("position_embedding.weight", positions),
        "embedding.position",
        name="embedding.position",
        attributes=(_int_attribute("axis", 0),),
    )
    hidden = builder.node(
        "Add",
        (token_embedding, position_embedding),
        "embedding.output",
        name="embedding.add",
    )

    for index in range(config.block_count):
        prefix = f"blocks.{index}"
        attention_input = _layer_norm(
            builder,
            hidden,
            f"{prefix}.attention_norm",
            config.layer_norm_epsilon,
        )
        query = _linear(builder, attention_input, f"{prefix}.attention.query")
        key = _linear(builder, attention_input, f"{prefix}.attention.key")
        value = _linear(builder, attention_input, f"{prefix}.attention.value")
        query_heads = _split_heads(builder, query, head_shape, f"{prefix}.query")
        key_heads = _split_heads(builder, key, head_shape, f"{prefix}.key")
        value_heads = _split_heads(builder, value, head_shape, f"{prefix}.value")
        transposed_key = builder.node(
            "Transpose",
            (key_heads,),
            f"{prefix}.key.transpose_for_scores",
            name=f"{prefix}.key.transpose_for_scores",
            attributes=(_ints_attribute("perm", (0, 1, 3, 2)),),
        )
        raw_scores = builder.node(
            "MatMul",
            (query_heads, transposed_key),
            f"{prefix}.attention.raw_scores",
            name=f"{prefix}.attention.raw_scores",
        )
        scaled_scores = builder.node(
            "Mul",
            (raw_scores, "_riftco.attention_scale"),
            f"{prefix}.attention.scaled_scores",
            name=f"{prefix}.attention.scale",
        )
        masked_scores = builder.node(
            "Add",
            (scaled_scores, causal_mask),
            f"{prefix}.attention.masked_scores",
            name=f"{prefix}.attention.mask",
        )
        probabilities = builder.node(
            "Softmax",
            (masked_scores,),
            f"{prefix}.attention.probabilities",
            name=f"{prefix}.attention.softmax",
            attributes=(_int_attribute("axis", -1),),
        )
        context_heads = builder.node(
            "MatMul",
            (probabilities, value_heads),
            f"{prefix}.attention.context_heads",
            name=f"{prefix}.attention.context",
        )
        context_time_major = builder.node(
            "Transpose",
            (context_heads,),
            f"{prefix}.attention.context_time_major",
            name=f"{prefix}.attention.merge.transpose",
            attributes=(_ints_attribute("perm", (0, 2, 1, 3)),),
        )
        context = builder.node(
            "Reshape",
            (context_time_major, model_shape),
            f"{prefix}.attention.context",
            name=f"{prefix}.attention.merge.reshape",
        )
        attention_output = _linear(
            builder,
            context,
            f"{prefix}.attention.output",
        )
        attention_state = builder.node(
            "Add",
            (hidden, attention_output),
            f"{prefix}.attention_residual",
            name=f"{prefix}.attention_residual",
        )
        feed_forward_input = _layer_norm(
            builder,
            attention_state,
            f"{prefix}.feed_forward_norm",
            config.layer_norm_epsilon,
        )
        expanded = _linear(
            builder,
            feed_forward_input,
            f"{prefix}.feed_forward.expand",
        )
        activated = _gelu(builder, expanded, f"{prefix}.feed_forward.gelu")
        projected = _linear(
            builder,
            activated,
            f"{prefix}.feed_forward.project",
        )
        hidden = builder.node(
            "Add",
            (attention_state, projected),
            f"{prefix}.output",
            name=f"{prefix}.feed_forward_residual",
        )

    normalized = _layer_norm(
        builder,
        hidden,
        "final_norm",
        config.layer_norm_epsilon,
    )
    _linear(
        builder,
        normalized,
        "language_model_head",
        output="logits",
    )
    return builder


def _linear(
    builder: _GraphBuilder,
    input_name: str,
    prefix: str,
    *,
    output: str | None = None,
) -> str:
    transposed_weight = builder.node(
        "Transpose",
        (f"{prefix}.weight",),
        f"{prefix}.weight_transposed",
        name=f"{prefix}.transpose_weight",
        attributes=(_ints_attribute("perm", (1, 0)),),
    )
    projected = builder.node(
        "MatMul",
        (input_name, transposed_weight),
        f"{prefix}.matmul",
        name=f"{prefix}.matmul",
    )
    return builder.node(
        "Add",
        (projected, f"{prefix}.bias"),
        output if output is not None else f"{prefix}.output",
        name=f"{prefix}.bias_add",
    )


def _layer_norm(
    builder: _GraphBuilder,
    input_name: str,
    prefix: str,
    epsilon: float,
) -> str:
    return builder.node(
        "LayerNormalization",
        (input_name, f"{prefix}.scale", f"{prefix}.bias"),
        f"{prefix}.output",
        name=f"{prefix}.layer_norm",
        attributes=(
            _int_attribute("axis", -1),
            _float_attribute("epsilon", epsilon),
        ),
    )


def _split_heads(
    builder: _GraphBuilder,
    input_name: str,
    head_shape: str,
    prefix: str,
) -> str:
    reshaped = builder.node(
        "Reshape",
        (input_name, head_shape),
        f"{prefix}.reshape",
        name=f"{prefix}.reshape",
    )
    return builder.node(
        "Transpose",
        (reshaped,),
        f"{prefix}.heads",
        name=f"{prefix}.transpose",
        attributes=(_ints_attribute("perm", (0, 2, 1, 3)),),
    )


def _gelu(
    builder: _GraphBuilder,
    input_name: str,
    prefix: str,
) -> str:
    scaled = builder.node(
        "Mul",
        (input_name, "_riftco.inverse_sqrt_two"),
        f"{prefix}.scaled",
        name=f"{prefix}.scale",
    )
    error_function = builder.node(
        "Erf",
        (scaled,),
        f"{prefix}.erf",
        name=f"{prefix}.erf",
    )
    shifted = builder.node(
        "Add",
        (error_function, "_riftco.one"),
        f"{prefix}.shifted",
        name=f"{prefix}.shift",
    )
    weighted = builder.node(
        "Mul",
        (input_name, shifted),
        f"{prefix}.weighted",
        name=f"{prefix}.weight",
    )
    return builder.node(
        "Mul",
        (weighted, "_riftco.half"),
        f"{prefix}.output",
        name=f"{prefix}.half",
    )


def _encode_model(
    bundle: ModelBundle,
    builder: _GraphBuilder,
    *,
    producer_version: str = ONNX_PRODUCER_VERSION,
) -> bytes:
    if _SEMANTIC_VERSION.fullmatch(producer_version) is None:
        raise ValueError("ONNX producer version must use MAJOR.MINOR.PATCH form")
    config = bundle.config
    graph = bytearray()
    for node in builder.nodes:
        graph.extend(_message_field(1, _encode_node(node)))
    graph.extend(_string_field(2, "riftco_decoder_v1_inference"))
    for initializer in builder.initializers:
        graph.extend(_message_field(5, _encode_tensor(initializer)))
    graph.extend(_message_field(
        11,
        _encode_value_info(
            "input_ids",
            _TENSOR_INT64,
            ("batch", "sequence"),
        ),
    ))
    graph.extend(_message_field(
        12,
        _encode_value_info(
            "logits",
            _TENSOR_FLOAT,
            ("batch", "sequence", config.vocabulary_size),
        ),
    ))

    model = bytearray()
    model.extend(_varint_field(1, ONNX_IR_VERSION))
    model.extend(_string_field(2, "riftco-transformer"))
    model.extend(_string_field(3, producer_version))
    model.extend(_string_field(4, "ai.riftco"))
    model.extend(_varint_field(5, ONNX_CANONICAL_GRAPH_VERSION))
    model.extend(_string_field(
        6,
        "Inference-only export of the Riftco decoder architecture. "
        "Weights are embedded FP32 initializers.",
    ))
    model.extend(_message_field(7, bytes(graph)))
    model.extend(_message_field(
        8,
        _varint_field(2, ONNX_OPSET_VERSION),
    ))
    metadata = {
        "ai.riftco.architecture": RIFTCO_ARCHITECTURE_ID,
        "ai.riftco.artifact_id": bundle.artifact_id,
        "ai.riftco.input_contract": ONNX_INPUT_CONTRACT,
        "ai.riftco.maximum_context": str(config.maximum_context),
        "ai.riftco.onnx_format": ONNX_CANONICAL_GRAPH_FORMAT,
        "ai.riftco.onnx_format_version": str(
            ONNX_CANONICAL_GRAPH_VERSION
        ),
        "ai.riftco.stage": bundle.stage,
        "ai.riftco.topology": (
            "learned_absolute_positions;pre_layer_norm;"
            "separate_qkvo;gelu;untied_lm_head"
        ),
        "ai.riftco.transformer_config": json.dumps(
            {
                "block_count": config.block_count,
                "feed_forward_width": config.feed_forward_width,
                "head_count": config.head_count,
                "layer_norm_epsilon": config.layer_norm_epsilon,
                "maximum_context": config.maximum_context,
                "model_width": config.model_width,
                "vocabulary_size": config.vocabulary_size,
            },
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    for key, value in sorted(metadata.items()):
        entry = _string_field(1, key) + _string_field(2, value)
        model.extend(_message_field(14, entry))
    return bytes(model)


def _encode_node(node: _Node) -> bytes:
    output = bytearray()
    for name in node.inputs:
        output.extend(_string_field(1, name))
    for name in node.outputs:
        output.extend(_string_field(2, name))
    output.extend(_string_field(3, node.name))
    output.extend(_string_field(4, node.op_type))
    for attribute in node.attributes:
        output.extend(_message_field(5, _encode_attribute(attribute)))
    return bytes(output)


def _encode_attribute(attribute: _Attribute) -> bytes:
    output = bytearray(_string_field(1, attribute.name))
    if attribute.kind == _ATTRIBUTE_FLOAT:
        output.extend(_fixed32_field(2, float(attribute.value)))
    elif attribute.kind == _ATTRIBUTE_INT:
        output.extend(_varint_field(3, int(attribute.value)))
    elif attribute.kind == _ATTRIBUTE_INTS:
        values = tuple(attribute.value)  # type: ignore[arg-type]
        packed = b"".join(_varint(value) for value in values)
        output.extend(_bytes_field(8, packed))
    else:
        raise RuntimeError("unsupported internal ONNX attribute kind")
    output.extend(_varint_field(20, attribute.kind))
    return bytes(output)


def _encode_tensor(initializer: _Initializer) -> bytes:
    output = bytearray()
    if initializer.shape:
        output.extend(_bytes_field(
            1,
            b"".join(_varint(dimension) for dimension in initializer.shape),
        ))
    output.extend(_varint_field(2, initializer.data_type))
    output.extend(_string_field(8, initializer.name))
    output.extend(_bytes_field(9, initializer.raw_data))
    return bytes(output)


def _encode_value_info(
    name: str,
    data_type: int,
    shape: Sequence[int | str],
) -> bytes:
    shape_message = bytearray()
    for dimension in shape:
        if isinstance(dimension, str):
            dimension_message = _string_field(2, dimension)
        else:
            dimension_message = _varint_field(1, dimension)
        shape_message.extend(_message_field(1, dimension_message))
    tensor_type = _varint_field(1, data_type) + _message_field(
        2, bytes(shape_message)
    )
    type_proto = _message_field(1, tensor_type)
    return _string_field(1, name) + _message_field(2, type_proto)


def _int64_initializer(
    builder: _GraphBuilder,
    name: str,
    shape: Sequence[int],
    values: Sequence[int],
) -> None:
    expected = math.prod(shape) if shape else 1
    if len(values) != expected:
        raise RuntimeError("invalid internal INT64 initializer")
    builder.initializer(
        name,
        shape,
        _TENSOR_INT64,
        b"".join(struct.pack("<q", value) for value in values),
    )


def _float_initializer(
    builder: _GraphBuilder,
    name: str,
    shape: Sequence[int],
    values: Sequence[float],
) -> None:
    expected = math.prod(shape) if shape else 1
    if len(values) != expected:
        raise RuntimeError("invalid internal FLOAT initializer")
    builder.initializer(
        name,
        shape,
        _TENSOR_FLOAT,
        b"".join(struct.pack("<f", value) for value in values),
    )


def _int_attribute(name: str, value: int) -> _Attribute:
    return _Attribute(name, _ATTRIBUTE_INT, value)


def _float_attribute(name: str, value: float) -> _Attribute:
    canonical = struct.unpack("<f", struct.pack("<f", value))[0]
    return _Attribute(name, _ATTRIBUTE_FLOAT, canonical)


def _ints_attribute(name: str, values: Iterable[int]) -> _Attribute:
    return _Attribute(name, _ATTRIBUTE_INTS, tuple(values))


def _inspect_node(data: memoryview) -> ONNXNodeInfo:
    fields = tuple(_protobuf_fields(data, depth=2))
    input_payloads = _all_bytes(fields, 1)
    output_payloads = _all_bytes(fields, 2)
    if (
        len(input_payloads) > _MAXIMUM_ONNX_NODE_VALUES
        or len(output_payloads) > _MAXIMUM_ONNX_NODE_VALUES
    ):
        raise ValueError("ONNX node contains too many inputs or outputs")
    inputs = tuple(_decode_string(payload) for payload in input_payloads)
    outputs = tuple(_decode_string(payload) for payload in output_payloads)
    name = _optional_single_string(fields, 3)
    op_type = _required_single_string(fields, 4, "NodeProto.op_type")
    if not outputs:
        raise ValueError(f"ONNX node {name!r} has no outputs")
    return ONNXNodeInfo(name, op_type, inputs, outputs)


def _decode_node(data: memoryview) -> _Node:
    fields = tuple(_protobuf_fields(data, depth=2))
    input_payloads = _all_bytes(fields, 1)
    output_payloads = _all_bytes(fields, 2)
    attribute_payloads = _all_bytes(fields, 5)
    if (
        len(input_payloads) > _MAXIMUM_ONNX_NODE_VALUES
        or len(output_payloads) > _MAXIMUM_ONNX_NODE_VALUES
    ):
        raise ValueError("ONNX node contains too many inputs or outputs")
    if len(attribute_payloads) > _MAXIMUM_ONNX_NODE_ATTRIBUTES:
        raise ValueError("ONNX node contains too many attributes")
    inputs = tuple(_decode_string(payload) for payload in input_payloads)
    outputs = tuple(_decode_string(payload) for payload in output_payloads)
    name = _required_single_string(fields, 3, "NodeProto.name")
    op_type = _required_single_string(fields, 4, "NodeProto.op_type")
    if not outputs:
        raise ValueError(f"ONNX node {name!r} has no outputs")
    attributes = tuple(
        _decode_attribute(payload)
        for payload in attribute_payloads
    )
    return _Node(inputs, outputs, name, op_type, attributes)


def _decode_attribute(data: memoryview) -> _Attribute:
    fields = tuple(_protobuf_fields(data, depth=3))
    name = _required_single_string(fields, 1, "AttributeProto.name")
    kind = _required_single_varint(fields, 20, "AttributeProto.type")
    if kind == _ATTRIBUTE_FLOAT:
        values = _all_fixed32(fields, 2)
        if len(values) != 1:
            raise ValueError(
                f"ONNX FLOAT attribute {name!r} must have one value"
            )
        value: float | int | tuple[int, ...] = struct.unpack(
            "<f",
            values[0],
        )[0]
    elif kind == _ATTRIBUTE_INT:
        raw_values = _all_varints(fields, 3)
        if len(raw_values) != 1:
            raise ValueError(
                f"ONNX INT attribute {name!r} must have one value"
            )
        value = _signed_int64(raw_values[0])
    elif kind == _ATTRIBUTE_INTS:
        packed_values = _all_bytes(fields, 8)
        if len(packed_values) != 1:
            raise ValueError(
                f"ONNX INTS attribute {name!r} must use one packed value"
            )
        packed = packed_values[0]
        decoded: list[int] = []
        position = 0
        while position < len(packed):
            raw, position = _decode_varint(packed, position)
            decoded.append(_signed_int64(raw))
            if len(decoded) > _MAXIMUM_ONNX_ATTRIBUTE_INTS:
                raise ValueError(
                    f"ONNX INTS attribute {name!r} has too many values"
                )
        value = tuple(decoded)
    else:
        raise ValueError(
            f"unsupported ONNX attribute type {kind} on {name!r}"
        )
    return _Attribute(name, kind, value)


def _inspect_tensor(data: memoryview) -> ONNXTensorInfo:
    fields = tuple(_protobuf_fields(data, depth=2))
    dimensions = _decode_tensor_dimensions(fields)
    data_type = _required_single_varint(fields, 2, "TensorProto.data_type")
    name = _required_single_string(fields, 8, "TensorProto.name")
    raw_values = _all_bytes(fields, 9)
    byte_count: int | None = None
    if raw_values:
        if len(raw_values) != 1:
            raise ValueError("TensorProto.raw_data appears more than once")
        byte_count = len(raw_values[0])
        element_count = math.prod(dimensions) if dimensions else 1
        width = {_TENSOR_FLOAT: 4, _TENSOR_INT64: 8}.get(data_type)
        if width is not None and byte_count != element_count * width:
            raise ValueError(f"ONNX initializer {name!r} has a wrong payload size")
    return ONNXTensorInfo(name, dimensions, data_type, byte_count)


def _decode_initializer(data: memoryview) -> _Initializer:
    fields = tuple(_protobuf_fields(data, depth=2))
    dimensions = _decode_tensor_dimensions(fields)
    data_type = _required_single_varint(fields, 2, "TensorProto.data_type")
    name = _required_single_string(fields, 8, "TensorProto.name")
    raw_values = _all_bytes(fields, 9)
    if len(raw_values) != 1:
        raise ValueError(
            f"ONNX initializer {name!r} must have exactly one raw_data payload"
        )
    raw_data = bytes(raw_values[0])
    element_count = math.prod(dimensions) if dimensions else 1
    width = {_TENSOR_FLOAT: 4, _TENSOR_INT64: 8}.get(data_type)
    if width is None:
        raise ValueError(
            f"ONNX initializer {name!r} has unsupported dtype {data_type}"
        )
    if len(raw_data) != element_count * width:
        raise ValueError(f"ONNX initializer {name!r} has a wrong payload size")
    return _Initializer(name, dimensions, data_type, raw_data)


def _decode_tensor_dimensions(
    fields: Sequence[tuple[int, int, int | memoryview]],
) -> tuple[int, ...]:
    dimensions: list[int] = []
    for field_number, wire_type, value in fields:
        if field_number != 1:
            continue
        if wire_type == 0:
            assert isinstance(value, int)
            dimensions.append(_signed_int64(value))
        elif wire_type == 2:
            assert isinstance(value, memoryview)
            position = 0
            while position < len(value):
                raw, position = _decode_varint(value, position)
                dimensions.append(_signed_int64(raw))
        else:
            raise ValueError("TensorProto.dims has the wrong wire type")
    if any(dimension < 0 for dimension in dimensions):
        raise ValueError("ONNX initializer dimensions must not be negative")
    if len(dimensions) > _MAXIMUM_ONNX_TENSOR_RANK:
        raise ValueError("ONNX initializer rank exceeds the safety limit")
    return tuple(dimensions)


def _inspect_value_info(data: memoryview) -> ONNXValueInfo:
    fields = tuple(_protobuf_fields(data, depth=2))
    name = _required_single_string(fields, 1, "ValueInfoProto.name")
    type_bytes = _required_single_bytes(fields, 2, "ValueInfoProto.type")
    type_fields = tuple(_protobuf_fields(type_bytes, depth=3))
    tensor_type = _required_single_bytes(
        fields=type_fields,
        number=1,
        name="tensor type",
    )
    tensor_fields = tuple(_protobuf_fields(tensor_type, depth=4))
    data_type = _required_single_varint(tensor_fields, 1, "tensor element type")
    shape_bytes = _required_single_bytes(tensor_fields, 2, "tensor shape")
    shape_fields = tuple(_protobuf_fields(shape_bytes, depth=5))
    shape: list[int | str] = []
    for dimension_bytes in _all_bytes(shape_fields, 1):
        if len(shape) >= _MAXIMUM_ONNX_TENSOR_RANK:
            raise ValueError("ONNX value rank exceeds the safety limit")
        dimension_fields = tuple(_protobuf_fields(dimension_bytes, depth=6))
        static_values = _all_varints(dimension_fields, 1)
        symbolic_values = _all_bytes(dimension_fields, 2)
        if len(static_values) + len(symbolic_values) != 1:
            raise ValueError("ONNX tensor dimension must have exactly one value")
        if static_values:
            shape.append(_signed_int64(static_values[0]))
        else:
            shape.append(_decode_string(symbolic_values[0]))
    return ONNXValueInfo(name, data_type, tuple(shape))


def _protobuf_fields(
    data: memoryview,
    *,
    depth: int,
) -> Iterable[tuple[int, int, int | memoryview]]:
    if depth > _MAXIMUM_PROTO_DEPTH:
        raise ValueError("ONNX protobuf nesting exceeds the safety limit")
    position = 0
    field_count = 0
    while position < len(data):
        field_count += 1
        if field_count > MAXIMUM_ONNX_PROTOBUF_FIELDS:
            raise ValueError("ONNX protobuf message contains too many fields")
        key, position = _decode_varint(data, position)
        field_number = key >> 3
        wire_type = key & 7
        if field_number == 0:
            raise ValueError("ONNX protobuf contains field number zero")
        if field_number > _MAXIMUM_PROTOBUF_FIELD_NUMBER:
            raise ValueError("ONNX protobuf field number exceeds the wire limit")
        if wire_type == 0:
            value, position = _decode_varint(data, position)
            yield field_number, wire_type, value
            continue
        if wire_type == 1:
            end = position + 8
            if end > len(data):
                raise ValueError("truncated ONNX fixed64 field")
            yield field_number, wire_type, data[position:end]
            position = end
            continue
        if wire_type == 2:
            length, position = _decode_varint(data, position)
            end = position + length
            if end < position or end > len(data):
                raise ValueError("truncated ONNX length-delimited field")
            yield field_number, wire_type, data[position:end]
            position = end
            continue
        if wire_type == 5:
            end = position + 4
            if end > len(data):
                raise ValueError("truncated ONNX fixed32 field")
            yield field_number, wire_type, data[position:end]
            position = end
            continue
        raise ValueError(f"unsupported ONNX protobuf wire type {wire_type}")


def _decode_varint(data: memoryview, position: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if position >= len(data):
            raise ValueError("truncated ONNX protobuf varint")
        byte = int(data[position])
        position += 1
        if shift == 63 and byte > 1:
            raise ValueError("ONNX protobuf varint exceeds 64 bits")
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, position
    raise ValueError("ONNX protobuf varint exceeds ten bytes")


def _varint(value: int) -> bytes:
    value &= (1 << 64) - 1
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _field_key(number: int, wire_type: int) -> bytes:
    return _varint((number << 3) | wire_type)


def _varint_field(number: int, value: int) -> bytes:
    return _field_key(number, 0) + _varint(value)


def _fixed32_field(number: int, value: float) -> bytes:
    return _field_key(number, 5) + struct.pack("<f", value)


def _bytes_field(number: int, value: bytes) -> bytes:
    return _field_key(number, 2) + _varint(len(value)) + value


def _message_field(number: int, value: bytes) -> bytes:
    return _bytes_field(number, value)


def _string_field(number: int, value: str) -> bytes:
    return _bytes_field(number, value.encode("utf-8", errors="strict"))


def _all_varints(
    fields: Sequence[tuple[int, int, int | memoryview]],
    number: int,
) -> tuple[int, ...]:
    values: list[int] = []
    for field_number, wire_type, value in fields:
        if field_number == number:
            if wire_type != 0 or not isinstance(value, int):
                raise ValueError(f"ONNX protobuf field {number} has a wrong wire type")
            values.append(value)
    return tuple(values)


def _all_fixed32(
    fields: Sequence[tuple[int, int, int | memoryview]],
    number: int,
) -> tuple[memoryview, ...]:
    values: list[memoryview] = []
    for field_number, wire_type, value in fields:
        if field_number == number:
            if wire_type != 5 or not isinstance(value, memoryview):
                raise ValueError(
                    f"ONNX protobuf field {number} has a wrong wire type"
                )
            values.append(value)
    return tuple(values)


def _all_bytes(
    fields: Sequence[tuple[int, int, int | memoryview]],
    number: int,
) -> tuple[memoryview, ...]:
    values: list[memoryview] = []
    for field_number, wire_type, value in fields:
        if field_number == number:
            if wire_type != 2 or not isinstance(value, memoryview):
                raise ValueError(f"ONNX protobuf field {number} has a wrong wire type")
            values.append(value)
    return tuple(values)


def _required_single_varint(
    fields: Sequence[tuple[int, int, int | memoryview]],
    number: int,
    name: str,
) -> int:
    values = _all_varints(fields, number)
    if len(values) != 1:
        raise ValueError(f"ONNX {name} must appear exactly once")
    return values[0]


def _optional_single_varint(
    fields: Sequence[tuple[int, int, int | memoryview]],
    number: int,
    *,
    default: int,
) -> int:
    values = _all_varints(fields, number)
    if len(values) > 1:
        raise ValueError(f"ONNX protobuf field {number} appears more than once")
    return default if not values else values[0]


def _required_single_bytes(
    fields: Sequence[tuple[int, int, int | memoryview]],
    number: int,
    name: str,
) -> memoryview:
    values = _all_bytes(fields, number)
    if len(values) != 1:
        raise ValueError(f"ONNX {name} must appear exactly once")
    return values[0]


def _decode_string(value: memoryview) -> str:
    if len(value) > _MAXIMUM_ONNX_STRING_BYTES:
        raise ValueError("ONNX protobuf string exceeds the safety limit")
    try:
        return bytes(value).decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("ONNX protobuf string is not valid UTF-8") from error


def _required_single_string(
    fields: Sequence[tuple[int, int, int | memoryview]],
    number: int,
    name: str,
) -> str:
    value = _decode_string(_required_single_bytes(fields, number, name))
    if not value:
        raise ValueError(f"ONNX {name} must be nonempty")
    return value


def _optional_single_string(
    fields: Sequence[tuple[int, int, int | memoryview]],
    number: int,
) -> str:
    values = _all_bytes(fields, number)
    if len(values) > 1:
        raise ValueError(f"ONNX protobuf field {number} appears more than once")
    return "" if not values else _decode_string(values[0])


def _signed_int64(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


__all__ = [
    "MAXIMUM_ONNX_FILE_BYTES",
    "MAXIMUM_ONNX_INITIALIZERS",
    "MAXIMUM_ONNX_NODES",
    "MAXIMUM_ONNX_PROTOBUF_FIELDS",
    "MAXIMUM_ONNX_SIDECAR_BYTES",
    "ONNX_CANONICAL_GRAPH_FORMAT",
    "ONNX_CANONICAL_GRAPH_VERSION",
    "ONNX_INPUT_CONTRACT",
    "ONNX_IR_VERSION",
    "ONNX_OPSET_VERSION",
    "ONNX_PRODUCER_VERSION",
    "ONNX_SIDECAR_FORMAT",
    "ONNX_SIDECAR_SUFFIX",
    "ONNX_SIDECAR_VERSION",
    "ONNXInspection",
    "ONNXNodeInfo",
    "ONNXTensorInfo",
    "ONNXValueInfo",
    "UnsupportedONNXImportError",
    "export_onnx",
    "inspect_onnx",
    "load_onnx",
    "onnx_sidecar_path",
]
