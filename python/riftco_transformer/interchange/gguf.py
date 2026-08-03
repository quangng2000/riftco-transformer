"""Strict, dependency-free GGUF v3 interchange for Riftco decoder models.

The module deliberately supports one narrow and honest boundary: ordinary
FP32 weights for ``riftco_decoder_v1``.  It emits a standards-compliant GGUF
container with a custom ``general.architecture`` value (``riftco``).  Existing
GGUF runtimes still need an implementation of that architecture before they
can execute the file.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import struct
import tempfile
from types import MappingProxyType
from typing import Mapping

from .._numeric import positive_float32
from ..artifacts import ModelBundle, TokenizerSpec
from ..native import TransformerConfig
from .contracts import (
    Float32Tensor,
    build_bundle_from_tensors,
    bundle_tensors,
    current_decoder_parameter_specs,
)


GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGUF_DEFAULT_ALIGNMENT = 32
RIFTCO_GGUF_ARCHITECTURE = "riftco"
RIFTCO_ARCHITECTURE_ID = "riftco_decoder_v1"

_MAGIC = b"GGUF"
_MAXIMUM_ENTRY_COUNT = 1 << 20
_MAXIMUM_STRING_BYTES = 1 << 26
_MAXIMUM_ARRAY_DEPTH = 8
_MAXIMUM_ARRAY_ELEMENTS = 1 << 26

_UINT8 = 0
_INT8 = 1
_UINT16 = 2
_INT16 = 3
_UINT32 = 4
_INT32 = 5
_FLOAT32 = 6
_BOOL = 7
_STRING = 8
_ARRAY = 9
_UINT64 = 10
_INT64 = 11
_FLOAT64 = 12

_METADATA_KEY = re.compile(
    r"^[a-z0-9_]+(?:\.[a-z0-9_]+)*$",
    flags=re.ASCII,
)


class UnsupportedGGUFError(ValueError):
    """The GGUF container is valid but outside Riftco's import boundary."""


@dataclass(frozen=True, slots=True)
class GGUFTensorInfo:
    """One GGUF tensor descriptor, expressed in row-major Python shape order."""

    name: str
    shape: tuple[int, ...]
    ggml_type: int
    data_offset: int
    byte_count: int | None


@dataclass(frozen=True, slots=True)
class GGUFInspection:
    """Dependency-free structural view of a GGUF v3 file."""

    version: int
    architecture: str | None
    alignment: int
    metadata: Mapping[str, object]
    metadata_types: Mapping[str, int]
    tensors: tuple[GGUFTensorInfo, ...]
    tensor_data_offset: int
    file_size: int


@dataclass(frozen=True, slots=True)
class _MetadataValue:
    value_type: int
    value: object
    element_type: int | None = None


@dataclass(frozen=True, slots=True)
class _ParsedGGUF:
    inspection: GGUFInspection
    data: bytes


class _Reader:
    __slots__ = ("_data", "offset")

    def __init__(self, data: bytes) -> None:
        self._data = memoryview(data)
        self.offset = 0

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def remaining(self) -> int:
        return self.size - self.offset

    def take(self, count: int) -> memoryview:
        if count < 0 or count > self.remaining:
            raise ValueError("truncated GGUF file")
        start = self.offset
        self.offset += count
        return self._data[start:self.offset]

    def unpack(self, format_string: str) -> object:
        size = struct.calcsize(format_string)
        return struct.unpack(format_string, self.take(size))[0]


def export_gguf(
    bundle: ModelBundle,
    path: str | os.PathLike[str],
    *,
    model_name: str = "Riftco Decoder",
) -> Path:
    """Export one deterministic, single-file, FP32 GGUF v3 model.

    The tensor bytes are ordinary row-major FP32.  GGUF tensor dimensions are
    written in GGML's innermost-first order, while Riftco parameter names and
    values are preserved exactly for lossless re-import.
    """

    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a nonempty string")

    tensors = bundle_tensors(bundle)
    expected = tuple(
        (spec.name, spec.shape)
        for spec in current_decoder_parameter_specs(bundle.config)
    )
    _validate_current_tensors(tensors, expected)

    metadata = _bundle_metadata(bundle, model_name)
    tensor_payloads: list[tuple[str, Float32Tensor, bytes]] = []
    for name, tensor in tensors.items():
        payload = tensor.to_little_endian_bytes()
        tensor_payloads.append((name, tensor, payload))

    header = bytearray(_MAGIC)
    header.extend(struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(metadata)))
    for key, value in metadata:
        header.extend(_encode_string(key))
        header.extend(struct.pack("<I", value.value_type))
        header.extend(_encode_metadata_payload(value))

    relative_offset = 0
    for name, tensor, payload in tensor_payloads:
        encoded_name = name.encode("utf-8", errors="strict")
        if len(encoded_name) > 64:
            raise ValueError(
                f"GGUF tensor name exceeds 64 UTF-8 bytes: {name!r}"
            )
        if not 0 <= len(tensor.shape) <= 4:
            raise ValueError("GGUF v3 tensors must have at most four dimensions")
        header.extend(_encode_string(name))
        header.extend(struct.pack("<I", len(tensor.shape)))
        for dimension in reversed(tensor.shape):
            header.extend(struct.pack("<Q", dimension))
        header.extend(struct.pack("<IQ", GGML_TYPE_F32, relative_offset))
        relative_offset = _align(relative_offset + len(payload))

    header.extend(b"\x00" * (_align(len(header)) - len(header)))
    body = bytearray()
    for _name, _tensor, payload in tensor_payloads:
        body.extend(payload)
        body.extend(b"\x00" * (_align(len(body)) - len(body)))

    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise IsADirectoryError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(header)
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return destination


def inspect_gguf(path: str | os.PathLike[str]) -> GGUFInspection:
    """Parse GGUF structure and metadata without accepting its architecture."""

    return _parse_gguf(path).inspection


def load_gguf(path: str | os.PathLike[str]) -> ModelBundle:
    """Load a Riftco-authored FP32 GGUF into a native ``ModelBundle``.

    Quantized tensors, foreign architectures, and general GGUF-to-Riftco
    architecture conversion are intentionally rejected.
    """

    parsed = _parse_gguf(path)
    inspection = parsed.inspection
    metadata = inspection.metadata
    if inspection.architecture != RIFTCO_GGUF_ARCHITECTURE:
        raise UnsupportedGGUFError(
            "GGUF import supports only general.architecture='riftco'"
        )
    if metadata.get("riftco.architecture_id") != RIFTCO_ARCHITECTURE_ID:
        raise UnsupportedGGUFError(
            "GGUF import supports only riftco_decoder_v1"
        )
    for tensor in inspection.tensors:
        if tensor.ggml_type != GGML_TYPE_F32:
            raise UnsupportedGGUFError(
                "GGUF import supports only unquantized F32 tensors; "
                f"{tensor.name!r} has GGML type {tensor.ggml_type}"
            )

    layer_norm_epsilon = _positive_metadata_float(
        metadata, "riftco.layer_norm_epsilon"
    )
    attention_layer_norm_epsilon = _positive_metadata_float(
        metadata, "riftco.attention.layer_norm_epsilon"
    )
    if attention_layer_norm_epsilon != positive_float32(
        layer_norm_epsilon,
        "GGUF metadata riftco.layer_norm_epsilon",
    ):
        raise ValueError(
            "GGUF layer-normalization epsilon metadata fields disagree"
        )

    config = TransformerConfig(
        vocabulary_size=_positive_metadata_integer(
            metadata, "riftco.vocabulary_size"
        ),
        maximum_context=_positive_metadata_integer(
            metadata, "riftco.context_length"
        ),
        model_width=_positive_metadata_integer(
            metadata, "riftco.embedding_length"
        ),
        head_count=_positive_metadata_integer(
            metadata, "riftco.attention.head_count"
        ),
        block_count=_positive_metadata_integer(
            metadata, "riftco.block_count"
        ),
        feed_forward_width=_positive_metadata_integer(
            metadata, "riftco.feed_forward_length"
        ),
        random_seed=_nonnegative_metadata_integer(
            metadata, "riftco.random_seed", maximum=(1 << 32) - 1
        ),
        layer_norm_epsilon=layer_norm_epsilon,
    )
    if config.model_width % config.head_count != 0:
        raise ValueError("GGUF model width must be divisible by head count")

    tokenizer = _load_tokenizer(metadata)
    if tokenizer.vocabulary_size != config.vocabulary_size:
        raise ValueError("GGUF tokenizer vocabulary does not match the model")

    expected_specs = current_decoder_parameter_specs(config)
    expected = tuple((spec.name, spec.shape) for spec in expected_specs)
    actual = tuple((tensor.name, tensor.shape) for tensor in inspection.tensors)
    if actual != expected:
        raise UnsupportedGGUFError(
            "GGUF tensors do not match the exact riftco_decoder_v1 schema"
        )

    loaded_tensors: dict[str, Float32Tensor] = {}
    for tensor in inspection.tensors:
        assert tensor.byte_count is not None
        start = tensor.data_offset
        end = start + tensor.byte_count
        loaded_tensors[tensor.name] = Float32Tensor.from_little_endian_bytes(
            tensor.shape,
            parsed.data[start:end],
        )

    stage = _required_metadata_string(metadata, "riftco.stage")
    parent_present = metadata.get("riftco.parent_artifact_present")
    if not isinstance(parent_present, bool):
        raise ValueError("GGUF metadata riftco.parent_artifact_present is invalid")
    parent_artifact_id = (
        _required_metadata_string(metadata, "riftco.parent_artifact_id")
        if parent_present
        else None
    )
    metadata_json = _required_metadata_string(metadata, "riftco.metadata_json")
    try:
        artifact_metadata = json.loads(metadata_json)
    except json.JSONDecodeError as error:
        raise ValueError("GGUF Riftco metadata is not valid JSON") from error
    if not isinstance(artifact_metadata, dict):
        raise ValueError("GGUF Riftco artifact metadata must be a JSON object")

    return build_bundle_from_tensors(
        config=config,
        tokenizer=tokenizer,
        tensors=loaded_tensors,
        stage=stage,
        parent_artifact_id=parent_artifact_id,
        metadata=artifact_metadata,
    )


def _parse_gguf(path: str | os.PathLike[str]) -> _ParsedGGUF:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    data = source.read_bytes()
    reader = _Reader(data)
    if bytes(reader.take(4)) != _MAGIC:
        raise ValueError("file is not GGUF")
    version = int(reader.unpack("<I"))
    if version != GGUF_VERSION:
        raise UnsupportedGGUFError(
            f"only little-endian GGUF v{GGUF_VERSION} is supported"
        )
    tensor_count = int(reader.unpack("<Q"))
    metadata_count = int(reader.unpack("<Q"))
    if tensor_count > _MAXIMUM_ENTRY_COUNT:
        raise ValueError("GGUF tensor count exceeds the safety limit")
    if metadata_count > _MAXIMUM_ENTRY_COUNT:
        raise ValueError("GGUF metadata count exceeds the safety limit")

    metadata: dict[str, object] = {}
    metadata_types: dict[str, int] = {}
    for _index in range(metadata_count):
        key = _read_string(reader, "metadata key")
        encoded_key = key.encode("ascii", errors="strict")
        if len(encoded_key) > 65535 or _METADATA_KEY.fullmatch(key) is None:
            raise ValueError(f"invalid GGUF metadata key: {key!r}")
        if key in metadata:
            raise ValueError(f"duplicate GGUF metadata key: {key!r}")
        value_type = int(reader.unpack("<I"))
        metadata[key] = _read_metadata_payload(reader, value_type, depth=0)
        metadata_types[key] = value_type

    raw_tensors: list[tuple[str, tuple[int, ...], int, int]] = []
    names: set[str] = set()
    for _index in range(tensor_count):
        name = _read_string(reader, "tensor name")
        if len(name.encode("utf-8")) > 64:
            raise ValueError("GGUF tensor name exceeds 64 UTF-8 bytes")
        if not name or name in names:
            raise ValueError("GGUF tensor names must be nonempty and unique")
        names.add(name)
        rank = int(reader.unpack("<I"))
        if rank > 4:
            raise UnsupportedGGUFError(
                "GGUF tensors with more than four dimensions are unsupported"
            )
        gguf_dimensions = tuple(int(reader.unpack("<Q")) for _ in range(rank))
        if any(dimension == 0 for dimension in gguf_dimensions):
            raise ValueError("GGUF tensor dimensions must be positive")
        shape = tuple(reversed(gguf_dimensions))
        ggml_type = int(reader.unpack("<I"))
        relative_offset = int(reader.unpack("<Q"))
        raw_tensors.append((name, shape, ggml_type, relative_offset))

    alignment_value = metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)
    if (
        isinstance(alignment_value, bool)
        or not isinstance(alignment_value, int)
        or alignment_value < 8
        or alignment_value % 8 != 0
    ):
        raise ValueError("GGUF general.alignment must be a positive multiple of 8")
    tensor_data_offset = _align(reader.offset, alignment_value)
    if tensor_data_offset > len(data):
        raise ValueError("truncated GGUF tensor-data alignment padding")

    tensors: list[GGUFTensorInfo] = []
    occupied: list[tuple[int, int, str]] = []
    for name, shape, ggml_type, relative_offset in raw_tensors:
        if relative_offset % alignment_value != 0:
            raise ValueError("GGUF tensor offset is not correctly aligned")
        absolute_offset = tensor_data_offset + relative_offset
        byte_count: int | None = None
        if ggml_type == GGML_TYPE_F32:
            element_count = math.prod(shape)
            byte_count = element_count * 4
            if absolute_offset > len(data) or byte_count > len(data) - absolute_offset:
                raise ValueError(f"truncated GGUF tensor payload: {name!r}")
            occupied.append((absolute_offset, absolute_offset + byte_count, name))
        elif absolute_offset > len(data):
            raise ValueError(f"GGUF tensor offset is outside the file: {name!r}")
        tensors.append(
            GGUFTensorInfo(
                name=name,
                shape=shape,
                ggml_type=ggml_type,
                data_offset=absolute_offset,
                byte_count=byte_count,
            )
        )
    occupied.sort()
    for left, right in zip(occupied, occupied[1:]):
        if left[1] > right[0]:
            raise ValueError(
                f"GGUF tensor payloads overlap: {left[2]!r} and {right[2]!r}"
            )

    architecture = metadata.get("general.architecture")
    if architecture is not None and not isinstance(architecture, str):
        raise ValueError("GGUF general.architecture must be a string")
    inspection = GGUFInspection(
        version=version,
        architecture=architecture,
        alignment=alignment_value,
        metadata=MappingProxyType(metadata),
        metadata_types=MappingProxyType(metadata_types),
        tensors=tuple(tensors),
        tensor_data_offset=tensor_data_offset,
        file_size=len(data),
    )
    return _ParsedGGUF(inspection=inspection, data=data)


def _bundle_metadata(
    bundle: ModelBundle,
    model_name: str,
) -> tuple[tuple[str, _MetadataValue], ...]:
    config = bundle.config
    tokenizer = bundle.tokenizer
    values: list[tuple[str, _MetadataValue]] = [
        ("general.architecture", _MetadataValue(_STRING, RIFTCO_GGUF_ARCHITECTURE)),
        ("general.name", _MetadataValue(_STRING, model_name)),
        ("general.description", _MetadataValue(
            _STRING,
            "Riftco decoder-only Transformer with learned absolute positions, "
            "pre-LayerNorm blocks, causal self-attention, and GELU "
            "feed-forward layers.",
        )),
        ("general.file_type", _MetadataValue(_UINT32, 0)),
        ("general.alignment", _MetadataValue(_UINT32, GGUF_DEFAULT_ALIGNMENT)),
        ("riftco.architecture_id", _MetadataValue(_STRING, RIFTCO_ARCHITECTURE_ID)),
        ("riftco.architecture_version", _MetadataValue(_UINT32, 1)),
        ("riftco.context_length", _MetadataValue(_UINT64, config.maximum_context)),
        ("riftco.embedding_length", _MetadataValue(_UINT64, config.model_width)),
        ("riftco.block_count", _MetadataValue(_UINT64, config.block_count)),
        ("riftco.feed_forward_length", _MetadataValue(
            _UINT64, config.feed_forward_width
        )),
        ("riftco.attention.head_count", _MetadataValue(_UINT64, config.head_count)),
        ("riftco.attention.layer_norm_epsilon", _MetadataValue(
            _FLOAT32, config.layer_norm_epsilon
        )),
        ("riftco.layer_norm_epsilon", _MetadataValue(
            _FLOAT64, config.layer_norm_epsilon
        )),
        ("riftco.vocabulary_size", _MetadataValue(_UINT64, config.vocabulary_size)),
        ("riftco.random_seed", _MetadataValue(_UINT64, config.random_seed)),
        ("riftco.activation", _MetadataValue(_STRING, "gelu")),
        ("riftco.position_embedding", _MetadataValue(_STRING, "learned_absolute")),
        ("riftco.normalization", _MetadataValue(_STRING, "pre_layer_norm")),
        ("riftco.tensor_data_layout", _MetadataValue(_STRING, "reference")),
        ("riftco.stage", _MetadataValue(_STRING, bundle.stage)),
        ("riftco.parent_artifact_present", _MetadataValue(
            _BOOL, bundle.parent_artifact_id is not None
        )),
        ("riftco.metadata_json", _MetadataValue(
            _STRING,
            json.dumps(
                bundle.metadata,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )),
        ("riftco.tokenizer.method", _MetadataValue(_STRING, tokenizer.method)),
    ]
    if bundle.parent_artifact_id is not None:
        values.append((
            "riftco.parent_artifact_id",
            _MetadataValue(_STRING, bundle.parent_artifact_id),
        ))
    if tokenizer.method == "byte":
        values.append((
            "riftco.tokenizer.byte_vocabulary",
            _MetadataValue(_ARRAY, tokenizer.byte_vocabulary, _UINT32),
        ))
    else:
        left = tuple(rule[0] for rule in tokenizer.merge_rules)
        right = tuple(rule[1] for rule in tokenizer.merge_rules)
        result = tuple(rule[2] for rule in tokenizer.merge_rules)
        values.extend((
            ("riftco.tokenizer.merge_left", _MetadataValue(_ARRAY, left, _UINT32)),
            ("riftco.tokenizer.merge_right", _MetadataValue(_ARRAY, right, _UINT32)),
            ("riftco.tokenizer.merge_result", _MetadataValue(_ARRAY, result, _UINT32)),
        ))
    return tuple(values)


def _load_tokenizer(metadata: Mapping[str, object]) -> TokenizerSpec:
    method = _required_metadata_string(metadata, "riftco.tokenizer.method")
    if method == "byte":
        vocabulary = _required_metadata_integer_array(
            metadata, "riftco.tokenizer.byte_vocabulary"
        )
        return TokenizerSpec(method="byte", byte_vocabulary=vocabulary)
    if method == "bpe":
        left = _required_metadata_integer_array(metadata, "riftco.tokenizer.merge_left")
        right = _required_metadata_integer_array(
            metadata, "riftco.tokenizer.merge_right"
        )
        result = _required_metadata_integer_array(
            metadata, "riftco.tokenizer.merge_result"
        )
        if not (len(left) == len(right) == len(result)):
            raise ValueError("GGUF BPE merge arrays have different lengths")
        return TokenizerSpec(
            method="bpe",
            merge_rules=tuple(zip(left, right, result)),
        )
    raise UnsupportedGGUFError("GGUF contains an unsupported Riftco tokenizer")


def _validate_current_tensors(
    tensors: Mapping[str, Float32Tensor],
    expected: tuple[tuple[str, tuple[int, ...]], ...],
) -> None:
    actual = tuple((name, tensor.shape) for name, tensor in tensors.items())
    if actual != expected:
        raise ValueError(
            "bundle parameters do not match the current riftco_decoder_v1 schema"
        )


def _encode_metadata_payload(value: _MetadataValue) -> bytes:
    value_type = value.value_type
    raw = value.value
    formats = {
        _UINT8: "<B",
        _INT8: "<b",
        _UINT16: "<H",
        _INT16: "<h",
        _UINT32: "<I",
        _INT32: "<i",
        _FLOAT32: "<f",
        _UINT64: "<Q",
        _INT64: "<q",
        _FLOAT64: "<d",
    }
    if value_type in formats:
        return struct.pack(formats[value_type], raw)
    if value_type == _BOOL:
        return struct.pack("<B", int(bool(raw)))
    if value_type == _STRING:
        if not isinstance(raw, str):
            raise TypeError("GGUF string metadata must be a string")
        return _encode_string(raw)
    if value_type == _ARRAY:
        if value.element_type is None:
            raise TypeError("GGUF array metadata requires an element type")
        elements = tuple(raw)  # type: ignore[arg-type]
        output = bytearray(struct.pack("<IQ", value.element_type, len(elements)))
        for element in elements:
            output.extend(_encode_metadata_payload(
                _MetadataValue(value.element_type, element)
            ))
        return bytes(output)
    raise TypeError(f"unsupported GGUF metadata type {value_type}")


def _read_metadata_payload(
    reader: _Reader,
    value_type: int,
    *,
    depth: int,
) -> object:
    formats = {
        _UINT8: "<B",
        _INT8: "<b",
        _UINT16: "<H",
        _INT16: "<h",
        _UINT32: "<I",
        _INT32: "<i",
        _FLOAT32: "<f",
        _UINT64: "<Q",
        _INT64: "<q",
        _FLOAT64: "<d",
    }
    if value_type in formats:
        return reader.unpack(formats[value_type])
    if value_type == _BOOL:
        value = int(reader.unpack("<B"))
        if value not in (0, 1):
            raise ValueError("GGUF boolean metadata must be zero or one")
        return bool(value)
    if value_type == _STRING:
        return _read_string(reader, "metadata string")
    if value_type == _ARRAY:
        if depth >= _MAXIMUM_ARRAY_DEPTH:
            raise ValueError("GGUF metadata arrays are nested too deeply")
        element_type = int(reader.unpack("<I"))
        count = int(reader.unpack("<Q"))
        if count > _MAXIMUM_ARRAY_ELEMENTS:
            raise ValueError("GGUF metadata array exceeds the safety limit")
        return tuple(
            _read_metadata_payload(reader, element_type, depth=depth + 1)
            for _ in range(count)
        )
    raise UnsupportedGGUFError(
        f"unsupported GGUF metadata value type {value_type}"
    )


def _encode_string(value: str) -> bytes:
    encoded = value.encode("utf-8", errors="strict")
    if len(encoded) > _MAXIMUM_STRING_BYTES:
        raise ValueError("GGUF string exceeds the safety limit")
    return struct.pack("<Q", len(encoded)) + encoded


def _read_string(reader: _Reader, description: str) -> str:
    length = int(reader.unpack("<Q"))
    if length > _MAXIMUM_STRING_BYTES:
        raise ValueError(f"GGUF {description} exceeds the safety limit")
    try:
        return bytes(reader.take(length)).decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"GGUF {description} is not valid UTF-8") from error


def _align(value: int, alignment: int = GGUF_DEFAULT_ALIGNMENT) -> int:
    return value + (alignment - value % alignment) % alignment


def _required_metadata_string(
    metadata: Mapping[str, object],
    key: str,
) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"GGUF metadata {key} must be a nonempty string")
    return value


def _required_metadata_integer_array(
    metadata: Mapping[str, object],
    key: str,
) -> tuple[int, ...]:
    value = metadata.get(key)
    if not isinstance(value, tuple):
        raise ValueError(f"GGUF metadata {key} must be an integer array")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"GGUF metadata {key} must be an integer array")
    return value


def _positive_metadata_integer(
    metadata: Mapping[str, object],
    key: str,
) -> int:
    value = _nonnegative_metadata_integer(metadata, key)
    if value == 0:
        raise ValueError(f"GGUF metadata {key} must be positive")
    return value


def _nonnegative_metadata_integer(
    metadata: Mapping[str, object],
    key: str,
    *,
    maximum: int = (1 << 63) - 1,
) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"GGUF metadata {key} must be an integer")
    if value < 0 or value > maximum:
        raise ValueError(f"GGUF metadata {key} is outside the supported range")
    return value


def _positive_metadata_float(
    metadata: Mapping[str, object],
    key: str,
) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"GGUF metadata {key} must be numeric")
    try:
        positive_float32(value, f"GGUF metadata {key}")
    except TypeError as error:
        raise ValueError(str(error)) from error
    result = float(value)
    return result


__all__ = [
    "GGML_TYPE_F32",
    "GGUF_VERSION",
    "GGUFInspection",
    "GGUFTensorInfo",
    "RIFTCO_GGUF_ARCHITECTURE",
    "UnsupportedGGUFError",
    "export_gguf",
    "inspect_gguf",
    "load_gguf",
]
