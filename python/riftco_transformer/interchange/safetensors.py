"""Strict, dependency-free SafeTensors F32 reader and writer."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import struct
import tempfile
from typing import NoReturn

from .contracts import Float32Tensor, MAXIMUM_TENSOR_ELEMENTS


SAFETENSORS_METADATA_KEY = "__metadata__"
SAFETENSORS_DTYPE_F32 = "F32"
MAXIMUM_HEADER_BYTES = 16 << 20
MAXIMUM_TENSOR_COUNT = 1 << 13


class SafeTensorsFile:
    """An immutable logical view of one parsed SafeTensors file."""

    __slots__ = ("_metadata_entries", "_tensor_entries")

    def __init__(
        self,
        tensors: Mapping[str, Float32Tensor],
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        checked_tensors = _validate_tensors(tensors)
        checked_metadata = _validate_metadata(metadata)
        self._tensor_entries = tuple(sorted(checked_tensors.items()))
        self._metadata_entries = tuple(sorted(checked_metadata.items()))

    @property
    def tensors(self) -> dict[str, Float32Tensor]:
        """Return a detached mapping of tensor names to immutable tensors."""

        return dict(self._tensor_entries)

    @property
    def metadata(self) -> dict[str, str]:
        """Return a detached copy of string metadata."""

        return dict(self._metadata_entries)


def encode_safetensors(
    tensors: Mapping[str, Float32Tensor],
    *,
    metadata: Mapping[str, str] | None = None,
) -> bytes:
    """Encode named F32 tensors deterministically in SafeTensors format."""

    checked_tensors = _validate_tensors(tensors)
    checked_metadata = _validate_metadata(metadata)

    header: dict[str, object] = {}
    if checked_metadata:
        header[SAFETENSORS_METADATA_KEY] = checked_metadata
    data_parts: list[bytes] = []
    offset = 0
    for name in sorted(checked_tensors):
        tensor = checked_tensors[name]
        next_offset = offset + tensor.byte_count
        header[name] = {
            "dtype": SAFETENSORS_DTYPE_F32,
            "shape": list(tensor.shape),
            "data_offsets": [offset, next_offset],
        }
        data_parts.append(tensor.to_little_endian_bytes())
        offset = next_offset

    try:
        unpadded_header = json.dumps(
            header,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise TypeError("SafeTensors header is not JSON-serializable") from error
    padding = (-len(unpadded_header)) % 8
    encoded_header = unpadded_header + (b" " * padding)
    if len(encoded_header) > MAXIMUM_HEADER_BYTES:
        raise ValueError("SafeTensors header exceeds the supported size limit")
    return (
        struct.pack("<Q", len(encoded_header))
        + encoded_header
        + b"".join(data_parts)
    )


def decode_safetensors(
    data: bytes | bytearray | memoryview,
) -> SafeTensorsFile:
    """Parse a complete SafeTensors byte sequence with strict bounds checks."""

    try:
        view = memoryview(data).cast("B")
    except (TypeError, ValueError) as error:
        raise TypeError("SafeTensors data must be bytes-like") from error
    if len(view) < 8:
        raise ValueError("SafeTensors data is shorter than its length prefix")
    header_length = struct.unpack_from("<Q", view, 0)[0]
    _validate_header_length(header_length, len(view) - 8)
    header_end = 8 + header_length
    header_bytes = bytes(view[8:header_end])
    payload = view[header_end:]
    return _decode_parts(header_bytes, payload)


def save_safetensors(
    tensors: Mapping[str, Float32Tensor],
    path: str | os.PathLike[str],
    *,
    metadata: Mapping[str, str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Atomically publish one deterministic SafeTensors file.

    Existing destinations are preserved unless ``overwrite`` is explicitly
    true. The temporary file is created beside the destination so publication
    stays on one filesystem.
    """

    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a bool")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = encode_safetensors(tensors, metadata=metadata)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        if overwrite:
            os.replace(temporary_path, destination)
        else:
            try:
                os.link(temporary_path, destination)
            except FileExistsError as error:
                raise FileExistsError(
                    f"SafeTensors destination already exists: {destination}"
                ) from error
            temporary_path.unlink()
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
    return destination


def load_safetensors(
    path: str | os.PathLike[str],
) -> SafeTensorsFile:
    """Load one SafeTensors file without trusting header-provided offsets."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as input_file:
        prefix = input_file.read(8)
        if len(prefix) != 8:
            raise ValueError(
                "SafeTensors file is shorter than its length prefix"
            )
        header_length = struct.unpack("<Q", prefix)[0]
        file_size = os.fstat(input_file.fileno()).st_size
        _validate_header_length(header_length, file_size - 8)
        header_bytes = input_file.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError("SafeTensors header is truncated")
        payload = input_file.read()
    return _decode_parts(header_bytes, memoryview(payload))


def _decode_parts(
    header_bytes: bytes,
    payload: memoryview,
) -> SafeTensorsFile:
    _preflight_top_level_entry_count(header_bytes)
    header = _parse_header(header_bytes)
    tensor_count = len(header) - int(SAFETENSORS_METADATA_KEY in header)
    if tensor_count > MAXIMUM_TENSOR_COUNT:
        raise ValueError(
            "SafeTensors tensor count exceeds the supported limit "
            f"{MAXIMUM_TENSOR_COUNT}"
        )
    raw_metadata = header.pop(SAFETENSORS_METADATA_KEY, {})
    if not isinstance(raw_metadata, dict):
        raise ValueError("SafeTensors __metadata__ must be an object")
    metadata = _validate_metadata(raw_metadata)

    descriptors: list[tuple[int, int, str, tuple[int, ...]]] = []
    for name, raw_descriptor in header.items():
        if not isinstance(name, str) or not name:
            raise ValueError("SafeTensors tensor names must be nonempty strings")
        if name == SAFETENSORS_METADATA_KEY:
            raise ValueError("reserved SafeTensors metadata key used as tensor")
        if not isinstance(raw_descriptor, dict):
            raise ValueError(
                f"SafeTensors descriptor for {name!r} must be an object"
            )
        required_fields = {"dtype", "shape", "data_offsets"}
        if set(raw_descriptor) != required_fields:
            missing = sorted(required_fields - set(raw_descriptor))
            extra = sorted(set(raw_descriptor) - required_fields)
            detail: list[str] = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("unexpected " + ", ".join(extra))
            raise ValueError(
                f"SafeTensors descriptor for {name!r} is invalid: "
                + "; ".join(detail)
            )
        dtype = raw_descriptor["dtype"]
        if dtype != SAFETENSORS_DTYPE_F32:
            raise ValueError(
                f"SafeTensors tensor {name!r} uses unsupported dtype "
                f"{dtype!r}; this runtime currently accepts only F32 weights"
            )
        shape = _parse_shape(name, raw_descriptor["shape"])
        offsets = raw_descriptor["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(
                f"SafeTensors tensor {name!r} data_offsets must contain "
                "exactly two integers"
            )
        start = _nonnegative_integer(offsets[0], f"{name!r} start offset")
        end = _nonnegative_integer(offsets[1], f"{name!r} end offset")
        if end < start:
            raise ValueError(
                f"SafeTensors tensor {name!r} end offset precedes its start"
            )
        expected_bytes = _element_count(shape) * 4
        if end - start != expected_bytes:
            raise ValueError(
                f"SafeTensors tensor {name!r} byte range has size "
                f"{end - start}; shape {shape} requires {expected_bytes}"
            )
        if end > len(payload):
            raise ValueError(
                f"SafeTensors tensor {name!r} range exceeds the data buffer"
            )
        descriptors.append((start, end, name, shape))

    descriptors.sort(key=lambda item: (item[0], item[1], item[2]))
    cursor = 0
    for start, end, name, _shape in descriptors:
        if start != cursor:
            relation = "overlaps prior data" if start < cursor else "leaves a gap"
            raise ValueError(
                f"SafeTensors tensor {name!r} {relation} in the data buffer"
            )
        cursor = end
    if cursor != len(payload):
        raise ValueError("SafeTensors data buffer contains unreferenced bytes")

    tensors: dict[str, Float32Tensor] = {}
    for start, end, name, shape in descriptors:
        tensors[name] = Float32Tensor.from_little_endian_bytes(
            shape,
            payload[start:end],
        )
    return SafeTensorsFile(tensors, metadata)


def _preflight_top_level_entry_count(header_bytes: bytes) -> None:
    """Bound root JSON entries before the decoder allocates Python objects.

    Every colon at root-object depth separates one top-level key from its
    value. Colons inside escaped strings or nested descriptor/metadata values
    are ignored. One extra entry is allowed for ``__metadata__``; the exact
    tensor count is checked again after strict JSON parsing.
    """

    maximum_entries = MAXIMUM_TENSOR_COUNT + 1
    nesting_depth = 0
    entry_count = 0
    in_string = False
    escaped = False
    for byte in header_bytes:
        if in_string:
            if escaped:
                escaped = False
            elif byte == ord("\\"):
                escaped = True
            elif byte == ord('"'):
                in_string = False
            continue

        if byte == ord('"'):
            in_string = True
        elif byte in (ord("{"), ord("[")):
            nesting_depth += 1
        elif byte in (ord("}"), ord("]")):
            nesting_depth -= 1
        elif byte == ord(":") and nesting_depth == 1:
            entry_count += 1
            if entry_count > maximum_entries:
                raise ValueError(
                    "SafeTensors tensor count exceeds the supported limit "
                    f"{MAXIMUM_TENSOR_COUNT}"
                )


def _parse_header(header_bytes: bytes) -> dict[str, object]:
    if not header_bytes:
        raise ValueError("SafeTensors header must not be empty")
    if header_bytes[:1] != b"{":
        raise ValueError("SafeTensors header must begin with a JSON object")
    try:
        decoded = header_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("SafeTensors header is not valid UTF-8") from error
    decoder = json.JSONDecoder(
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )
    try:
        value, end = decoder.raw_decode(decoded)
    except (
        json.JSONDecodeError,
        _DuplicateJsonKey,
        RecursionError,
        ValueError,
    ) as error:
        raise ValueError("SafeTensors header is not valid strict JSON") from error
    if any(character != " " for character in decoded[end:]):
        raise ValueError(
            "SafeTensors header padding must contain only ASCII spaces"
        )
    if not isinstance(value, dict):
        raise ValueError("SafeTensors header must be a JSON object")
    return value


def _validate_header_length(header_length: int, remaining_bytes: int) -> None:
    if header_length == 0:
        raise ValueError("SafeTensors header length must be greater than zero")
    if header_length > MAXIMUM_HEADER_BYTES:
        raise ValueError("SafeTensors header exceeds the supported size limit")
    if header_length % 8:
        raise ValueError("SafeTensors header length must be divisible by eight")
    if header_length > remaining_bytes:
        raise ValueError("SafeTensors header length exceeds the file size")


def _validate_tensors(
    tensors: Mapping[str, Float32Tensor],
) -> dict[str, Float32Tensor]:
    if not isinstance(tensors, Mapping):
        raise TypeError("tensors must be a mapping")
    result: dict[str, Float32Tensor] = {}
    for index, (name, tensor) in enumerate(tensors.items()):
        if index >= MAXIMUM_TENSOR_COUNT:
            raise ValueError(
                "SafeTensors tensor count exceeds the supported limit "
                f"{MAXIMUM_TENSOR_COUNT}"
            )
        if not isinstance(name, str) or not name:
            raise ValueError("SafeTensors tensor names must be nonempty strings")
        if name == SAFETENSORS_METADATA_KEY:
            raise ValueError(
                f"{SAFETENSORS_METADATA_KEY!r} is reserved for metadata"
            )
        if not isinstance(tensor, Float32Tensor):
            raise TypeError(
                f"SafeTensors tensor {name!r} must be a Float32Tensor"
            )
        result[name] = tensor
    return result


def _validate_metadata(
    metadata: Mapping[str, str] | None,
) -> dict[str, str]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("SafeTensors metadata must be a mapping")
    result: dict[str, str] = {}
    for name, value in metadata.items():
        if not isinstance(name, str):
            raise TypeError("SafeTensors metadata keys must be strings")
        if not isinstance(value, str):
            raise TypeError(
                f"SafeTensors metadata value for {name!r} must be a string"
            )
        result[name] = value
    return result


def _parse_shape(name: str, raw_shape: object) -> tuple[int, ...]:
    if not isinstance(raw_shape, list):
        raise ValueError(
            f"SafeTensors tensor {name!r} shape must be an array"
        )
    shape = tuple(
        _nonnegative_integer(value, f"{name!r} shape dimension")
        for value in raw_shape
    )
    _element_count(shape)
    return shape


def _element_count(shape: tuple[int, ...]) -> int:
    count = 1
    for dimension in shape:
        if dimension == 0:
            count = 0
        elif count:
            if count > MAXIMUM_TENSOR_ELEMENTS // dimension:
                raise ValueError("SafeTensors tensor element count is too large")
            count *= dimension
    return count


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"SafeTensors {name} must be an integer")
    if value < 0:
        raise ValueError(f"SafeTensors {name} must not be negative")
    if value > MAXIMUM_TENSOR_ELEMENTS:
        raise ValueError(f"SafeTensors {name} is too large")
    return value


class _DuplicateJsonKey(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in pairs:
        if name in result:
            raise _DuplicateJsonKey(name)
        result[name] = value
    return result


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


__all__ = [
    "MAXIMUM_HEADER_BYTES",
    "MAXIMUM_TENSOR_COUNT",
    "SAFETENSORS_DTYPE_F32",
    "SAFETENSORS_METADATA_KEY",
    "SafeTensorsFile",
    "decode_safetensors",
    "encode_safetensors",
    "load_safetensors",
    "save_safetensors",
]
