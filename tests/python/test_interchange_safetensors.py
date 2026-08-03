from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import riftco_transformer.interchange.safetensors as safetensors_module
from riftco_transformer.interchange import (
    Float32Tensor,
    SAFETENSORS_MAXIMUM_HEADER_BYTES,
    SAFETENSORS_MAXIMUM_TENSOR_COUNT,
    decode_safetensors,
    encode_safetensors,
    load_safetensors,
    save_safetensors,
)


def encoded_header(source: str, payload: bytes = b"") -> bytes:
    header = source.encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    return struct.pack("<Q", len(header)) + header + payload


def encoded_object(header: object, payload: bytes = b"") -> bytes:
    source = json.dumps(header, separators=(",", ":"))
    return encoded_header(source, payload)


class Float32TensorTests(unittest.TestCase):
    def test_values_are_canonical_f32_and_shape_checked(self) -> None:
        tensor = Float32Tensor((2,), (0.1, -0.2))
        expected = tuple(
            value[0]
            for value in struct.iter_unpack(
                "<f",
                struct.pack("<ff", 0.1, -0.2),
            )
        )
        self.assertEqual(tensor.values, expected)
        self.assertEqual(tensor.byte_count, 8)

        with self.assertRaisesRegex(ValueError, "value count"):
            Float32Tensor((2,), (1.0,))
        with self.assertRaisesRegex(ValueError, "finite"):
            Float32Tensor((1,), (float("nan"),))
        with self.assertRaisesRegex(TypeError, "dimension"):
            Float32Tensor((True,), (1.0,))

    def test_zero_extent_and_scalar_tensors_are_unambiguous(self) -> None:
        self.assertEqual(Float32Tensor((), (2.0,)).value_count, 1)
        self.assertEqual(Float32Tensor((2, 0, 3), ()).value_count, 0)


class SafeTensorsTests(unittest.TestCase):
    def test_deterministic_round_trip_with_metadata(self) -> None:
        first = {
            "z": Float32Tensor((2,), (1.0, 2.0)),
            "a": Float32Tensor((1, 1), (-3.5,)),
        }
        second = {"a": first["a"], "z": first["z"]}
        encoded_first = encode_safetensors(
            first,
            metadata={"z": "last", "a": "first"},
        )
        encoded_second = encode_safetensors(
            second,
            metadata={"a": "first", "z": "last"},
        )
        self.assertEqual(encoded_first, encoded_second)
        header_length = struct.unpack_from("<Q", encoded_first)[0]
        self.assertEqual(header_length % 8, 0)

        parsed = decode_safetensors(encoded_first)
        self.assertEqual(parsed.tensors, second)
        self.assertEqual(parsed.metadata, {"a": "first", "z": "last"})
        detached = parsed.tensors
        detached.clear()
        self.assertEqual(set(parsed.tensors), {"a", "z"})

    def test_file_publication_preserves_existing_destination(self) -> None:
        tensor = {"x": Float32Tensor((1,), (4.0,))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            save_safetensors(tensor, path)
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                save_safetensors(
                    {"x": Float32Tensor((1,), (5.0,))},
                    path,
                )
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(load_safetensors(path).tensors, tensor)

    def test_rejects_unaligned_header_length(self) -> None:
        malformed = struct.pack("<Q", 2) + b"{}"
        with self.assertRaisesRegex(ValueError, "divisible by eight"):
            decode_safetensors(malformed)

    def test_rejects_duplicate_header_keys(self) -> None:
        malformed = encoded_header(
            '{"__metadata__":{},"__metadata__":{}}'
        )
        with self.assertRaisesRegex(ValueError, "strict JSON"):
            decode_safetensors(malformed)

    def test_rejects_non_space_header_padding(self) -> None:
        header = b"{}\n     "
        self.assertEqual(len(header) % 8, 0)
        malformed = struct.pack("<Q", len(header)) + header
        with self.assertRaisesRegex(ValueError, "ASCII spaces"):
            decode_safetensors(malformed)

    def test_rejects_unsupported_dtype(self) -> None:
        malformed = encoded_object(
            {
                "x": {
                    "dtype": "F16",
                    "shape": [1],
                    "data_offsets": [0, 2],
                }
            },
            b"\0\0",
        )
        with self.assertRaisesRegex(ValueError, "only F32"):
            decode_safetensors(malformed)

    def test_rejects_shape_range_mismatch_and_out_of_bounds(self) -> None:
        mismatch = encoded_object(
            {
                "x": {
                    "dtype": "F32",
                    "shape": [2],
                    "data_offsets": [0, 4],
                }
            },
            b"\0" * 4,
        )
        with self.assertRaisesRegex(ValueError, "requires 8"):
            decode_safetensors(mismatch)

        out_of_bounds = encoded_object(
            {
                "x": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "exceeds the data buffer"):
            decode_safetensors(out_of_bounds)

    def test_rejects_gaps_overlaps_and_unreferenced_data(self) -> None:
        gap = encoded_object(
            {
                "x": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [4, 8],
                }
            },
            b"\0" * 8,
        )
        with self.assertRaisesRegex(ValueError, "leaves a gap"):
            decode_safetensors(gap)

        overlap = encoded_object(
            {
                "x": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                },
                "y": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                },
            },
            b"\0" * 4,
        )
        with self.assertRaisesRegex(ValueError, "overlaps prior data"):
            decode_safetensors(overlap)

        unreferenced = encoded_object({}, b"\0")
        with self.assertRaisesRegex(ValueError, "unreferenced bytes"):
            decode_safetensors(unreferenced)

    def test_rejects_nonfinite_f32_payload(self) -> None:
        malformed = encoded_object(
            {
                "x": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                }
            },
            struct.pack("<f", float("inf")),
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            decode_safetensors(malformed)

    def test_rejects_excessive_zero_sized_tensor_descriptors(self) -> None:
        descriptor = {
            "dtype": "F32",
            "shape": [0],
            "data_offsets": [0, 0],
        }
        malformed = encoded_object(
            {
                f"empty_{index:05d}": descriptor
                for index in range(SAFETENSORS_MAXIMUM_TENSOR_COUNT + 2)
            }
        )
        header_length = struct.unpack_from("<Q", malformed)[0]
        self.assertLess(header_length, SAFETENSORS_MAXIMUM_HEADER_BYTES)
        with mock.patch.object(
            safetensors_module,
            "_parse_header",
            side_effect=AssertionError("JSON decoder must not run"),
        ):
            with self.assertRaisesRegex(ValueError, "tensor count exceeds"):
                decode_safetensors(malformed)

        exact_overflow = encoded_object(
            {
                f"empty_{index:05d}": descriptor
                for index in range(SAFETENSORS_MAXIMUM_TENSOR_COUNT + 1)
            }
        )
        with self.assertRaisesRegex(ValueError, "tensor count exceeds"):
            decode_safetensors(exact_overflow)

    def test_descriptor_preflight_ignores_nested_and_escaped_punctuation(
        self,
    ) -> None:
        valid = encoded_object(
            {
                "__metadata__": {
                    "punctuation": '\\"colon: braces{} brackets[]',
                },
                "x": {
                    "dtype": "F32",
                    "shape": [1],
                    "data_offsets": [0, 4],
                },
            },
            struct.pack("<f", 3.0),
        )
        parsed = decode_safetensors(valid)
        self.assertEqual(parsed.tensors["x"].values, (3.0,))


if __name__ == "__main__":
    unittest.main()
