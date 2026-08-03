from __future__ import annotations

from pathlib import Path
import struct
import tempfile
import unittest

from riftco_transformer.artifacts import ModelBundle, TokenizerSpec
from riftco_transformer.interchange.contracts import (
    MAXIMUM_DECODER_BLOCKS,
    current_decoder_parameter_specs,
)
from riftco_transformer.interchange.gguf import (
    GGML_TYPE_F32,
    GGUF_VERSION,
    RIFTCO_GGUF_ARCHITECTURE,
    UnsupportedGGUFError,
    export_gguf,
    inspect_gguf,
    load_gguf,
)
from riftco_transformer.native import TransformerConfig


def make_bundle() -> ModelBundle:
    config = TransformerConfig(
        vocabulary_size=4,
        maximum_context=4,
        model_width=4,
        head_count=2,
        block_count=1,
        feed_forward_width=8,
        random_seed=73,
        layer_norm_epsilon=1.0e-5,
    )
    parameters = current_decoder_parameter_specs(config)
    value_count = sum(parameter.value_count for parameter in parameters)
    return ModelBundle(
        config=config,
        tokenizer=TokenizerSpec(
            method="byte",
            byte_vocabulary=(97, 98, 99, 100),
        ),
        parameters=parameters,
        weights=(
            ((index % 29) - 14) / 31.0
            for index in range(value_count)
        ),
        stage="interchange-test",
        parent_artifact_id="a" * 64,
        metadata={"purpose": "GGUF round trip", "nested": {"value": 7}},
    )


def make_bpe_bundle() -> ModelBundle:
    config = TransformerConfig(
        vocabulary_size=258,
        maximum_context=3,
        model_width=2,
        head_count=1,
        block_count=1,
        feed_forward_width=4,
        random_seed=83,
    )
    parameters = current_decoder_parameter_specs(config)
    value_count = sum(parameter.value_count for parameter in parameters)
    return ModelBundle(
        config=config,
        tokenizer=TokenizerSpec(
            method="bpe",
            merge_rules=((97, 98, 256), (256, 99, 257)),
        ),
        parameters=parameters,
        weights=(0.0 for _ in range(value_count)),
        stage="bpe-test",
    )


def rewrite_uint64_metadata(path: Path, key: str, value: int) -> None:
    data = bytearray(path.read_bytes())
    encoded_key = key.encode("utf-8")
    uint64_metadata_type = 10
    marker = (
        struct.pack("<Q", len(encoded_key))
        + encoded_key
        + struct.pack("<I", uint64_metadata_type)
    )
    if data.count(marker) != 1:
        raise AssertionError(f"expected one GGUF metadata field {key!r}")
    value_offset = data.index(marker) + len(marker)
    struct.pack_into("<Q", data, value_offset, value)
    path.write_bytes(data)


def rewrite_float64_metadata(path: Path, key: str, value: float) -> None:
    data = bytearray(path.read_bytes())
    encoded_key = key.encode("utf-8")
    float64_metadata_type = 12
    marker = (
        struct.pack("<Q", len(encoded_key))
        + encoded_key
        + struct.pack("<I", float64_metadata_type)
    )
    if data.count(marker) != 1:
        raise AssertionError(f"expected one GGUF metadata field {key!r}")
    value_offset = data.index(marker) + len(marker)
    struct.pack_into("<d", data, value_offset, value)
    path.write_bytes(data)


def rewrite_float32_metadata(path: Path, key: str, value: float) -> None:
    data = bytearray(path.read_bytes())
    encoded_key = key.encode("utf-8")
    float32_metadata_type = 6
    marker = (
        struct.pack("<Q", len(encoded_key))
        + encoded_key
        + struct.pack("<I", float32_metadata_type)
    )
    if data.count(marker) != 1:
        raise AssertionError(f"expected one GGUF metadata field {key!r}")
    value_offset = data.index(marker) + len(marker)
    struct.pack_into("<f", data, value_offset, value)
    path.write_bytes(data)


class GGUFInterchangeTests(unittest.TestCase):
    def test_import_and_export_reject_non_f32_epsilon(self) -> None:
        for description, invalid in (
            ("overflow", 1.0e300),
            ("underflow", 1.0e-50),
        ):
            with self.subTest(description=description):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    bundle = make_bundle()
                    path = export_gguf(bundle, root / "valid.gguf")
                    rewrite_float64_metadata(
                        path,
                        "riftco.layer_norm_epsilon",
                        invalid,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "finite, strictly positive float32",
                    ):
                        load_gguf(path)

                    standard_path = export_gguf(
                        bundle,
                        root / "invalid-standard.gguf",
                    )
                    rewrite_float32_metadata(
                        standard_path,
                        "riftco.attention.layer_norm_epsilon",
                        float("inf") if description == "overflow" else 0.0,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "finite, strictly positive float32",
                    ):
                        load_gguf(standard_path)

                    object.__setattr__(
                        bundle.config,
                        "layer_norm_epsilon",
                        invalid,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "finite, strictly positive float32",
                    ):
                        export_gguf(bundle, root / "invalid-export.gguf")

    def test_export_is_deterministic_and_round_trips_exact_bundle(self) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = export_gguf(bundle, root / "first.gguf")
            second = export_gguf(bundle, root / "second.gguf")
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()
            loaded = load_gguf(first)

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first_bytes[:4], b"GGUF")
        self.assertEqual(loaded.artifact_id, bundle.artifact_id)
        self.assertEqual(loaded.config, bundle.config)
        self.assertEqual(loaded.tokenizer, bundle.tokenizer)
        self.assertEqual(loaded.parameters, bundle.parameters)
        self.assertEqual(loaded.weights, bundle.weights)
        self.assertEqual(loaded.metadata, bundle.metadata)

    def test_bpe_merge_ids_round_trip_in_single_file_metadata(self) -> None:
        bundle = make_bpe_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = export_gguf(bundle, Path(directory) / "bpe.gguf")
            inspection = inspect_gguf(path)
            loaded = load_gguf(path)

        self.assertEqual(loaded.artifact_id, bundle.artifact_id)
        self.assertEqual(loaded.tokenizer, bundle.tokenizer)
        self.assertEqual(
            inspection.metadata["riftco.tokenizer.merge_result"],
            (256, 257),
        )

    def test_inspection_reports_standard_layout_and_current_architecture(
        self,
    ) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = export_gguf(bundle, Path(directory) / "model.gguf")
            inspection = inspect_gguf(path)

        self.assertEqual(inspection.version, GGUF_VERSION)
        self.assertEqual(
            inspection.architecture,
            RIFTCO_GGUF_ARCHITECTURE,
        )
        self.assertEqual(inspection.alignment, 32)
        self.assertEqual(
            inspection.metadata["riftco.architecture_id"],
            "riftco_decoder_v1",
        )
        self.assertEqual(
            inspection.metadata["riftco.position_embedding"],
            "learned_absolute",
        )
        self.assertEqual(
            inspection.metadata["riftco.normalization"],
            "pre_layer_norm",
        )
        self.assertEqual(
            tuple(tensor.name for tensor in inspection.tensors),
            tuple(parameter.name for parameter in bundle.parameters),
        )
        self.assertTrue(
            all(tensor.ggml_type == GGML_TYPE_F32 for tensor in inspection.tensors)
        )
        self.assertTrue(
            all(tensor.data_offset % 32 == 0 for tensor in inspection.tensors)
        )
        self.assertEqual(
            inspection.tensors[0].shape,
            (bundle.config.vocabulary_size, bundle.config.model_width),
        )

    def test_loader_rejects_foreign_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_gguf(make_bundle(), Path(directory) / "model.gguf")
            data = path.read_bytes()
            marker = struct.pack("<Q", 6) + b"riftco"
            self.assertEqual(data.count(marker), 1)
            path.write_bytes(data.replace(marker, struct.pack("<Q", 6) + b"llamax", 1))
            self.assertEqual(inspect_gguf(path).architecture, "llamax")
            with self.assertRaisesRegex(UnsupportedGGUFError, "architecture"):
                load_gguf(path)

    def test_loader_rejects_non_f32_tensor_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_gguf(make_bundle(), Path(directory) / "model.gguf")
            data = bytearray(path.read_bytes())
            name = b"token_embedding.weight"
            marker = struct.pack("<Q", len(name)) + name
            descriptor = data.find(marker)
            self.assertGreaterEqual(descriptor, 0)
            type_offset = descriptor + len(marker) + 4 + 2 * 8
            self.assertEqual(struct.unpack_from("<I", data, type_offset)[0], 0)
            struct.pack_into("<I", data, type_offset, 1)
            path.write_bytes(data)
            with self.assertRaisesRegex(UnsupportedGGUFError, "F32"):
                load_gguf(path)

    def test_loader_rejects_decoder_depth_above_safety_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_gguf(make_bundle(), Path(directory) / "model.gguf")
            rewrite_uint64_metadata(
                path,
                "riftco.block_count",
                MAXIMUM_DECODER_BLOCKS + 1,
            )
            with self.assertRaisesRegex(ValueError, "block_count exceeds"):
                load_gguf(path)

    def test_inspector_rejects_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_gguf(make_bundle(), Path(directory) / "model.gguf")
            # The writer leaves 16 bytes of optional tail alignment after the
            # final four-value bias, so remove that padding plus one data byte.
            path.write_bytes(path.read_bytes()[:-17])
            with self.assertRaisesRegex(ValueError, "truncated"):
                inspect_gguf(path)


if __name__ == "__main__":
    unittest.main()
