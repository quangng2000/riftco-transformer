from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from riftco_transformer.artifacts import ModelBundle, TokenizerSpec
from riftco_transformer.interchange.contracts import (
    current_decoder_parameter_specs,
)
from riftco_transformer.interchange.onnx import (
    MAXIMUM_ONNX_PROTOBUF_FIELDS,
    ONNX_CANONICAL_GRAPH_FORMAT,
    ONNX_CANONICAL_GRAPH_VERSION,
    ONNX_INPUT_CONTRACT,
    ONNX_IR_VERSION,
    ONNX_OPSET_VERSION,
    ONNX_PRODUCER_VERSION,
    UnsupportedONNXImportError,
    export_onnx,
    inspect_onnx,
    load_onnx,
    onnx_sidecar_path,
)
import riftco_transformer.interchange.onnx as onnx_module
from riftco_transformer.native import TransformerConfig


def make_bundle(
    *,
    stage: str = "onnx-test",
    weight_offset: int = 0,
) -> ModelBundle:
    config = TransformerConfig(
        vocabulary_size=4,
        maximum_context=5,
        model_width=4,
        head_count=2,
        block_count=1,
        feed_forward_width=8,
        random_seed=79,
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
            (((index + weight_offset) % 23) - 11) / 29.0
            for index in range(value_count)
        ),
        stage=stage,
        metadata={"purpose": "structural ONNX test"},
    )


class ONNXInterchangeTests(unittest.TestCase):
    def test_canonical_export_round_trips_complete_bundle_identity(self) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(bundle, Path(directory) / "model.onnx")
            sidecar = onnx_sidecar_path(path)
            restored = load_onnx(path)

        self.assertTrue(sidecar.name.endswith(".onnx.riftco.json"))
        self.assertEqual(restored.artifact_id, bundle.artifact_id)
        self.assertEqual(restored.config, bundle.config)
        self.assertEqual(restored.tokenizer, bundle.tokenizer)
        self.assertEqual(restored.parameters, bundle.parameters)
        self.assertEqual(restored.weights, bundle.weights)
        self.assertEqual(restored.stage, bundle.stage)
        self.assertEqual(restored.metadata, bundle.metadata)

    def test_export_rejects_epsilon_that_cannot_remain_positive_f32(
        self,
    ) -> None:
        for description, invalid in (
            ("overflow", 1.0e300),
            ("underflow", 1.0e-50),
        ):
            with self.subTest(description=description):
                bundle = make_bundle()
                object.__setattr__(
                    bundle.config,
                    "layer_norm_epsilon",
                    invalid,
                )
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(
                        ValueError,
                        "finite, strictly positive float32",
                    ):
                        export_onnx(
                            bundle,
                            Path(directory) / "invalid.onnx",
                        )

    def test_export_is_deterministic_real_inference_graph(self) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = export_onnx(bundle, root / "first.onnx")
            second = export_onnx(bundle, root / "second.onnx")
            first_bytes = first.read_bytes()
            second_bytes = second.read_bytes()
            inspection = inspect_onnx(first)

        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(inspection.ir_version, ONNX_IR_VERSION)
        self.assertEqual(inspection.opset_version, ONNX_OPSET_VERSION)
        self.assertEqual(inspection.producer_name, "riftco-transformer")
        self.assertEqual(inspection.producer_version, ONNX_PRODUCER_VERSION)
        self.assertEqual(inspection.domain, "ai.riftco")
        self.assertEqual(
            inspection.metadata["ai.riftco.architecture"],
            "riftco_decoder_v1",
        )
        self.assertEqual(
            inspection.inputs[0].shape,
            ("batch", "sequence"),
        )
        self.assertEqual(inspection.inputs[0].data_type, 7)
        self.assertEqual(
            inspection.outputs[0].shape,
            ("batch", "sequence", bundle.config.vocabulary_size),
        )
        self.assertEqual(inspection.outputs[0].data_type, 1)

        operator_types = {node.op_type for node in inspection.nodes}
        self.assertTrue(
            {
                "Erf",
                "Gather",
                "GreaterOrEqual",
                "LayerNormalization",
                "MatMul",
                "Range",
                "Softmax",
                "Where",
            }.issubset(operator_types)
        )
        self.assertTrue(
            any(
                node.name == "blocks.0.attention.mask"
                and node.op_type == "Add"
                for node in inspection.nodes
            )
        )
        self.assertEqual(inspection.nodes[-1].outputs, ("logits",))

    def test_graph_declares_and_enforces_its_input_contract(self) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(bundle, Path(directory) / "model.onnx")
            inspection = inspect_onnx(path)

        self.assertEqual(
            inspection.metadata["ai.riftco.input_contract"],
            ONNX_INPUT_CONTRACT,
        )
        self.assertEqual(
            inspection.metadata["ai.riftco.onnx_format"],
            ONNX_CANONICAL_GRAPH_FORMAT,
        )
        self.assertEqual(
            inspection.metadata["ai.riftco.onnx_format_version"],
            str(ONNX_CANONICAL_GRAPH_VERSION),
        )
        nodes = {node.name: node for node in inspection.nodes}
        self.assertEqual(
            nodes["riftco.checked_token_ids"].op_type,
            "Where",
        )
        self.assertEqual(
            nodes["embedding.token"].inputs,
            ("token_embedding.weight", "_riftco.checked_token_ids"),
        )
        self.assertEqual(
            nodes["riftco.checked_sequence_length"].op_type,
            "Gather",
        )
        self.assertEqual(
            nodes["riftco.positions"].inputs[1],
            "_riftco.checked_sequence_length",
        )
        self.assertEqual(
            {
                nodes["riftco.token_ids_nonnegative"].op_type,
                nodes["riftco.token_ids_below_vocabulary"].op_type,
                nodes["riftco.token_ids_valid"].op_type,
                nodes["riftco.input_element_count"].op_type,
            },
            {"GreaterOrEqual", "Less", "And", "Size"},
        )

    def test_bpe_lineage_and_unicode_metadata_round_trip(self) -> None:
        config = TransformerConfig(
            vocabulary_size=257,
            maximum_context=3,
            model_width=2,
            head_count=1,
            block_count=1,
            feed_forward_width=4,
            random_seed=83,
        )
        parameters = current_decoder_parameter_specs(config)
        value_count = sum(parameter.value_count for parameter in parameters)
        bundle = ModelBundle(
            config=config,
            tokenizer=TokenizerSpec(
                method="bpe",
                merge_rules=((97, 98, 256),),
            ),
            parameters=parameters,
            weights=(
                -0.0 if index == 0 else ((index % 13) - 6) / 31.0
                for index in range(value_count)
            ),
            stage="onnx-bpe-δ",
            parent_artifact_id="ab" * 32,
            metadata={
                "language": "日本語",
                "nested": {"emoji": "🧪", "values": [1, -0.0]},
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(bundle, Path(directory) / "model.onnx")
            relocated = Path(directory) / "relocated-sidecar.json"
            os.replace(onnx_sidecar_path(path), relocated)
            restored = load_onnx(path, sidecar_path=relocated)

        self.assertEqual(restored.artifact_id, bundle.artifact_id)
        self.assertEqual(restored.tokenizer, bundle.tokenizer)
        self.assertEqual(restored.parent_artifact_id, bundle.parent_artifact_id)
        self.assertEqual(restored.metadata, bundle.metadata)

    def test_import_accepts_prior_producer_version_for_same_graph_format(
        self,
    ) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(bundle, Path(directory) / "model.onnx")
            original = path.read_bytes()
            previous = "0.5.9"
            mutated = original.replace(
                ONNX_PRODUCER_VERSION.encode("ascii"),
                previous.encode("ascii"),
                1,
            )
            self.assertNotEqual(mutated, original)
            path.write_bytes(mutated)
            self._update_sidecar_digest(path)
            restored = load_onnx(path)
            inspection = inspect_onnx(path)

        self.assertEqual(restored.artifact_id, bundle.artifact_id)
        self.assertEqual(inspection.producer_version, previous)

    def test_export_restores_existing_pair_when_sidecar_replace_fails(
        self,
    ) -> None:
        original_bundle = make_bundle()
        replacement_bundle = make_bundle(
            stage="replacement",
            weight_offset=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = export_onnx(original_bundle, root / "model.onnx")
            sidecar = onnx_sidecar_path(path)
            original_model_bytes = path.read_bytes()
            original_sidecar_bytes = sidecar.read_bytes()
            real_replace = os.replace
            injected = False

            def failing_replace(
                source: str | os.PathLike[str],
                destination: str | os.PathLike[str],
            ) -> None:
                nonlocal injected
                if (
                    not injected
                    and Path(destination) == sidecar
                    and Path(source).suffix == ".tmp"
                ):
                    injected = True
                    raise OSError("injected sidecar replace failure")
                real_replace(source, destination)

            with mock.patch.object(
                onnx_module.os,
                "replace",
                side_effect=failing_replace,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "injected sidecar replace failure",
                ):
                    export_onnx(replacement_bundle, path)

            self.assertTrue(injected)
            self.assertEqual(path.read_bytes(), original_model_bytes)
            self.assertEqual(sidecar.read_bytes(), original_sidecar_bytes)
            self.assertEqual(
                load_onnx(path).artifact_id,
                original_bundle.artifact_id,
            )
            leftovers = tuple(
                candidate.name
                for candidate in root.iterdir()
                if candidate.suffix in {".tmp", ".rollback"}
            )
            self.assertEqual(leftovers, ())

    def test_sidecar_is_validated_before_model_bytes_are_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(make_bundle(), Path(directory) / "model.onnx")
            onnx_sidecar_path(path).unlink()
            with mock.patch.object(
                onnx_module,
                "_read_onnx_snapshot",
                side_effect=AssertionError("model bytes were read first"),
            ) as snapshot:
                with self.assertRaises(UnsupportedONNXImportError):
                    load_onnx(path)
            snapshot.assert_not_called()

    def test_failed_new_export_leaves_no_partial_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.onnx"
            sidecar = onnx_sidecar_path(path)
            real_replace = os.replace
            injected = False

            def failing_replace(
                source: str | os.PathLike[str],
                destination: str | os.PathLike[str],
            ) -> None:
                nonlocal injected
                if (
                    not injected
                    and Path(destination) == sidecar
                    and Path(source).suffix == ".tmp"
                ):
                    injected = True
                    raise OSError("injected new-sidecar failure")
                real_replace(source, destination)

            with mock.patch.object(
                onnx_module.os,
                "replace",
                side_effect=failing_replace,
            ):
                with self.assertRaisesRegex(
                    OSError,
                    "injected new-sidecar failure",
                ):
                    export_onnx(make_bundle(), path)

            self.assertTrue(injected)
            self.assertFalse(path.exists())
            self.assertFalse(sidecar.exists())
            self.assertEqual(tuple(root.iterdir()), ())

    def test_import_rejects_file_above_config_derived_size_before_reading(
        self,
    ) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(bundle, Path(directory) / "model.onnx")
            limit = onnx_module._maximum_onnx_file_bytes(bundle.config)
            with path.open("r+b") as output:
                output.truncate(limit + 1)
            with self.assertRaisesRegex(
                UnsupportedONNXImportError,
                "bounded canonical size",
            ):
                load_onnx(path)

    def test_parser_rejects_excessive_repeated_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(make_bundle(), Path(directory) / "model.onnx")
            path.write_bytes(
                path.read_bytes()
                + b"\x78\x00" * (MAXIMUM_ONNX_PROTOBUF_FIELDS + 1)
            )
            self._update_sidecar_digest(path)
            with self.assertRaisesRegex(ValueError, "too many fields"):
                inspect_onnx(path)
            with self.assertRaisesRegex(
                UnsupportedONNXImportError,
                "malformed or unsupported",
            ):
                load_onnx(path)

    def test_every_native_weight_is_an_embedded_f32_initializer(self) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(bundle, Path(directory) / "model.onnx")
            inspection = inspect_onnx(path)

        by_name = {tensor.name: tensor for tensor in inspection.initializers}
        for parameter in bundle.parameters:
            with self.subTest(parameter=parameter.name):
                initializer = by_name[parameter.name]
                self.assertEqual(initializer.shape, parameter.shape)
                self.assertEqual(initializer.data_type, 1)
                self.assertEqual(initializer.byte_count, parameter.value_count * 4)
        self.assertIn("_riftco.negative_infinity", by_name)
        self.assertIn("_riftco.head_shape_tail", by_name)

    def test_import_requires_exporter_sidecar_for_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(make_bundle(), Path(directory) / "model.onnx")
            onnx_sidecar_path(path).unlink()
            with self.assertRaisesRegex(
                UnsupportedONNXImportError,
                "requires.*sidecar.*tokenizer",
            ):
                load_onnx(path)

    def test_graph_mutation_is_rejected_even_with_updated_sidecar_hash(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(make_bundle(), Path(directory) / "model.onnx")
            original = path.read_bytes()
            mutated = original.replace(b"Erf", b"Add", 1)
            self.assertNotEqual(mutated, original)
            path.write_bytes(mutated)
            self._update_sidecar_digest(path)

            with self.assertRaisesRegex(
                UnsupportedONNXImportError,
                "node topology|canonical",
            ):
                load_onnx(path)

    def test_initializer_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(make_bundle(), Path(directory) / "model.onnx")
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            with self.assertRaisesRegex(
                UnsupportedONNXImportError,
                "checksum.*edited|wrong file",
            ):
                load_onnx(path)

    def test_tokenizer_sidecar_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(make_bundle(), Path(directory) / "model.onnx")
            sidecar_path = onnx_sidecar_path(path)
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["tokenizer"]["byte_vocabulary"][0] = 101
            sidecar_path.write_text(
                json.dumps(sidecar, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                UnsupportedONNXImportError,
                "source_artifact_id",
            ):
                load_onnx(path)

    def test_inspector_rejects_truncated_protobuf(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = export_onnx(make_bundle(), Path(directory) / "model.onnx")
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated"):
                inspect_onnx(path)

    @staticmethod
    def _update_sidecar_digest(path: Path) -> None:
        sidecar_path = onnx_sidecar_path(path)
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["onnx_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        sidecar_path.write_text(
            json.dumps(sidecar, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
