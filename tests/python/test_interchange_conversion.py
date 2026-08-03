from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from riftco_transformer.artifacts import ModelBundle, TokenizerSpec
from riftco_transformer.interchange import (
    convert_model,
    export_model,
    load_model,
)
from riftco_transformer.interchange.contracts import (
    current_decoder_parameter_specs,
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
        random_seed=97,
    )
    parameters = current_decoder_parameter_specs(config)
    count = sum(parameter.value_count for parameter in parameters)
    return ModelBundle(
        config=config,
        tokenizer=TokenizerSpec(
            method="byte",
            byte_vocabulary=(97, 98, 99, 100),
        ),
        parameters=parameters,
        weights=(
            ((index % 19) - 9) / 23.0
            for index in range(count)
        ),
        stage="conversion-test",
        metadata={"purpose": "conversion facade"},
    )


class ModelConversionTests(unittest.TestCase):
    def test_complete_model_conversion_chain_preserves_identity(self) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            native = export_model(bundle, root / "model.rift", format="rift")
            hf = convert_model(
                native,
                root / "huggingface",
                source_format="rift",
                destination_format="huggingface",
            )
            gguf = convert_model(
                hf,
                root / "model.gguf",
                source_format="huggingface",
                destination_format="gguf",
            )
            restored = load_model(gguf, format="gguf")

        self.assertEqual(restored.artifact_id, bundle.artifact_id)

    def test_onnx_conversion_round_trip_preserves_identity(self) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = export_model(
                bundle,
                Path(directory) / "model.onnx",
                format="onnx",
            )
            restored = load_model(path, format="onnx")

        self.assertEqual(restored.artifact_id, bundle.artifact_id)

    def test_formats_are_not_inferred_or_case_folded(self) -> None:
        with self.assertRaisesRegex(ValueError, "source format"):
            load_model("model.rift", format="RIFT")
        with self.assertRaisesRegex(ValueError, "destination format"):
            export_model(make_bundle(), "model.bin", format="safetensors")


if __name__ == "__main__":
    unittest.main()
