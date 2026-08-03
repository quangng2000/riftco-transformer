from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from riftco_transformer import TransformerConfig
from riftco_transformer.artifacts import ModelBundle, ParameterSpec, TokenizerSpec
from riftco_transformer.interchange import (
    Float32Tensor,
    MAXIMUM_DECODER_BLOCKS,
    RIFTCO_ARCHITECTURE,
    UnsupportedHuggingFaceModelError,
    current_decoder_parameter_specs,
    export_huggingface_directory,
    huggingface_parameter_map,
    load_huggingface_directory,
    load_safetensors,
    save_safetensors,
)


def make_bundle(*, bpe: bool = False) -> ModelBundle:
    tokenizer = (
        TokenizerSpec(method="bpe", merge_rules=((97, 98, 256),))
        if bpe
        else TokenizerSpec(
            method="byte",
            byte_vocabulary=(97, 98, 99, 100),
        )
    )
    config = TransformerConfig(
        vocabulary_size=tokenizer.vocabulary_size,
        maximum_context=4,
        model_width=4,
        head_count=2,
        block_count=2,
        feed_forward_width=7,
        random_seed=101,
        layer_norm_epsilon=1.0e-5,
    )
    parameters = current_decoder_parameter_specs(config)
    value_count = sum(parameter.value_count for parameter in parameters)
    return ModelBundle(
        config=config,
        tokenizer=tokenizer,
        parameters=parameters,
        weights=(
            ((index % 31) - 15) / 31.0
            for index in range(value_count)
        ),
        stage="post_training",
        parent_artifact_id="a" * 64,
        metadata={"purpose": "interchange-test", "nested": {"ok": True}},
    )


def rewrite_json(path: Path, update: object) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    update(value)  # type: ignore[operator]
    path.write_text(json.dumps(value), encoding="utf-8")


class DecoderContractTests(unittest.TestCase):
    def test_contract_matches_current_native_parameter_layout(self) -> None:
        bundle = make_bundle()
        specs = current_decoder_parameter_specs(bundle.config)
        self.assertEqual(len(specs), 2 + (16 * 2) + 4)
        self.assertEqual(specs[0], ParameterSpec("token_embedding.weight", (4, 4)))
        self.assertIn(
            ParameterSpec(
                "blocks.1.feed_forward.project.weight",
                (4, 7),
            ),
            specs,
        )
        self.assertEqual(
            specs[-1],
            ParameterSpec("language_model_head.bias", (4,)),
        )

    def test_huggingface_parameter_map_is_exact_and_bijective(self) -> None:
        bundle = make_bundle()
        mapping = huggingface_parameter_map(bundle.config)
        self.assertEqual(set(mapping), {spec.name for spec in bundle.parameters})
        self.assertEqual(len(mapping), len(set(mapping.values())))
        self.assertEqual(
            mapping["blocks.1.attention.query.weight"],
            "transformer.h.1.attn.q_proj.weight",
        )
        self.assertEqual(
            mapping["blocks.0.feed_forward.expand.weight"],
            "transformer.h.0.mlp.fc_in.weight",
        )

    def test_contract_rejects_depth_before_expanding_parameter_specs(self) -> None:
        config = TransformerConfig(
            vocabulary_size=1,
            maximum_context=1,
            model_width=1,
            head_count=1,
            block_count=MAXIMUM_DECODER_BLOCKS + 1,
            feed_forward_width=1,
        )
        with self.assertRaisesRegex(ValueError, "block_count exceeds"):
            current_decoder_parameter_specs(config)


class HuggingFaceDirectoryTests(unittest.TestCase):
    def test_import_and_export_reject_non_f32_epsilon(self) -> None:
        for description, invalid in (
            ("overflow", 1.0e300),
            ("underflow", 1.0e-50),
        ):
            with self.subTest(description=description):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    bundle = make_bundle()
                    exported = export_huggingface_directory(
                        bundle,
                        root / "valid",
                    )
                    rewrite_json(
                        exported / "config.json",
                        lambda value: value.__setitem__(
                            "layer_norm_eps",
                            invalid,
                        ),
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "finite, strictly positive float32",
                    ):
                        load_huggingface_directory(exported)

                    object.__setattr__(
                        bundle.config,
                        "layer_norm_epsilon",
                        invalid,
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        "finite, strictly positive float32",
                    ):
                        export_huggingface_directory(
                            bundle,
                            root / "invalid-export",
                        )

    def test_byte_tokenizer_bundle_round_trips_losslessly(self) -> None:
        original = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            exported = export_huggingface_directory(
                original,
                Path(directory) / "model",
            )
            self.assertEqual(
                {path.name for path in exported.iterdir()},
                {
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                },
            )
            config = json.loads(
                (exported / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["model_type"], "riftco_decoder")
            self.assertEqual(
                config["riftco_architecture"],
                RIFTCO_ARCHITECTURE,
            )
            self.assertFalse(config["tie_word_embeddings"])
            self.assertEqual(config["position_embedding_type"], "learned_absolute")

            restored = load_huggingface_directory(exported)
        self.assertEqual(restored.artifact_id, original.artifact_id)
        self.assertEqual(restored.config, original.config)
        self.assertEqual(restored.tokenizer, original.tokenizer)
        self.assertEqual(restored.parameters, original.parameters)
        self.assertEqual(restored.weights, original.weights)
        self.assertEqual(restored.metadata, original.metadata)

    def test_bpe_tokenizer_round_trips_exact_merge_ids(self) -> None:
        original = make_bundle(bpe=True)
        with tempfile.TemporaryDirectory() as directory:
            exported = export_huggingface_directory(
                original,
                Path(directory) / "model",
            )
            tokenizer = json.loads(
                (exported / "tokenizer.json").read_text(encoding="utf-8")
            )
            self.assertEqual(tokenizer["method"], "bpe")
            self.assertEqual(tokenizer["merge_rules"], [[97, 98, 256]])
            restored = load_huggingface_directory(exported)
        self.assertEqual(restored.artifact_id, original.artifact_id)

    def test_export_is_deterministic_and_never_replaces_directory(self) -> None:
        bundle = make_bundle()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = export_huggingface_directory(bundle, root / "first")
            second = export_huggingface_directory(bundle, root / "second")
            for name in (
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                self.assertEqual(
                    (first / name).read_bytes(),
                    (second / name).read_bytes(),
                )
            with self.assertRaises(FileExistsError):
                export_huggingface_directory(bundle, first)

    def test_rejects_llama_and_mistral_with_topology_explanation(self) -> None:
        for model_type in ("llama", "mistral"):
            with self.subTest(model_type=model_type):
                with tempfile.TemporaryDirectory() as directory:
                    exported = export_huggingface_directory(
                        make_bundle(),
                        Path(directory) / "model",
                    )
                    rewrite_json(
                        exported / "config.json",
                        lambda value: value.__setitem__(
                            "model_type",
                            model_type,
                        ),
                    )
                    with self.assertRaisesRegex(
                        UnsupportedHuggingFaceModelError,
                        "learned absolute positions",
                    ):
                        load_huggingface_directory(exported)

    def test_rejects_incompatible_topology_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exported = export_huggingface_directory(
                make_bundle(),
                Path(directory) / "model",
            )
            rewrite_json(
                exported / "config.json",
                lambda value: value.__setitem__(
                    "tie_word_embeddings",
                    True,
                ),
            )
            with self.assertRaisesRegex(
                UnsupportedHuggingFaceModelError,
                "tie_word_embeddings",
            ):
                load_huggingface_directory(exported)

    def test_rejects_decoder_depth_above_import_safety_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exported = export_huggingface_directory(
                make_bundle(),
                Path(directory) / "model",
            )
            rewrite_json(
                exported / "config.json",
                lambda value: value.__setitem__(
                    "num_hidden_layers",
                    MAXIMUM_DECODER_BLOCKS + 1,
                ),
            )
            with self.assertRaisesRegex(ValueError, "block_count exceeds"):
                load_huggingface_directory(exported)

    def test_rejects_invalid_or_duplicate_model_type(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exported = export_huggingface_directory(
                make_bundle(),
                Path(directory) / "model",
            )
            rewrite_json(
                exported / "config.json",
                lambda value: value.__setitem__("model_type", []),
            )
            with self.assertRaisesRegex(
                UnsupportedHuggingFaceModelError,
                "unsupported Hugging Face model_type",
            ):
                load_huggingface_directory(exported)

        with tempfile.TemporaryDirectory() as directory:
            exported = export_huggingface_directory(
                make_bundle(),
                Path(directory) / "model",
            )
            (exported / "config.json").write_text(
                '{"model_type":"riftco_decoder","model_type":"llama"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "strict JSON"):
                load_huggingface_directory(exported)

    def test_rejects_missing_extra_and_wrong_shape_weights(self) -> None:
        for mutation, expected_error in (
            ("missing", "missing 'lm_head.bias'"),
            ("extra", "unexpected 'foreign.weight'"),
            ("shape", "has shape"),
        ):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    exported = export_huggingface_directory(
                        make_bundle(),
                        Path(directory) / "model",
                    )
                    weight_path = exported / "model.safetensors"
                    document = load_safetensors(weight_path)
                    tensors = document.tensors
                    if mutation == "missing":
                        del tensors["lm_head.bias"]
                    elif mutation == "extra":
                        tensors["foreign.weight"] = Float32Tensor((1,), (0.0,))
                    else:
                        values = tensors["transformer.ln_f.weight"].values
                        tensors["transformer.ln_f.weight"] = Float32Tensor(
                            (2, 2),
                            values,
                        )
                    save_safetensors(
                        tensors,
                        weight_path,
                        metadata=document.metadata,
                        overwrite=True,
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        load_huggingface_directory(exported)

    def test_rejects_tokenizer_vocab_or_context_mismatch(self) -> None:
        for filename, field, value, expected_error in (
            (
                "tokenizer_config.json",
                "model_max_length",
                8,
                "must match",
            ),
            (
                "tokenizer.json",
                "byte_vocabulary",
                [97, 98],
                "vocabulary size",
            ),
        ):
            with self.subTest(filename=filename):
                with tempfile.TemporaryDirectory() as directory:
                    exported = export_huggingface_directory(
                        make_bundle(),
                        Path(directory) / "model",
                    )
                    rewrite_json(
                        exported / filename,
                        lambda document: document.__setitem__(field, value),
                    )
                    with self.assertRaisesRegex(ValueError, expected_error):
                        load_huggingface_directory(exported)


if __name__ == "__main__":
    unittest.main()
