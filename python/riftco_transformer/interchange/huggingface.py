"""Hugging Face-style directory adapter for the current Riftco decoder."""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import NoReturn

from .._atomic_publish import publish_directory_no_replace
from .._numeric import positive_float32
from ..artifacts import ModelBundle, TokenizerSpec
from ..native import TransformerConfig
from .contracts import (
    Float32Tensor,
    build_bundle_from_tensors,
    bundle_tensors,
    current_decoder_parameter_specs,
)
from .safetensors import load_safetensors, save_safetensors


CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"
TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
TOKENIZER_NAME = "tokenizer.json"
RIFTCO_ARCHITECTURE = "riftco_decoder_v1"
RIFTCO_INTERCHANGE_VERSION = 1
HUGGINGFACE_MODEL_TYPE = "riftco_decoder"
HUGGINGFACE_ARCHITECTURE_CLASS = "RiftcoDecoderForCausalLM"
MAXIMUM_JSON_BYTES = 16 << 20


class UnsupportedHuggingFaceModelError(ValueError):
    """Raised when a directory describes a non-Riftco model topology."""


def huggingface_parameter_map(
    config: TransformerConfig,
) -> dict[str, str]:
    """Map every native decoder parameter to its stable HF-style name."""

    # This validates the dimensions before constructing a topology map.
    current_decoder_parameter_specs(config)
    result = {
        "token_embedding.weight": "transformer.wte.weight",
        "position_embedding.weight": "transformer.wpe.weight",
    }
    projection_names = {
        "query": "q_proj",
        "key": "k_proj",
        "value": "v_proj",
        "output": "out_proj",
    }
    for block_index in range(config.block_count):
        native = f"blocks.{block_index}."
        huggingface = f"transformer.h.{block_index}."
        result[native + "attention_norm.scale"] = huggingface + "ln_1.weight"
        result[native + "attention_norm.bias"] = huggingface + "ln_1.bias"
        for native_projection, hf_projection in projection_names.items():
            for suffix in ("weight", "bias"):
                result[
                    native + f"attention.{native_projection}.{suffix}"
                ] = huggingface + f"attn.{hf_projection}.{suffix}"
        result[native + "feed_forward_norm.scale"] = (
            huggingface + "ln_2.weight"
        )
        result[native + "feed_forward_norm.bias"] = (
            huggingface + "ln_2.bias"
        )
        result[native + "feed_forward.expand.weight"] = (
            huggingface + "mlp.fc_in.weight"
        )
        result[native + "feed_forward.expand.bias"] = (
            huggingface + "mlp.fc_in.bias"
        )
        result[native + "feed_forward.project.weight"] = (
            huggingface + "mlp.fc_out.weight"
        )
        result[native + "feed_forward.project.bias"] = (
            huggingface + "mlp.fc_out.bias"
        )
    result.update(
        {
            "final_norm.scale": "transformer.ln_f.weight",
            "final_norm.bias": "transformer.ln_f.bias",
            "language_model_head.weight": "lm_head.weight",
            "language_model_head.bias": "lm_head.bias",
        }
    )

    expected_names = {
        parameter.name
        for parameter in current_decoder_parameter_specs(config)
    }
    if set(result) != expected_names or len(set(result.values())) != len(result):
        raise RuntimeError(
            "internal Hugging Face parameter map does not cover the current "
            "decoder exactly"
        )
    return result


def export_huggingface_directory(
    bundle: ModelBundle,
    directory: str | os.PathLike[str],
) -> Path:
    """Atomically export a native bundle as a HF-style SafeTensors directory.

    The destination must not already exist. This adapter describes Riftco's
    current decoder explicitly; it does not relabel the model as Llama,
    Mistral, GPT-2, or another incompatible architecture.
    """

    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle")
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(
            f"Hugging Face export destination already exists: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    native_tensors = bundle_tensors(bundle)
    parameter_map = huggingface_parameter_map(bundle.config)
    hf_tensors = {
        parameter_map[name]: tensor
        for name, tensor in native_tensors.items()
    }

    temporary_path = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
    )
    try:
        save_safetensors(
            hf_tensors,
            temporary_path / WEIGHTS_NAME,
            metadata={
                "architecture": RIFTCO_ARCHITECTURE,
                "format": "riftco-huggingface-weights",
                "format_version": str(RIFTCO_INTERCHANGE_VERSION),
                "source_artifact_id": bundle.artifact_id,
            },
        )
        _write_json(temporary_path / CONFIG_NAME, _config_entry(bundle))
        _write_json(
            temporary_path / TOKENIZER_CONFIG_NAME,
            _tokenizer_config_entry(bundle),
        )
        _write_json(
            temporary_path / TOKENIZER_NAME,
            _tokenizer_entry(bundle.tokenizer),
        )
        publish_directory_no_replace(temporary_path, destination)
    finally:
        shutil.rmtree(temporary_path, ignore_errors=True)
    return destination


def load_huggingface_directory(
    directory: str | os.PathLike[str],
) -> ModelBundle:
    """Import an explicitly compatible Riftco decoder HF directory."""

    source = Path(directory)
    if not source.is_dir():
        raise NotADirectoryError(source)
    config_entry = _read_json(source / CONFIG_NAME)
    config = _parse_config(config_entry)
    tokenizer = _parse_tokenizer(
        _read_json(source / TOKENIZER_CONFIG_NAME),
        _read_json(source / TOKENIZER_NAME),
        config,
    )

    safe_file = load_safetensors(source / WEIGHTS_NAME)
    _validate_weight_metadata(safe_file.metadata, config_entry)
    parameter_map = huggingface_parameter_map(config)
    hf_to_native = {hf: native for native, hf in parameter_map.items()}
    actual_hf_names = set(safe_file.tensors)
    expected_hf_names = set(hf_to_native)
    missing = sorted(expected_hf_names - actual_hf_names)
    extra = sorted(actual_hf_names - expected_hf_names)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(
                "missing " + ", ".join(repr(name) for name in missing)
            )
        if extra:
            details.append(
                "unexpected " + ", ".join(repr(name) for name in extra)
            )
        raise ValueError(
            "Hugging Face weight names do not match "
            f"{RIFTCO_ARCHITECTURE}: " + "; ".join(details)
        )
    native_tensors: dict[str, Float32Tensor] = {
        hf_to_native[hf_name]: tensor
        for hf_name, tensor in safe_file.tensors.items()
    }

    artifact = config_entry.get("riftco_artifact", {})
    if not isinstance(artifact, dict):
        raise ValueError("config.json riftco_artifact must be an object")
    stage = artifact.get("stage", "huggingface_import")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("config.json riftco_artifact.stage must be nonempty")
    parent_artifact_id = artifact.get("parent_artifact_id")
    if parent_artifact_id is not None and not isinstance(
        parent_artifact_id, str
    ):
        raise ValueError(
            "config.json riftco_artifact.parent_artifact_id must be a "
            "string or null"
        )
    metadata = artifact.get(
        "metadata",
        {"interchange_source": "huggingface_directory"},
    )
    if not isinstance(metadata, dict):
        raise ValueError("config.json riftco_artifact.metadata must be an object")

    bundle = build_bundle_from_tensors(
        config=config,
        tokenizer=tokenizer,
        tensors=native_tensors,
        stage=stage,
        parent_artifact_id=parent_artifact_id,
        metadata=metadata,
    )
    source_artifact_id = artifact.get("source_artifact_id")
    if source_artifact_id is not None:
        _validate_sha256(source_artifact_id, "source_artifact_id")
        if bundle.artifact_id != source_artifact_id:
            raise ValueError(
                "Hugging Face directory source_artifact_id does not match "
                "the reconstructed model"
            )
    return bundle


def import_huggingface_directory(
    directory: str | os.PathLike[str],
) -> ModelBundle:
    """Alias for :func:`load_huggingface_directory`."""

    return load_huggingface_directory(directory)


def _config_entry(bundle: ModelBundle) -> dict[str, object]:
    config = bundle.config
    return {
        "architectures": [HUGGINGFACE_ARCHITECTURE_CLASS],
        "attention_projection": "separate_qkv",
        "dtype": "float32",
        "hidden_act": "gelu",
        "hidden_size": config.model_width,
        "initializer_seed": config.random_seed,
        "intermediate_size": config.feed_forward_width,
        "layer_norm_eps": config.layer_norm_epsilon,
        "max_position_embeddings": config.maximum_context,
        "model_type": HUGGINGFACE_MODEL_TYPE,
        "normalization_type": "layer_norm_with_bias",
        "num_attention_heads": config.head_count,
        "num_hidden_layers": config.block_count,
        "position_embedding_type": "learned_absolute",
        "riftco_architecture": RIFTCO_ARCHITECTURE,
        "riftco_artifact": {
            "metadata": bundle.metadata,
            "parent_artifact_id": bundle.parent_artifact_id,
            "source_artifact_id": bundle.artifact_id,
            "stage": bundle.stage,
        },
        "riftco_interchange_version": RIFTCO_INTERCHANGE_VERSION,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "use_bias": True,
        "vocab_size": config.vocabulary_size,
    }


def _tokenizer_config_entry(bundle: ModelBundle) -> dict[str, object]:
    tokenizer_class = (
        "RiftcoByteTokenizer"
        if bundle.tokenizer.method == "byte"
        else "RiftcoByteLevelBPETokenizer"
    )
    return {
        "model_max_length": bundle.config.maximum_context,
        "riftco_tokenizer_format": "riftco_tokenizer_v1",
        "tokenizer_class": tokenizer_class,
        "tokenizer_file": TOKENIZER_NAME,
    }


def _tokenizer_entry(tokenizer: TokenizerSpec) -> dict[str, object]:
    result: dict[str, object] = {
        "format": "riftco-tokenizer",
        "format_version": 1,
        "method": tokenizer.method,
    }
    if tokenizer.method == "byte":
        result["byte_vocabulary"] = list(tokenizer.byte_vocabulary)
    else:
        result["merge_rules"] = [
            list(rule) for rule in tokenizer.merge_rules
        ]
    return result


def _parse_config(entry: dict[str, object]) -> TransformerConfig:
    model_type = entry.get("model_type")
    if model_type != HUGGINGFACE_MODEL_TYPE:
        raise _unsupported_model(model_type)
    architecture = entry.get("riftco_architecture")
    if architecture != RIFTCO_ARCHITECTURE:
        raise UnsupportedHuggingFaceModelError(
            "unsupported or missing riftco_architecture "
            f"{architecture!r}; expected {RIFTCO_ARCHITECTURE!r}. "
            "Architecture IDs are required because similarly shaped decoder "
            "weights can still have incompatible semantics."
        )
    _require_exact(
        entry,
        "riftco_interchange_version",
        RIFTCO_INTERCHANGE_VERSION,
    )
    _require_exact(
        entry,
        "architectures",
        [HUGGINGFACE_ARCHITECTURE_CLASS],
    )
    for name, expected in (
        ("attention_projection", "separate_qkv"),
        ("dtype", "float32"),
        ("hidden_act", "gelu"),
        ("normalization_type", "layer_norm_with_bias"),
        ("position_embedding_type", "learned_absolute"),
        ("tie_word_embeddings", False),
        ("torch_dtype", "float32"),
        ("use_bias", True),
    ):
        _require_exact(entry, name, expected)

    config = TransformerConfig(
        vocabulary_size=_positive_integer(entry, "vocab_size"),
        maximum_context=_positive_integer(
            entry,
            "max_position_embeddings",
        ),
        model_width=_positive_integer(entry, "hidden_size"),
        head_count=_positive_integer(entry, "num_attention_heads"),
        block_count=_positive_integer(entry, "num_hidden_layers"),
        feed_forward_width=_positive_integer(entry, "intermediate_size"),
        random_seed=_bounded_integer(
            entry,
            "initializer_seed",
            (1 << 32) - 1,
        ),
        layer_norm_epsilon=_positive_number(entry, "layer_norm_eps"),
    )
    # Validate divisibility and the complete current parameter schema now.
    current_decoder_parameter_specs(config)
    return config


def _parse_tokenizer(
    config_entry: dict[str, object],
    tokenizer_entry: dict[str, object],
    model_config: TransformerConfig,
) -> TokenizerSpec:
    _require_exact(
        config_entry,
        "riftco_tokenizer_format",
        "riftco_tokenizer_v1",
    )
    _require_exact(config_entry, "tokenizer_file", TOKENIZER_NAME)
    if (
        _positive_integer(config_entry, "model_max_length")
        != model_config.maximum_context
    ):
        raise ValueError(
            "tokenizer_config.json model_max_length must match "
            "config.json max_position_embeddings"
        )
    _require_exact(tokenizer_entry, "format", "riftco-tokenizer")
    _require_exact(tokenizer_entry, "format_version", 1)
    method = tokenizer_entry.get("method")
    if method == "byte":
        _require_exact(
            config_entry,
            "tokenizer_class",
            "RiftcoByteTokenizer",
        )
        vocabulary = _integer_array(tokenizer_entry, "byte_vocabulary")
        tokenizer = TokenizerSpec(
            method="byte",
            byte_vocabulary=tuple(vocabulary),
        )
    elif method == "bpe":
        _require_exact(
            config_entry,
            "tokenizer_class",
            "RiftcoByteLevelBPETokenizer",
        )
        raw_rules = tokenizer_entry.get("merge_rules")
        if not isinstance(raw_rules, list):
            raise ValueError("tokenizer.json merge_rules must be an array")
        rules: list[tuple[int, int, int]] = []
        for index, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, list) or len(raw_rule) != 3:
                raise ValueError(
                    f"tokenizer.json merge_rules[{index}] must contain "
                    "three integers"
                )
            rules.append(
                tuple(
                    _json_integer(value, f"merge_rules[{index}]")
                    for value in raw_rule
                )
            )
        tokenizer = TokenizerSpec(method="bpe", merge_rules=tuple(rules))
    else:
        raise ValueError(
            "tokenizer.json method must be Riftco 'byte' or 'bpe'"
        )
    if tokenizer.vocabulary_size != model_config.vocabulary_size:
        raise ValueError(
            "tokenizer vocabulary size does not match config.json vocab_size"
        )
    return tokenizer


def _validate_weight_metadata(
    metadata: Mapping[str, str],
    config_entry: Mapping[str, object],
) -> None:
    expected = {
        "architecture": RIFTCO_ARCHITECTURE,
        "format": "riftco-huggingface-weights",
        "format_version": str(RIFTCO_INTERCHANGE_VERSION),
    }
    for name, expected_value in expected.items():
        actual = metadata.get(name)
        if actual is not None and actual != expected_value:
            raise ValueError(
                f"model.safetensors metadata {name!r} is {actual!r}; "
                f"expected {expected_value!r}"
            )
    artifact = config_entry.get("riftco_artifact")
    if isinstance(artifact, dict):
        source_id = artifact.get("source_artifact_id")
        weight_source_id = metadata.get("source_artifact_id")
        if (
            source_id is not None
            and weight_source_id is not None
            and source_id != weight_source_id
        ):
            raise ValueError(
                "model.safetensors source_artifact_id does not match "
                "config.json"
            )


def _unsupported_model(model_type: object) -> UnsupportedHuggingFaceModelError:
    topology = (
        "Riftco decoder v1 uses learned absolute positions, biased LayerNorm, "
        "a GELU two-linear MLP, separate biased Q/K/V projections, and an "
        "untied biased language-model head"
    )
    if isinstance(model_type, str) and model_type in {"llama", "mistral"}:
        return UnsupportedHuggingFaceModelError(
            f"Hugging Face model_type {model_type!r} is not compatible: "
            f"{topology}. Add a dedicated {model_type} architecture adapter "
            "instead of reinterpreting its weights."
        )
    return UnsupportedHuggingFaceModelError(
        f"unsupported Hugging Face model_type {model_type!r}; only "
        f"{HUGGINGFACE_MODEL_TYPE!r} ({RIFTCO_ARCHITECTURE}) is accepted. "
        f"{topology}."
    )


def _write_json(path: Path, value: object) -> None:
    try:
        encoded = (
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError) as error:
        raise TypeError("Hugging Face sidecar is not finite JSON") from error
    if len(encoded) > MAXIMUM_JSON_BYTES:
        raise ValueError(
            f"Hugging Face sidecar exceeds the size limit: {path.name}"
        )
    path.write_bytes(encoded)


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as input_file:
        encoded = input_file.read(MAXIMUM_JSON_BYTES + 1)
    if len(encoded) > MAXIMUM_JSON_BYTES:
        raise ValueError(f"Hugging Face sidecar is too large: {path.name}")
    try:
        decoded = encoded.decode("utf-8", errors="strict")
        value = json.loads(
            decoded,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        _DuplicateJsonKey,
        RecursionError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Hugging Face sidecar is not valid strict JSON: {path.name}"
        ) from error
    if not isinstance(value, dict):
        raise ValueError(
            f"Hugging Face sidecar must contain a JSON object: {path.name}"
        )
    return value


def _require_exact(
    mapping: Mapping[str, object],
    name: str,
    expected: object,
) -> None:
    actual = mapping.get(name)
    if actual != expected or type(actual) is not type(expected):
        raise UnsupportedHuggingFaceModelError(
            f"config field {name!r} is {actual!r}; expected {expected!r} "
            f"for {RIFTCO_ARCHITECTURE}"
        )


def _positive_integer(mapping: Mapping[str, object], name: str) -> int:
    value = _json_integer(mapping.get(name), name)
    if value <= 0:
        raise ValueError(f"config field {name!r} must be greater than zero")
    return value


def _bounded_integer(
    mapping: Mapping[str, object],
    name: str,
    maximum: int,
) -> int:
    value = _json_integer(mapping.get(name), name)
    if value < 0 or value > maximum:
        raise ValueError(f"config field {name!r} is outside its supported range")
    return value


def _positive_number(
    mapping: Mapping[str, object],
    name: str,
) -> int | float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"config field {name!r} must be a number")
    try:
        positive_float32(value, f"config field {name!r}")
    except TypeError as error:
        raise ValueError(str(error)) from error
    return value


def _integer_array(mapping: Mapping[str, object], name: str) -> list[int]:
    value = mapping.get(name)
    if not isinstance(value, list):
        raise ValueError(f"tokenizer.json {name} must be an array")
    return [_json_integer(item, name) for item in value]


def _json_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _validate_sha256(value: object, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


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
    "CONFIG_NAME",
    "HUGGINGFACE_ARCHITECTURE_CLASS",
    "HUGGINGFACE_MODEL_TYPE",
    "MAXIMUM_JSON_BYTES",
    "RIFTCO_ARCHITECTURE",
    "RIFTCO_INTERCHANGE_VERSION",
    "TOKENIZER_CONFIG_NAME",
    "TOKENIZER_NAME",
    "WEIGHTS_NAME",
    "UnsupportedHuggingFaceModelError",
    "export_huggingface_directory",
    "huggingface_parameter_map",
    "import_huggingface_directory",
    "load_huggingface_directory",
]
