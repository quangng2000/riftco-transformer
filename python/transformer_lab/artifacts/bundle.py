"""Versioned, dependency-free model artifacts shared between all stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import tempfile
from typing import Iterable, Mapping
import zipfile

from ..native import DecoderOnlyTransformer, Tokenizer, TransformerConfig


# Stable wire identifier retained so existing v1 model bundles remain readable.
FORMAT_NAME = "transformer-lab-model-bundle"
FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
WEIGHTS_NAME = "weights.f32le"
MAXIMUM_MANIFEST_BYTES = 1 << 20


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """The stable identity and shape of one named model parameter."""

    name: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("parameter name must be a nonempty string")
        try:
            shape = tuple(self.shape)
        except TypeError as error:
            raise TypeError("parameter shape must be an iterable") from error
        for dimension in shape:
            _positive_integer(dimension, "parameter dimension")
        object.__setattr__(self, "shape", shape)

    @property
    def value_count(self) -> int:
        count = 1
        for dimension in self.shape:
            count *= dimension
        return count


@dataclass(frozen=True, slots=True)
class TokenizerSpec:
    """A restorable byte or byte-level BPE tokenizer definition."""

    method: str
    byte_vocabulary: tuple[int, ...] = ()
    merge_rules: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self) -> None:
        try:
            byte_vocabulary = tuple(self.byte_vocabulary)
        except TypeError as error:
            raise TypeError(
                "byte vocabulary must be an iterable"
            ) from error
        try:
            merge_rules = tuple(
                tuple(rule) for rule in self.merge_rules
            )
        except TypeError as error:
            raise TypeError("merge rules must be iterable triples") from error
        object.__setattr__(self, "byte_vocabulary", byte_vocabulary)
        object.__setattr__(self, "merge_rules", merge_rules)

        if self.method == "byte":
            if not byte_vocabulary:
                raise ValueError(
                    "byte tokenizer state requires a nonempty vocabulary"
                )
            if len(byte_vocabulary) > 256:
                raise ValueError(
                    "byte tokenizer vocabulary cannot exceed 256 entries"
                )
            seen: set[int] = set()
            for byte in byte_vocabulary:
                checked = _bounded_integer(byte, 255, "vocabulary byte")
                if checked in seen:
                    raise ValueError(
                        "byte tokenizer vocabulary contains a duplicate"
                    )
                seen.add(checked)
            if merge_rules:
                raise ValueError(
                    "byte tokenizer state must not contain BPE merge rules"
                )
            return

        if self.method != "bpe":
            raise ValueError("tokenizer method must be 'byte' or 'bpe'")
        if byte_vocabulary:
            raise ValueError(
                "BPE state uses its fixed 256-byte base vocabulary"
            )
        seen_pairs: set[tuple[int, int]] = set()
        for index, rule in enumerate(merge_rules):
            if len(rule) != 3:
                raise ValueError(
                    "each BPE merge rule must contain left, right, and result"
                )
            left = _bounded_integer(rule[0], (1 << 32) - 1, "merge left")
            right = _bounded_integer(
                rule[1],
                (1 << 32) - 1,
                "merge right",
            )
            result = _bounded_integer(
                rule[2],
                (1 << 32) - 1,
                "merge result",
            )
            expected_result = 256 + index
            if result != expected_result:
                raise ValueError(
                    "BPE merge result IDs must be sequential from 256"
                )
            if left >= result or right >= result:
                raise ValueError(
                    "BPE merge operands must precede their result token"
                )
            pair = (left, right)
            if pair in seen_pairs:
                raise ValueError("BPE merge rules contain a duplicate pair")
            seen_pairs.add(pair)

    @property
    def vocabulary_size(self) -> int:
        if self.method == "byte":
            return len(self.byte_vocabulary)
        return 256 + len(self.merge_rules)

    @classmethod
    def capture(cls, tokenizer: Tokenizer) -> TokenizerSpec:
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("tokenizer must be a Tokenizer")
        if tokenizer.method == "byte":
            return cls(
                method="byte",
                byte_vocabulary=tuple(tokenizer.vocabulary_bytes),
            )
        return cls(
            method="bpe",
            merge_rules=tuple(tokenizer.merge_rules),
        )

    def instantiate(self) -> Tokenizer:
        return Tokenizer.from_state(
            method=self.method,
            byte_vocabulary=self.byte_vocabulary,
            merge_rules=self.merge_rules,
        )


class ModelBundle:
    """Immutable inference artifact: config, tokenizer, and named weights."""

    __slots__ = (
        "_artifact_id",
        "_config",
        "_metadata_json",
        "_parameters",
        "_parent_artifact_id",
        "_stage",
        "_tokenizer",
        "_weights",
        "_weights_bytes",
    )

    def __init__(
        self,
        *,
        config: TransformerConfig,
        tokenizer: TokenizerSpec,
        parameters: Iterable[ParameterSpec],
        weights: Iterable[float],
        stage: str,
        parent_artifact_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        _validate_config(config)
        if not isinstance(tokenizer, TokenizerSpec):
            raise TypeError("tokenizer must be a TokenizerSpec")
        if config.vocabulary_size != tokenizer.vocabulary_size:
            raise ValueError(
                "model vocabulary size must match tokenizer state"
            )
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a nonempty string")
        _validate_artifact_id(parent_artifact_id, "parent_artifact_id")

        parameter_tuple = tuple(parameters)
        if not parameter_tuple:
            raise ValueError("model bundle must contain parameters")
        if any(
            not isinstance(parameter, ParameterSpec)
            for parameter in parameter_tuple
        ):
            raise TypeError(
                "parameters must contain only ParameterSpec values"
            )
        names = [parameter.name for parameter in parameter_tuple]
        if len(set(names)) != len(names):
            raise ValueError("model bundle contains duplicate parameter names")

        expected_value_count = sum(
            parameter.value_count for parameter in parameter_tuple
        )
        weights_tuple, weights_bytes = _canonical_weights(weights)
        if len(weights_tuple) != expected_value_count:
            raise ValueError(
                "weight count does not match parameter shapes"
            )

        detached_metadata = {} if metadata is None else dict(metadata)
        metadata_json = _canonical_json(detached_metadata)
        parsed_metadata = json.loads(metadata_json)
        if not isinstance(parsed_metadata, dict):
            raise TypeError("metadata must be a JSON object")

        self._config = config
        self._tokenizer = tokenizer
        self._parameters = parameter_tuple
        self._weights = weights_tuple
        self._weights_bytes = weights_bytes
        self._stage = stage
        self._parent_artifact_id = parent_artifact_id
        self._metadata_json = metadata_json
        self._artifact_id = self._calculate_artifact_id()

    @property
    def artifact_id(self) -> str:
        return self._artifact_id

    @property
    def config(self) -> TransformerConfig:
        return self._config

    @property
    def tokenizer(self) -> TokenizerSpec:
        return self._tokenizer

    @property
    def parameters(self) -> tuple[ParameterSpec, ...]:
        return self._parameters

    @property
    def weights(self) -> tuple[float, ...]:
        return self._weights

    @property
    def stage(self) -> str:
        return self._stage

    @property
    def parent_artifact_id(self) -> str | None:
        return self._parent_artifact_id

    @property
    def metadata(self) -> dict[str, object]:
        return json.loads(self._metadata_json)

    @classmethod
    def capture(
        cls,
        model: DecoderOnlyTransformer,
        tokenizer: Tokenizer,
        *,
        stage: str,
        parent_artifact_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> ModelBundle:
        if not isinstance(model, DecoderOnlyTransformer):
            raise TypeError("model must be a DecoderOnlyTransformer")
        if model.lora_attached:
            raise RuntimeError(
                "cannot capture a model bundle with active LoRA adapters; "
                "merge them first"
            )
        tokenizer_spec = TokenizerSpec.capture(tokenizer)
        with model.parameters() as parameters:
            names = parameters.names
            shapes = parameters.shapes
            values = parameters.flat_values()
        specs = tuple(
            ParameterSpec(name, shape)
            for name, shape in zip(names, shapes)
        )
        return cls(
            config=model.config,
            tokenizer=tokenizer_spec,
            parameters=specs,
            weights=values,
            stage=stage,
            parent_artifact_id=parent_artifact_id,
            metadata=metadata,
        )

    def instantiate(self, backend: str = "cpu") -> ModelRuntime:
        return ModelRuntime(self, backend)

    def save(self, path: str | os.PathLike[str]) -> Path:
        destination = Path(path)
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        manifest_bytes = self._manifest_bytes(include_artifact_id=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            with zipfile.ZipFile(
                temporary_path,
                mode="w",
                compression=zipfile.ZIP_STORED,
                allowZip64=True,
            ) as archive:
                archive.writestr(
                    _deterministic_zip_info(MANIFEST_NAME),
                    manifest_bytes,
                )
                archive.writestr(
                    _deterministic_zip_info(WEIGHTS_NAME),
                    self._weights_bytes,
                )
            os.replace(temporary_path, destination)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        return destination

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> ModelBundle:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            with zipfile.ZipFile(source, mode="r") as archive:
                infos = archive.infolist()
                names = [info.filename for info in infos]
                if names != [MANIFEST_NAME, WEIGHTS_NAME]:
                    raise ValueError(
                        "model bundle must contain exactly manifest.json "
                        "followed by weights.f32le"
                    )
                if any(
                    info.compress_type != zipfile.ZIP_STORED
                    for info in infos
                ):
                    raise ValueError(
                        "model bundle entries must use stored compression"
                    )
                manifest_info, weights_info = infos
                if manifest_info.file_size > MAXIMUM_MANIFEST_BYTES:
                    raise ValueError("model bundle manifest is too large")
                manifest_bytes = archive.read(manifest_info)
                manifest = _parse_manifest(manifest_bytes)
                expected_weight_bytes = (
                    sum(
                        parameter.value_count
                        for parameter in manifest["parameters"]
                    )
                    * 4
                )
                if weights_info.file_size != expected_weight_bytes:
                    raise ValueError(
                        "model bundle weight size does not match manifest"
                    )
                weight_bytes = archive.read(weights_info)
        except zipfile.BadZipFile as error:
            raise ValueError("model bundle is not a valid ZIP artifact") from error

        weight_digest = hashlib.sha256(weight_bytes).hexdigest()
        if weight_digest != manifest["weights_sha256"]:
            raise ValueError("model bundle weight checksum does not match")
        weights = _unpack_weights(weight_bytes)
        result = cls(
            config=manifest["config"],
            tokenizer=manifest["tokenizer"],
            parameters=manifest["parameters"],
            weights=weights,
            stage=manifest["stage"],
            parent_artifact_id=manifest["parent_artifact_id"],
            metadata=manifest["metadata"],
        )
        if result.artifact_id != manifest["artifact_id"]:
            raise ValueError("model bundle artifact ID does not match")
        return result

    def _calculate_artifact_id(self) -> str:
        return hashlib.sha256(
            self._manifest_bytes(include_artifact_id=False)
        ).hexdigest()

    def _manifest_bytes(self, *, include_artifact_id: bool) -> bytes:
        offset = 0
        parameter_entries: list[dict[str, object]] = []
        for parameter in self._parameters:
            parameter_entries.append(
                {
                    "name": parameter.name,
                    "shape": list(parameter.shape),
                    "offset": offset,
                    "count": parameter.value_count,
                }
            )
            offset += parameter.value_count

        tokenizer_entry: dict[str, object] = {
            "method": self._tokenizer.method,
        }
        if self._tokenizer.method == "byte":
            tokenizer_entry["byte_vocabulary"] = list(
                self._tokenizer.byte_vocabulary
            )
        else:
            tokenizer_entry["merge_rules"] = [
                list(rule) for rule in self._tokenizer.merge_rules
            ]

        manifest: dict[str, object] = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "stage": self._stage,
            "parent_artifact_id": self._parent_artifact_id,
            "config": asdict(self._config),
            "tokenizer": tokenizer_entry,
            "parameters": parameter_entries,
            "weights": {
                "file": WEIGHTS_NAME,
                "dtype": "float32",
                "byte_order": "little",
                "count": len(self._weights),
                "sha256": hashlib.sha256(
                    self._weights_bytes
                ).hexdigest(),
            },
            "metadata": json.loads(self._metadata_json),
        }
        if include_artifact_id:
            manifest["artifact_id"] = self._artifact_id
        return _canonical_json(manifest).encode("utf-8")


class ModelRuntime:
    """A closeable live tokenizer/model pair instantiated from an artifact."""

    __slots__ = ("_model", "_tokenizer")

    def __init__(self, bundle: ModelBundle, backend: str) -> None:
        if not isinstance(bundle, ModelBundle):
            raise TypeError("bundle must be a ModelBundle")
        self._model: DecoderOnlyTransformer | None = None
        self._tokenizer: Tokenizer | None = None
        tokenizer = bundle.tokenizer.instantiate()
        try:
            model = DecoderOnlyTransformer(bundle.config).to(backend)
            try:
                with model.parameters() as parameters:
                    expected_names = tuple(
                        parameter.name for parameter in bundle.parameters
                    )
                    expected_shapes = tuple(
                        parameter.shape for parameter in bundle.parameters
                    )
                    if parameters.names != expected_names:
                        raise ValueError(
                            "artifact parameter names do not match model"
                        )
                    if parameters.shapes != expected_shapes:
                        raise ValueError(
                            "artifact parameter shapes do not match model"
                        )
                    parameters.load_flat_values(bundle.weights)
            except BaseException:
                model.close()
                raise
        except BaseException:
            tokenizer.close()
            raise
        self._tokenizer = tokenizer
        self._model = model

    @property
    def tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise RuntimeError("model runtime is closed")
        return self._tokenizer

    @property
    def model(self) -> DecoderOnlyTransformer:
        if self._model is None:
            raise RuntimeError("model runtime is closed")
        return self._model

    def close(self) -> None:
        model = self._model
        tokenizer = self._tokenizer
        self._model = None
        self._tokenizer = None
        if model is not None:
            model.close()
        if tokenizer is not None:
            tokenizer.close()

    def __enter__(self) -> ModelRuntime:
        _ = self.model
        _ = self.tokenizer
        return self

    def __exit__(
        self,
        _type: object,
        _value: object,
        _traceback: object,
    ) -> None:
        self.close()


def _parse_manifest(manifest_bytes: bytes) -> dict[str, object]:
    try:
        decoded = manifest_bytes.decode("utf-8", errors="strict")
        value = json.loads(decoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("model bundle manifest is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("model bundle manifest must be a JSON object")
    if value.get("format") != FORMAT_NAME:
        raise ValueError("unknown model bundle format")
    if value.get("format_version") != FORMAT_VERSION:
        raise ValueError("unsupported model bundle format version")

    artifact_id = value.get("artifact_id")
    _validate_artifact_id(artifact_id, "artifact_id", allow_none=False)
    parent_artifact_id = value.get("parent_artifact_id")
    _validate_artifact_id(parent_artifact_id, "parent_artifact_id")

    config_entry = _required_mapping(value, "config")
    config = TransformerConfig(
        vocabulary_size=_required_integer(
            config_entry,
            "vocabulary_size",
        ),
        maximum_context=_required_integer(
            config_entry,
            "maximum_context",
        ),
        model_width=_required_integer(config_entry, "model_width"),
        head_count=_required_integer(config_entry, "head_count"),
        block_count=_required_integer(config_entry, "block_count"),
        feed_forward_width=_required_integer(
            config_entry,
            "feed_forward_width",
        ),
        random_seed=_required_integer(config_entry, "random_seed"),
        layer_norm_epsilon=_required_number(
            config_entry,
            "layer_norm_epsilon",
        ),
    )
    _validate_config(config)

    tokenizer_entry = _required_mapping(value, "tokenizer")
    method = tokenizer_entry.get("method")
    if method == "byte":
        vocabulary = _required_integer_list(
            tokenizer_entry,
            "byte_vocabulary",
        )
        tokenizer = TokenizerSpec(
            method="byte",
            byte_vocabulary=tuple(vocabulary),
        )
    elif method == "bpe":
        raw_rules = tokenizer_entry.get("merge_rules")
        if not isinstance(raw_rules, list):
            raise ValueError("tokenizer.merge_rules must be an array")
        rules: list[tuple[int, int, int]] = []
        for raw_rule in raw_rules:
            if not isinstance(raw_rule, list) or len(raw_rule) != 3:
                raise ValueError(
                    "each tokenizer merge rule must be a three-item array"
                )
            rules.append(
                tuple(
                    _manifest_integer(item, "tokenizer merge value")
                    for item in raw_rule
                )
            )
        tokenizer = TokenizerSpec(
            method="bpe",
            merge_rules=tuple(rules),
        )
    else:
        raise ValueError("unknown tokenizer method in model bundle")

    raw_parameters = value.get("parameters")
    if not isinstance(raw_parameters, list) or not raw_parameters:
        raise ValueError("parameters must be a nonempty array")
    parameters: list[ParameterSpec] = []
    expected_offset = 0
    for entry in raw_parameters:
        if not isinstance(entry, dict):
            raise ValueError("each parameter must be an object")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("parameter name must be a nonempty string")
        shape = tuple(_required_integer_list(entry, "shape"))
        parameter = ParameterSpec(name, shape)
        offset = _required_integer(entry, "offset")
        count = _required_integer(entry, "count")
        if offset != expected_offset or count != parameter.value_count:
            raise ValueError(
                "parameter offsets/counts do not match their shapes"
            )
        expected_offset += count
        parameters.append(parameter)

    weights_entry = _required_mapping(value, "weights")
    if (
        weights_entry.get("file") != WEIGHTS_NAME
        or weights_entry.get("dtype") != "float32"
        or weights_entry.get("byte_order") != "little"
    ):
        raise ValueError("unsupported model bundle weight encoding")
    if _required_integer(weights_entry, "count") != expected_offset:
        raise ValueError("manifest weight count does not match parameters")
    weight_digest = weights_entry.get("sha256")
    _validate_artifact_id(weight_digest, "weights.sha256", allow_none=False)

    stage = value.get("stage")
    if not isinstance(stage, str) or not stage:
        raise ValueError("stage must be a nonempty string")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")

    return {
        "artifact_id": artifact_id,
        "parent_artifact_id": parent_artifact_id,
        "config": config,
        "tokenizer": tokenizer,
        "parameters": tuple(parameters),
        "weights_sha256": weight_digest,
        "stage": stage,
        "metadata": metadata,
    }


def _canonical_weights(
    weights: Iterable[float],
) -> tuple[tuple[float, ...], bytes]:
    canonical: list[float] = []
    packed = bytearray()
    for index, value in enumerate(weights):
        if isinstance(value, bool):
            raise TypeError(f"weight {index} must be a number")
        try:
            converted = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"weight {index} must be a number") from error
        if not math.isfinite(converted):
            raise ValueError(f"weight {index} must be finite")
        try:
            encoded = struct.pack("<f", converted)
        except (OverflowError, struct.error) as error:
            raise ValueError(
                f"weight {index} is outside float32 range"
            ) from error
        rounded = struct.unpack("<f", encoded)[0]
        if not math.isfinite(rounded):
            raise ValueError(f"weight {index} is outside float32 range")
        canonical.append(rounded)
        packed.extend(encoded)
    return tuple(canonical), bytes(packed)


def _unpack_weights(weight_bytes: bytes) -> tuple[float, ...]:
    if len(weight_bytes) % 4 != 0:
        raise ValueError("float32 weight data must be four-byte aligned")
    return tuple(
        value[0]
        for value in struct.iter_unpack("<f", weight_bytes)
    )


def _validate_config(config: TransformerConfig) -> None:
    if not isinstance(config, TransformerConfig):
        raise TypeError("config must be a TransformerConfig")
    for name in (
        "vocabulary_size",
        "maximum_context",
        "model_width",
        "head_count",
        "block_count",
        "feed_forward_width",
    ):
        _positive_integer(getattr(config, name), name)
    _bounded_integer(config.random_seed, (1 << 32) - 1, "random_seed")
    if config.model_width % config.head_count != 0:
        raise ValueError("model_width must be divisible by head_count")
    epsilon = float(config.layer_norm_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            "layer_norm_epsilon must be finite and greater than zero"
        )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypeError("value must contain only finite JSON data") from error


def _deterministic_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 0
    info.external_attr = 0
    return info


def _validate_artifact_id(
    value: object,
    name: str,
    *,
    allow_none: bool = True,
) -> None:
    if value is None and allow_none:
        return
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _positive_integer(value: object, name: str) -> int:
    checked = _manifest_integer(value, name)
    if checked <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return checked


def _bounded_integer(value: object, maximum: int, name: str) -> int:
    checked = _manifest_integer(value, name)
    if checked < 0 or checked > maximum:
        raise ValueError(f"{name} is outside its supported range")
    return checked


def _manifest_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _required_mapping(
    mapping: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _required_integer(
    mapping: Mapping[str, object],
    name: str,
) -> int:
    if name not in mapping:
        raise ValueError(f"missing required integer {name}")
    return _manifest_integer(mapping[name], name)


def _required_number(
    mapping: Mapping[str, object],
    name: str,
) -> float:
    value = mapping.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _required_integer_list(
    mapping: Mapping[str, object],
    name: str,
) -> list[int]:
    value = mapping.get(name)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return [_manifest_integer(item, name) for item in value]


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "MAXIMUM_MANIFEST_BYTES",
    "WEIGHTS_NAME",
    "ModelBundle",
    "ModelRuntime",
    "ParameterSpec",
    "TokenizerSpec",
]
