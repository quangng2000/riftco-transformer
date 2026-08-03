"""Shared, dependency-free contracts for model interchange adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct
from typing import Iterable, Mapping

from .._numeric import positive_float32
from ..artifacts import ModelBundle, ParameterSpec, TokenizerSpec
from ..native import TransformerConfig


MAXIMUM_TENSOR_ELEMENTS = (1 << 63) - 1
# Interchange metadata is untrusted. Each decoder block expands into sixteen
# parameter descriptors, so keep the topology bound intentionally modest while
# still leaving ample room above teaching- and research-scale configurations.
MAXIMUM_DECODER_BLOCKS = 256


@dataclass(frozen=True, slots=True)
class Float32Tensor:
    """One immutable, row-major F32 tensor independent of any runtime."""

    shape: tuple[int, ...]
    values: tuple[float, ...]
    _little_endian_bytes: bytes = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        try:
            shape = tuple(self.shape)
        except TypeError as error:
            raise TypeError("tensor shape must be an iterable") from error

        value_count = 1
        for index, dimension in enumerate(shape):
            if isinstance(dimension, bool) or not isinstance(dimension, int):
                raise TypeError(
                    f"tensor shape dimension {index} must be an integer"
                )
            if dimension < 0:
                raise ValueError(
                    f"tensor shape dimension {index} must not be negative"
                )
            if dimension > MAXIMUM_TENSOR_ELEMENTS:
                raise ValueError(
                    f"tensor shape dimension {index} is too large"
                )
            if dimension == 0:
                value_count = 0
            elif value_count:
                if value_count > MAXIMUM_TENSOR_ELEMENTS // dimension:
                    raise ValueError("tensor element count is too large")
                value_count *= dimension

        try:
            raw_values = tuple(self.values)
        except TypeError as error:
            raise TypeError("tensor values must be an iterable") from error
        if len(raw_values) != value_count:
            raise ValueError(
                "tensor value count does not match its shape: "
                f"expected {value_count}, received {len(raw_values)}"
            )

        canonical_values: list[float] = []
        encoded = bytearray()
        for index, raw_value in enumerate(raw_values):
            if isinstance(raw_value, bool):
                raise TypeError(f"tensor value {index} must be a number")
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"tensor value {index} must be a number"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"tensor value {index} must be finite")
            try:
                packed = struct.pack("<f", value)
            except (OverflowError, struct.error) as error:
                raise ValueError(
                    f"tensor value {index} is outside float32 range"
                ) from error
            rounded = struct.unpack("<f", packed)[0]
            if not math.isfinite(rounded):
                raise ValueError(
                    f"tensor value {index} is outside float32 range"
                )
            canonical_values.append(rounded)
            encoded.extend(packed)

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "values", tuple(canonical_values))
        object.__setattr__(self, "_little_endian_bytes", bytes(encoded))

    @property
    def value_count(self) -> int:
        """Return the number of scalar F32 values in the tensor."""

        return len(self.values)

    @property
    def byte_count(self) -> int:
        """Return the encoded payload size."""

        return len(self._little_endian_bytes)

    def to_little_endian_bytes(self) -> bytes:
        """Return canonical IEEE-754 little-endian F32 bytes."""

        return self._little_endian_bytes

    @classmethod
    def from_little_endian_bytes(
        cls,
        shape: Iterable[int],
        data: bytes | bytearray | memoryview,
    ) -> Float32Tensor:
        """Decode a tensor after validating four-byte alignment."""

        try:
            view = memoryview(data).cast("B")
        except (TypeError, ValueError) as error:
            raise TypeError("tensor data must be bytes-like") from error
        if len(view) % 4:
            raise ValueError("F32 tensor data must be four-byte aligned")
        values = tuple(
            value[0]
            for value in struct.iter_unpack("<f", view)
        )
        return cls(tuple(shape), values)


def current_decoder_parameter_specs(
    config: TransformerConfig,
) -> tuple[ParameterSpec, ...]:
    """Return the exact native parameter contract for today's decoder."""

    _validate_transformer_config(config)
    vocabulary_size = config.vocabulary_size
    maximum_context = config.maximum_context
    model_width = config.model_width
    feed_forward_width = config.feed_forward_width

    specs: list[ParameterSpec] = [
        ParameterSpec(
            "token_embedding.weight",
            (vocabulary_size, model_width),
        ),
        ParameterSpec(
            "position_embedding.weight",
            (maximum_context, model_width),
        ),
    ]
    for block_index in range(config.block_count):
        prefix = f"blocks.{block_index}."
        specs.extend(
            (
                ParameterSpec(prefix + "attention_norm.scale", (model_width,)),
                ParameterSpec(prefix + "attention_norm.bias", (model_width,)),
                ParameterSpec(
                    prefix + "attention.query.weight",
                    (model_width, model_width),
                ),
                ParameterSpec(prefix + "attention.query.bias", (model_width,)),
                ParameterSpec(
                    prefix + "attention.key.weight",
                    (model_width, model_width),
                ),
                ParameterSpec(prefix + "attention.key.bias", (model_width,)),
                ParameterSpec(
                    prefix + "attention.value.weight",
                    (model_width, model_width),
                ),
                ParameterSpec(prefix + "attention.value.bias", (model_width,)),
                ParameterSpec(
                    prefix + "attention.output.weight",
                    (model_width, model_width),
                ),
                ParameterSpec(prefix + "attention.output.bias", (model_width,)),
                ParameterSpec(
                    prefix + "feed_forward_norm.scale",
                    (model_width,),
                ),
                ParameterSpec(
                    prefix + "feed_forward_norm.bias",
                    (model_width,),
                ),
                ParameterSpec(
                    prefix + "feed_forward.expand.weight",
                    (feed_forward_width, model_width),
                ),
                ParameterSpec(
                    prefix + "feed_forward.expand.bias",
                    (feed_forward_width,),
                ),
                ParameterSpec(
                    prefix + "feed_forward.project.weight",
                    (model_width, feed_forward_width),
                ),
                ParameterSpec(
                    prefix + "feed_forward.project.bias",
                    (model_width,),
                ),
            )
        )
    specs.extend(
        (
            ParameterSpec("final_norm.scale", (model_width,)),
            ParameterSpec("final_norm.bias", (model_width,)),
            ParameterSpec(
                "language_model_head.weight",
                (vocabulary_size, model_width),
            ),
            ParameterSpec("language_model_head.bias", (vocabulary_size,)),
        )
    )
    return tuple(specs)


def bundle_tensors(bundle: ModelBundle) -> dict[str, Float32Tensor]:
    """Split a current-decoder bundle into exact named F32 tensors."""

    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle")
    expected_specs = current_decoder_parameter_specs(bundle.config)
    if bundle.parameters != expected_specs:
        raise ValueError(_parameter_contract_error(bundle.parameters, expected_specs))

    tensors: dict[str, Float32Tensor] = {}
    offset = 0
    for parameter in expected_specs:
        next_offset = offset + parameter.value_count
        tensors[parameter.name] = Float32Tensor(
            parameter.shape,
            bundle.weights[offset:next_offset],
        )
        offset = next_offset
    return tensors


def build_bundle_from_tensors(
    *,
    config: TransformerConfig,
    tokenizer: TokenizerSpec,
    tensors: Mapping[str, Float32Tensor],
    stage: str,
    parent_artifact_id: str | None = None,
    metadata: Mapping[str, object] | None = None,
) -> ModelBundle:
    """Build a native bundle after exact-name and exact-shape validation."""

    if not isinstance(tensors, Mapping):
        raise TypeError("tensors must be a mapping")
    expected_specs = current_decoder_parameter_specs(config)
    expected_names = tuple(spec.name for spec in expected_specs)
    expected_name_set = set(expected_names)
    actual_names: set[str] = set()
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not name:
            raise ValueError("tensor names must be nonempty strings")
        if not isinstance(tensor, Float32Tensor):
            raise TypeError(
                f"tensor {name!r} must be a Float32Tensor"
            )
        actual_names.add(name)

    missing = sorted(expected_name_set - actual_names)
    extra = sorted(actual_names - expected_name_set)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(repr(name) for name in missing))
        if extra:
            details.append("unexpected " + ", ".join(repr(name) for name in extra))
        raise ValueError(
            "tensor names do not match the Riftco decoder contract: "
            + "; ".join(details)
        )

    flattened: list[float] = []
    for spec in expected_specs:
        tensor = tensors[spec.name]
        if tensor.shape != spec.shape:
            raise ValueError(
                f"tensor {spec.name!r} has shape {tensor.shape}; "
                f"expected {spec.shape}"
            )
        flattened.extend(tensor.values)

    return ModelBundle(
        config=config,
        tokenizer=tokenizer,
        parameters=expected_specs,
        weights=flattened,
        stage=stage,
        parent_artifact_id=parent_artifact_id,
        metadata=metadata,
    )


def _validate_transformer_config(config: TransformerConfig) -> None:
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
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero")
        if value > MAXIMUM_TENSOR_ELEMENTS:
            raise ValueError(f"{name} is too large")
    if config.block_count > MAXIMUM_DECODER_BLOCKS:
        raise ValueError(
            f"block_count exceeds the safety limit {MAXIMUM_DECODER_BLOCKS}"
        )
    if config.model_width % config.head_count:
        raise ValueError("model_width must be divisible by head_count")

    vocabulary_size = config.vocabulary_size
    maximum_context = config.maximum_context
    model_width = config.model_width
    block_count = config.block_count
    feed_forward_width = config.feed_forward_width
    total_values = (
        (2 * vocabulary_size * model_width)
        + (maximum_context * model_width)
        + block_count
        * (
            (4 * model_width * model_width)
            + (2 * model_width * feed_forward_width)
            + (9 * model_width)
            + feed_forward_width
        )
        + (2 * model_width)
        + vocabulary_size
    )
    if total_values > MAXIMUM_TENSOR_ELEMENTS:
        raise ValueError("decoder parameter count is too large")
    if (
        isinstance(config.random_seed, bool)
        or not isinstance(config.random_seed, int)
    ):
        raise TypeError("random_seed must be an integer")
    if config.random_seed < 0 or config.random_seed > (1 << 32) - 1:
        raise ValueError("random_seed is outside uint32 range")
    positive_float32(config.layer_norm_epsilon, "layer_norm_epsilon")


def _parameter_contract_error(
    actual: tuple[ParameterSpec, ...],
    expected: tuple[ParameterSpec, ...],
) -> str:
    actual_by_name = {spec.name: spec.shape for spec in actual}
    expected_by_name = {spec.name: spec.shape for spec in expected}
    missing = sorted(set(expected_by_name) - set(actual_by_name))
    extra = sorted(set(actual_by_name) - set(expected_by_name))
    wrong_shape = sorted(
        name
        for name in set(actual_by_name) & set(expected_by_name)
        if actual_by_name[name] != expected_by_name[name]
    )
    details: list[str] = []
    if missing:
        details.append("missing " + ", ".join(repr(name) for name in missing))
    if extra:
        details.append("unexpected " + ", ".join(repr(name) for name in extra))
    if wrong_shape:
        details.append(
            "wrong shape for "
            + ", ".join(
                f"{name!r} ({actual_by_name[name]} != {expected_by_name[name]})"
                for name in wrong_shape
            )
        )
    if not details:
        details.append("parameter order differs from native stable order")
    return (
        "model bundle does not match the current Riftco decoder topology: "
        + "; ".join(details)
    )


__all__ = [
    "MAXIMUM_DECODER_BLOCKS",
    "MAXIMUM_TENSOR_ELEMENTS",
    "Float32Tensor",
    "build_bundle_from_tensors",
    "bundle_tensors",
    "current_decoder_parameter_specs",
]
