"""Execution adapter from lab-owned specs to the public programmed API."""

from __future__ import annotations

from dataclasses import dataclass

from riftco_transformer import (
    ABI_VERSION,
    ABI_VERSION_MAJOR,
    ABI_VERSION_MINOR,
    backend_available,
)
from riftco_transformer.programmed import (
    MultilinearMap,
    NeuralLoweringConfig,
    ProgramAugmentedModel,
    ProgramAugmentedModelConfig,
    ProgramBranch,
    ProgramInputLayout as NativeProgramInputLayout,
)

from .config import ModelConfig, Variant
from .data import TokenCodec
from .programs import ProgramSpec, build_program_spec
from .protocol import ProtocolConfig


@dataclass(frozen=True, slots=True)
class ParameterManifest:
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    total_numel: int


@dataclass(slots=True)
class NativeProgram:
    """Native objects produced from one sparse lab program description."""

    map: MultilinearMap
    branch: ProgramBranch

    def close(self) -> None:
        self.map.close()


class ModelRuntime:
    """Own a program-augmented model and its source map in close order."""

    __slots__ = ("config", "model", "program", "program_spec", "protocol")

    def __init__(
        self,
        *,
        protocol: ProtocolConfig,
        config: ModelConfig,
        model: ProgramAugmentedModel,
        program: NativeProgram | None,
        program_spec: ProgramSpec | None,
    ) -> None:
        self.protocol = protocol
        self.config = config
        self.model = model
        self.program = program
        self.program_spec = program_spec

    @property
    def variant(self) -> Variant:
        return self.config.variant

    @property
    def has_program(self) -> bool:
        return self.model.has_program

    @property
    def backend(self) -> str:
        return self.model.backend

    def parameter_manifest(self) -> ParameterManifest:
        with self.model.parameters() as parameters:
            return ParameterManifest(
                names=parameters.names,
                shapes=parameters.shapes,
                total_numel=parameters.total_numel,
            )

    def close(self) -> None:
        model = self.model
        program = self.program
        try:
            model.close()
        finally:
            if program is not None:
                program.close()

    def __enter__(self) -> ModelRuntime:
        if self.model.closed:
            raise RuntimeError("model runtime is closed")
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()


def translate_program_spec(spec: ProgramSpec) -> NativeProgram:
    """Copy sparse unit entries into a generic native multilinear map."""

    if not isinstance(spec, ProgramSpec):
        raise TypeError("spec must be a ProgramSpec")
    native_map = MultilinearMap.from_sparse(
        spec.reference_map.input_dimensions,
        spec.reference_map.output_dimension,
        spec.reference_map.nonzero_flat_indices,
        (1.0 for _ in spec.reference_map.nonzero_flat_indices),
    )
    try:
        lowering = NeuralLoweringConfig(
            strategy=("linear" if spec.variant is Variant.P else "linear_attention"),
            precision="exact",
            initialization=spec.initialization.value,
            trainable=spec.trainable,
            random_seed=spec.random_seed,
            random_scale=spec.random_scale,
            attention_query_axis=spec.attention_query_axis,
        )
        inputs = tuple(
            NativeProgramInputLayout(
                source=layout.source.value,
                position=layout.position,
                projection_group=projection_group,
            )
            for layout, projection_group in zip(
                spec.input_layouts,
                spec.input_projection_groups,
            )
        )
        branch = ProgramBranch(
            map=native_map,
            source_offset=spec.source_offset,
            source_length=spec.source_length,
            target_offset=spec.target_offset,
            output_length=spec.output_length,
            inputs=inputs,
            lowering=lowering,
            input_projection_bias=spec.input_projection_bias,
            merge_bias=spec.program_merge_bias,
        )
        return NativeProgram(native_map, branch)
    except BaseException:
        native_map.close()
        raise


def build_model(
    protocol: ProtocolConfig,
    config: ModelConfig,
) -> ModelRuntime:
    """Build one F/P/T/I runtime; all variant choices remain in the lab."""

    if not isinstance(protocol, ProtocolConfig):
        raise TypeError("protocol must be a ProtocolConfig")
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be a ModelConfig")
    codec = TokenCodec.from_protocol(protocol)
    program_spec = build_program_spec(
        config.variant,
        protocol.sequence_length,
        config.program_width,
        seed=config.seed,
        random_scale=config.random_program_scale,
    )
    native_program = (
        None if program_spec is None else translate_program_spec(program_spec)
    )
    try:
        model = ProgramAugmentedModel(
            ProgramAugmentedModelConfig(
                vocabulary_size=codec.vocabulary_size,
                context_length=protocol.context_length,
                model_width=config.model_width,
                head_count=config.head_count,
                attention_branch_count=config.parallel_attention_count,
                feed_forward_width=config.feed_forward_width,
                random_seed=config.seed,
                attention="materialized",
            ),
            None if native_program is None else native_program.branch,
        )
        model.to(config.backend)
        return ModelRuntime(
            protocol=protocol,
            config=config,
            model=model,
            program=native_program,
            program_spec=program_spec,
        )
    except BaseException:
        if native_program is not None:
            native_program.close()
        raise


def resolve_backend(requested: str) -> str:
    """Resolve auto acceleration or reject an unavailable explicit backend."""

    if not isinstance(requested, str):
        raise TypeError("requested backend must be a str")
    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "metal", "cuda", "tpu"}:
        raise ValueError("backend must be auto, cpu, metal, cuda, or tpu")
    if normalized == "auto":
        for candidate in ("tpu", "cuda", "metal"):
            if backend_available(candidate):
                return candidate
        return "cpu"
    if not backend_available(normalized):
        raise RuntimeError(f"requested backend is unavailable: {normalized}")
    return normalized


def framework_abi() -> dict[str, int]:
    return {
        "version": ABI_VERSION,
        "major": ABI_VERSION_MAJOR,
        "minor": ABI_VERSION_MINOR,
    }


__all__ = [
    "ModelRuntime",
    "NativeProgram",
    "ParameterManifest",
    "build_model",
    "framework_abi",
    "resolve_backend",
    "translate_program_spec",
]
