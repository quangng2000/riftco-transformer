"""Checksummed, atomic, dependency-free exact-resume checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import tempfile
from typing import Iterable, Mapping
import zipfile

from .._numeric import positive_float32
from ..native import (
    Adam,
    AdamOptions,
    AdamState,
    DecoderOnlyTransformer,
    LoraConfig,
    TransformerConfig,
)
from ..training import BatchSourceState


FORMAT_NAME = "riftco-transformer-training-checkpoint"
FORMAT_VERSION = 2
LEGACY_FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
MODEL_VALUES_NAME = "model.f32le"
PACKED_MODEL_STATE_NAME = "model-nf4.rtnf4"
ADAPTER_VALUES_NAME = "adapter.f32le"
FIRST_MOMENTS_NAME = "adam-first-moments.f32le"
SECOND_MOMENTS_NAME = "adam-second-moments.f32le"
MAXIMUM_MANIFEST_BYTES = 1 << 20
_BOUNDARY = "post_optimizer_step"


@dataclass(frozen=True, slots=True)
class _ParameterSpec:
    name: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("parameter name must be a nonempty string")
        try:
            shape = tuple(self.shape)
        except TypeError as error:
            raise TypeError("parameter shape must be iterable") from error
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
class TrainingCheckpointRestore:
    """Identity and counters returned after one successful restoration."""

    checkpoint_id: str
    global_step: int
    optimizer_step: int


class TrainingCheckpoint:
    """Immutable logical state for exact continuation of one Adam loop.

    A checkpoint restores into an already-constructed model and optimizer.
    This keeps architecture and physical backend selection in ordinary code
    while the archive remains backend-neutral. ``.rift`` model-bundle v1 is a
    separate inference/stage artifact and is not modified by this format.
    """

    __slots__ = (
        "_adapter_parameters",
        "_adapter_values",
        "_batch_source_state",
        "_checkpoint_id",
        "_config",
        "_activation_checkpointing",
        "_first_moments",
        "_full_sequence_attention",
        "_global_step",
        "_lora_config",
        "_metadata_json",
        "_model_parameters",
        "_model_values",
        "_packed_model_state",
        "_format_version",
        "_optimizer_options",
        "_optimizer_scope",
        "_optimizer_step",
        "_python_random_state",
        "_second_moments",
        "_beta1_power",
        "_beta2_power",
    )

    def __init__(
        self,
        *,
        config: TransformerConfig,
        full_sequence_attention: str,
        activation_checkpointing: str,
        model_parameters: Iterable[_ParameterSpec],
        model_values: Iterable[float],
        packed_model_state: bytes | None,
        lora_config: LoraConfig | None,
        adapter_parameters: Iterable[_ParameterSpec],
        adapter_values: Iterable[float],
        optimizer_scope: str,
        optimizer_options: AdamOptions,
        optimizer_step: int,
        beta1_power: float,
        beta2_power: float,
        first_moments: Iterable[float],
        second_moments: Iterable[float],
        global_step: int,
        python_random_state: tuple[object, ...],
        batch_source_state: BatchSourceState | None,
        metadata: Mapping[str, object] | None = None,
        format_version: int = FORMAT_VERSION,
    ) -> None:
        if (
            isinstance(format_version, bool)
            or not isinstance(format_version, int)
            or format_version
            not in {LEGACY_FORMAT_VERSION, FORMAT_VERSION}
        ):
            raise ValueError("unsupported training checkpoint format version")
        if not isinstance(config, TransformerConfig):
            raise TypeError("config must be a TransformerConfig")
        _validate_config(config)
        if full_sequence_attention not in {"materialized", "flash"}:
            raise ValueError(
                "full_sequence_attention must be 'materialized' or 'flash'"
            )
        if activation_checkpointing not in {"disabled", "block"}:
            raise ValueError(
                "activation_checkpointing must be 'disabled' or 'block'"
            )
        model_specs = _parameter_specs(model_parameters, "model")
        adapter_specs = _parameter_specs(
            adapter_parameters, "adapter", allow_empty=True
        )
        model_tuple, _ = _canonical_f32_values(model_values, "model values")
        adapter_tuple, _ = _canonical_f32_values(
            adapter_values, "adapter values"
        )
        _require_value_count(model_specs, model_tuple, "model")
        _require_value_count(adapter_specs, adapter_tuple, "adapter")
        if packed_model_state is None:
            packed_state = None
        elif not isinstance(packed_model_state, bytes):
            raise TypeError("packed_model_state must be bytes or None")
        elif not packed_model_state:
            raise ValueError("packed_model_state must not be empty")
        else:
            packed_state = packed_model_state
        if format_version == LEGACY_FORMAT_VERSION and packed_state is not None:
            raise ValueError("checkpoint v1 cannot contain packed NF4 state")

        if lora_config is None:
            if adapter_specs or adapter_tuple:
                raise ValueError(
                    "adapter state requires an active LoRA configuration"
                )
        elif not isinstance(lora_config, LoraConfig):
            raise TypeError("lora_config must be a LoraConfig or None")
        elif not adapter_specs:
            raise ValueError("active LoRA requires adapter parameters")
        if packed_state is not None and lora_config is None:
            raise ValueError("packed QLoRA state requires an active adapter")

        if optimizer_scope not in {"model", "adapter"}:
            raise ValueError("optimizer_scope must be 'model' or 'adapter'")
        if optimizer_scope == "adapter" and lora_config is None:
            raise ValueError("adapter optimizer scope requires active LoRA")
        if lora_config is not None and optimizer_scope != "adapter":
            raise ValueError(
                "active LoRA checkpoints require Adam to optimize adapters"
            )
        if packed_state is not None and optimizer_scope != "adapter":
            raise ValueError("packed QLoRA state requires adapter-only Adam")
        optimizer_specs = (
            model_specs if optimizer_scope == "model" else adapter_specs
        )
        optimizer_values = (
            model_tuple if optimizer_scope == "model" else adapter_tuple
        )

        if not isinstance(optimizer_options, AdamOptions):
            raise TypeError("optimizer_options must be an AdamOptions")
        _validate_optimizer_options(optimizer_options)
        checked_optimizer_step = _nonnegative_integer(
            optimizer_step, "optimizer_step"
        )
        checked_beta1_power = _beta_power(
            beta1_power,
            optimizer_options.beta1,
            checked_optimizer_step,
            "beta1_power",
        )
        checked_beta2_power = _beta_power(
            beta2_power,
            optimizer_options.beta2,
            checked_optimizer_step,
            "beta2_power",
        )
        first_tuple, _ = _canonical_f32_values(
            first_moments, "Adam first moments"
        )
        second_tuple, _ = _canonical_f32_values(
            second_moments, "Adam second moments"
        )
        expected_optimizer_values = sum(
            parameter.value_count for parameter in optimizer_specs
        )
        if (
            len(optimizer_values) != expected_optimizer_values
            or len(first_tuple) != expected_optimizer_values
            or len(second_tuple) != expected_optimizer_values
        ):
            raise ValueError(
                "Adam state counts do not match optimizer parameters"
            )

        checked_global_step = _nonnegative_integer(
            global_step, "global_step"
        )
        checked_random_state = _checked_random_state(
            python_random_state,
            "python_random_state",
        )
        if batch_source_state is not None and not isinstance(
            batch_source_state, BatchSourceState
        ):
            raise TypeError(
                "batch_source_state must be a BatchSourceState or None"
            )

        metadata_json = _canonical_json(
            {} if metadata is None else dict(metadata)
        )
        if not isinstance(json.loads(metadata_json), dict):
            raise TypeError("metadata must be a JSON object")

        self._config = config
        self._full_sequence_attention = full_sequence_attention
        self._activation_checkpointing = activation_checkpointing
        self._model_parameters = model_specs
        self._model_values = model_tuple
        self._packed_model_state = packed_state
        self._format_version = format_version
        self._lora_config = lora_config
        self._adapter_parameters = adapter_specs
        self._adapter_values = adapter_tuple
        self._optimizer_scope = optimizer_scope
        self._optimizer_options = optimizer_options
        self._optimizer_step = checked_optimizer_step
        self._beta1_power = checked_beta1_power
        self._beta2_power = checked_beta2_power
        self._first_moments = first_tuple
        self._second_moments = second_tuple
        self._global_step = checked_global_step
        self._python_random_state = checked_random_state
        self._batch_source_state = batch_source_state
        self._metadata_json = metadata_json
        manifest_without_id = self._manifest_bytes(
            include_checkpoint_id=False
        )
        self._checkpoint_id = hashlib.sha256(manifest_without_id).hexdigest()
        if len(self._manifest_bytes(include_checkpoint_id=True)) > (
            MAXIMUM_MANIFEST_BYTES
        ):
            raise ValueError("training checkpoint manifest is too large")

    @property
    def checkpoint_id(self) -> str:
        return self._checkpoint_id

    @property
    def global_step(self) -> int:
        return self._global_step

    @property
    def optimizer_step(self) -> int:
        return self._optimizer_step

    @property
    def full_sequence_attention(self) -> str:
        return self._full_sequence_attention

    @property
    def activation_checkpointing(self) -> str:
        return self._activation_checkpointing

    @property
    def optimizer_options(self) -> AdamOptions:
        return self._optimizer_options

    @property
    def optimizer_scope(self) -> str:
        return self._optimizer_scope

    @property
    def quantized(self) -> bool:
        return self._packed_model_state is not None

    @property
    def packed_model_state(self) -> bytes | None:
        """Return immutable canonical NF4 bytes, or ``None`` for FP32 bases."""

        return self._packed_model_state

    @property
    def batch_source_state(self) -> BatchSourceState | None:
        return self._batch_source_state

    @property
    def metadata(self) -> dict[str, object]:
        return json.loads(self._metadata_json)

    @classmethod
    def capture(
        cls,
        model: DecoderOnlyTransformer,
        optimizer: Adam,
        *,
        source: object | None = None,
        global_step: int | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> TrainingCheckpoint:
        """Capture model/Adam/RNG/data state at a clean post-step boundary."""

        if not isinstance(model, DecoderOnlyTransformer):
            raise TypeError("model must be a DecoderOnlyTransformer")
        if not isinstance(optimizer, Adam):
            raise TypeError("optimizer must be an Adam")
        if not optimizer.owns_parameters_of(model):
            raise ValueError(
                "optimizer must own parameters from this exact model"
            )
        if model.backend != optimizer.backend:
            raise ValueError("model and optimizer backends must match")
        optimizer_state = optimizer.state()
        packed_model_state = (
            model.packed_quantized_state
            if model.quantized_linear_weights
            else None
        )
        with model.parameters() as parameters:
            model_specs = _specs_from_parameter_list(parameters)
            model_values = parameters.flat_values()

        lora_config = model.lora_config
        adapter_specs: tuple[_ParameterSpec, ...] = ()
        adapter_values: tuple[float, ...] = ()
        if lora_config is not None:
            with model.adapter_parameters() as parameters:
                adapter_specs = _specs_from_parameter_list(parameters)
                adapter_values = parameters.flat_values()

        optimizer_specs = tuple(
            _ParameterSpec(name, shape)
            for name, shape in zip(
                optimizer.parameter_names,
                optimizer.parameter_shapes,
            )
        )
        if optimizer_specs == model_specs and lora_config is None:
            optimizer_scope = "model"
            expected_values = model_values
        elif optimizer_specs == adapter_specs and lora_config is not None:
            optimizer_scope = "adapter"
            expected_values = adapter_values
        else:
            raise ValueError(
                "Adam must own the complete model parameter list or the "
                "complete active LoRA adapter parameter list"
            )
        if optimizer_state.parameter_values != expected_values:
            raise RuntimeError(
                "model values changed while checkpoint state was captured"
            )

        batch_state = _capture_batch_source_state(source)
        return cls(
            config=model.config,
            full_sequence_attention=model.full_sequence_attention,
            activation_checkpointing=model.activation_checkpointing,
            model_parameters=model_specs,
            model_values=model_values,
            packed_model_state=packed_model_state,
            lora_config=lora_config,
            adapter_parameters=adapter_specs,
            adapter_values=adapter_values,
            optimizer_scope=optimizer_scope,
            optimizer_options=optimizer.options,
            optimizer_step=optimizer_state.step_count,
            beta1_power=optimizer_state.beta1_power,
            beta2_power=optimizer_state.beta2_power,
            first_moments=optimizer_state.first_moments,
            second_moments=optimizer_state.second_moments,
            global_step=(
                optimizer_state.step_count
                if global_step is None
                else global_step
            ),
            python_random_state=random.getstate(),
            batch_source_state=batch_state,
            metadata=metadata,
        )

    def validate(
        self,
        model: DecoderOnlyTransformer,
        optimizer: Adam,
        source: object | None = None,
    ) -> BatchSourceState | None:
        """Validate every live-state compatibility check without mutation."""

        if not isinstance(model, DecoderOnlyTransformer):
            raise TypeError("model must be a DecoderOnlyTransformer")
        if not isinstance(optimizer, Adam):
            raise TypeError("optimizer must be an Adam")
        if not optimizer.owns_parameters_of(model):
            raise ValueError(
                "optimizer must own parameters from this exact model"
            )
        if not _runtime_configs_compatible(model.config, self._config):
            raise ValueError("model configuration does not match checkpoint")
        if model.full_sequence_attention != self._full_sequence_attention:
            raise ValueError(
                "model full-sequence attention does not match checkpoint"
            )
        if model.activation_checkpointing != self._activation_checkpointing:
            raise ValueError(
                "model activation checkpointing does not match checkpoint"
            )
        if model.backend != optimizer.backend:
            raise ValueError("model and optimizer backends must match")
        if model.quantized_linear_weights != (
            self._packed_model_state is not None
        ):
            raise ValueError(
                "model packed-NF4 state does not match checkpoint"
            )
        if not _lora_configs_compatible(
            model.lora_config, self._lora_config
        ):
            raise ValueError("model LoRA configuration does not match checkpoint")

        with model.parameters() as parameters:
            current_model_specs = _specs_from_parameter_list(parameters)
        if current_model_specs != self._model_parameters:
            raise ValueError("model parameter signature does not match checkpoint")

        current_adapter_specs: tuple[_ParameterSpec, ...] = ()
        if self._lora_config is not None:
            with model.adapter_parameters() as parameters:
                current_adapter_specs = _specs_from_parameter_list(parameters)
        if current_adapter_specs != self._adapter_parameters:
            raise ValueError(
                "adapter parameter signature does not match checkpoint"
            )

        expected_optimizer_specs = (
            self._model_parameters
            if self._optimizer_scope == "model"
            else self._adapter_parameters
        )
        actual_optimizer_specs = tuple(
            _ParameterSpec(name, shape)
            for name, shape in zip(
                optimizer.parameter_names,
                optimizer.parameter_shapes,
            )
        )
        if actual_optimizer_specs != expected_optimizer_specs:
            raise ValueError(
                "optimizer parameter signature does not match checkpoint"
            )
        _validate_semantic_options(
            optimizer.options, self._optimizer_options
        )

        previous_source_state = _validate_batch_source(
            source, self._batch_source_state
        )
        _checked_random_state(
            self._python_random_state, "python_random_state"
        )
        return previous_source_state

    def restore(
        self,
        model: DecoderOnlyTransformer,
        optimizer: Adam,
        source: object | None = None,
    ) -> TrainingCheckpointRestore:
        """Validate first, then restore one clean post-step boundary."""

        previous_source_state = self.validate(model, optimizer, source)
        previous_optimizer_state = optimizer.state()
        previous_python_random_state = random.getstate()
        previous_model_values: tuple[float, ...] | None = None
        if self._optimizer_scope == "adapter":
            with model.parameters() as parameters:
                previous_model_values = parameters.flat_values()
        previous_packed_model_state = (
            model.packed_quantized_state
            if self._packed_model_state is not None
            else None
        )
        parameter_values = (
            self._model_values
            if self._optimizer_scope == "model"
            else self._adapter_values
        )
        checkpoint_optimizer_state = AdamState(
            step_count=self._optimizer_step,
            beta1_power=self._beta1_power,
            beta2_power=self._beta2_power,
            parameter_values=parameter_values,
            first_moments=self._first_moments,
            second_moments=self._second_moments,
        )

        source_restore_attempted = False
        frozen_model_committed = False
        packed_model_committed = False
        optimizer_committed = False
        try:
            # A custom source callback is the least constrained component.
            # Restore it first so a failure cannot leave Adam at checkpoint
            # state, and retain its pre-call snapshot for rollback if it
            # mutates before raising.
            if self._batch_source_state is not None:
                source_restore_attempted = True
                restore = getattr(source, "restore_checkpoint_state")
                restore(self._batch_source_state)
            if self._optimizer_scope == "adapter":
                model._load_frozen_parameter_values(
                    optimizer, self._model_values
                )
                frozen_model_committed = True
            if self._packed_model_state is not None:
                model.load_packed_quantized_state(self._packed_model_state)
                packed_model_committed = True
            optimizer.load_state(checkpoint_optimizer_state)
            optimizer_committed = True
            random.setstate(self._python_random_state)
        except BaseException as error:
            rollback_errors: list[BaseException] = []
            try:
                random.setstate(previous_python_random_state)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
            if optimizer_committed:
                try:
                    optimizer.load_state(previous_optimizer_state)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if packed_model_committed:
                try:
                    model.load_packed_quantized_state(
                        previous_packed_model_state
                    )
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if frozen_model_committed:
                try:
                    assert previous_model_values is not None
                    model._load_frozen_parameter_values(
                        optimizer, previous_model_values
                    )
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if source_restore_attempted:
                try:
                    restore = getattr(source, "restore_checkpoint_state")
                    restore(previous_source_state)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                raise RuntimeError(
                    "training checkpoint restore failed and rollback could "
                    "not recover every component; live state may be "
                    "indeterminate"
                ) from error
            raise
        return TrainingCheckpointRestore(
            checkpoint_id=self._checkpoint_id,
            global_step=self._global_step,
            optimizer_step=self._optimizer_step,
        )

    def save(self, path: str | os.PathLike[str]) -> Path:
        _validate_config(self._config)
        destination = Path(path)
        if destination.exists() and destination.is_dir():
            raise IsADirectoryError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)

        members = self._member_bytes()
        manifest = self._manifest_bytes(include_checkpoint_id=True)
        if len(manifest) > MAXIMUM_MANIFEST_BYTES:
            raise ValueError("training checkpoint manifest is too large")
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
                    _deterministic_zip_info(MANIFEST_NAME), manifest
                )
                for name, payload in members:
                    archive.writestr(
                        _deterministic_zip_info(name), payload
                    )
            _fsync_file(temporary_path)
            os.replace(temporary_path, destination)
            _fsync_directory(destination.parent)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        return destination

    @classmethod
    def load(
        cls, path: str | os.PathLike[str]
    ) -> TrainingCheckpoint:
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(source)
        try:
            with zipfile.ZipFile(source, mode="r") as archive:
                infos = archive.infolist()
                if not infos or infos[0].filename != MANIFEST_NAME:
                    raise ValueError(
                        "training checkpoint must begin with manifest.json"
                    )
                if any(
                    info.compress_type != zipfile.ZIP_STORED
                    for info in infos
                ):
                    raise ValueError(
                        "training checkpoint entries must use stored compression"
                    )
                if infos[0].file_size > MAXIMUM_MANIFEST_BYTES:
                    raise ValueError("training checkpoint manifest is too large")
                manifest = _parse_manifest(archive.read(infos[0]))
                expected_names = [MANIFEST_NAME]
                expected_names.extend(manifest["member_names"])
                actual_names = [info.filename for info in infos]
                if actual_names != expected_names:
                    raise ValueError(
                        "training checkpoint members/order do not match manifest"
                    )
                payloads: dict[str, bytes] = {}
                for info in infos[1:]:
                    descriptor = manifest["members"][info.filename]
                    if info.file_size != descriptor["size"]:
                        raise ValueError(
                            f"checkpoint member {info.filename} has wrong size"
                        )
                    payload = archive.read(info)
                    if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
                        raise ValueError(
                            f"checkpoint member {info.filename} checksum does not match"
                        )
                    payloads[info.filename] = payload
        except zipfile.BadZipFile as error:
            raise ValueError(
                "training checkpoint is not a valid ZIP archive"
            ) from error

        result = cls(
            config=manifest["config"],
            full_sequence_attention=manifest["full_sequence_attention"],
            activation_checkpointing=manifest["activation_checkpointing"],
            model_parameters=manifest["model_parameters"],
            model_values=_unpack_f32_values(
                payloads[MODEL_VALUES_NAME], MODEL_VALUES_NAME
            ),
            packed_model_state=payloads.get(PACKED_MODEL_STATE_NAME),
            lora_config=manifest["lora_config"],
            adapter_parameters=manifest["adapter_parameters"],
            adapter_values=(
                _unpack_f32_values(
                    payloads[ADAPTER_VALUES_NAME], ADAPTER_VALUES_NAME
                )
                if ADAPTER_VALUES_NAME in payloads
                else ()
            ),
            optimizer_scope=manifest["optimizer_scope"],
            optimizer_options=manifest["optimizer_options"],
            optimizer_step=manifest["optimizer_step"],
            beta1_power=manifest["beta1_power"],
            beta2_power=manifest["beta2_power"],
            first_moments=_unpack_f32_values(
                payloads[FIRST_MOMENTS_NAME], FIRST_MOMENTS_NAME
            ),
            second_moments=_unpack_f32_values(
                payloads[SECOND_MOMENTS_NAME], SECOND_MOMENTS_NAME
            ),
            global_step=manifest["global_step"],
            python_random_state=manifest["python_random_state"],
            batch_source_state=manifest["batch_source_state"],
            metadata=manifest["metadata"],
            format_version=manifest["format_version"],
        )
        if result.checkpoint_id != manifest["checkpoint_id"]:
            raise ValueError("training checkpoint ID does not match")
        return result

    def _member_bytes(self) -> tuple[tuple[str, bytes], ...]:
        members: list[tuple[str, bytes]] = [
            (MODEL_VALUES_NAME, _pack_f32_values(self._model_values)),
        ]
        if self._packed_model_state is not None:
            members.append(
                (PACKED_MODEL_STATE_NAME, self._packed_model_state)
            )
        if self._lora_config is not None:
            members.append(
                (
                    ADAPTER_VALUES_NAME,
                    _pack_f32_values(self._adapter_values),
                )
            )
        members.extend(
            (
                (
                    FIRST_MOMENTS_NAME,
                    _pack_f32_values(self._first_moments),
                ),
                (
                    SECOND_MOMENTS_NAME,
                    _pack_f32_values(self._second_moments),
                ),
            )
        )
        return tuple(members)

    def _manifest_bytes(self, *, include_checkpoint_id: bool) -> bytes:
        members = self._member_bytes()
        member_entries: dict[str, dict[str, object]] = {}
        for name, payload in members:
            if name == PACKED_MODEL_STATE_NAME:
                member_entries[name] = {
                    "dtype": "uint8",
                    "encoding": "riftco-packed-nf4-v1",
                    "count": len(payload),
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            else:
                member_entries[name] = {
                    "dtype": "float32",
                    "byte_order": "little",
                    "count": len(payload) // 4,
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
        manifest: dict[str, object] = {
            "format": FORMAT_NAME,
            "format_version": self._format_version,
            "boundary": _BOUNDARY,
            "global_step": self._global_step,
            "model": {
                "config": asdict(self._config),
                "full_sequence_attention": self._full_sequence_attention,
                "activation_checkpointing": self._activation_checkpointing,
                "quantized": self._packed_model_state is not None,
                "parameters": _parameter_entries(self._model_parameters),
                "values_file": MODEL_VALUES_NAME,
                "lora": _lora_entry(self._lora_config),
                "adapter_parameters": _parameter_entries(
                    self._adapter_parameters
                ),
                "adapter_values_file": (
                    ADAPTER_VALUES_NAME
                    if self._lora_config is not None
                    else None
                ),
            },
            "optimizer": {
                "type": "adam",
                "scope": self._optimizer_scope,
                "options": asdict(self._optimizer_options),
                "step_count": self._optimizer_step,
                "beta1_power": self._beta1_power,
                "beta2_power": self._beta2_power,
                "first_moments_file": FIRST_MOMENTS_NAME,
                "second_moments_file": SECOND_MOMENTS_NAME,
            },
            "python_random_state": _encode_random_state(
                self._python_random_state
            ),
            "batch_source": _batch_source_entry(
                self._batch_source_state
            ),
            "members": member_entries,
            "metadata": json.loads(self._metadata_json),
        }
        if self._format_version >= FORMAT_VERSION:
            model_entry = manifest["model"]
            assert isinstance(model_entry, dict)
            model_entry["packed_state_file"] = (
                PACKED_MODEL_STATE_NAME
                if self._packed_model_state is not None
                else None
            )
        if include_checkpoint_id:
            manifest["checkpoint_id"] = self._checkpoint_id
        return _canonical_json(manifest).encode("utf-8")


def _capture_batch_source_state(
    source: object | None,
) -> BatchSourceState | None:
    if source is None:
        return None
    capture = getattr(source, "checkpoint_state", None)
    restore = getattr(source, "restore_checkpoint_state", None)
    if not callable(capture) or not callable(restore):
        raise TypeError(
            "a supplied source must implement checkpoint_state() and "
            "restore_checkpoint_state(); pass source=None to explicitly "
            "omit data position"
        )
    state = capture()
    if not isinstance(state, BatchSourceState):
        raise TypeError(
            "batch source checkpoint_state() must return BatchSourceState"
        )
    return state


def _validate_batch_source(
    source: object | None,
    checkpoint_state: BatchSourceState | None,
) -> BatchSourceState | None:
    if checkpoint_state is None:
        return None
    if source is None:
        raise ValueError("checkpoint requires its batch source for restoration")
    capture = getattr(source, "checkpoint_state", None)
    restore = getattr(source, "restore_checkpoint_state", None)
    if not callable(capture) or not callable(restore):
        raise TypeError(
            "source must implement checkpoint_state() and "
            "restore_checkpoint_state()"
        )
    current = capture()
    if not isinstance(current, BatchSourceState):
        raise TypeError(
            "batch source checkpoint_state() must return BatchSourceState"
        )
    if current.source_type != checkpoint_state.source_type:
        raise ValueError("batch-source type does not match checkpoint")
    if current.fingerprint != checkpoint_state.fingerprint:
        raise ValueError("batch-source dataset fingerprint does not match")
    return current


def _specs_from_parameter_list(parameters: object) -> tuple[_ParameterSpec, ...]:
    return tuple(
        _ParameterSpec(name, shape)
        for name, shape in zip(parameters.names, parameters.shapes)
    )


def _parse_manifest(manifest_bytes: bytes) -> dict[str, object]:
    try:
        value = _strict_json_loads(
            manifest_bytes.decode("utf-8", errors="strict")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "training checkpoint manifest is not valid JSON"
        ) from error
    if not isinstance(value, dict):
        raise ValueError("training checkpoint manifest must be an object")
    if value.get("format") != FORMAT_NAME:
        raise ValueError("unknown training checkpoint format")
    format_version = value.get("format_version")
    if (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version not in {LEGACY_FORMAT_VERSION, FORMAT_VERSION}
    ):
        raise ValueError("unsupported training checkpoint format version")
    if value.get("boundary") != _BOUNDARY:
        raise ValueError("unsupported training checkpoint boundary")
    checkpoint_id = value.get("checkpoint_id")
    _sha256(checkpoint_id, "checkpoint_id")

    model = _mapping(value, "model")
    quantized = model.get("quantized")
    if not isinstance(quantized, bool):
        raise ValueError("checkpoint model quantized flag must be boolean")
    if format_version == LEGACY_FORMAT_VERSION and quantized:
        raise ValueError("checkpoint v1 requires an FP32 base model")
    expected_packed_file = PACKED_MODEL_STATE_NAME if quantized else None
    if format_version >= FORMAT_VERSION and (
        model.get("packed_state_file") != expected_packed_file
    ):
        raise ValueError("packed model state member does not match NF4 state")
    config_value = _mapping(model, "config")
    config = TransformerConfig(
        vocabulary_size=_integer(config_value, "vocabulary_size"),
        maximum_context=_integer(config_value, "maximum_context"),
        model_width=_integer(config_value, "model_width"),
        head_count=_integer(config_value, "head_count"),
        block_count=_integer(config_value, "block_count"),
        feed_forward_width=_integer(
            config_value, "feed_forward_width"
        ),
        random_seed=_integer(config_value, "random_seed"),
        layer_norm_epsilon=_number(
            config_value, "layer_norm_epsilon"
        ),
    )
    _validate_config(config)
    full_sequence_attention = model.get("full_sequence_attention")
    if full_sequence_attention not in {"materialized", "flash"}:
        raise ValueError("unsupported full-sequence attention policy")
    activation_checkpointing = model.get("activation_checkpointing")
    if activation_checkpointing not in {"disabled", "block"}:
        raise ValueError("unsupported activation-checkpointing policy")
    model_parameters = _parse_parameter_entries(
        model.get("parameters"), "model parameters"
    )
    if model.get("values_file") != MODEL_VALUES_NAME:
        raise ValueError("unsupported model values member")
    lora_config = _parse_lora_entry(model.get("lora"))
    adapter_parameters = _parse_parameter_entries(
        model.get("adapter_parameters"),
        "adapter parameters",
        allow_empty=True,
    )
    expected_adapter_file = (
        ADAPTER_VALUES_NAME if lora_config is not None else None
    )
    if model.get("adapter_values_file") != expected_adapter_file:
        raise ValueError("adapter values member does not match LoRA state")

    optimizer = _mapping(value, "optimizer")
    if optimizer.get("type") != "adam":
        raise ValueError("unsupported checkpoint optimizer")
    optimizer_scope = optimizer.get("scope")
    if optimizer_scope not in {"model", "adapter"}:
        raise ValueError("unsupported optimizer parameter scope")
    options_value = _mapping(optimizer, "options")
    state_storage = options_value.get("state_storage")
    if state_storage not in {"contiguous", "paged"}:
        raise ValueError("unknown Adam state storage")
    optimizer_options = AdamOptions(
        learning_rate=_number(options_value, "learning_rate"),
        beta1=_number(options_value, "beta1"),
        beta2=_number(options_value, "beta2"),
        epsilon=_number(options_value, "epsilon"),
        maximum_gradient_norm=_number(
            options_value, "maximum_gradient_norm"
        ),
        state_storage=state_storage,
        page_size=_integer(options_value, "page_size"),
    )
    _validate_optimizer_options(optimizer_options)
    if optimizer.get("first_moments_file") != FIRST_MOMENTS_NAME or (
        optimizer.get("second_moments_file") != SECOND_MOMENTS_NAME
    ):
        raise ValueError("unsupported Adam moment members")

    members_value = _mapping(value, "members")
    expected_member_names = [MODEL_VALUES_NAME]
    if quantized:
        expected_member_names.append(PACKED_MODEL_STATE_NAME)
    if lora_config is not None:
        expected_member_names.append(ADAPTER_VALUES_NAME)
    expected_member_names.extend(
        [FIRST_MOMENTS_NAME, SECOND_MOMENTS_NAME]
    )
    if set(members_value) != set(expected_member_names) or (
        len(members_value) != len(expected_member_names)
    ):
        raise ValueError("checkpoint member manifest has wrong members/order")
    members: dict[str, dict[str, object]] = {}
    expected_counts = {
        MODEL_VALUES_NAME: sum(
            parameter.value_count for parameter in model_parameters
        ),
        FIRST_MOMENTS_NAME: sum(
            parameter.value_count
            for parameter in (
                model_parameters
                if optimizer_scope == "model"
                else adapter_parameters
            )
        ),
        SECOND_MOMENTS_NAME: sum(
            parameter.value_count
            for parameter in (
                model_parameters
                if optimizer_scope == "model"
                else adapter_parameters
            )
        ),
    }
    if lora_config is not None:
        expected_counts[ADAPTER_VALUES_NAME] = sum(
            parameter.value_count for parameter in adapter_parameters
        )
    for name in expected_member_names:
        entry = _mapping(members_value, name)
        count = _integer(entry, "count")
        size = _integer(entry, "size")
        if name == PACKED_MODEL_STATE_NAME:
            if (
                entry.get("dtype") != "uint8"
                or entry.get("encoding") != "riftco-packed-nf4-v1"
                or count <= 0
                or size != count
            ):
                raise ValueError(
                    "unsupported encoding for packed NF4 model state"
                )
        else:
            if (
                entry.get("dtype") != "float32"
                or entry.get("byte_order") != "little"
            ):
                raise ValueError(f"unsupported encoding for {name}")
            if count != expected_counts[name] or size != count * 4:
                raise ValueError(
                    f"checkpoint member {name} count is invalid"
                )
        digest = entry.get("sha256")
        _sha256(digest, f"members.{name}.sha256")
        members[name] = {"size": size, "sha256": digest}

    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    return {
        "format_version": format_version,
        "checkpoint_id": checkpoint_id,
        "config": config,
        "full_sequence_attention": full_sequence_attention,
        "activation_checkpointing": activation_checkpointing,
        "model_parameters": model_parameters,
        "lora_config": lora_config,
        "adapter_parameters": adapter_parameters,
        "optimizer_scope": optimizer_scope,
        "optimizer_options": optimizer_options,
        "optimizer_step": _integer(optimizer, "step_count"),
        "beta1_power": _number(optimizer, "beta1_power"),
        "beta2_power": _number(optimizer, "beta2_power"),
        "global_step": _integer(value, "global_step"),
        "python_random_state": _decode_random_state(
            value.get("python_random_state"), "python_random_state"
        ),
        "batch_source_state": _parse_batch_source_entry(
            value.get("batch_source")
        ),
        "members": members,
        "member_names": tuple(expected_member_names),
        "metadata": metadata,
    }


def _parameter_specs(
    values: Iterable[_ParameterSpec],
    description: str,
    *,
    allow_empty: bool = False,
) -> tuple[_ParameterSpec, ...]:
    try:
        result = tuple(values)
    except TypeError as error:
        raise TypeError(f"{description} parameters must be iterable") from error
    if not result and not allow_empty:
        raise ValueError(f"{description} parameters must not be empty")
    if any(not isinstance(value, _ParameterSpec) for value in result):
        raise TypeError(
            f"{description} parameters must contain parameter specs"
        )
    names = tuple(value.name for value in result)
    if len(set(names)) != len(names):
        raise ValueError(f"{description} parameter names must be unique")
    return result


def _parameter_entries(
    parameters: tuple[_ParameterSpec, ...],
) -> list[dict[str, object]]:
    offset = 0
    entries: list[dict[str, object]] = []
    for parameter in parameters:
        entries.append(
            {
                "name": parameter.name,
                "shape": list(parameter.shape),
                "offset": offset,
                "count": parameter.value_count,
            }
        )
        offset += parameter.value_count
    return entries


def _parse_parameter_entries(
    value: object,
    description: str,
    *,
    allow_empty: bool = False,
) -> tuple[_ParameterSpec, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{description} must be an array")
    if not value and not allow_empty:
        raise ValueError(f"{description} must not be empty")
    result: list[_ParameterSpec] = []
    offset = 0
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(f"each {description} entry must be an object")
        name = entry.get("name")
        shape_value = entry.get("shape")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{description} name must be nonempty")
        if not isinstance(shape_value, list):
            raise ValueError(f"{description} shape must be an array")
        shape = tuple(
            _manifest_integer(item, f"{description} dimension")
            for item in shape_value
        )
        parameter = _ParameterSpec(name, shape)
        if (
            _integer(entry, "offset") != offset
            or _integer(entry, "count") != parameter.value_count
        ):
            raise ValueError(
                f"{description} offsets/counts do not match shapes"
            )
        result.append(parameter)
        offset += parameter.value_count
    return _parameter_specs(result, description, allow_empty=allow_empty)


def _require_value_count(
    parameters: tuple[_ParameterSpec, ...],
    values: tuple[float, ...],
    description: str,
) -> None:
    expected = sum(parameter.value_count for parameter in parameters)
    if len(values) != expected:
        raise ValueError(
            f"{description} values do not match parameter shapes"
        )


def _canonical_f32_values(
    values: Iterable[float], description: str
) -> tuple[tuple[float, ...], bytes]:
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{description} must be iterable") from error
    canonical: list[float] = []
    payload = bytearray()
    for index, value in enumerate(raw_values):
        if isinstance(value, bool):
            raise TypeError(f"{description}[{index}] must be a number")
        try:
            converted = float(value)
            encoded = struct.pack("<f", converted)
        except (TypeError, ValueError, OverflowError, struct.error) as error:
            raise ValueError(
                f"{description}[{index}] is outside float32 range"
            ) from error
        rounded = struct.unpack("<f", encoded)[0]
        if not math.isfinite(rounded):
            raise ValueError(f"{description}[{index}] must be finite")
        canonical.append(rounded)
        payload.extend(encoded)
    return tuple(canonical), bytes(payload)


def _pack_f32_values(values: tuple[float, ...]) -> bytes:
    return b"".join(struct.pack("<f", value) for value in values)


def _unpack_f32_values(payload: bytes, description: str) -> tuple[float, ...]:
    if len(payload) % 4 != 0:
        raise ValueError(f"{description} is not float32 aligned")
    values = tuple(value[0] for value in struct.iter_unpack("<f", payload))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{description} contains non-finite values")
    return values


def _lora_entry(config: LoraConfig | None) -> dict[str, object] | None:
    if config is None:
        return None
    return {
        "rank": config.rank,
        "alpha": config.alpha,
        "targets": list(config.targets),
        "random_seed": config.random_seed,
    }


def _parse_lora_entry(value: object) -> LoraConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("model.lora must be an object or null")
    targets = value.get("targets")
    if not isinstance(targets, list) or not all(
        isinstance(target, str) for target in targets
    ):
        raise ValueError("model.lora.targets must be a string array")
    return LoraConfig(
        rank=_integer(value, "rank"),
        alpha=_number(value, "alpha"),
        targets=tuple(targets),
        random_seed=_integer(value, "random_seed"),
    )


def _encode_random_state(state: tuple[object, ...]) -> dict[str, object]:
    checked = _checked_random_state(state, "random state")
    return {
        "version": checked[0],
        "internal": list(checked[1]),
        "gauss_next": checked[2],
    }


def _decode_random_state(value: object, description: str) -> tuple[object, ...]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    internal = value.get("internal")
    if not isinstance(internal, list):
        raise ValueError(f"{description}.internal must be an array")
    state: tuple[object, ...] = (
        _integer(value, "version"),
        tuple(
            _manifest_integer(item, f"{description}.internal")
            for item in internal
        ),
        value.get("gauss_next"),
    )
    return _checked_random_state(state, description)


def _checked_random_state(
    state: object, description: str
) -> tuple[object, ...]:
    if not isinstance(state, tuple) or len(state) != 3:
        raise ValueError(f"{description} is not a Python RNG state")
    gauss_next = state[2]
    if gauss_next is not None and (
        isinstance(gauss_next, bool)
        or not isinstance(gauss_next, (int, float))
        or not math.isfinite(float(gauss_next))
    ):
        raise ValueError(
            f"{description} has an invalid cached Gaussian value"
        )
    candidate = random.Random()
    try:
        candidate.setstate(state)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{description} is not a valid Python RNG state"
        ) from error
    return state


def _batch_source_entry(
    state: BatchSourceState | None,
) -> dict[str, object] | None:
    if state is None:
        return None
    return {
        "source_type": state.source_type,
        "fingerprint": state.fingerprint,
        "batches_emitted": state.batches_emitted,
        "random_state": _encode_random_state(state.random_state),
    }


def _parse_batch_source_entry(value: object) -> BatchSourceState | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("batch_source must be an object or null")
    source_type = value.get("source_type")
    fingerprint = value.get("fingerprint")
    if not isinstance(source_type, str):
        raise ValueError("batch_source.source_type must be a string")
    if not isinstance(fingerprint, str):
        raise ValueError("batch_source.fingerprint must be a string")
    return BatchSourceState(
        source_type=source_type,
        fingerprint=fingerprint,
        batches_emitted=_integer(value, "batches_emitted"),
        random_state=_decode_random_state(
            value.get("random_state"), "batch_source.random_state"
        ),
    )


def _runtime_configs_compatible(
    actual: TransformerConfig,
    expected: TransformerConfig,
) -> bool:
    """Compare execution semantics while ignoring initializer provenance."""

    return (
        actual.vocabulary_size == expected.vocabulary_size
        and actual.maximum_context == expected.maximum_context
        and actual.model_width == expected.model_width
        and actual.head_count == expected.head_count
        and actual.block_count == expected.block_count
        and actual.feed_forward_width == expected.feed_forward_width
        and actual.layer_norm_epsilon == expected.layer_norm_epsilon
    )


def _lora_configs_compatible(
    actual: LoraConfig | None,
    expected: LoraConfig | None,
) -> bool:
    if actual is None or expected is None:
        return actual is expected
    return (
        actual.rank == expected.rank
        and actual.alpha == expected.alpha
        and actual.targets == expected.targets
    )


def _validate_config(config: TransformerConfig) -> None:
    for name in (
        "vocabulary_size",
        "maximum_context",
        "model_width",
        "head_count",
        "block_count",
        "feed_forward_width",
    ):
        _positive_integer(getattr(config, name), name)
    _nonnegative_integer(config.random_seed, "random_seed")
    positive_float32(config.layer_norm_epsilon, "layer_norm_epsilon")
    if config.model_width % config.head_count != 0:
        raise ValueError("model_width must be divisible by head_count")


def _validate_optimizer_options(options: AdamOptions) -> None:
    for name in ("learning_rate", "epsilon", "maximum_gradient_norm"):
        value = float(getattr(options, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Adam {name} must be finite and positive")
    for name in ("beta1", "beta2"):
        value = float(getattr(options, name))
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"Adam {name} must be finite and in (0, 1)")
    if options.state_storage not in {"contiguous", "paged"}:
        raise ValueError("Adam state_storage is invalid")
    _positive_integer(options.page_size, "Adam page_size")


def _validate_semantic_options(
    actual: AdamOptions, expected: AdamOptions
) -> None:
    # State storage/page size are physical and may change across backends.
    for name in (
        "learning_rate",
        "beta1",
        "beta2",
        "epsilon",
        "maximum_gradient_norm",
    ):
        if getattr(actual, name) != getattr(expected, name):
            raise ValueError(f"Adam option {name} does not match checkpoint")


def _beta_power(
    value: object,
    beta: float,
    step: int,
    description: str,
) -> float:
    number = _finite_number(value, description)
    expected = math.pow(beta, step)
    if expected <= sys_float_min():
        if number < 0.0 or number > sys_float_min():
            raise ValueError(f"{description} does not match beta**step")
        return number
    tolerance = max(abs(expected) * 1.0e-12, 5.0e-324)
    if not math.isclose(number, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"{description} does not match beta**step")
    return number


def sys_float_min() -> float:
    # Smallest positive normal IEEE-754 binary64 value without importing sys.
    return float.fromhex("0x1.0p-1022")


def _finite_number(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _mapping(mapping: Mapping[str, object], name: str) -> dict[str, object]:
    value = mapping.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(mapping: Mapping[str, object], name: str) -> int:
    if name not in mapping:
        raise ValueError(f"missing required integer {name}")
    return _manifest_integer(mapping[name], name)


def _number(mapping: Mapping[str, object], name: str) -> float:
    if name not in mapping:
        raise ValueError(f"missing required number {name}")
    return _finite_number(mapping[name], name)


def _manifest_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer")
    return value


def _positive_integer(value: object, description: str) -> int:
    result = _nonnegative_integer(value, description)
    if result == 0:
        raise ValueError(f"{description} must be greater than zero")
    return result


def _nonnegative_integer(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{description} must be an int")
    if value < 0:
        raise ValueError(f"{description} must not be negative")
    return value


def _sha256(value: object, description: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")


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


def _strict_json_loads(text: str) -> object:
    def reject_pairs(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON number: {value}")

    return json.loads(
        text,
        object_pairs_hook=reject_pairs,
        parse_constant=reject_constant,
    )


def _deterministic_zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 0
    info.external_attr = 0
    return info


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "TrainingCheckpoint",
    "TrainingCheckpointRestore",
]
