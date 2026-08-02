"""Small, dependency-free Python binding for Riftco Transformer's C ABI."""

from __future__ import annotations

import ctypes
import ctypes.util
import math
import operator
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


ABI_VERSION_MAJOR = 2
ABI_VERSION_MINOR = 5
ABI_VERSION = (ABI_VERSION_MAJOR << 16) | ABI_VERSION_MINOR

STATUS_OK = 0
STATUS_INVALID_ARGUMENT = 1
STATUS_OUT_OF_RANGE = 2
STATUS_OVERFLOW = 3
STATUS_BACKEND_UNAVAILABLE = 4
STATUS_OUT_OF_MEMORY = 5
STATUS_RUNTIME_ERROR = 6
STATUS_UNKNOWN_ERROR = 255

BACKEND_CPU = 0
BACKEND_METAL = 1
BACKEND_CUDA = 2
BACKEND_TPU = 3

FULL_SEQUENCE_ATTENTION_MATERIALIZED = 0
FULL_SEQUENCE_ATTENTION_FLASH = 1

ACTIVATION_CHECKPOINTING_DISABLED = 0
ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK = 1

TOKENIZER_METHOD_BYTE = 0
TOKENIZER_METHOD_BPE = 1

KV_CACHE_CONTIGUOUS = 0
KV_CACHE_PAGED = 1

ADAM_STATE_CONTIGUOUS = 0
ADAM_STATE_PAGED = 1

PROGRAM_INPUT_WHOLE_SOURCE = 0
PROGRAM_INPUT_SOURCE_POSITION = 1

LOWERING_STRATEGY_AUTO = 0
LOWERING_STRATEGY_DENSE = 1
LOWERING_STRATEGY_LINEAR = 2
LOWERING_STRATEGY_LINEAR_ATTENTION = 3

COEFFICIENT_REQUIRE_EXACT_F32 = 0
COEFFICIENT_ALLOW_ROUNDED_F32 = 1
COEFFICIENT_INITIALIZATION_COMPILED = 0
COEFFICIENT_INITIALIZATION_RANDOM_UNIFORM = 1

LORA_TARGET_ATTENTION_QUERY = 1 << 0
LORA_TARGET_ATTENTION_KEY = 1 << 1
LORA_TARGET_ATTENTION_VALUE = 1 << 2
LORA_TARGET_ATTENTION_OUTPUT = 1 << 3
LORA_TARGET_FF_EXPAND = 1 << 4
LORA_TARGET_FF_PROJECT = 1 << 5
LORA_TARGET_LM_HEAD = 1 << 6
LORA_TARGET_DEFAULT = (
    LORA_TARGET_ATTENTION_QUERY | LORA_TARGET_ATTENTION_VALUE
)
LORA_TARGET_ALL_LINEAR = (1 << 7) - 1

_BACKEND_CODES = {
    "cpu": BACKEND_CPU,
    "metal": BACKEND_METAL,
    "cuda": BACKEND_CUDA,
    "tpu": BACKEND_TPU,
}
_BACKEND_NAMES = {
    value: key for key, value in _BACKEND_CODES.items()
}
_FULL_SEQUENCE_ATTENTION_CODES = {
    "materialized": FULL_SEQUENCE_ATTENTION_MATERIALIZED,
    "flash": FULL_SEQUENCE_ATTENTION_FLASH,
}
_FULL_SEQUENCE_ATTENTION_NAMES = {
    value: key for key, value in _FULL_SEQUENCE_ATTENTION_CODES.items()
}
_ACTIVATION_CHECKPOINTING_CODES = {
    "disabled": ACTIVATION_CHECKPOINTING_DISABLED,
    "block": ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK,
}
_ACTIVATION_CHECKPOINTING_NAMES = {
    value: key for key, value in _ACTIVATION_CHECKPOINTING_CODES.items()
}
_TOKENIZER_METHOD_CODES = {
    "byte": TOKENIZER_METHOD_BYTE,
    "bpe": TOKENIZER_METHOD_BPE,
}
_TOKENIZER_METHOD_NAMES = {
    value: key for key, value in _TOKENIZER_METHOD_CODES.items()
}
_KV_CACHE_CODES = {
    "contiguous": KV_CACHE_CONTIGUOUS,
    "paged": KV_CACHE_PAGED,
}
_KV_CACHE_NAMES = {
    value: key for key, value in _KV_CACHE_CODES.items()
}
_ADAM_STATE_CODES = {
    "contiguous": ADAM_STATE_CONTIGUOUS,
    "paged": ADAM_STATE_PAGED,
}
_ADAM_STATE_NAMES = {
    value: key for key, value in _ADAM_STATE_CODES.items()
}
_PROGRAM_INPUT_CODES = {
    "whole_source": PROGRAM_INPUT_WHOLE_SOURCE,
    "source_position": PROGRAM_INPUT_SOURCE_POSITION,
}
_LOWERING_STRATEGY_CODES = {
    "auto": LOWERING_STRATEGY_AUTO,
    "dense": LOWERING_STRATEGY_DENSE,
    "linear": LOWERING_STRATEGY_LINEAR,
    "linear_attention": LOWERING_STRATEGY_LINEAR_ATTENTION,
}
_COEFFICIENT_PRECISION_CODES = {
    "exact": COEFFICIENT_REQUIRE_EXACT_F32,
    "rounded": COEFFICIENT_ALLOW_ROUNDED_F32,
}
_COEFFICIENT_INITIALIZATION_CODES = {
    "compiled": COEFFICIENT_INITIALIZATION_COMPILED,
    "random_uniform": COEFFICIENT_INITIALIZATION_RANDOM_UNIFORM,
}
_LORA_TARGET_CODES = {
    "attention.query": LORA_TARGET_ATTENTION_QUERY,
    "attention.key": LORA_TARGET_ATTENTION_KEY,
    "attention.value": LORA_TARGET_ATTENTION_VALUE,
    "attention.output": LORA_TARGET_ATTENTION_OUTPUT,
    "feed_forward.expand": LORA_TARGET_FF_EXPAND,
    "feed_forward.project": LORA_TARGET_FF_PROJECT,
    "language_model_head": LORA_TARGET_LM_HEAD,
}
_LORA_TARGET_NAMES = {
    value: key for key, value in _LORA_TARGET_CODES.items()
}
LORA_TARGET_NAMES = tuple(_LORA_TARGET_CODES)


class RiftcoTransformerError(RuntimeError):
    """An error reported by the Riftco Transformer native library."""

    def __init__(self, status: int, status_name: str, detail: str) -> None:
        self.status = status
        self.status_name = status_name
        self.detail = detail
        message = status_name
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class _NativeContext(ctypes.Structure):
    pass


class _NativeTokenizer(ctypes.Structure):
    pass


class _NativeTensor(ctypes.Structure):
    pass


class _NativeModel(ctypes.Structure):
    pass


class _NativeDecodeSession(ctypes.Structure):
    pass


class _NativeParameterList(ctypes.Structure):
    pass


class _NativeVariable(ctypes.Structure):
    pass


class _NativeAdam(ctypes.Structure):
    pass


class _NativeMultilinearMap(ctypes.Structure):
    pass


class _NativeProgramAugmentedModel(ctypes.Structure):
    pass


class _NativeRepresentationTrace(ctypes.Structure):
    pass


class _NativeTransformerConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("vocabulary_size", ctypes.c_uint64),
        ("maximum_context", ctypes.c_uint64),
        ("model_width", ctypes.c_uint64),
        ("head_count", ctypes.c_uint64),
        ("block_count", ctypes.c_uint64),
        ("feed_forward_width", ctypes.c_uint64),
        ("random_seed", ctypes.c_uint32),
        ("layer_norm_epsilon", ctypes.c_float),
    ]


class _NativeLoraConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("rank", ctypes.c_uint64),
        ("alpha", ctypes.c_float),
        ("random_seed", ctypes.c_uint32),
        ("targets", ctypes.c_uint64),
        ("reserved", ctypes.c_uint64),
    ]


class _NativeQuantizedMemoryStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("weight_count", ctypes.c_uint64),
        ("packed_code_bytes", ctypes.c_uint64),
        ("scale_bytes", ctypes.c_uint64),
        ("logical_payload_bytes", ctypes.c_uint64),
        ("resident_payload_bytes", ctypes.c_uint64),
        ("fp32_equivalent_bytes", ctypes.c_uint64),
        ("fp32_scale_bytes", ctypes.c_uint64),
        ("scale_code_bytes", ctypes.c_uint64),
        ("second_level_scale_bytes", ctypes.c_uint64),
        ("scale_offset_bytes", ctypes.c_uint64),
        ("double_quantized_weight_count", ctypes.c_uint64),
    ]


class _NativeDecodeSessionOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("kind", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("block_size", ctypes.c_uint64),
    ]


class _NativeAdamOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("learning_rate", ctypes.c_float),
        ("beta1", ctypes.c_float),
        ("beta2", ctypes.c_float),
        ("epsilon", ctypes.c_float),
        ("maximum_gradient_norm", ctypes.c_float),
        ("reserved", ctypes.c_uint32),
        ("state_storage", ctypes.c_int32),
        ("reserved2", ctypes.c_uint32),
        ("page_size", ctypes.c_uint64),
    ]


class _NativeTokenizerOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("method", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("vocabulary_size", ctypes.c_uint64),
        ("minimum_pair_frequency", ctypes.c_uint64),
    ]


class _NativeBpeMergeRule(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_uint32),
        ("right", ctypes.c_uint32),
        ("result", ctypes.c_uint32),
    ]


class _NativeAdamStepStats(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("step", ctypes.c_uint64),
        ("gradient_norm", ctypes.c_double),
        ("clip_scale", ctypes.c_double),
    ]


class _NativeProgramAugmentedModelConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("vocabulary_size", ctypes.c_uint64),
        ("context_length", ctypes.c_uint64),
        ("model_width", ctypes.c_uint64),
        ("head_count", ctypes.c_uint64),
        ("attention_branch_count", ctypes.c_uint64),
        ("feed_forward_width", ctypes.c_uint64),
        ("random_seed", ctypes.c_uint32),
        ("attention_kind", ctypes.c_int32),
    ]


class _NativeProgramInputLayout(ctypes.Structure):
    _fields_ = [
        ("source", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
        ("position", ctypes.c_uint64),
        ("projection_group", ctypes.c_uint64),
    ]


class _NativeProgramBranchConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("source_offset", ctypes.c_uint64),
        ("source_length", ctypes.c_uint64),
        ("target_offset", ctypes.c_uint64),
        ("output_length", ctypes.c_uint64),
        ("inputs", ctypes.POINTER(_NativeProgramInputLayout)),
        ("input_count", ctypes.c_uint64),
        ("input_projection_bias", ctypes.c_int32),
        ("merge_bias", ctypes.c_int32),
    ]


class _NativeNeuralLoweringOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("strategy", ctypes.c_int32),
        ("precision", ctypes.c_int32),
        ("initialization", ctypes.c_int32),
        ("trainable", ctypes.c_int32),
        ("random_seed", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("random_scale", ctypes.c_float),
        ("reserved2", ctypes.c_uint32),
        ("max_coefficient_elements", ctypes.c_uint64),
        ("attention_query_axis", ctypes.c_int64),
    ]


class _NativeProgramInputSteering(ctypes.Structure):
    _fields_ = [
        ("input_index", ctypes.c_uint64),
        ("position", ctypes.c_uint64),
        ("scales", ctypes.POINTER(ctypes.c_float)),
        ("scale_count", ctypes.c_uint64),
        ("offsets", ctypes.POINTER(ctypes.c_float)),
        ("offset_count", ctypes.c_uint64),
    ]


class _NativeProgramAugmentedForwardOptions(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint64),
        ("capture_representations", ctypes.c_int32),
        ("batch_roll_learned_attention", ctypes.c_int32),
        ("batch_roll_shift", ctypes.c_uint64),
        ("steering", ctypes.POINTER(_NativeProgramInputSteering)),
        ("steering_count", ctypes.c_uint64),
        ("ablated_program_inputs", ctypes.POINTER(ctypes.c_uint64)),
        ("ablated_program_input_count", ctypes.c_uint64),
        ("ablate_program_output", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


_ContextHandle = ctypes.POINTER(_NativeContext)
_TokenizerHandle = ctypes.POINTER(_NativeTokenizer)
_TensorHandle = ctypes.POINTER(_NativeTensor)
_ModelHandle = ctypes.POINTER(_NativeModel)
_DecodeSessionHandle = ctypes.POINTER(_NativeDecodeSession)
_ParameterListHandle = ctypes.POINTER(_NativeParameterList)
_VariableHandle = ctypes.POINTER(_NativeVariable)
_AdamHandle = ctypes.POINTER(_NativeAdam)
_MultilinearMapHandle = ctypes.POINTER(_NativeMultilinearMap)
_ProgramAugmentedModelHandle = ctypes.POINTER(_NativeProgramAugmentedModel)
_RepresentationTraceHandle = ctypes.POINTER(_NativeRepresentationTrace)

_library: ctypes.CDLL | None = None
_library_lock = threading.Lock()


def _library_filenames(postfix: str = "") -> tuple[str, ...]:
    if sys.platform == "darwin":
        return (f"libriftco_transformer_c{postfix}.dylib",)
    if os.name == "nt":
        return (
            f"riftco_transformer_c{postfix}.dll",
            f"libriftco_transformer_c{postfix}.dll",
        )
    return (f"libriftco_transformer_c{postfix}.so",)


def _source_project_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[3]
    if (
        (candidate / "CMakeLists.txt").is_file()
        and (candidate / "include" / "riftco_transformer").is_dir()
    ):
        return candidate
    return None


def _sanitizers_enabled_for_build(directory: Path) -> bool:
    for candidate in (directory, directory.parent):
        cache = candidate / "CMakeCache.txt"
        if not cache.is_file():
            continue
        try:
            contents = cache.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        return (
            "RIFTCO_TRANSFORMER_ENABLE_SANITIZERS:BOOL=ON" in contents
            or "-fsanitize=address" in contents
        )
    return False


def _candidate_libraries() -> list[str]:
    override = os.environ.get("RIFTCO_TRANSFORMER_LIBRARY")
    if override:
        return [str(Path(override).expanduser())]

    root = _source_project_root()
    package_directory = Path(__file__).resolve().parents[1]
    release_names = _library_filenames()
    directory_candidates = [
        (package_directory / ".libs", release_names, False),
        (package_directory, release_names, False),
    ]
    if root is not None:
        debug_names = (
            *_library_filenames("_debug"),
            *release_names,
        )
        relwithdebinfo_names = (
            *_library_filenames("_relwithdebinfo"),
            *release_names,
        )
        minsizerel_names = (
            *_library_filenames("_minsizerel"),
            *release_names,
        )
        directory_candidates.extend(
            [
                (root / "build" / "release", release_names, True),
                (root / "build" / "Release", release_names, True),
                (root / "cmake-build-release", release_names, True),
                (root / "build", release_names, True),
                (
                    root / "build" / "RelWithDebInfo",
                    relwithdebinfo_names,
                    True,
                ),
                (
                    root / "build" / "MinSizeRel",
                    minsizerel_names,
                    True,
                ),
                (root / "build" / "debug", debug_names, True),
                (root / "build" / "Debug", debug_names, True),
                (root / "cmake-build-debug", debug_names, True),
            ]
        )

    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: Path | str) -> None:
        value = str(path)
        if value not in seen:
            seen.add(value)
            candidates.append(value)

    for directory, names, is_build_directory in directory_candidates:
        if (
            is_build_directory
            and _sanitizers_enabled_for_build(directory)
        ):
            continue
        for name in names:
            path = directory / name
            if path.is_file():
                add(path)

    system_library = ctypes.util.find_library("riftco_transformer_c")
    if system_library:
        add(system_library)
    return candidates


def _abi_version_is_compatible(actual_version: int) -> bool:
    actual_major = actual_version >> 16
    actual_minor = actual_version & 0xFFFF
    return (
        actual_major == ABI_VERSION_MAJOR
        and actual_minor >= ABI_VERSION_MINOR
    )


def _configure_library(library: ctypes.CDLL) -> None:
    library.rt_abi_version.argtypes = []
    library.rt_abi_version.restype = ctypes.c_uint32
    actual_version = int(library.rt_abi_version())
    if not _abi_version_is_compatible(actual_version):
        raise RuntimeError(
            "riftco_transformer C ABI mismatch: "
            f"Python requires {ABI_VERSION_MAJOR}."
            f"{ABI_VERSION_MINOR} or a newer compatible minor, "
            f"library provides 0x{actual_version:08x}"
        )

    library.rt_status_string.argtypes = [ctypes.c_int32]
    library.rt_status_string.restype = ctypes.c_char_p
    library.rt_last_error.argtypes = []
    library.rt_last_error.restype = ctypes.c_char_p

    library.rt_tokenizer_create.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(_TokenizerHandle),
    ]
    library.rt_tokenizer_create.restype = ctypes.c_int32
    library.rt_tokenizer_options_init.argtypes = [
        ctypes.POINTER(_NativeTokenizerOptions),
        ctypes.c_uint64,
    ]
    library.rt_tokenizer_options_init.restype = ctypes.c_int32
    library.rt_tokenizer_create_with_options.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(_NativeTokenizerOptions),
        ctypes.POINTER(_TokenizerHandle),
    ]
    library.rt_tokenizer_create_with_options.restype = ctypes.c_int32
    library.rt_tokenizer_create_from_byte_vocabulary.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(_TokenizerHandle),
    ]
    library.rt_tokenizer_create_from_byte_vocabulary.restype = (
        ctypes.c_int32
    )
    library.rt_tokenizer_create_from_bpe_merges.argtypes = [
        ctypes.POINTER(_NativeBpeMergeRule),
        ctypes.c_uint64,
        ctypes.POINTER(_TokenizerHandle),
    ]
    library.rt_tokenizer_create_from_bpe_merges.restype = ctypes.c_int32
    library.rt_tokenizer_release.argtypes = [_TokenizerHandle]
    library.rt_tokenizer_release.restype = None
    library.rt_tokenizer_get_method.argtypes = [
        _TokenizerHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_tokenizer_get_method.restype = ctypes.c_int32
    library.rt_tokenizer_vocabulary_size.argtypes = [
        _TokenizerHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_tokenizer_vocabulary_size.restype = ctypes.c_int32
    library.rt_tokenizer_bpe_merge_count.argtypes = [
        _TokenizerHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_tokenizer_bpe_merge_count.restype = ctypes.c_int32
    library.rt_tokenizer_bpe_merge_rule.argtypes = [
        _TokenizerHandle,
        ctypes.c_uint64,
        ctypes.POINTER(_NativeBpeMergeRule),
    ]
    library.rt_tokenizer_bpe_merge_rule.restype = ctypes.c_int32
    library.rt_tokenizer_vocabulary.argtypes = [
        _TokenizerHandle,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_tokenizer_vocabulary.restype = ctypes.c_int32
    library.rt_tokenizer_token_bytes.argtypes = [
        _TokenizerHandle,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_tokenizer_token_bytes.restype = ctypes.c_int32
    library.rt_tokenizer_encode.argtypes = [
        _TokenizerHandle,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_tokenizer_encode.restype = ctypes.c_int32
    library.rt_tokenizer_decode.argtypes = [
        _TokenizerHandle,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_tokenizer_decode.restype = ctypes.c_int32

    library.rt_backend_is_available.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_backend_is_available.restype = ctypes.c_int32

    library.rt_context_create.argtypes = [
        ctypes.c_int32,
        ctypes.POINTER(_ContextHandle),
    ]
    library.rt_context_create.restype = ctypes.c_int32
    library.rt_context_release.argtypes = [_ContextHandle]
    library.rt_context_release.restype = None
    library.rt_context_backend.argtypes = [
        _ContextHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_context_backend.restype = ctypes.c_int32

    library.rt_tensor_create_f32.argtypes = [
        _ContextHandle,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.POINTER(_TensorHandle),
    ]
    library.rt_tensor_create_f32.restype = ctypes.c_int32
    library.rt_tensor_zeros_f32.argtypes = [
        _ContextHandle,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.POINTER(_TensorHandle),
    ]
    library.rt_tensor_zeros_f32.restype = ctypes.c_int32
    library.rt_tensor_release.argtypes = [_TensorHandle]
    library.rt_tensor_release.restype = None
    library.rt_tensor_backend.argtypes = [
        _TensorHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_tensor_backend.restype = ctypes.c_int32
    library.rt_tensor_rank.argtypes = [
        _TensorHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_tensor_rank.restype = ctypes.c_int32
    library.rt_tensor_shape.argtypes = [
        _TensorHandle,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
    ]
    library.rt_tensor_shape.restype = ctypes.c_int32
    library.rt_tensor_numel.argtypes = [
        _TensorHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_tensor_numel.restype = ctypes.c_int32
    library.rt_tensor_copy_to_host_f32.argtypes = [
        _TensorHandle,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    library.rt_tensor_copy_to_host_f32.restype = ctypes.c_int32
    library.rt_tensor_matmul.argtypes = [
        _TensorHandle,
        _TensorHandle,
        ctypes.POINTER(_TensorHandle),
    ]
    library.rt_tensor_matmul.restype = ctypes.c_int32

    library.rt_multilinear_map_create_dense.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint64,
        ctypes.POINTER(_MultilinearMapHandle),
    ]
    library.rt_multilinear_map_create_dense.restype = ctypes.c_int32
    library.rt_multilinear_map_create_sparse.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint64,
        ctypes.POINTER(_MultilinearMapHandle),
    ]
    library.rt_multilinear_map_create_sparse.restype = ctypes.c_int32
    library.rt_multilinear_map_release.argtypes = [_MultilinearMapHandle]
    library.rt_multilinear_map_release.restype = None

    library.rt_program_augmented_model_config_init.argtypes = [
        ctypes.POINTER(_NativeProgramAugmentedModelConfig),
        ctypes.c_uint64,
    ]
    library.rt_program_augmented_model_config_init.restype = ctypes.c_int32
    library.rt_program_branch_config_init.argtypes = [
        ctypes.POINTER(_NativeProgramBranchConfig),
        ctypes.c_uint64,
    ]
    library.rt_program_branch_config_init.restype = ctypes.c_int32
    library.rt_neural_lowering_options_init.argtypes = [
        ctypes.POINTER(_NativeNeuralLoweringOptions),
        ctypes.c_uint64,
    ]
    library.rt_neural_lowering_options_init.restype = ctypes.c_int32
    library.rt_program_augmented_forward_options_init.argtypes = [
        ctypes.POINTER(_NativeProgramAugmentedForwardOptions),
        ctypes.c_uint64,
    ]
    library.rt_program_augmented_forward_options_init.restype = (
        ctypes.c_int32
    )
    library.rt_program_augmented_model_create.argtypes = [
        ctypes.POINTER(_NativeProgramAugmentedModelConfig),
        _MultilinearMapHandle,
        ctypes.POINTER(_NativeProgramBranchConfig),
        ctypes.POINTER(_NativeNeuralLoweringOptions),
        ctypes.POINTER(_ProgramAugmentedModelHandle),
    ]
    library.rt_program_augmented_model_create.restype = ctypes.c_int32
    library.rt_program_augmented_model_release.argtypes = [
        _ProgramAugmentedModelHandle,
    ]
    library.rt_program_augmented_model_release.restype = None
    library.rt_program_augmented_model_to.argtypes = [
        _ProgramAugmentedModelHandle,
        ctypes.c_int32,
    ]
    library.rt_program_augmented_model_to.restype = ctypes.c_int32
    library.rt_program_augmented_model_backend.argtypes = [
        _ProgramAugmentedModelHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_program_augmented_model_backend.restype = ctypes.c_int32
    library.rt_program_augmented_model_has_program.argtypes = [
        _ProgramAugmentedModelHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_program_augmented_model_has_program.restype = ctypes.c_int32
    library.rt_program_augmented_model_parameters.argtypes = [
        _ProgramAugmentedModelHandle,
        ctypes.POINTER(_ParameterListHandle),
    ]
    library.rt_program_augmented_model_parameters.restype = ctypes.c_int32
    library.rt_program_augmented_model_forward.argtypes = [
        _ProgramAugmentedModelHandle,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(_NativeProgramAugmentedForwardOptions),
        ctypes.POINTER(_VariableHandle),
        ctypes.POINTER(_RepresentationTraceHandle),
    ]
    library.rt_program_augmented_model_forward.restype = ctypes.c_int32

    library.rt_representation_trace_release.argtypes = [
        _RepresentationTraceHandle,
    ]
    library.rt_representation_trace_release.restype = None
    library.rt_representation_trace_count.argtypes = [
        _RepresentationTraceHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_representation_trace_count.restype = ctypes.c_int32
    library.rt_representation_trace_name.argtypes = [
        _RepresentationTraceHandle,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_representation_trace_name.restype = ctypes.c_int32
    library.rt_representation_trace_rank.argtypes = [
        _RepresentationTraceHandle,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_representation_trace_rank.restype = ctypes.c_int32
    library.rt_representation_trace_shape.argtypes = [
        _RepresentationTraceHandle,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
    ]
    library.rt_representation_trace_shape.restype = ctypes.c_int32
    library.rt_representation_trace_numel.argtypes = [
        _RepresentationTraceHandle,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_representation_trace_numel.restype = ctypes.c_int32
    library.rt_representation_trace_copy_to_host_f32.argtypes = [
        _RepresentationTraceHandle,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    library.rt_representation_trace_copy_to_host_f32.restype = (
        ctypes.c_int32
    )

    library.rt_transformer_config_init.argtypes = [
        ctypes.POINTER(_NativeTransformerConfig),
        ctypes.c_uint64,
    ]
    library.rt_transformer_config_init.restype = ctypes.c_int32
    library.rt_decode_session_options_init.argtypes = [
        ctypes.POINTER(_NativeDecodeSessionOptions),
        ctypes.c_uint64,
    ]
    library.rt_decode_session_options_init.restype = ctypes.c_int32
    library.rt_model_create.argtypes = [
        ctypes.POINTER(_NativeTransformerConfig),
        ctypes.POINTER(_ModelHandle),
    ]
    library.rt_model_create.restype = ctypes.c_int32
    library.rt_model_release.argtypes = [_ModelHandle]
    library.rt_model_release.restype = None
    library.rt_model_to.argtypes = [
        _ModelHandle,
        ctypes.c_int32,
    ]
    library.rt_model_to.restype = ctypes.c_int32
    library.rt_model_backend.argtypes = [
        _ModelHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_model_backend.restype = ctypes.c_int32
    library.rt_model_set_full_sequence_attention.argtypes = [
        _ModelHandle,
        ctypes.c_int32,
    ]
    library.rt_model_set_full_sequence_attention.restype = ctypes.c_int32
    library.rt_model_full_sequence_attention.argtypes = [
        _ModelHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_model_full_sequence_attention.restype = ctypes.c_int32
    library.rt_model_set_activation_checkpointing.argtypes = [
        _ModelHandle,
        ctypes.c_int32,
    ]
    library.rt_model_set_activation_checkpointing.restype = ctypes.c_int32
    library.rt_model_activation_checkpointing.argtypes = [
        _ModelHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_model_activation_checkpointing.restype = ctypes.c_int32
    library.rt_lora_config_init.argtypes = [
        ctypes.POINTER(_NativeLoraConfig),
        ctypes.c_uint64,
    ]
    library.rt_lora_config_init.restype = ctypes.c_int32
    library.rt_model_attach_lora.argtypes = [
        _ModelHandle,
        ctypes.POINTER(_NativeLoraConfig),
    ]
    library.rt_model_attach_lora.restype = ctypes.c_int32
    library.rt_model_has_lora.argtypes = [
        _ModelHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_model_has_lora.restype = ctypes.c_int32
    library.rt_model_lora_config.argtypes = [
        _ModelHandle,
        ctypes.POINTER(_NativeLoraConfig),
    ]
    library.rt_model_lora_config.restype = ctypes.c_int32
    library.rt_model_quantize_linear_weights_nf4.argtypes = [
        _ModelHandle,
        ctypes.c_uint64,
    ]
    library.rt_model_quantize_linear_weights_nf4.restype = ctypes.c_int32
    library.rt_model_quantize_linear_weights_nf4_double_quantized.argtypes = [
        _ModelHandle,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    library.rt_model_quantize_linear_weights_nf4_double_quantized.restype = (
        ctypes.c_int32
    )
    library.rt_model_has_quantized_linear_weights.argtypes = [
        _ModelHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_model_has_quantized_linear_weights.restype = ctypes.c_int32
    library.rt_model_quantized_memory_stats.argtypes = [
        _ModelHandle,
        ctypes.POINTER(_NativeQuantizedMemoryStats),
    ]
    library.rt_model_quantized_memory_stats.restype = ctypes.c_int32
    library.rt_model_forward.argtypes = [
        _ModelHandle,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(_VariableHandle),
    ]
    library.rt_model_forward.restype = ctypes.c_int32
    library.rt_model_decode_session_create.argtypes = [
        _ModelHandle,
        ctypes.POINTER(_NativeDecodeSessionOptions),
        ctypes.POINTER(_DecodeSessionHandle),
    ]
    library.rt_model_decode_session_create.restype = ctypes.c_int32
    library.rt_decode_session_release.argtypes = [_DecodeSessionHandle]
    library.rt_decode_session_release.restype = None
    library.rt_decode_session_reset.argtypes = [_DecodeSessionHandle]
    library.rt_decode_session_reset.restype = ctypes.c_int32
    library.rt_decode_session_size.argtypes = [
        _DecodeSessionHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_decode_session_size.restype = ctypes.c_int32
    library.rt_decode_session_capacity.argtypes = [
        _DecodeSessionHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_decode_session_capacity.restype = ctypes.c_int32
    library.rt_decode_session_cache_kind.argtypes = [
        _DecodeSessionHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_decode_session_cache_kind.restype = ctypes.c_int32
    library.rt_decode_session_block_size.argtypes = [
        _DecodeSessionHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_decode_session_block_size.restype = ctypes.c_int32
    library.rt_decode_session_step.argtypes = [
        _DecodeSessionHandle,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_decode_session_step.restype = ctypes.c_int32
    library.rt_model_parameters.argtypes = [
        _ModelHandle,
        ctypes.POINTER(_ParameterListHandle),
    ]
    library.rt_model_parameters.restype = ctypes.c_int32
    library.rt_model_lora_parameters.argtypes = [
        _ModelHandle,
        ctypes.POINTER(_ParameterListHandle),
    ]
    library.rt_model_lora_parameters.restype = ctypes.c_int32
    library.rt_model_merge_lora.argtypes = [_ModelHandle]
    library.rt_model_merge_lora.restype = ctypes.c_int32

    library.rt_parameter_list_release.argtypes = [
        _ParameterListHandle,
    ]
    library.rt_parameter_list_release.restype = None
    library.rt_parameter_list_count.argtypes = [
        _ParameterListHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_parameter_list_count.restype = ctypes.c_int32
    library.rt_parameter_list_backend.argtypes = [
        _ParameterListHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_parameter_list_backend.restype = ctypes.c_int32
    library.rt_parameter_list_name.argtypes = [
        _ParameterListHandle,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_char),
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_parameter_list_name.restype = ctypes.c_int32
    library.rt_parameter_list_rank.argtypes = [
        _ParameterListHandle,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_parameter_list_rank.restype = ctypes.c_int32
    library.rt_parameter_list_shape.argtypes = [
        _ParameterListHandle,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
    ]
    library.rt_parameter_list_shape.restype = ctypes.c_int32
    library.rt_parameter_list_numel.argtypes = [
        _ParameterListHandle,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_parameter_list_numel.restype = ctypes.c_int32
    library.rt_parameter_list_total_numel.argtypes = [
        _ParameterListHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_parameter_list_total_numel.restype = ctypes.c_int32
    library.rt_parameter_list_copy_to_host_f32.argtypes = [
        _ParameterListHandle,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    library.rt_parameter_list_copy_to_host_f32.restype = ctypes.c_int32
    library.rt_parameter_list_load_from_host_f32.argtypes = [
        _ParameterListHandle,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    library.rt_parameter_list_load_from_host_f32.restype = ctypes.c_int32

    library.rt_variable_release.argtypes = [_VariableHandle]
    library.rt_variable_release.restype = None
    library.rt_variable_backend.argtypes = [
        _VariableHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_variable_backend.restype = ctypes.c_int32
    library.rt_variable_rank.argtypes = [
        _VariableHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_variable_rank.restype = ctypes.c_int32
    library.rt_variable_shape.argtypes = [
        _VariableHandle,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_uint64,
    ]
    library.rt_variable_shape.restype = ctypes.c_int32
    library.rt_variable_numel.argtypes = [
        _VariableHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_variable_numel.restype = ctypes.c_int32
    library.rt_variable_copy_to_host_f32.argtypes = [
        _VariableHandle,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint64,
    ]
    library.rt_variable_copy_to_host_f32.restype = ctypes.c_int32
    library.rt_variable_backward.argtypes = [_VariableHandle]
    library.rt_variable_backward.restype = ctypes.c_int32
    library.rt_cross_entropy.argtypes = [
        _VariableHandle,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.POINTER(_VariableHandle),
    ]
    library.rt_cross_entropy.restype = ctypes.c_int32
    library.rt_cross_entropy_time_range.argtypes = [
        _VariableHandle,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.POINTER(_VariableHandle),
    ]
    library.rt_cross_entropy_time_range.restype = ctypes.c_int32

    library.rt_adam_options_init.argtypes = [
        ctypes.POINTER(_NativeAdamOptions),
        ctypes.c_uint64,
    ]
    library.rt_adam_options_init.restype = ctypes.c_int32
    library.rt_adam_create.argtypes = [
        _ParameterListHandle,
        ctypes.POINTER(_NativeAdamOptions),
        ctypes.POINTER(_AdamHandle),
    ]
    library.rt_adam_create.restype = ctypes.c_int32
    library.rt_adam_release.argtypes = [_AdamHandle]
    library.rt_adam_release.restype = None
    library.rt_adam_backend.argtypes = [
        _AdamHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_adam_backend.restype = ctypes.c_int32
    library.rt_adam_step_count.argtypes = [
        _AdamHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_adam_step_count.restype = ctypes.c_int32
    library.rt_adam_parameter_count.argtypes = [
        _AdamHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_adam_parameter_count.restype = ctypes.c_int32
    library.rt_adam_state_storage.argtypes = [
        _AdamHandle,
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.rt_adam_state_storage.restype = ctypes.c_int32
    library.rt_adam_state_page_size.argtypes = [
        _AdamHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_adam_state_page_size.restype = ctypes.c_int32
    library.rt_adam_state_page_count.argtypes = [
        _AdamHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_adam_state_page_count.restype = ctypes.c_int32
    library.rt_adam_state_payload_bytes.argtypes = [
        _AdamHandle,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    library.rt_adam_state_payload_bytes.restype = ctypes.c_int32
    library.rt_adam_step.argtypes = [
        _AdamHandle,
        ctypes.POINTER(_NativeAdamStepStats),
    ]
    library.rt_adam_step.restype = ctypes.c_int32
    library.rt_adam_zero_gradients.argtypes = [_AdamHandle]
    library.rt_adam_zero_gradients.restype = ctypes.c_int32


def _load_library() -> ctypes.CDLL:
    candidates = _candidate_libraries()
    override = os.environ.get("RIFTCO_TRANSFORMER_LIBRARY")
    if override and not candidates:
        candidates = [override]

    failures: list[str] = []
    for candidate in candidates:
        try:
            library = ctypes.CDLL(candidate)
            _configure_library(library)
            return library
        except (OSError, AttributeError, RuntimeError) as error:
            failures.append(f"{candidate}: {error}")
            if override:
                break

    guidance = (
        "could not load libriftco_transformer_c; set "
        "RIFTCO_TRANSFORMER_LIBRARY to the shared-library path"
    )
    if failures:
        guidance += "\n" + "\n".join(failures)
    raise OSError(guidance)


def _native() -> ctypes.CDLL:
    global _library
    if _library is None:
        with _library_lock:
            if _library is None:
                _library = _load_library()
    return _library


def _decode(value: bytes | None) -> str:
    return "" if value is None else value.decode("utf-8", errors="replace")


def _check(status: int) -> None:
    if status == STATUS_OK:
        return
    library = _native()
    # Copy the thread-local detail before making another native call.
    detail = _decode(library.rt_last_error())
    status_name = _decode(library.rt_status_string(status))
    raise RiftcoTransformerError(int(status), status_name, detail)


def _backend_code(backend: str | int) -> int:
    if isinstance(backend, str):
        try:
            return _BACKEND_CODES[backend.strip().lower()]
        except KeyError as error:
            raise ValueError(
                f"unknown backend {backend!r}; expected 'cpu', 'metal', "
                "'cuda', or 'tpu'"
            ) from error

    if isinstance(backend, bool):
        raise TypeError(
            "backend must be 'cpu', 'metal', 'cuda', 'tpu', "
            "0, 1, 2, or 3"
        )
    try:
        value = operator.index(backend)
    except TypeError as error:
        raise TypeError(
            "backend must be 'cpu', 'metal', 'cuda', 'tpu', "
            "0, 1, 2, or 3"
        ) from error
    if value not in _BACKEND_NAMES:
        raise ValueError(
            f"unknown backend {value!r}; expected 0, 1, 2, or 3"
        )
    return value


def _backend_name(backend: int) -> str:
    try:
        return _BACKEND_NAMES[backend]
    except KeyError as error:
        raise RuntimeError(
            f"native library returned unknown backend {backend}"
        ) from error


def _full_sequence_attention_code(attention: str | int) -> int:
    if isinstance(attention, str):
        try:
            return _FULL_SEQUENCE_ATTENTION_CODES[
                attention.strip().lower()
            ]
        except KeyError as error:
            raise ValueError(
                "unknown full-sequence attention "
                f"{attention!r}; expected 'materialized' or 'flash'"
            ) from error

    if isinstance(attention, bool):
        raise TypeError(
            "attention must be 'materialized', 'flash', 0, or 1"
        )
    try:
        value = operator.index(attention)
    except TypeError as error:
        raise TypeError(
            "attention must be 'materialized', 'flash', 0, or 1"
        ) from error
    if value not in _FULL_SEQUENCE_ATTENTION_NAMES:
        raise ValueError(
            f"unknown full-sequence attention {value!r}; expected 0 or 1"
        )
    return value


def _full_sequence_attention_name(attention: int) -> str:
    try:
        return _FULL_SEQUENCE_ATTENTION_NAMES[attention]
    except KeyError as error:
        raise RuntimeError(
            "native library returned unknown full-sequence attention "
            f"{attention}"
        ) from error


def _activation_checkpointing_code(
    activation_checkpointing: str | int,
) -> int:
    if isinstance(activation_checkpointing, str):
        try:
            return _ACTIVATION_CHECKPOINTING_CODES[
                activation_checkpointing.strip().lower()
            ]
        except KeyError as error:
            raise ValueError(
                "unknown activation checkpointing "
                f"{activation_checkpointing!r}; expected "
                "'disabled' or 'block'"
            ) from error

    if isinstance(activation_checkpointing, bool):
        raise TypeError(
            "activation_checkpointing must be 'disabled', "
            "'block', 0, or 1"
        )
    try:
        value = operator.index(activation_checkpointing)
    except TypeError as error:
        raise TypeError(
            "activation_checkpointing must be 'disabled', "
            "'block', 0, or 1"
        ) from error
    if value not in _ACTIVATION_CHECKPOINTING_NAMES:
        raise ValueError(
            "unknown activation checkpointing "
            f"{value!r}; expected 0 or 1"
        )
    return value


def _activation_checkpointing_name(
    activation_checkpointing: int,
) -> str:
    try:
        return _ACTIVATION_CHECKPOINTING_NAMES[
            activation_checkpointing
        ]
    except KeyError as error:
        raise RuntimeError(
            "native library returned unknown activation checkpointing "
            f"{activation_checkpointing}"
        ) from error


def _kv_cache_code(cache: str | int) -> int:
    if isinstance(cache, str):
        try:
            return _KV_CACHE_CODES[cache.strip().lower()]
        except KeyError as error:
            raise ValueError(
                f"unknown KV cache {cache!r}; expected "
                "'contiguous' or 'paged'"
            ) from error
    if isinstance(cache, bool):
        raise TypeError(
            "cache must be 'contiguous', 'paged', 0, or 1"
        )
    try:
        value = operator.index(cache)
    except TypeError as error:
        raise TypeError(
            "cache must be 'contiguous', 'paged', 0, or 1"
        ) from error
    if value not in _KV_CACHE_NAMES:
        raise ValueError(
            f"unknown KV cache {value!r}; expected 0 or 1"
        )
    return value


def _kv_cache_name(cache: int) -> str:
    try:
        return _KV_CACHE_NAMES[cache]
    except KeyError as error:
        raise RuntimeError(
            f"native library returned unknown KV cache kind {cache}"
        ) from error


def _adam_state_storage_code(storage: str | int) -> int:
    if isinstance(storage, str):
        try:
            return _ADAM_STATE_CODES[storage.strip().lower()]
        except KeyError as error:
            raise ValueError(
                f"unknown Adam state storage {storage!r}; expected "
                "'contiguous' or 'paged'"
            ) from error
    if isinstance(storage, bool):
        raise TypeError(
            "state_storage must be 'contiguous', 'paged', 0, or 1"
        )
    try:
        value = operator.index(storage)
    except TypeError as error:
        raise TypeError(
            "state_storage must be 'contiguous', 'paged', 0, or 1"
        ) from error
    if value not in _ADAM_STATE_NAMES:
        raise ValueError(
            f"unknown Adam state storage {value!r}; expected 0 or 1"
        )
    return value


def _adam_state_storage_name(storage: int) -> str:
    try:
        return _ADAM_STATE_NAMES[storage]
    except KeyError as error:
        raise RuntimeError(
            "native library returned unknown Adam state storage "
            f"{storage}"
        ) from error


def _tokenizer_method_code(method: object) -> int:
    if not isinstance(method, str):
        raise TypeError("method must be 'byte' or 'bpe'")
    try:
        return _TOKENIZER_METHOD_CODES[method]
    except KeyError as error:
        raise ValueError(
            f"unknown tokenizer method {method!r}; "
            "expected 'byte' or 'bpe'"
        ) from error


def _tokenizer_method_name(method: int) -> str:
    try:
        return _TOKENIZER_METHOD_NAMES[method]
    except KeyError as error:
        raise RuntimeError(
            f"native library returned unknown tokenizer method {method}"
        ) from error


def _choice_code(
    value: object,
    choices: dict[str, int],
    name: str,
) -> tuple[str, int]:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    normalized = value.strip().lower()
    try:
        return normalized, choices[normalized]
    except KeyError as error:
        expected = ", ".join(repr(choice) for choice in choices)
        raise ValueError(
            f"unknown {name} {value!r}; expected {expected}"
        ) from error


def _lora_targets(
    targets: Iterable[object],
) -> tuple[tuple[str, ...], int]:
    if isinstance(targets, (str, bytes)):
        raise TypeError("targets must be an iterable of target names")
    try:
        values = tuple(targets)
    except TypeError as error:
        raise TypeError(
            "targets must be an iterable of target names"
        ) from error
    if not values:
        raise ValueError("targets must not be empty")

    mask = 0
    for index, target in enumerate(values):
        if not isinstance(target, str):
            raise TypeError(f"targets[{index}] must be a str")
        try:
            bit = _LORA_TARGET_CODES[target]
        except KeyError as error:
            expected = ", ".join(repr(name) for name in _LORA_TARGET_CODES)
            raise ValueError(
                f"unknown LoRA target {target!r}; expected one of "
                f"{expected}"
            ) from error
        if mask & bit:
            raise ValueError(f"duplicate LoRA target {target!r}")
        mask |= bit
    canonical = tuple(
        name
        for name, bit in _LORA_TARGET_CODES.items()
        if mask & bit
    )
    return canonical, mask


def _lora_target_names(mask: int) -> tuple[str, ...]:
    if mask == 0 or mask & ~LORA_TARGET_ALL_LINEAR:
        raise RuntimeError(
            f"native library returned invalid LoRA targets 0x{mask:x}"
        )
    return tuple(
        name
        for bit, name in _LORA_TARGET_NAMES.items()
        if mask & bit
    )


def _shape_values(
    shape: Sequence[int] | Iterable[int],
) -> tuple[tuple[int, ...], ctypes.Array[ctypes.c_uint64] | None]:
    try:
        dimensions = tuple(operator.index(value) for value in shape)
    except TypeError as error:
        raise TypeError("shape must be an iterable of integers") from error

    maximum = (1 << 64) - 1
    for dimension in dimensions:
        if dimension < 0:
            raise ValueError("tensor dimensions must not be negative")
        if dimension > maximum:
            raise OverflowError("tensor dimension exceeds uint64")

    if not dimensions:
        return dimensions, None
    array_type = ctypes.c_uint64 * len(dimensions)
    return dimensions, array_type(*dimensions)


def _unsigned_integer(value: object, maximum: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 0:
        raise ValueError(f"{name} must not be negative")
    if result > maximum:
        raise OverflowError(f"{name} exceeds its native integer range")
    return result


def _token_id(value: object, name: str) -> int:
    return _unsigned_integer(value, (1 << 32) - 1, name)


def _token_matrix(
    tokens: Iterable[object],
    name: str,
) -> tuple[tuple[int, ...], int, int]:
    try:
        outer = tuple(tokens)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a token row or a rectangular batch"
        ) from error
    if not outer:
        raise ValueError(f"{name} must not be empty")

    try:
        first = _token_id(outer[0], f"{name}[0]")
    except TypeError:
        rows: list[tuple[object, ...]] = []
        for row_index, row in enumerate(outer):
            try:
                rows.append(tuple(row))  # type: ignore[arg-type]
            except TypeError as error:
                raise TypeError(
                    f"{name}[{row_index}] must be an iterable token row"
                ) from error
        width = len(rows[0])
        if width == 0:
            raise ValueError(f"{name} rows must not be empty")
        flattened: list[int] = []
        for row_index, row in enumerate(rows):
            if len(row) != width:
                raise ValueError(f"{name} must be rectangular")
            for column_index, token in enumerate(row):
                flattened.append(
                    _token_id(
                        token,
                        f"{name}[{row_index}][{column_index}]",
                    )
                )
        return tuple(flattened), len(rows), width

    flattened = [first]
    for index, token in enumerate(outer[1:], start=1):
        flattened.append(_token_id(token, f"{name}[{index}]"))
    return tuple(flattened), 1, len(flattened)


def _native_u32_values(
    values: tuple[int, ...],
) -> ctypes.Array[ctypes.c_uint32] | None:
    if not values:
        return None
    array_type = ctypes.c_uint32 * len(values)
    return array_type(*values)


def _bytes_like(value: object, name: str) -> bytes:
    if isinstance(value, str):
        raise TypeError(f"{name} must be bytes-like, not str")
    try:
        view = memoryview(value)
    except TypeError as error:
        raise TypeError(f"{name} must be bytes-like") from error
    try:
        return view.tobytes()
    finally:
        view.release()


def _native_u8_values(
    values: bytes,
) -> ctypes.Array[ctypes.c_uint8] | None:
    if not values:
        return None
    array_type = ctypes.c_uint8 * len(values)
    return array_type.from_buffer_copy(values)


def _token_ids(
    tokens: Iterable[object],
    name: str,
) -> tuple[int, ...]:
    try:
        values = tuple(tokens)
    except TypeError as error:
        raise TypeError(f"{name} must be an iterable of token IDs") from error
    return tuple(
        _token_id(token, f"{name}[{index}]")
        for index, token in enumerate(values)
    )


@dataclass(frozen=True, slots=True)
class TransformerConfig:
    """Dimensions and deterministic initialization for a decoder model."""

    vocabulary_size: int
    maximum_context: int
    model_width: int
    head_count: int
    block_count: int
    feed_forward_width: int
    random_seed: int = 5489
    layer_norm_epsilon: float = 1.0e-5


@dataclass(frozen=True, slots=True)
class ProgramAugmentedModelConfig:
    """Dimensions for a learned sequence model with an optional program."""

    vocabulary_size: int
    context_length: int
    model_width: int
    head_count: int
    feed_forward_width: int
    attention_branch_count: int = 2
    random_seed: int = 42
    attention: str = "materialized"

    def __post_init__(self) -> None:
        maximum = (1 << 64) - 1
        for name in (
            "vocabulary_size",
            "context_length",
            "model_width",
            "head_count",
            "feed_forward_width",
            "attention_branch_count",
        ):
            value = _unsigned_integer(getattr(self, name), maximum, name)
            if value == 0:
                raise ValueError(f"{name} must be greater than zero")
            object.__setattr__(self, name, value)
        if self.model_width % self.head_count != 0:
            raise ValueError("model_width must be divisible by head_count")
        object.__setattr__(
            self,
            "random_seed",
            _unsigned_integer(self.random_seed, (1 << 32) - 1, "random_seed"),
        )
        attention, _code = _choice_code(
            self.attention,
            _FULL_SEQUENCE_ATTENTION_CODES,
            "attention",
        )
        object.__setattr__(self, "attention", attention)


@dataclass(frozen=True, slots=True)
class NeuralLoweringConfig:
    """Task-neutral policy for lowering one multilinear map."""

    strategy: str = "auto"
    precision: str = "exact"
    initialization: str = "compiled"
    trainable: bool = False
    random_seed: int = 42
    random_scale: float = 0.02
    max_coefficient_elements: int = 1 << 24
    attention_query_axis: int | None = None

    def __post_init__(self) -> None:
        for attribute, choices in (
            ("strategy", _LOWERING_STRATEGY_CODES),
            ("precision", _COEFFICIENT_PRECISION_CODES),
            ("initialization", _COEFFICIENT_INITIALIZATION_CODES),
        ):
            value, _code = _choice_code(
                getattr(self, attribute), choices, attribute
            )
            object.__setattr__(self, attribute, value)
        if not isinstance(self.trainable, bool):
            raise TypeError("trainable must be a bool")
        object.__setattr__(
            self,
            "random_seed",
            _unsigned_integer(self.random_seed, (1 << 32) - 1, "random_seed"),
        )
        if isinstance(self.random_scale, bool):
            raise TypeError("random_scale must be a real number")
        try:
            scale = float(ctypes.c_float(float(self.random_scale)).value)
        except (TypeError, ValueError) as error:
            raise TypeError("random_scale must be a real number") from error
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("random_scale must be finite and positive")
        object.__setattr__(self, "random_scale", scale)
        limit = _unsigned_integer(
            self.max_coefficient_elements,
            (1 << 64) - 1,
            "max_coefficient_elements",
        )
        if limit == 0:
            raise ValueError("max_coefficient_elements must be positive")
        object.__setattr__(self, "max_coefficient_elements", limit)
        if self.attention_query_axis is not None:
            axis = _unsigned_integer(
                self.attention_query_axis,
                (1 << 63) - 1,
                "attention_query_axis",
            )
            object.__setattr__(self, "attention_query_axis", axis)


@dataclass(frozen=True, slots=True)
class ProgramInputLayout:
    """Source selection and learned-projection sharing for one map input."""

    source: str = "whole_source"
    position: int = 0
    projection_group: int = 0

    def __post_init__(self) -> None:
        source, _code = _choice_code(
            self.source, _PROGRAM_INPUT_CODES, "program input source"
        )
        position = _unsigned_integer(
            self.position, (1 << 64) - 1, "position"
        )
        group = _unsigned_integer(
            self.projection_group, (1 << 64) - 1, "projection_group"
        )
        if source == "whole_source" and position != 0:
            raise ValueError("whole_source input position must be zero")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "projection_group", group)


@dataclass(frozen=True, slots=True)
class ProgramBranch:
    """Placement, projections, map, and lowering for one optional branch."""

    map: MultilinearMap
    source_offset: int
    source_length: int
    target_offset: int
    output_length: int
    inputs: tuple[ProgramInputLayout, ...]
    lowering: NeuralLoweringConfig = NeuralLoweringConfig()
    input_projection_bias: bool = False
    merge_bias: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.map, MultilinearMap):
            raise TypeError("map must be a MultilinearMap")
        maximum = (1 << 64) - 1
        for name in (
            "source_offset",
            "source_length",
            "target_offset",
            "output_length",
        ):
            value = _unsigned_integer(getattr(self, name), maximum, name)
            if name == "output_length" and value == 0:
                raise ValueError(f"{name} must be greater than zero")
            object.__setattr__(self, name, value)
        try:
            inputs = tuple(self.inputs)
        except TypeError as error:
            raise TypeError("inputs must be an iterable") from error
        if len(inputs) != len(self.map.input_dimensions):
            raise ValueError("inputs must match the multilinear map arity")
        if self.map.input_dimensions and self.source_length == 0:
            raise ValueError("nonconstant programs need a positive source_length")
        if any(not isinstance(item, ProgramInputLayout) for item in inputs):
            raise TypeError("inputs must contain ProgramInputLayout values")
        if not isinstance(self.lowering, NeuralLoweringConfig):
            raise TypeError("lowering must be a NeuralLoweringConfig")
        if not isinstance(self.input_projection_bias, bool):
            raise TypeError("input_projection_bias must be a bool")
        if not isinstance(self.merge_bias, bool):
            raise TypeError("merge_bias must be a bool")
        object.__setattr__(self, "inputs", inputs)


@dataclass(frozen=True, slots=True)
class ProgramInputSteering:
    """Affine intervention on one projected input and source position."""

    input_index: int
    position: int
    scales: tuple[float, ...] = ()
    offsets: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "input_index",
            _unsigned_integer(
                self.input_index, (1 << 64) - 1, "input_index"
            ),
        )
        object.__setattr__(
            self,
            "position",
            _unsigned_integer(self.position, (1 << 64) - 1, "position"),
        )
        for name in ("scales", "offsets"):
            try:
                raw_values = tuple(getattr(self, name))
            except TypeError as error:
                raise TypeError(f"{name} must be an iterable") from error
            values: list[float] = []
            for index, raw in enumerate(raw_values):
                if isinstance(raw, bool):
                    raise TypeError(f"{name}[{index}] must be a number")
                try:
                    value = float(ctypes.c_float(float(raw)).value)
                except (TypeError, ValueError) as error:
                    raise TypeError(
                        f"{name}[{index}] must be a number"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(f"{name}[{index}] must be finite")
                values.append(value)
            object.__setattr__(self, name, tuple(values))


@dataclass(frozen=True, slots=True)
class ProgramAugmentedForwardOptions:
    capture_representations: bool = False
    batch_roll_learned_attention: bool = False
    batch_roll_shift: int = 1
    steering: tuple[ProgramInputSteering, ...] = ()
    ablated_program_inputs: tuple[int, ...] = ()
    ablate_program_output: bool = False

    def __post_init__(self) -> None:
        for name in (
            "capture_representations",
            "batch_roll_learned_attention",
            "ablate_program_output",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        shift = _unsigned_integer(
            self.batch_roll_shift, (1 << 64) - 1, "batch_roll_shift"
        )
        if shift == 0:
            raise ValueError("batch_roll_shift must be greater than zero")
        try:
            steering = tuple(self.steering)
        except TypeError as error:
            raise TypeError("steering must be an iterable") from error
        if any(not isinstance(item, ProgramInputSteering) for item in steering):
            raise TypeError("steering must contain ProgramInputSteering values")
        try:
            raw_ablations = tuple(self.ablated_program_inputs)
        except TypeError as error:
            raise TypeError(
                "ablated_program_inputs must be an iterable"
            ) from error
        ablations = tuple(
            _unsigned_integer(value, (1 << 64) - 1, "program input index")
            for value in raw_ablations
        )
        if len(set(ablations)) != len(ablations):
            raise ValueError("ablated_program_inputs must be unique")
        object.__setattr__(self, "batch_roll_shift", shift)
        object.__setattr__(self, "steering", steering)
        object.__setattr__(self, "ablated_program_inputs", ablations)


@dataclass(frozen=True, slots=True)
class Representation:
    name: str
    shape: tuple[int, ...]
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class RepresentationTrace:
    entries: tuple[Representation, ...] = ()

    def at(self, name: str) -> Representation:
        for entry in self.entries:
            if entry.name == name:
                return entry
        raise KeyError(name)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and any(
            entry.name == name for entry in self.entries
        )


@dataclass(frozen=True, slots=True)
class ProgramAugmentedForwardResult:
    logits: Variable
    representations: RepresentationTrace


@dataclass(frozen=True, slots=True)
class LoraConfig:
    """Low-rank adapter dimensions, scaling, targets, and initialization."""

    rank: int = 4
    alpha: float = 8.0
    targets: tuple[str, ...] = (
        "attention.query",
        "attention.value",
    )
    random_seed: int = 5489

    def __post_init__(self) -> None:
        rank = _unsigned_integer(
            self.rank,
            (1 << (ctypes.sizeof(ctypes.c_size_t) * 8)) - 1,
            "rank",
        )
        if rank == 0:
            raise ValueError("rank must be greater than zero")
        if isinstance(self.alpha, bool):
            raise TypeError("alpha must be a real number")
        try:
            alpha = float(self.alpha)
        except (TypeError, ValueError) as error:
            raise TypeError("alpha must be a real number") from error
        alpha = float(ctypes.c_float(alpha).value)
        if not math.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        native_alpha = float(ctypes.c_float(alpha).value)
        native_scale = float(
            ctypes.c_float(native_alpha / rank).value
        )
        if (
            not math.isfinite(native_alpha)
            or native_alpha <= 0.0
            or not math.isfinite(native_scale)
            or native_scale <= 0.0
        ):
            raise ValueError(
                "alpha and alpha divided by rank must be finite, "
                "positive float32 values"
            )
        targets, _mask = _lora_targets(self.targets)
        random_seed = _unsigned_integer(
            self.random_seed,
            (1 << 32) - 1,
            "random_seed",
        )

        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "alpha", native_alpha)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "random_seed", random_seed)


@dataclass(frozen=True, slots=True)
class QuantizedMemoryStats:
    """Persistent NF4 payload bytes for a quantized model."""

    weight_count: int
    packed_code_bytes: int
    scale_bytes: int
    logical_payload_bytes: int
    resident_payload_bytes: int
    fp32_equivalent_bytes: int
    fp32_scale_bytes: int
    scale_code_bytes: int
    second_level_scale_bytes: int
    scale_offset_bytes: int
    double_quantized_weight_count: int

    @property
    def compression_ratio(self) -> float:
        """Return FP32 bytes divided by the logical packed payload bytes."""

        if self.logical_payload_bytes == 0:
            return 0.0
        return self.fp32_equivalent_bytes / self.logical_payload_bytes


@dataclass(frozen=True, slots=True)
class AdamStepStats:
    """Diagnostics returned after one successful Adam update."""

    step: int
    gradient_norm: float
    clip_scale: float


def backend_available(backend: str | int) -> bool:
    """Return whether a native execution backend is available."""

    available = ctypes.c_int32()
    _check(
        _native().rt_backend_is_available(
            _backend_code(backend),
            ctypes.byref(available),
        )
    )
    return bool(available.value)


class Tokenizer:
    """A deterministic byte or byte-pair-encoding tokenizer."""

    __slots__ = ("_handle", "_lock", "__weakref__")

    def __init__(
        self,
        corpus: str | bytes | bytearray | memoryview,
        *,
        method: str = "byte",
        vocabulary_size: int | None = None,
        minimum_pair_frequency: int = 2,
    ) -> None:
        self._handle: _TokenizerHandle | None = None
        self._lock = threading.RLock()

        method_code = _tokenizer_method_code(method)
        pair_frequency = _unsigned_integer(
            minimum_pair_frequency,
            (1 << 64) - 1,
            "minimum_pair_frequency",
        )
        if pair_frequency < 1:
            raise ValueError(
                "minimum_pair_frequency must be at least 1"
            )

        if method_code == TOKENIZER_METHOD_BYTE:
            if vocabulary_size is not None:
                raise ValueError(
                    "vocabulary_size does not apply to the byte tokenizer"
                )
            if pair_frequency != 2:
                raise ValueError(
                    "minimum_pair_frequency does not apply to the byte "
                    "tokenizer; leave it at the default value 2"
                )
            configured_vocabulary_size = 512
        else:
            configured_vocabulary_size = (
                512
                if vocabulary_size is None
                else _unsigned_integer(
                    vocabulary_size,
                    (1 << 32) - 1,
                    "vocabulary_size",
                )
            )
            if configured_vocabulary_size < 256:
                raise ValueError(
                    "BPE vocabulary_size must be at least 256"
                )

        corpus_bytes = (
            corpus.encode("utf-8")
            if isinstance(corpus, str)
            else _bytes_like(corpus, "corpus")
        )
        native_corpus = _native_u8_values(corpus_bytes)
        options = _NativeTokenizerOptions()
        _check(
            _native().rt_tokenizer_options_init(
                ctypes.byref(options),
                ctypes.sizeof(options),
            )
        )
        options.method = method_code
        options.vocabulary_size = configured_vocabulary_size
        options.minimum_pair_frequency = pair_frequency
        output = _TokenizerHandle()
        _check(
            _native().rt_tokenizer_create_with_options(
                native_corpus,
                len(corpus_bytes),
                ctypes.byref(options),
                ctypes.byref(output),
            )
        )
        if not output:
            raise RuntimeError(
                "native tokenizer creation succeeded without a handle"
            )
        self._handle = output

    @classmethod
    def from_state(
        cls,
        *,
        method: str,
        byte_vocabulary: Iterable[object] = (),
        merge_rules: Iterable[Iterable[object]] = (),
    ) -> Tokenizer:
        """Restore exact tokenizer state without its training corpus."""

        result = cls.__new__(cls)
        result._handle = None
        result._lock = threading.RLock()
        method_code = _tokenizer_method_code(method)
        try:
            raw_vocabulary = tuple(byte_vocabulary)
        except TypeError as error:
            raise TypeError(
                "byte_vocabulary must be an iterable of bytes"
            ) from error
        try:
            raw_rules = tuple(merge_rules)
        except TypeError as error:
            raise TypeError(
                "merge_rules must be an iterable of merge triples"
            ) from error

        output = _TokenizerHandle()
        if method_code == TOKENIZER_METHOD_BYTE:
            if raw_rules:
                raise ValueError(
                    "byte tokenizer state must not contain merge rules"
                )
            vocabulary = bytes(
                _unsigned_integer(value, 255, "vocabulary byte")
                for value in raw_vocabulary
            )
            native_vocabulary = _native_u8_values(vocabulary)
            _check(
                _native().rt_tokenizer_create_from_byte_vocabulary(
                    native_vocabulary,
                    len(vocabulary),
                    ctypes.byref(output),
                )
            )
        else:
            if raw_vocabulary:
                raise ValueError(
                    "BPE tokenizer state uses its fixed byte vocabulary"
                )
            checked_rules: list[tuple[int, int, int]] = []
            for index, raw_rule in enumerate(raw_rules):
                try:
                    values = tuple(raw_rule)
                except TypeError as error:
                    raise TypeError(
                        f"merge_rules[{index}] must be an iterable"
                    ) from error
                if len(values) != 3:
                    raise ValueError(
                        f"merge_rules[{index}] must have three values"
                    )
                checked_rules.append(
                    (
                        _token_id(values[0], f"merge_rules[{index}].left"),
                        _token_id(values[1], f"merge_rules[{index}].right"),
                        _token_id(
                            values[2],
                            f"merge_rules[{index}].result",
                        ),
                    )
                )
            array_type = _NativeBpeMergeRule * len(checked_rules)
            native_rules = (
                array_type(
                    *(
                        _NativeBpeMergeRule(*rule)
                        for rule in checked_rules
                    )
                )
                if checked_rules
                else None
            )
            _check(
                _native().rt_tokenizer_create_from_bpe_merges(
                    native_rules,
                    len(checked_rules),
                    ctypes.byref(output),
                )
            )
        if not output:
            raise RuntimeError(
                "native tokenizer restoration succeeded without a handle"
            )
        result._handle = output
        return result

    def _native_handle(self) -> _TokenizerHandle:
        if self._handle is None:
            raise RuntimeError("tokenizer is closed")
        return self._handle

    @staticmethod
    def _check_second_pass_size(
        required: ctypes.c_uint64,
        capacity: int,
        operation: str,
    ) -> None:
        if int(required.value) != capacity:
            raise RuntimeError(
                f"native tokenizer {operation} size changed between passes"
            )

    @property
    def vocab_size(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_tokenizer_vocabulary_size(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    @property
    def method(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_tokenizer_get_method(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _tokenizer_method_name(int(output.value))

    def _token_bytes(self, token: int) -> bytes:
        required = ctypes.c_uint64()
        _check(
            _native().rt_tokenizer_token_bytes(
                self._native_handle(),
                token,
                None,
                0,
                ctypes.byref(required),
            )
        )
        capacity = int(required.value)
        array_type = ctypes.c_uint8 * capacity
        output = array_type() if capacity else None
        _check(
            _native().rt_tokenizer_token_bytes(
                self._native_handle(),
                token,
                output,
                capacity,
                ctypes.byref(required),
            )
        )
        self._check_second_pass_size(required, capacity, "token bytes")
        return b"" if output is None else bytes(output)

    @property
    def vocabulary(self) -> tuple[bytes, ...]:
        with self._lock:
            size = self.vocab_size
            return tuple(self._token_bytes(token) for token in range(size))

    @property
    def merge_rules(self) -> tuple[tuple[int, int, int], ...]:
        """Return ordered BPE merges as ``(left, right, result)`` triples."""

        with self._lock:
            if self.method != "bpe":
                raise RuntimeError(
                    "merge_rules is only available for the BPE tokenizer"
                )
            count = ctypes.c_uint64()
            _check(
                _native().rt_tokenizer_bpe_merge_count(
                    self._native_handle(),
                    ctypes.byref(count),
                )
            )
            rules: list[tuple[int, int, int]] = []
            for index in range(int(count.value)):
                rule = _NativeBpeMergeRule()
                _check(
                    _native().rt_tokenizer_bpe_merge_rule(
                        self._native_handle(),
                        index,
                        ctypes.byref(rule),
                    )
                )
                rules.append(
                    (
                        int(rule.left),
                        int(rule.right),
                        int(rule.result),
                    )
                )
            return tuple(rules)

    @property
    def vocabulary_bytes(self) -> bytes:
        with self._lock:
            if self.method != "byte":
                raise RuntimeError(
                    "vocabulary_bytes is only available for the byte "
                    "tokenizer; use vocabulary for BPE token pieces"
                )
            required = ctypes.c_uint64()
            _check(
                _native().rt_tokenizer_vocabulary(
                    self._native_handle(),
                    None,
                    0,
                    ctypes.byref(required),
                )
            )
            capacity = int(required.value)
            array_type = ctypes.c_uint8 * capacity
            output = array_type() if capacity else None
            _check(
                _native().rt_tokenizer_vocabulary(
                    self._native_handle(),
                    output,
                    capacity,
                    ctypes.byref(required),
                )
            )
            self._check_second_pass_size(
                required,
                capacity,
                "vocabulary",
            )
            return b"" if output is None else bytes(output)

    def encode_bytes(self, value: object) -> list[int]:
        source = _bytes_like(value, "value")
        native_source = _native_u8_values(source)
        with self._lock:
            required = ctypes.c_uint64()
            _check(
                _native().rt_tokenizer_encode(
                    self._native_handle(),
                    native_source,
                    len(source),
                    None,
                    0,
                    ctypes.byref(required),
                )
            )
            capacity = int(required.value)
            array_type = ctypes.c_uint32 * capacity
            output = array_type() if capacity else None
            _check(
                _native().rt_tokenizer_encode(
                    self._native_handle(),
                    native_source,
                    len(source),
                    output,
                    capacity,
                    ctypes.byref(required),
                )
            )
            self._check_second_pass_size(required, capacity, "encode")
            return [] if output is None else [int(token) for token in output]

    def decode_bytes(self, tokens: Iterable[object]) -> bytes:
        values = _token_ids(tokens, "tokens")
        native_tokens = _native_u32_values(values)
        with self._lock:
            required = ctypes.c_uint64()
            _check(
                _native().rt_tokenizer_decode(
                    self._native_handle(),
                    native_tokens,
                    len(values),
                    None,
                    0,
                    ctypes.byref(required),
                )
            )
            capacity = int(required.value)
            array_type = ctypes.c_uint8 * capacity
            output = array_type() if capacity else None
            _check(
                _native().rt_tokenizer_decode(
                    self._native_handle(),
                    native_tokens,
                    len(values),
                    output,
                    capacity,
                    ctypes.byref(required),
                )
            )
            self._check_second_pass_size(required, capacity, "decode")
            return b"" if output is None else bytes(output)

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a str")
        return self.encode_bytes(text.encode("utf-8"))

    def decode(self, tokens: Iterable[object]) -> str:
        return self.decode_bytes(tokens).decode("utf-8", errors="strict")

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native().rt_tokenizer_release(handle)

    def __enter__(self) -> Tokenizer:
        with self._lock:
            self._native_handle()
            return self

    def __exit__(
        self,
        _type: object,
        _value: object,
        _traceback: object,
    ) -> None:
        self.close()

    def __copy__(self) -> Tokenizer:
        raise TypeError("Tokenizer handles cannot be copied")

    def __deepcopy__(self, _memo: object) -> Tokenizer:
        raise TypeError("Tokenizer handles cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Context:
    """Owns a CPU, Metal, CUDA, or TPU selection for tensor construction."""

    __slots__ = ("_handle", "_lock", "__weakref__")

    def __init__(self, backend: str | int = "cpu") -> None:
        self._handle: _ContextHandle | None = None
        self._lock = threading.RLock()
        output = _ContextHandle()
        _check(
            _native().rt_context_create(
                _backend_code(backend),
                ctypes.byref(output),
            )
        )
        if not output:
            raise RuntimeError(
                "native context creation succeeded without a handle"
            )
        self._handle = output

    def _native_handle(self) -> _ContextHandle:
        if self._handle is None:
            raise RuntimeError("context is closed")
        return self._handle

    @property
    def backend(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_context_backend(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _backend_name(int(output.value))

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native().rt_context_release(handle)

    def __enter__(self) -> Context:
        with self._lock:
            self._native_handle()
            return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def __copy__(self) -> Context:
        raise TypeError("Context handles cannot be copied")

    def __deepcopy__(self, _memo: object) -> Context:
        raise TypeError("Context handles cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Tensor:
    """An immutable float32 tensor owned by the native library."""

    __slots__ = ("_context", "_handle", "_lock", "__weakref__")

    def __init__(self) -> None:
        raise TypeError(
            "construct tensors with Tensor.from_data() or Tensor.zeros()"
        )

    @classmethod
    def _from_native(
        cls,
        context: Context,
        handle: _TensorHandle,
    ) -> Tensor:
        if not handle:
            raise RuntimeError(
                "native tensor operation succeeded without a handle"
            )
        try:
            tensor = cls.__new__(cls)
            tensor._lock = threading.RLock()
            tensor._context = context
            tensor._handle = handle
            return tensor
        except BaseException:
            _native().rt_tensor_release(handle)
            raise

    @classmethod
    def from_data(
        cls,
        context: Context,
        shape: Sequence[int] | Iterable[int],
        values: Iterable[float],
    ) -> Tensor:
        if not isinstance(context, Context):
            raise TypeError("context must be a Context")
        dimensions, native_shape = _shape_values(shape)
        try:
            copied_values = tuple(float(value) for value in values)
        except TypeError as error:
            raise TypeError("values must be an iterable of numbers") from error

        native_values = None
        if copied_values:
            array_type = ctypes.c_float * len(copied_values)
            native_values = array_type(*copied_values)

        with context._lock:
            output = _TensorHandle()
            _check(
                _native().rt_tensor_create_f32(
                    context._native_handle(),
                    native_shape,
                    len(dimensions),
                    native_values,
                    len(copied_values),
                    ctypes.byref(output),
                )
            )
        return cls._from_native(context, output)

    @classmethod
    def zeros(
        cls,
        context: Context,
        shape: Sequence[int] | Iterable[int],
    ) -> Tensor:
        if not isinstance(context, Context):
            raise TypeError("context must be a Context")
        dimensions, native_shape = _shape_values(shape)
        with context._lock:
            output = _TensorHandle()
            _check(
                _native().rt_tensor_zeros_f32(
                    context._native_handle(),
                    native_shape,
                    len(dimensions),
                    ctypes.byref(output),
                )
            )
        return cls._from_native(context, output)

    def _native_handle(self) -> _TensorHandle:
        if self._handle is None:
            raise RuntimeError("tensor is closed")
        return self._handle

    @property
    def context(self) -> Context:
        with self._lock:
            context = self._context
            if context is None:
                raise RuntimeError("tensor is closed")
            return context

    @property
    def backend(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_tensor_backend(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _backend_name(int(output.value))

    @property
    def rank(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_tensor_rank(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    @property
    def shape(self) -> tuple[int, ...]:
        with self._lock:
            rank = self.rank
            if rank == 0:
                _check(
                    _native().rt_tensor_shape(
                        self._native_handle(),
                        None,
                        0,
                    )
                )
                return ()

            array_type = ctypes.c_uint64 * rank
            output = array_type()
            _check(
                _native().rt_tensor_shape(
                    self._native_handle(),
                    output,
                    rank,
                )
            )
            return tuple(int(value) for value in output)

    @property
    def numel(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_tensor_numel(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    def tolist(self) -> list[float]:
        with self._lock:
            count = self.numel
            array_type = ctypes.c_float * count
            output = array_type()
            _check(
                _native().rt_tensor_copy_to_host_f32(
                    self._native_handle(),
                    output,
                    count,
                )
            )
            return [float(value) for value in output]

    def matmul(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            raise TypeError("matmul operand must be a Tensor")

        def run_locked() -> Tensor:
            context = self.context
            output = _TensorHandle()
            _check(
                _native().rt_tensor_matmul(
                    self._native_handle(),
                    other._native_handle(),
                    ctypes.byref(output),
                )
            )
            return type(self)._from_native(context, output)

        if self is other:
            with self._lock:
                return run_locked()

        first, second = (
            (self, other)
            if id(self) < id(other)
            else (other, self)
        )
        with first._lock:
            with second._lock:
                return run_locked()

    def __matmul__(self, other: object) -> Tensor:
        if not isinstance(other, Tensor):
            return NotImplemented
        return self.matmul(other)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native().rt_tensor_release(handle)
            self._context = None

    def __enter__(self) -> Tensor:
        with self._lock:
            self._native_handle()
            return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def __copy__(self) -> Tensor:
        raise TypeError("Tensor handles cannot be copied")

    def __deepcopy__(self, _memo: object) -> Tensor:
        raise TypeError("Tensor handles cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MultilinearMap:
    """An immutable output-major multilinear map owned by the native runtime."""

    __slots__ = (
        "_handle",
        "_lock",
        "_input_dimensions",
        "_output_dimension",
        "_coefficient_count",
        "__weakref__",
    )

    def __init__(self) -> None:
        raise TypeError("construct maps with from_dense() or from_sparse()")

    @staticmethod
    def _checked_dimensions(
        input_dimensions: Iterable[object],
        output_dimension: object,
    ) -> tuple[tuple[int, ...], ctypes.Array[ctypes.c_uint64] | None, int, int]:
        dimensions, native_dimensions = _shape_values(input_dimensions)
        if any(dimension == 0 for dimension in dimensions):
            raise ValueError("map input dimensions must be greater than zero")
        output = _unsigned_integer(
            output_dimension, (1 << 64) - 1, "output_dimension"
        )
        if output == 0:
            raise ValueError("output_dimension must be greater than zero")
        coefficient_count = output
        for dimension in dimensions:
            coefficient_count *= dimension
            if coefficient_count > (1 << 64) - 1:
                raise OverflowError("multilinear coefficient count exceeds uint64")
        return dimensions, native_dimensions, output, coefficient_count

    @classmethod
    def from_dense(
        cls,
        input_dimensions: Iterable[object],
        output_dimension: object,
        coefficients: Iterable[object],
    ) -> MultilinearMap:
        dimensions, native_dimensions, output_dimension_value, expected = (
            cls._checked_dimensions(input_dimensions, output_dimension)
        )
        try:
            raw = tuple(coefficients)
        except TypeError as error:
            raise TypeError("coefficients must be an iterable") from error
        if len(raw) != expected:
            raise ValueError(f"coefficients must contain exactly {expected} values")
        converted: list[float] = []
        for index, item in enumerate(raw):
            if isinstance(item, bool):
                raise TypeError(f"coefficients[{index}] must be a number")
            try:
                value = float(item)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"coefficients[{index}] must be a number"
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"coefficients[{index}] must be finite")
            converted.append(value)
        array_type = ctypes.c_double * len(converted)
        native_coefficients = array_type(*converted)
        output = _MultilinearMapHandle()
        _check(
            _native().rt_multilinear_map_create_dense(
                native_dimensions,
                len(dimensions),
                output_dimension_value,
                native_coefficients,
                len(converted),
                ctypes.byref(output),
            )
        )
        return cls._from_native(
            output, dimensions, output_dimension_value, expected
        )

    @classmethod
    def from_sparse(
        cls,
        input_dimensions: Iterable[object],
        output_dimension: object,
        flat_indices: Iterable[object],
        values: Iterable[object],
    ) -> MultilinearMap:
        dimensions, native_dimensions, output_dimension_value, expected = (
            cls._checked_dimensions(input_dimensions, output_dimension)
        )
        try:
            raw_indices = tuple(flat_indices)
            raw_values = tuple(values)
        except TypeError as error:
            raise TypeError("sparse indices and values must be iterables") from error
        if len(raw_indices) != len(raw_values):
            raise ValueError("flat_indices and values must have the same length")
        indices = tuple(
            _unsigned_integer(value, (1 << 64) - 1, "flat index")
            for value in raw_indices
        )
        converted: list[float] = []
        for index, item in enumerate(raw_values):
            if isinstance(item, bool):
                raise TypeError(f"values[{index}] must be a number")
            try:
                value = float(item)
            except (TypeError, ValueError) as error:
                raise TypeError(f"values[{index}] must be a number") from error
            if not math.isfinite(value):
                raise ValueError(f"values[{index}] must be finite")
            if value == 0.0:
                raise ValueError(f"values[{index}] must be nonzero")
            converted.append(value)
        index_type = ctypes.c_uint64 * len(indices)
        value_type = ctypes.c_double * len(converted)
        native_indices = index_type(*indices) if indices else None
        native_values = value_type(*converted) if converted else None
        output = _MultilinearMapHandle()
        _check(
            _native().rt_multilinear_map_create_sparse(
                native_dimensions,
                len(dimensions),
                output_dimension_value,
                native_indices,
                native_values,
                len(indices),
                ctypes.byref(output),
            )
        )
        return cls._from_native(
            output, dimensions, output_dimension_value, expected
        )

    @classmethod
    def _from_native(
        cls,
        handle: _MultilinearMapHandle,
        input_dimensions: tuple[int, ...],
        output_dimension: int,
        coefficient_count: int,
    ) -> MultilinearMap:
        if not handle:
            raise RuntimeError("native map creation succeeded without a handle")
        try:
            result = cls.__new__(cls)
            result._handle = None
            result._lock = threading.RLock()
            result._input_dimensions = input_dimensions
            result._output_dimension = output_dimension
            result._coefficient_count = coefficient_count
            result._handle = handle
            return result
        except BaseException:
            _native().rt_multilinear_map_release(handle)
            raise

    def _native_handle(self) -> _MultilinearMapHandle:
        if self._handle is None:
            raise RuntimeError("multilinear map is closed")
        return self._handle

    @property
    def input_dimensions(self) -> tuple[int, ...]:
        return self._input_dimensions

    @property
    def output_dimension(self) -> int:
        return self._output_dimension

    @property
    def coefficient_count(self) -> int:
        return self._coefficient_count

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native().rt_multilinear_map_release(handle)

    def __enter__(self) -> MultilinearMap:
        self._native_handle()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def __copy__(self) -> MultilinearMap:
        raise TypeError("multilinear-map handles cannot be copied")

    def __deepcopy__(self, _memo: object) -> MultilinearMap:
        raise TypeError("multilinear-map handles cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _copy_representation_trace(
    handle: _RepresentationTraceHandle,
) -> RepresentationTrace:
    count = ctypes.c_uint64()
    _check(_native().rt_representation_trace_count(handle, ctypes.byref(count)))
    entries: list[Representation] = []
    for index in range(int(count.value)):
        required = ctypes.c_uint64()
        _check(
            _native().rt_representation_trace_name(
                handle, index, None, 0, ctypes.byref(required)
            )
        )
        name_buffer = ctypes.create_string_buffer(int(required.value))
        _check(
            _native().rt_representation_trace_name(
                handle,
                index,
                name_buffer,
                len(name_buffer),
                ctypes.byref(required),
            )
        )
        rank = ctypes.c_uint64()
        _check(
            _native().rt_representation_trace_rank(
                handle, index, ctypes.byref(rank)
            )
        )
        shape_type = ctypes.c_uint64 * int(rank.value)
        native_shape = shape_type()
        _check(
            _native().rt_representation_trace_shape(
                handle, index, native_shape, len(native_shape)
            )
        )
        numel = ctypes.c_uint64()
        _check(
            _native().rt_representation_trace_numel(
                handle, index, ctypes.byref(numel)
            )
        )
        value_type = ctypes.c_float * int(numel.value)
        native_values = value_type()
        _check(
            _native().rt_representation_trace_copy_to_host_f32(
                handle, index, native_values, len(native_values)
            )
        )
        entries.append(
            Representation(
                name=name_buffer.value.decode("utf-8"),
                shape=tuple(int(value) for value in native_shape),
                values=tuple(float(value) for value in native_values),
            )
        )
    return RepresentationTrace(tuple(entries))


class ProgramAugmentedModel:
    """A learned sequence model composed with an optional compiled program."""

    __slots__ = ("_handle", "_lock", "_config", "_program", "__weakref__")

    def __init__(
        self,
        config: ProgramAugmentedModelConfig,
        program: ProgramBranch | None = None,
    ) -> None:
        if not isinstance(config, ProgramAugmentedModelConfig):
            raise TypeError("config must be a ProgramAugmentedModelConfig")
        if program is not None and not isinstance(program, ProgramBranch):
            raise TypeError("program must be a ProgramBranch or None")

        # Establish a destructible empty state before any native allocation.
        # The handle is adopted only after every fallible Python-side field has
        # been initialized, mirroring Tensor._from_native's ownership pattern.
        self._handle: _ProgramAugmentedModelHandle | None = None
        self._lock = threading.RLock()
        self._config = config
        self._program = program

        if program is not None:
            if program.source_offset + program.source_length > config.context_length:
                raise ValueError("program source span exceeds context_length")
            if program.target_offset + program.output_length > config.context_length:
                raise ValueError("program target span exceeds context_length")

        native_config = _NativeProgramAugmentedModelConfig()
        _check(
            _native().rt_program_augmented_model_config_init(
                ctypes.byref(native_config), ctypes.sizeof(native_config)
            )
        )
        native_config.vocabulary_size = config.vocabulary_size
        native_config.context_length = config.context_length
        native_config.model_width = config.model_width
        native_config.head_count = config.head_count
        native_config.attention_branch_count = config.attention_branch_count
        native_config.feed_forward_width = config.feed_forward_width
        native_config.random_seed = config.random_seed
        native_config.attention_kind = _FULL_SEQUENCE_ATTENTION_CODES[
            config.attention
        ]

        native_branch = None
        native_lowering = None
        native_layouts = None
        if program is not None:
            layout_type = _NativeProgramInputLayout * len(program.inputs)
            native_layouts = layout_type(
                *(
                    _NativeProgramInputLayout(
                        _PROGRAM_INPUT_CODES[item.source],
                        0,
                        item.position,
                        item.projection_group,
                    )
                    for item in program.inputs
                )
            )
            native_branch = _NativeProgramBranchConfig()
            _check(
                _native().rt_program_branch_config_init(
                    ctypes.byref(native_branch), ctypes.sizeof(native_branch)
                )
            )
            native_branch.source_offset = program.source_offset
            native_branch.source_length = program.source_length
            native_branch.target_offset = program.target_offset
            native_branch.output_length = program.output_length
            native_branch.inputs = native_layouts
            native_branch.input_count = len(program.inputs)
            native_branch.input_projection_bias = int(
                program.input_projection_bias
            )
            native_branch.merge_bias = int(program.merge_bias)

            lowering = program.lowering
            native_lowering = _NativeNeuralLoweringOptions()
            _check(
                _native().rt_neural_lowering_options_init(
                    ctypes.byref(native_lowering),
                    ctypes.sizeof(native_lowering),
                )
            )
            native_lowering.strategy = _LOWERING_STRATEGY_CODES[
                lowering.strategy
            ]
            native_lowering.precision = _COEFFICIENT_PRECISION_CODES[
                lowering.precision
            ]
            native_lowering.initialization = (
                _COEFFICIENT_INITIALIZATION_CODES[lowering.initialization]
            )
            native_lowering.trainable = int(lowering.trainable)
            native_lowering.random_seed = lowering.random_seed
            native_lowering.random_scale = lowering.random_scale
            native_lowering.max_coefficient_elements = (
                lowering.max_coefficient_elements
            )
            native_lowering.attention_query_axis = (
                -1
                if lowering.attention_query_axis is None
                else lowering.attention_query_axis
            )

        output = _ProgramAugmentedModelHandle()
        if program is None:
            status = _native().rt_program_augmented_model_create(
                ctypes.byref(native_config),
                None,
                None,
                None,
                ctypes.byref(output),
            )
        else:
            # ctypes may release the GIL for the native lowering call. Keep the
            # map handle alive until that call has copied and lowered it.
            with program.map._lock:
                status = _native().rt_program_augmented_model_create(
                    ctypes.byref(native_config),
                    program.map._native_handle(),
                    ctypes.byref(native_branch),
                    ctypes.byref(native_lowering),
                    ctypes.byref(output),
                )
        try:
            _check(status)
            if not output:
                raise RuntimeError(
                    "native model creation succeeded without a handle"
                )
        except BaseException:
            if output:
                _native().rt_program_augmented_model_release(output)
            raise
        self._handle = output

    def _native_handle(self) -> _ProgramAugmentedModelHandle:
        if self._handle is None:
            raise RuntimeError("program-augmented model is closed")
        return self._handle

    @property
    def config(self) -> ProgramAugmentedModelConfig:
        return self._config

    @property
    def has_program(self) -> bool:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_program_augmented_model_has_program(
                    self._native_handle(), ctypes.byref(output)
                )
            )
            return bool(output.value)

    @property
    def backend(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_program_augmented_model_backend(
                    self._native_handle(), ctypes.byref(output)
                )
            )
            return _backend_name(int(output.value))

    def to(self, backend: str | int) -> ProgramAugmentedModel:
        with self._lock:
            _check(
                _native().rt_program_augmented_model_to(
                    self._native_handle(), _backend_code(backend)
                )
            )
        return self

    def parameters(self) -> ParameterList:
        with self._lock:
            output = _ParameterListHandle()
            _check(
                _native().rt_program_augmented_model_parameters(
                    self._native_handle(), ctypes.byref(output)
                )
            )
            return ParameterList._from_native(self, output)

    def forward(
        self,
        tokens: Iterable[object],
        options: ProgramAugmentedForwardOptions | None = None,
    ) -> ProgramAugmentedForwardResult:
        configured = ProgramAugmentedForwardOptions() if options is None else options
        if not isinstance(configured, ProgramAugmentedForwardOptions):
            raise TypeError("options must be ProgramAugmentedForwardOptions or None")
        values, batch_size, sequence_length = _token_matrix(tokens, "tokens")
        if sequence_length != self._config.context_length:
            raise ValueError("tokens must match the configured context_length")
        native_tokens = _native_u32_values(values)

        scale_arrays: list[ctypes.Array[ctypes.c_float]] = []
        offset_arrays: list[ctypes.Array[ctypes.c_float]] = []
        steering_values: list[_NativeProgramInputSteering] = []
        for steering in configured.steering:
            scale_type = ctypes.c_float * len(steering.scales)
            offset_type = ctypes.c_float * len(steering.offsets)
            native_scales = scale_type(*steering.scales)
            native_offsets = offset_type(*steering.offsets)
            scale_arrays.append(native_scales)
            offset_arrays.append(native_offsets)
            steering_values.append(
                _NativeProgramInputSteering(
                    steering.input_index,
                    steering.position,
                    native_scales if steering.scales else None,
                    len(steering.scales),
                    native_offsets if steering.offsets else None,
                    len(steering.offsets),
                )
            )
        steering_type = _NativeProgramInputSteering * len(steering_values)
        native_steering = (
            steering_type(*steering_values) if steering_values else None
        )
        ablation_type = ctypes.c_uint64 * len(
            configured.ablated_program_inputs
        )
        native_ablations = (
            ablation_type(*configured.ablated_program_inputs)
            if configured.ablated_program_inputs
            else None
        )
        native_options = _NativeProgramAugmentedForwardOptions()
        _check(
            _native().rt_program_augmented_forward_options_init(
                ctypes.byref(native_options), ctypes.sizeof(native_options)
            )
        )
        native_options.capture_representations = int(
            configured.capture_representations
        )
        native_options.batch_roll_learned_attention = int(
            configured.batch_roll_learned_attention
        )
        native_options.batch_roll_shift = configured.batch_roll_shift
        native_options.steering = native_steering
        native_options.steering_count = len(steering_values)
        native_options.ablated_program_inputs = native_ablations
        native_options.ablated_program_input_count = len(
            configured.ablated_program_inputs
        )
        native_options.ablate_program_output = int(
            configured.ablate_program_output
        )

        with self._lock:
            logits = _VariableHandle()
            trace = _RepresentationTraceHandle()
            _check(
                _native().rt_program_augmented_model_forward(
                    self._native_handle(),
                    native_tokens,
                    len(values),
                    batch_size,
                    sequence_length,
                    ctypes.byref(native_options),
                    ctypes.byref(logits),
                    ctypes.byref(trace)
                    if configured.capture_representations
                    else None,
                )
            )
            try:
                representations = (
                    _copy_representation_trace(trace)
                    if configured.capture_representations
                    else RepresentationTrace()
                )
            except BaseException:
                if logits:
                    _native().rt_variable_release(logits)
                raise
            finally:
                if trace:
                    _native().rt_representation_trace_release(trace)
            variable = Variable._from_native(logits, self, self._lock)
            return ProgramAugmentedForwardResult(variable, representations)

    def __call__(self, tokens: Iterable[object]) -> Variable:
        return self.forward(tokens).logits

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native().rt_program_augmented_model_release(handle)

    def __enter__(self) -> ProgramAugmentedModel:
        self._native_handle()
        return self

    def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
        self.close()

    def __copy__(self) -> ProgramAugmentedModel:
        raise TypeError("program-augmented model handles cannot be copied")

    def __deepcopy__(self, _memo: object) -> ProgramAugmentedModel:
        raise TypeError("program-augmented model handles cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DecoderOnlyTransformer:
    """A native decoder-only Transformer with backend-independent layers."""

    __slots__ = ("_config", "_handle", "_lock", "__weakref__")

    def __init__(
        self,
        config: TransformerConfig,
        *,
        attention: str | int = "materialized",
        activation_checkpointing: str | int = "disabled",
    ) -> None:
        if not isinstance(config, TransformerConfig):
            raise TypeError("config must be a TransformerConfig")
        attention_code = _full_sequence_attention_code(attention)
        activation_checkpointing_code = _activation_checkpointing_code(
            activation_checkpointing
        )

        self._handle: _ModelHandle | None = None
        self._lock = threading.RLock()
        native_config = _NativeTransformerConfig()
        _check(
            _native().rt_transformer_config_init(
                ctypes.byref(native_config),
                ctypes.sizeof(native_config),
            )
        )
        maximum_u64 = (1 << 64) - 1
        native_config.vocabulary_size = _unsigned_integer(
            config.vocabulary_size,
            maximum_u64,
            "vocabulary_size",
        )
        native_config.maximum_context = _unsigned_integer(
            config.maximum_context,
            maximum_u64,
            "maximum_context",
        )
        native_config.model_width = _unsigned_integer(
            config.model_width,
            maximum_u64,
            "model_width",
        )
        native_config.head_count = _unsigned_integer(
            config.head_count,
            maximum_u64,
            "head_count",
        )
        native_config.block_count = _unsigned_integer(
            config.block_count,
            maximum_u64,
            "block_count",
        )
        native_config.feed_forward_width = _unsigned_integer(
            config.feed_forward_width,
            maximum_u64,
            "feed_forward_width",
        )
        native_config.random_seed = _unsigned_integer(
            config.random_seed,
            (1 << 32) - 1,
            "random_seed",
        )
        epsilon = float(config.layer_norm_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError(
                "layer_norm_epsilon must be finite and positive"
            )
        native_config.layer_norm_epsilon = epsilon

        output = _ModelHandle()
        _check(
            _native().rt_model_create(
                ctypes.byref(native_config),
                ctypes.byref(output),
            )
        )
        if not output:
            raise RuntimeError(
                "native model creation succeeded without a handle"
            )
        try:
            _check(
                _native().rt_model_set_full_sequence_attention(
                    output,
                    attention_code,
                )
            )
            _check(
                _native().rt_model_set_activation_checkpointing(
                    output,
                    activation_checkpointing_code,
                )
            )
        except BaseException:
            _native().rt_model_release(output)
            raise
        self._config = config
        self._handle = output

    def _native_handle(self) -> _ModelHandle:
        if self._handle is None:
            raise RuntimeError("model is closed")
        return self._handle

    @property
    def config(self) -> TransformerConfig:
        return self._config

    @property
    def backend(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_model_backend(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _backend_name(int(output.value))

    @property
    def full_sequence_attention(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_model_full_sequence_attention(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _full_sequence_attention_name(int(output.value))

    def set_full_sequence_attention(
        self,
        attention: str | int,
    ) -> DecoderOnlyTransformer:
        """Select the implementation used by future full-sequence forwards."""

        attention_code = _full_sequence_attention_code(attention)
        with self._lock:
            _check(
                _native().rt_model_set_full_sequence_attention(
                    self._native_handle(),
                    attention_code,
                )
            )
        return self

    @property
    def activation_checkpointing(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_model_activation_checkpointing(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _activation_checkpointing_name(int(output.value))

    def set_activation_checkpointing(
        self,
        activation_checkpointing: str | int,
    ) -> DecoderOnlyTransformer:
        """Select activation retention for future training forwards."""

        checkpointing_code = _activation_checkpointing_code(
            activation_checkpointing
        )
        with self._lock:
            _check(
                _native().rt_model_set_activation_checkpointing(
                    self._native_handle(),
                    checkpointing_code,
                )
            )
        return self

    def to(self, backend: str | int) -> DecoderOnlyTransformer:
        """Move all parameters transactionally and return this model."""

        backend_code = _backend_code(backend)
        with self._lock:
            _check(
                _native().rt_model_to(
                    self._native_handle(),
                    backend_code,
                )
            )
        return self

    def attach_lora(
        self,
        config: LoraConfig | None = None,
    ) -> DecoderOnlyTransformer:
        """Attach one trainable low-rank adapter and return this model."""

        configured = LoraConfig() if config is None else config
        if not isinstance(configured, LoraConfig):
            raise TypeError("config must be a LoraConfig or None")

        native_config = _NativeLoraConfig()
        _check(
            _native().rt_lora_config_init(
                ctypes.byref(native_config),
                ctypes.sizeof(native_config),
            )
        )
        _targets, target_mask = _lora_targets(configured.targets)
        native_config.rank = configured.rank
        native_config.alpha = configured.alpha
        native_config.random_seed = configured.random_seed
        native_config.targets = target_mask

        with self._lock:
            _check(
                _native().rt_model_attach_lora(
                    self._native_handle(),
                    ctypes.byref(native_config),
                )
            )
        return self

    @property
    def lora_attached(self) -> bool:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_model_has_lora(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return bool(output.value)

    @property
    def lora_config(self) -> LoraConfig | None:
        with self._lock:
            if not self.lora_attached:
                return None
            output = _NativeLoraConfig()
            output.struct_size = ctypes.sizeof(output)
            _check(
                _native().rt_model_lora_config(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return LoraConfig(
                rank=int(output.rank),
                alpha=float(output.alpha),
                targets=_lora_target_names(int(output.targets)),
                random_seed=int(output.random_seed),
            )

    def quantize_nf4(
        self,
        block_size: int = 64,
        *,
        double_quantization: bool = True,
        scale_block_size: int = 256,
    ) -> DecoderOnlyTransformer:
        """Pack eligible Linear weights as immutable blockwise NF4."""

        checked_block_size = _unsigned_integer(
            block_size,
            (1 << 64) - 1,
            "block_size",
        )
        if checked_block_size == 0:
            raise ValueError("block_size must be greater than zero")
        if not isinstance(double_quantization, bool):
            raise TypeError("double_quantization must be a bool")
        checked_scale_block_size = _unsigned_integer(
            scale_block_size,
            (1 << 64) - 1,
            "scale_block_size",
        )
        if checked_scale_block_size == 0:
            raise ValueError("scale_block_size must be greater than zero")
        with self._lock:
            if double_quantization:
                _check(
                    _native()
                    .rt_model_quantize_linear_weights_nf4_double_quantized(
                        self._native_handle(),
                        checked_block_size,
                        checked_scale_block_size,
                    )
                )
            else:
                _check(
                    _native().rt_model_quantize_linear_weights_nf4(
                        self._native_handle(),
                        checked_block_size,
                    )
                )
        return self

    @property
    def quantized_linear_weights(self) -> bool:
        """Return whether the model currently owns packed Linear weights."""

        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_model_has_quantized_linear_weights(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return bool(output.value)

    @property
    def quantized_memory(self) -> QuantizedMemoryStats:
        """Return exact persistent packed-weight memory accounting."""

        with self._lock:
            output = _NativeQuantizedMemoryStats()
            output.struct_size = ctypes.sizeof(output)
            _check(
                _native().rt_model_quantized_memory_stats(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return QuantizedMemoryStats(
                weight_count=int(output.weight_count),
                packed_code_bytes=int(output.packed_code_bytes),
                scale_bytes=int(output.scale_bytes),
                logical_payload_bytes=int(output.logical_payload_bytes),
                resident_payload_bytes=int(output.resident_payload_bytes),
                fp32_equivalent_bytes=int(output.fp32_equivalent_bytes),
                fp32_scale_bytes=int(output.fp32_scale_bytes),
                scale_code_bytes=int(output.scale_code_bytes),
                second_level_scale_bytes=int(
                    output.second_level_scale_bytes
                ),
                scale_offset_bytes=int(output.scale_offset_bytes),
                double_quantized_weight_count=int(
                    output.double_quantized_weight_count
                ),
            )

    def parameters(self) -> ParameterList:
        """Return the stable base-model parameter collection."""

        with self._lock:
            output = _ParameterListHandle()
            _check(
                _native().rt_model_parameters(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return ParameterList._from_native(self, output)

    def adapter_parameters(self) -> ParameterList:
        """Return parameters belonging to the attached LoRA adapter."""

        with self._lock:
            output = _ParameterListHandle()
            _check(
                _native().rt_model_lora_parameters(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return ParameterList._from_native(self, output)

    def merge_lora(self) -> DecoderOnlyTransformer:
        """Merge the attached adapter into base weights and return this model."""

        with self._lock:
            _check(
                _native().rt_model_merge_lora(
                    self._native_handle()
                )
            )
        return self

    def decode_session(
        self,
        *,
        cache: str | int = "paged",
        block_size: int = 16,
    ) -> DecodeSession:
        """Create a request-local incremental decoding session."""

        cache_code = _kv_cache_code(cache)
        configured_block_size = _unsigned_integer(
            block_size,
            (1 << 64) - 1,
            "block_size",
        )
        if configured_block_size == 0:
            raise ValueError("block_size must be greater than zero")

        options = _NativeDecodeSessionOptions()
        _check(
            _native().rt_decode_session_options_init(
                ctypes.byref(options),
                ctypes.sizeof(options),
            )
        )
        options.kind = cache_code
        options.block_size = configured_block_size

        with self._lock:
            output = _DecodeSessionHandle()
            _check(
                _native().rt_model_decode_session_create(
                    self._native_handle(),
                    ctypes.byref(options),
                    ctypes.byref(output),
                )
            )
            return DecodeSession._from_native(
                output,
                self,
                self._lock,
                self._config.vocabulary_size,
            )

    def __call__(self, tokens: Iterable[object]) -> Variable:
        values, batch_size, sequence_length = _token_matrix(
            tokens,
            "tokens",
        )
        native_values = _native_u32_values(values)
        with self._lock:
            output = _VariableHandle()
            _check(
                _native().rt_model_forward(
                    self._native_handle(),
                    native_values,
                    len(values),
                    batch_size,
                    sequence_length,
                    ctypes.byref(output),
                )
            )
            return Variable._from_native(
                output,
                self,
                self._lock,
            )

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native().rt_model_release(handle)

    def __enter__(self) -> DecoderOnlyTransformer:
        with self._lock:
            self._native_handle()
            return self

    def __exit__(
        self,
        _type: object,
        _value: object,
        _traceback: object,
    ) -> None:
        self.close()

    def __copy__(self) -> DecoderOnlyTransformer:
        raise TypeError("model handles cannot be copied")

    def __deepcopy__(self, _memo: object) -> DecoderOnlyTransformer:
        raise TypeError("model handles cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DecodeSession:
    """A model-owning, single-sequence incremental decoding cache."""

    __slots__ = (
        "_handle",
        "_lock",
        "_model",
        "_vocabulary_size",
        "__weakref__",
    )

    def __init__(self) -> None:
        raise TypeError(
            "construct decode sessions with model.decode_session()"
        )

    @classmethod
    def _from_native(
        cls,
        handle: _DecodeSessionHandle,
        model: DecoderOnlyTransformer,
        lock: threading.RLock,
        vocabulary_size: int,
    ) -> DecodeSession:
        if not handle:
            raise RuntimeError(
                "native decode-session creation succeeded without a handle"
            )
        try:
            result = cls.__new__(cls)
            result._handle = handle
            result._model = model
            result._lock = lock
            result._vocabulary_size = vocabulary_size
            return result
        except BaseException:
            _native().rt_decode_session_release(handle)
            raise

    def _native_handle(self) -> _DecodeSessionHandle:
        if self._handle is None:
            raise RuntimeError("decode session is closed")
        return self._handle

    @property
    def size(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_decode_session_size(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    @property
    def capacity(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_decode_session_capacity(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    @property
    def cache(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_decode_session_cache_kind(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _kv_cache_name(int(output.value))

    @property
    def cache_kind(self) -> str:
        """Alias for ``cache``."""

        return self.cache

    @property
    def block_size(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_decode_session_block_size(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    def reset(self) -> None:
        with self._lock:
            _check(
                _native().rt_decode_session_reset(
                    self._native_handle()
                )
            )

    def step(self, token: object) -> tuple[float, ...]:
        """Append one token and return logits predicting the next token."""

        token_id = _token_id(token, "token")
        with self._lock:
            count = self._vocabulary_size
            array_type = ctypes.c_float * count
            output = array_type()
            required = ctypes.c_uint64()
            _check(
                _native().rt_decode_session_step(
                    self._native_handle(),
                    token_id,
                    output,
                    count,
                    ctypes.byref(required),
                )
            )
            if int(required.value) != count:
                raise RuntimeError(
                    "native decode-session vocabulary size changed"
                )
            return tuple(float(value) for value in output)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native().rt_decode_session_release(handle)
            self._model = None

    def __enter__(self) -> DecodeSession:
        with self._lock:
            self._native_handle()
            return self

    def __exit__(
        self,
        _type: object,
        _value: object,
        _traceback: object,
    ) -> None:
        self.close()

    def __copy__(self) -> DecodeSession:
        raise TypeError("decode-session handles cannot be copied")

    def __deepcopy__(self, _memo: object) -> DecodeSession:
        raise TypeError("decode-session handles cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ParameterList:
    """A native-module-owning view of trainable parameters."""

    __slots__ = ("_handle", "_lock", "_model", "__weakref__")

    def __init__(self) -> None:
        raise TypeError(
            "construct parameter lists with model.parameters() or "
            "model.adapter_parameters()"
        )

    @classmethod
    def _from_native(
        cls,
        model: object,
        handle: _ParameterListHandle,
    ) -> ParameterList:
        if not handle:
            raise RuntimeError(
                "native parameter query succeeded without a handle"
            )
        try:
            result = cls.__new__(cls)
            result._handle = handle
            result._model = model
            result._lock = model._lock  # type: ignore[attr-defined]
            return result
        except BaseException:
            _native().rt_parameter_list_release(handle)
            raise

    def _native_handle(self) -> _ParameterListHandle:
        if self._handle is None:
            raise RuntimeError("parameter list is closed")
        return self._handle

    def __len__(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_parameter_list_count(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    @property
    def backend(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_parameter_list_backend(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _backend_name(int(output.value))

    @property
    def names(self) -> tuple[str, ...]:
        with self._lock:
            names: list[str] = []
            for index in range(len(self)):
                required = ctypes.c_uint64()
                _check(
                    _native().rt_parameter_list_name(
                        self._native_handle(),
                        index,
                        None,
                        0,
                        ctypes.byref(required),
                    )
                )
                buffer = ctypes.create_string_buffer(
                    int(required.value)
                )
                _check(
                    _native().rt_parameter_list_name(
                        self._native_handle(),
                        index,
                        buffer,
                        len(buffer),
                        ctypes.byref(required),
                    )
                )
                names.append(
                    buffer.value.decode(
                        "utf-8",
                        errors="strict",
                    )
                )
            return tuple(names)

    @property
    def shapes(self) -> tuple[tuple[int, ...], ...]:
        """Return parameter tensor shapes in stable parameter-list order."""

        with self._lock:
            shapes: list[tuple[int, ...]] = []
            for index in range(len(self)):
                rank = ctypes.c_uint64()
                _check(
                    _native().rt_parameter_list_rank(
                        self._native_handle(),
                        index,
                        ctypes.byref(rank),
                    )
                )
                native_rank = int(rank.value)
                if native_rank == 0:
                    _check(
                        _native().rt_parameter_list_shape(
                            self._native_handle(),
                            index,
                            None,
                            0,
                        )
                    )
                    shapes.append(())
                    continue
                array_type = ctypes.c_uint64 * native_rank
                dimensions = array_type()
                _check(
                    _native().rt_parameter_list_shape(
                        self._native_handle(),
                        index,
                        dimensions,
                        native_rank,
                    )
                )
                shapes.append(
                    tuple(int(dimension) for dimension in dimensions)
                )
            return tuple(shapes)

    @property
    def total_numel(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_parameter_list_total_numel(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    def flat_values(self) -> tuple[float, ...]:
        """Copy all parameters to host in stable list/flat tensor order."""

        with self._lock:
            count = self.total_numel
            array_type = ctypes.c_float * count
            output = array_type() if count else None
            _check(
                _native().rt_parameter_list_copy_to_host_f32(
                    self._native_handle(),
                    output,
                    count,
                )
            )
            return (
                ()
                if output is None
                else tuple(float(value) for value in output)
            )

    def load_flat_values(self, values: Iterable[object]) -> None:
        """Transactionally replace every parameter from flattened float32."""

        try:
            raw_values = tuple(values)
        except TypeError as error:
            raise TypeError("values must be an iterable of numbers") from error
        converted: list[float] = []
        for index, value in enumerate(raw_values):
            if isinstance(value, bool):
                raise TypeError(f"values[{index}] must be a number")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise TypeError(
                    f"values[{index}] must be a number"
                ) from error
            if not math.isfinite(number):
                raise ValueError(f"values[{index}] must be finite")
            converted.append(number)

        with self._lock:
            expected = self.total_numel
            if len(converted) != expected:
                raise ValueError(
                    f"values must contain exactly {expected} floats"
                )
            array_type = ctypes.c_float * len(converted)
            native_values = (
                array_type(*converted) if converted else None
            )
            _check(
                _native().rt_parameter_list_load_from_host_f32(
                    self._native_handle(),
                    native_values,
                    len(converted),
                )
            )

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native().rt_parameter_list_release(handle)

    def __enter__(self) -> ParameterList:
        with self._lock:
            self._native_handle()
            return self

    def __exit__(
        self,
        _type: object,
        _value: object,
        _traceback: object,
    ) -> None:
        self.close()

    def __copy__(self) -> ParameterList:
        raise TypeError("parameter-list handles cannot be copied")

    def __deepcopy__(self, _memo: object) -> ParameterList:
        raise TypeError("parameter-list handles cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class Variable:
    """A native tensor value and its reverse-mode computation graph."""

    __slots__ = ("_handle", "_lock", "_owner", "__weakref__")

    def __init__(self) -> None:
        raise TypeError("variables are returned by models and operations")

    @classmethod
    def _from_native(
        cls,
        handle: _VariableHandle,
        owner: object,
        lock: threading.RLock,
    ) -> Variable:
        if not handle:
            raise RuntimeError(
                "native variable operation succeeded without a handle"
            )
        try:
            result = cls.__new__(cls)
            result._handle = handle
            result._owner = owner
            result._lock = lock
            return result
        except BaseException:
            _native().rt_variable_release(handle)
            raise

    def _native_handle(self) -> _VariableHandle:
        if self._handle is None:
            raise RuntimeError("variable is closed")
        return self._handle

    @property
    def backend(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_variable_backend(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _backend_name(int(output.value))

    @property
    def rank(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_variable_rank(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    @property
    def shape(self) -> tuple[int, ...]:
        with self._lock:
            rank = self.rank
            if rank == 0:
                _check(
                    _native().rt_variable_shape(
                        self._native_handle(),
                        None,
                        0,
                    )
                )
                return ()
            array_type = ctypes.c_uint64 * rank
            output = array_type()
            _check(
                _native().rt_variable_shape(
                    self._native_handle(),
                    output,
                    rank,
                )
            )
            return tuple(int(value) for value in output)

    @property
    def numel(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_variable_numel(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    def tolist(self) -> list[float]:
        with self._lock:
            count = self.numel
            array_type = ctypes.c_float * count
            output = array_type()
            _check(
                _native().rt_variable_copy_to_host_f32(
                    self._native_handle(),
                    output,
                    count,
                )
            )
            return [float(value) for value in output]

    def item(self) -> float:
        if self.numel != 1:
            raise ValueError("item() requires exactly one value")
        return self.tolist()[0]

    def backward(self) -> None:
        with self._lock:
            _check(
                _native().rt_variable_backward(
                    self._native_handle()
                )
            )

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native().rt_variable_release(handle)
            self._owner = None

    def __enter__(self) -> Variable:
        with self._lock:
            self._native_handle()
            return self

    def __exit__(
        self,
        _type: object,
        _value: object,
        _traceback: object,
    ) -> None:
        self.close()

    def __copy__(self) -> Variable:
        raise TypeError("variable handles cannot be copied")

    def __deepcopy__(self, _memo: object) -> Variable:
        raise TypeError("variable handles cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def cross_entropy(
    logits: Variable,
    targets: Iterable[object],
) -> Variable:
    """Return mean cross-entropy for flattened or rectangular targets."""

    if not isinstance(logits, Variable):
        raise TypeError("logits must be a Variable")
    values, _batch_size, _sequence_length = _token_matrix(
        targets,
        "targets",
    )
    native_values = _native_u32_values(values)
    with logits._lock:
        output = _VariableHandle()
        _check(
            _native().rt_cross_entropy(
                logits._native_handle(),
                native_values,
                len(values),
                ctypes.byref(output),
            )
        )
        return Variable._from_native(
            output,
            logits,
            logits._lock,
        )


def cross_entropy_time_range(
    logits: Variable,
    targets: Iterable[object],
    time_offset: int,
    time_count: int,
) -> Variable:
    """Return mean cross entropy for one contiguous time span per batch."""

    if not isinstance(logits, Variable):
        raise TypeError("logits must be a Variable")
    values, _batch_size, _sequence_length = _token_matrix(
        targets,
        "targets",
    )
    offset = _unsigned_integer(
        time_offset, (1 << 64) - 1, "time_offset"
    )
    count = _unsigned_integer(time_count, (1 << 64) - 1, "time_count")
    if count == 0:
        raise ValueError("time_count must be greater than zero")
    native_values = _native_u32_values(values)
    with logits._lock:
        output = _VariableHandle()
        _check(
            _native().rt_cross_entropy_time_range(
                logits._native_handle(),
                native_values,
                len(values),
                offset,
                count,
                ctypes.byref(output),
            )
        )
        return Variable._from_native(output, logits, logits._lock)


class Adam:
    """Adam for a base-model or LoRA adapter parameter collection."""

    __slots__ = ("_handle", "_lock", "_parameters", "__weakref__")

    def __init__(
        self,
        parameters: ParameterList,
        *,
        learning_rate: float = 1.0e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1.0e-8,
        maximum_gradient_norm: float = 1.0,
        state_storage: str | int = "contiguous",
        page_size: int = 4096,
    ) -> None:
        if not isinstance(parameters, ParameterList):
            raise TypeError(
                "parameters must be returned by model.parameters() or "
                "model.adapter_parameters()"
            )
        self._handle: _AdamHandle | None = None
        self._lock = parameters._lock
        self._parameters = parameters

        native_options = _NativeAdamOptions()
        _check(
            _native().rt_adam_options_init(
                ctypes.byref(native_options),
                ctypes.sizeof(native_options),
            )
        )
        native_options.learning_rate = float(learning_rate)
        native_options.beta1 = float(beta1)
        native_options.beta2 = float(beta2)
        native_options.epsilon = float(epsilon)
        native_options.maximum_gradient_norm = float(
            maximum_gradient_norm
        )
        native_options.state_storage = _adam_state_storage_code(
            state_storage
        )
        native_options.page_size = _unsigned_integer(
            page_size,
            (1 << 64) - 1,
            "page_size",
        )

        with self._lock:
            output = _AdamHandle()
            _check(
                _native().rt_adam_create(
                    parameters._native_handle(),
                    ctypes.byref(native_options),
                    ctypes.byref(output),
                )
            )
            if not output:
                raise RuntimeError(
                    "native Adam creation succeeded without a handle"
                )
            self._handle = output

    def _native_handle(self) -> _AdamHandle:
        if self._handle is None:
            raise RuntimeError("Adam optimizer is closed")
        return self._handle

    @property
    def backend(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_adam_backend(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _backend_name(int(output.value))

    @property
    def step_count(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_adam_step_count(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    @property
    def parameter_count(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_adam_parameter_count(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    @property
    def state_storage(self) -> str:
        with self._lock:
            output = ctypes.c_int32()
            _check(
                _native().rt_adam_state_storage(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return _adam_state_storage_name(int(output.value))

    @property
    def state_page_size(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_adam_state_page_size(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    @property
    def state_page_count(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_adam_state_page_count(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    @property
    def state_payload_bytes(self) -> int:
        with self._lock:
            output = ctypes.c_uint64()
            _check(
                _native().rt_adam_state_payload_bytes(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return int(output.value)

    def step(self) -> AdamStepStats:
        with self._lock:
            output = _NativeAdamStepStats()
            output.struct_size = ctypes.sizeof(output)
            _check(
                _native().rt_adam_step(
                    self._native_handle(),
                    ctypes.byref(output),
                )
            )
            return AdamStepStats(
                step=int(output.step),
                gradient_norm=float(output.gradient_norm),
                clip_scale=float(output.clip_scale),
            )

    def zero_gradients(self) -> None:
        with self._lock:
            _check(
                _native().rt_adam_zero_gradients(
                    self._native_handle()
                )
            )

    def zero_grad(self) -> None:
        self.zero_gradients()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._handle is None

    def close(self) -> None:
        with self._lock:
            handle = self._handle
            if handle is None:
                return
            self._handle = None
            _native().rt_adam_release(handle)

    def __enter__(self) -> Adam:
        with self._lock:
            self._native_handle()
            return self

    def __exit__(
        self,
        _type: object,
        _value: object,
        _traceback: object,
    ) -> None:
        self.close()

    def __copy__(self) -> Adam:
        raise TypeError("Adam handles cannot be copied")

    def __deepcopy__(self, _memo: object) -> Adam:
        raise TypeError("Adam handles cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "ADAM_STATE_CONTIGUOUS",
    "ADAM_STATE_PAGED",
    "ABI_VERSION",
    "ABI_VERSION_MAJOR",
    "ABI_VERSION_MINOR",
    "BACKEND_CPU",
    "BACKEND_METAL",
    "BACKEND_CUDA",
    "BACKEND_TPU",
    "ACTIVATION_CHECKPOINTING_DISABLED",
    "ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK",
    "FULL_SEQUENCE_ATTENTION_FLASH",
    "FULL_SEQUENCE_ATTENTION_MATERIALIZED",
    "COEFFICIENT_ALLOW_ROUNDED_F32",
    "COEFFICIENT_INITIALIZATION_COMPILED",
    "COEFFICIENT_INITIALIZATION_RANDOM_UNIFORM",
    "COEFFICIENT_REQUIRE_EXACT_F32",
    "KV_CACHE_CONTIGUOUS",
    "KV_CACHE_PAGED",
    "LORA_TARGET_ALL_LINEAR",
    "LORA_TARGET_ATTENTION_KEY",
    "LORA_TARGET_ATTENTION_OUTPUT",
    "LORA_TARGET_ATTENTION_QUERY",
    "LORA_TARGET_ATTENTION_VALUE",
    "LORA_TARGET_DEFAULT",
    "LORA_TARGET_FF_EXPAND",
    "LORA_TARGET_FF_PROJECT",
    "LORA_TARGET_LM_HEAD",
    "LORA_TARGET_NAMES",
    "LOWERING_STRATEGY_AUTO",
    "LOWERING_STRATEGY_DENSE",
    "LOWERING_STRATEGY_LINEAR",
    "LOWERING_STRATEGY_LINEAR_ATTENTION",
    "PROGRAM_INPUT_SOURCE_POSITION",
    "PROGRAM_INPUT_WHOLE_SOURCE",
    "TOKENIZER_METHOD_BPE",
    "TOKENIZER_METHOD_BYTE",
    "Adam",
    "AdamStepStats",
    "Context",
    "DecodeSession",
    "DecoderOnlyTransformer",
    "LoraConfig",
    "MultilinearMap",
    "NeuralLoweringConfig",
    "ParameterList",
    "ProgramAugmentedForwardOptions",
    "ProgramAugmentedForwardResult",
    "ProgramAugmentedModel",
    "ProgramAugmentedModelConfig",
    "ProgramBranch",
    "ProgramInputLayout",
    "ProgramInputSteering",
    "QuantizedMemoryStats",
    "Representation",
    "RepresentationTrace",
    "STATUS_BACKEND_UNAVAILABLE",
    "STATUS_INVALID_ARGUMENT",
    "STATUS_OK",
    "STATUS_OUT_OF_MEMORY",
    "STATUS_OUT_OF_RANGE",
    "STATUS_OVERFLOW",
    "STATUS_RUNTIME_ERROR",
    "STATUS_UNKNOWN_ERROR",
    "Tensor",
    "RiftcoTransformerError",
    "Tokenizer",
    "TransformerConfig",
    "Variable",
    "backend_available",
    "cross_entropy",
    "cross_entropy_time_range",
]
