"""Supervised post-training built on the shared causal training engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Protocol

from ..native import Adam, LoraConfig
from ..artifacts import ModelBundle
from ..training import (
    CausalLanguageModelTrainer,
    ExampleWindowBatchSource,
    SequenceWindowBatchSource,
    TrainingLoopConfig,
    TrainingMetric,
    selected_backend,
)


FULL_SEQUENCE_OBJECTIVE = "full_sequence_causal_sft"
_ASCII_WHITESPACE = " \t\n\r\f\v"


def _selected_post_training_backend(
    config: PostTrainingConfig,
    *,
    require_qlora_capability: bool = False,
) -> str:
    del require_qlora_capability
    return selected_backend(config.backend)


def _trim_ascii(value: str) -> str:
    """Match the native UTF-8 formatter's locale-independent trimming."""

    return value.strip(_ASCII_WHITESPACE)


@dataclass(frozen=True, slots=True)
class InstructionExample:
    """One supervised prompt/response example."""

    prompt: str
    response: str

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("instruction prompt must be a str")
        if not isinstance(self.response, str):
            raise TypeError("instruction response must be a str")
        if not _trim_ascii(self.prompt):
            raise ValueError("instruction prompt must not be blank")
        if not _trim_ascii(self.response):
            raise ValueError("instruction response must not be blank")


class InstructionFormatter(Protocol):
    """Composition point for alternative chat and instruction templates."""

    def format(self, example: InstructionExample) -> str:
        """Convert one example to a causal-language-model sequence."""


class PlainChatFormatter:
    """Readable delimiter-based format without unsupported special tokens."""

    __slots__ = ()

    def format_prompt(self, prompt: str) -> str:
        """Format an inference prompt with the same training prefix."""

        if not isinstance(prompt, str):
            raise TypeError("prompt must be a str")
        if not _trim_ascii(prompt):
            raise ValueError("prompt must not be blank")
        return (
            "### User:\n"
            f"{_trim_ascii(prompt)}\n"
            "### Assistant:\n"
        )

    def format(self, example: InstructionExample) -> str:
        if not isinstance(example, InstructionExample):
            raise TypeError("example must be an InstructionExample")
        return (
            self.format_prompt(example.prompt)
            + f"{_trim_ascii(example.response)}\n"
        )


@dataclass(frozen=True, slots=True)
class PostTrainingConfig:
    """Controls full-sequence supervised post-training."""

    steps: int = 20
    context_size: int = 16
    batch_size: int = 2
    evaluation_interval: int = 10
    loss_average_window: int = 10
    learning_rate: float = 1.0e-3
    random_seed: int = 29
    backend: str = "auto"
    fine_tuning_method: str = "full"
    sampling_strategy: str = "example_uniform"
    lora: LoraConfig = field(default_factory=LoraConfig)
    attention: str = "materialized"
    activation_checkpointing: str = "disabled"
    nf4_block_size: int = 64
    nf4_scale_block_size: int = 256
    double_quantization: bool = True
    optimizer_state: str = "auto"
    optimizer_page_size: int = 4096

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "context_size",
            "batch_size",
            "evaluation_interval",
            "loss_average_window",
        ):
            _positive_integer(getattr(self, name), name)
        _nonnegative_integer(self.random_seed, "random_seed")
        learning_rate = _finite_real(
            self.learning_rate,
            "learning_rate",
        )
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero")
        if self.backend not in {"auto", "cpu", "metal", "cuda", "tpu"}:
            raise ValueError(
                "backend must be 'auto', 'cpu', 'metal', 'cuda', or 'tpu'"
            )
        if self.attention not in {"materialized", "flash"}:
            raise ValueError(
                "attention must be 'materialized' or 'flash'"
            )
        if self.activation_checkpointing not in {"disabled", "block"}:
            raise ValueError(
                "activation_checkpointing must be 'disabled' or 'block'"
            )
        if self.fine_tuning_method not in {"full", "lora", "qlora"}:
            raise ValueError(
                "fine_tuning_method must be 'full', 'lora', or 'qlora'"
            )
        if self.sampling_strategy not in {
            "example_uniform",
            "window_uniform",
        }:
            raise ValueError(
                "sampling_strategy must be 'example_uniform' or "
                "'window_uniform'"
            )
        if not isinstance(self.lora, LoraConfig):
            raise TypeError("lora must be a LoraConfig")
        block_size = _positive_integer(
            self.nf4_block_size,
            "nf4_block_size",
        )
        if block_size not in {32, 64, 128, 256, 512, 1024, 2048, 4096}:
            raise ValueError(
                "nf4_block_size must be one of 32, 64, 128, 256, "
                "512, 1024, 2048, or 4096"
            )
        scale_block_size = _positive_integer(
            self.nf4_scale_block_size,
            "nf4_scale_block_size",
        )
        if scale_block_size not in {
            32,
            64,
            128,
            256,
            512,
            1024,
            2048,
            4096,
        }:
            raise ValueError(
                "nf4_scale_block_size must be one of 32, 64, 128, 256, "
                "512, 1024, 2048, or 4096"
            )
        if not isinstance(self.double_quantization, bool):
            raise TypeError("double_quantization must be a bool")
        if self.optimizer_state not in {"auto", "contiguous", "paged"}:
            raise ValueError(
                "optimizer_state must be 'auto', 'contiguous', or 'paged'"
            )
        _positive_integer(
            self.optimizer_page_size,
            "optimizer_page_size",
        )


@dataclass(frozen=True, slots=True)
class PostTrainingResult:
    """A derived assistant artifact and its supervised metrics."""

    bundle: ModelBundle
    metrics: tuple[TrainingMetric, ...]
    example_count: int
    objective: str = FULL_SEQUENCE_OBJECTIVE
    fine_tuning_method: str = "full"


def load_instruction_jsonl(
    path: str | Path,
) -> tuple[InstructionExample, ...]:
    """Load strict UTF-8 JSONL records containing prompt and response."""

    source = Path(path)
    return load_instruction_jsonl_bytes(
        source.read_bytes(),
        source=source,
    )


def load_instruction_jsonl_bytes(
    content: bytes,
    *,
    source: str | Path = "<instruction JSONL>",
) -> tuple[InstructionExample, ...]:
    """Parse instruction records from one fixed UTF-8 JSONL buffer."""

    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    source_name = str(source)
    examples: list[InstructionExample] = []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{source_name} is not valid UTF-8") from error
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{source_name}:{line_number} is not valid JSON"
            ) from error
        if not isinstance(value, dict):
            raise ValueError(
                f"{source_name}:{line_number} must be a JSON object"
            )
        if "prompt" not in value or "response" not in value:
            raise ValueError(
                f"{source_name}:{line_number} requires prompt and response"
            )
        try:
            examples.append(
                InstructionExample(
                    prompt=value["prompt"],
                    response=value["response"],
                )
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{source_name}:{line_number}: {error}"
            ) from error
    if not examples:
        raise ValueError("instruction JSONL must contain at least one example")
    return tuple(examples)


def post_train(
    base_bundle: ModelBundle,
    examples: Iterable[InstructionExample],
    config: PostTrainingConfig | None = None,
    *,
    formatter: InstructionFormatter | None = None,
    metric_sink: Callable[[TrainingMetric], None] | None = None,
) -> PostTrainingResult:
    """Continue training on formatted prompt/response sequences.

    This initial objective applies causal loss to the complete formatted
    sequence. It is deliberately named ``full_sequence_causal_sft`` and does
    not claim response-only masking.
    """

    config = PostTrainingConfig() if config is None else config
    if not isinstance(base_bundle, ModelBundle):
        raise TypeError("base_bundle must be a ModelBundle")
    if not isinstance(config, PostTrainingConfig):
        raise TypeError("config must be a PostTrainingConfig")
    if config.context_size > base_bundle.config.maximum_context:
        raise ValueError(
            "post-training context_size exceeds the model maximum_context"
        )
    configured_formatter = (
        PlainChatFormatter() if formatter is None else formatter
    )
    if not callable(getattr(configured_formatter, "format", None)):
        raise TypeError("formatter must provide a callable format() method")

    example_tuple = tuple(examples)
    if not example_tuple:
        raise ValueError("post-training requires at least one example")
    if any(
        not isinstance(example, InstructionExample)
        for example in example_tuple
    ):
        raise TypeError("examples must contain InstructionExample values")

    backend = _selected_post_training_backend(config)
    with base_bundle.instantiate(backend) as runtime:
        runtime.model.set_full_sequence_attention(config.attention)
        runtime.model.set_activation_checkpointing(
            config.activation_checkpointing
        )
        sequences = tuple(
            tuple(
                runtime.tokenizer.encode(
                    configured_formatter.format(example)
                )
            )
            for example in example_tuple
        )
        too_short = tuple(
            index
            for index, sequence in enumerate(sequences)
            if len(sequence) <= config.context_size
        )
        if too_short:
            displayed = ", ".join(str(index) for index in too_short[:8])
            suffix = "..." if len(too_short) > 8 else ""
            raise ValueError(
                "each formatted example must encode to at least "
                "context_size + 1 tokens; too-short example indices: "
                f"{displayed}{suffix}"
            )
        source_type = (
            ExampleWindowBatchSource
            if config.sampling_strategy == "example_uniform"
            else SequenceWindowBatchSource
        )
        source = source_type(
            sequences,
            batch_size=config.batch_size,
            context_size=config.context_size,
            random_seed=config.random_seed,
        )
        loop_config = TrainingLoopConfig(
            steps=config.steps,
            evaluation_interval=config.evaluation_interval,
            loss_average_window=config.loss_average_window,
        )
        qlora_memory: dict[str, int | float] | None = None
        if config.fine_tuning_method == "qlora":
            runtime.model.quantize_nf4(
                config.nf4_block_size,
                double_quantization=config.double_quantization,
                scale_block_size=config.nf4_scale_block_size,
            )
            packed = runtime.model.quantized_memory
            qlora_memory = {
                "weight_count": packed.weight_count,
                "packed_code_bytes": packed.packed_code_bytes,
                "scale_bytes": packed.scale_bytes,
                "logical_payload_bytes": packed.logical_payload_bytes,
                "resident_payload_bytes": packed.resident_payload_bytes,
                "fp32_equivalent_bytes": packed.fp32_equivalent_bytes,
                "fp32_scale_bytes": packed.fp32_scale_bytes,
                "scale_code_bytes": packed.scale_code_bytes,
                "second_level_scale_bytes": (
                    packed.second_level_scale_bytes
                ),
                "scale_offset_bytes": packed.scale_offset_bytes,
                "double_quantized_weight_count": (
                    packed.double_quantized_weight_count
                ),
                "compression_ratio": packed.compression_ratio,
            }
            runtime.model.attach_lora(config.lora)
            parameter_source = runtime.model.adapter_parameters
        elif config.fine_tuning_method == "lora":
            runtime.model.attach_lora(config.lora)
            parameter_source = runtime.model.adapter_parameters
        else:
            parameter_source = runtime.model.parameters

        optimizer_state = (
            "paged"
            if config.optimizer_state == "auto"
            and config.fine_tuning_method == "qlora"
            else (
                "contiguous"
                if config.optimizer_state == "auto"
                else config.optimizer_state
            )
        )
        optimizer_diagnostics: dict[str, int | str] | None = None
        with parameter_source() as parameters:
            with Adam(
                parameters,
                learning_rate=config.learning_rate,
                state_storage=optimizer_state,
                page_size=config.optimizer_page_size,
            ) as optimizer:
                optimizer_diagnostics = {
                    "state_storage": optimizer.state_storage,
                    "page_size": optimizer.state_page_size,
                    "page_count": optimizer.state_page_count,
                    "state_payload_bytes": optimizer.state_payload_bytes,
                }
                trainer = CausalLanguageModelTrainer(
                    runtime.model,
                    optimizer,
                )
                metrics = trainer.run(
                    source,
                    loop_config,
                    metric_sink=metric_sink,
                )
        if config.fine_tuning_method in {"lora", "qlora"}:
            runtime.model.merge_lora()
        tuned_bundle = ModelBundle.capture(
            runtime.model,
            runtime.tokenizer,
            stage="post_training",
            parent_artifact_id=base_bundle.artifact_id,
            metadata={
                "objective": FULL_SEQUENCE_OBJECTIVE,
                "example_count": len(example_tuple),
                "formatter": type(configured_formatter).__name__,
                "training_config": asdict(config),
                "resolved_backend": backend,
                "optimizer": "adam",
                "optimizer_state": optimizer_diagnostics,
                "optimizer_parameter_scope": (
                    "lora_adapters"
                    if config.fine_tuning_method in {"lora", "qlora"}
                    else "all_model_parameters"
                ),
                "fine_tuning_method": config.fine_tuning_method,
                "lora_merged": config.fine_tuning_method in {"lora", "qlora"},
                "quantization": (
                    {
                        "format": "nf4",
                        "block_size": config.nf4_block_size,
                        "double_quantization": (
                            config.double_quantization
                        ),
                        "scale_block_size": (
                            config.nf4_scale_block_size
                            if config.double_quantization
                            else None
                        ),
                        "training_memory": qlora_memory,
                        "export_materialized_to_fp32": True,
                    }
                    if config.fine_tuning_method == "qlora"
                    else None
                ),
                "adapter_state_included": False,
                "optimizer_state_included": False,
                "response_only_loss": False,
            },
        )
    return PostTrainingResult(
        bundle=tuned_bundle,
        metrics=metrics,
        example_count=len(example_tuple),
        fine_tuning_method=config.fine_tuning_method,
    )


def post_train_jsonl(
    base_bundle: ModelBundle,
    path: str | Path,
    config: PostTrainingConfig | None = None,
    *,
    formatter: InstructionFormatter | None = None,
    metric_sink: Callable[[TrainingMetric], None] | None = None,
) -> PostTrainingResult:
    return post_train(
        base_bundle,
        load_instruction_jsonl(path),
        config,
        formatter=formatter,
        metric_sink=metric_sink,
    )


def _positive_integer(value: object, name: str) -> int:
    checked = _nonnegative_integer(value, name)
    if checked == 0:
        raise ValueError(f"{name} must be greater than zero")
    return checked


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "FULL_SEQUENCE_OBJECTIVE",
    "InstructionExample",
    "InstructionFormatter",
    "PlainChatFormatter",
    "PostTrainingConfig",
    "PostTrainingResult",
    "load_instruction_jsonl",
    "load_instruction_jsonl_bytes",
    "post_train",
    "post_train_jsonl",
]
