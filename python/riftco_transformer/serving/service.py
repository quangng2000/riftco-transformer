"""Thread-safe in-process generation over an immutable model artifact."""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading

from ..artifacts import ModelBundle, ModelRuntime
from ..native import backend_available
from .generation import (
    GenerationResult,
    GreedySampler,
    TemperatureSampler,
    TextGenerator,
)


@dataclass(frozen=True, slots=True)
class ServingConfig:
    """Resource and backend limits for one serving process."""

    backend: str = "auto"
    maximum_new_tokens: int = 256
    maximum_request_bytes: int = 1 << 20
    kv_cache: str = "paged"
    kv_cache_block_size: int = 16

    def __post_init__(self) -> None:
        if self.backend not in {"auto", "cpu", "metal"}:
            raise ValueError("backend must be 'auto', 'cpu', or 'metal'")
        _positive_integer(self.maximum_new_tokens, "maximum_new_tokens")
        _positive_integer(
            self.maximum_request_bytes,
            "maximum_request_bytes",
        )
        if not isinstance(self.kv_cache, str):
            raise TypeError(
                "kv_cache must be 'contiguous' or 'paged'"
            )
        if self.kv_cache not in {"contiguous", "paged"}:
            raise ValueError(
                "kv_cache must be 'contiguous' or 'paged'"
            )
        _positive_integer(
            self.kv_cache_block_size,
            "kv_cache_block_size",
        )


class ModelService:
    """Thread-safe single-model generation service."""

    __slots__ = (
        "_bundle",
        "_config",
        "_generation_lock",
        "_runtime",
    )

    def __init__(
        self,
        bundle: ModelBundle,
        config: ServingConfig | None = None,
    ) -> None:
        config = ServingConfig() if config is None else config
        if not isinstance(bundle, ModelBundle):
            raise TypeError("bundle must be a ModelBundle")
        if not isinstance(config, ServingConfig):
            raise TypeError("config must be a ServingConfig")
        backend = _selected_backend(config.backend)
        self._bundle = bundle
        self._config = config
        self._generation_lock = threading.Lock()
        self._runtime: ModelRuntime | None = bundle.instantiate(backend)

    @property
    def artifact_id(self) -> str:
        return self._bundle.artifact_id

    @property
    def stage(self) -> str:
        return self._bundle.stage

    @property
    def backend(self) -> str:
        with self._generation_lock:
            return self._active_runtime().model.backend

    @property
    def config(self) -> ServingConfig:
        return self._config

    def health(self) -> dict[str, object]:
        with self._generation_lock:
            runtime = self._active_runtime()
            return {
                "status": "ok",
                "artifact_id": self.artifact_id,
                "artifact_stage": self.stage,
                "backend": runtime.model.backend,
                "vocabulary_size": runtime.tokenizer.vocab_size,
                "maximum_context": runtime.model.config.maximum_context,
                "maximum_new_tokens": self._config.maximum_new_tokens,
                "kv_cache": self._config.kv_cache,
                "kv_cache_block_size": self._config.kv_cache_block_size,
            }

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        top_k: int | None = None,
        seed: int = 0,
    ) -> GenerationResult:
        count = _nonnegative_integer(max_new_tokens, "max_new_tokens")
        if count > self._config.maximum_new_tokens:
            raise ValueError(
                "max_new_tokens exceeds the serving limit "
                f"{self._config.maximum_new_tokens}"
            )
        configured_temperature = _finite_real(
            temperature,
            "temperature",
        )
        if configured_temperature < 0.0:
            raise ValueError("temperature must not be negative")
        if configured_temperature == 0.0:
            if top_k is not None:
                raise ValueError(
                    "top_k requires temperature greater than zero"
                )
            sampler = GreedySampler()
        else:
            sampler = TemperatureSampler(
                configured_temperature,
                top_k=top_k,
                seed=seed,
            )

        with self._generation_lock:
            runtime = self._active_runtime()
            return TextGenerator(
                runtime.model,
                runtime.tokenizer,
                sampler,
                kv_cache=self._config.kv_cache,
                kv_cache_block_size=self._config.kv_cache_block_size,
            ).generate(prompt, max_new_tokens=count)

    def close(self) -> None:
        with self._generation_lock:
            runtime = self._runtime
            self._runtime = None
            if runtime is not None:
                runtime.close()

    def __enter__(self) -> ModelService:
        self._active_runtime()
        return self

    def __exit__(
        self,
        _type: object,
        _value: object,
        _traceback: object,
    ) -> None:
        self.close()

    def _active_runtime(self) -> ModelRuntime:
        if self._runtime is None:
            raise RuntimeError("model service is closed")
        return self._runtime


def _selected_backend(requested: str) -> str:
    if requested == "auto":
        return "metal" if backend_available("metal") else "cpu"
    if requested == "metal" and not backend_available("metal"):
        raise RuntimeError("Metal was requested but is unavailable")
    return requested


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
    "ModelService",
    "ServingConfig",
]
