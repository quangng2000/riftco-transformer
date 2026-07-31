"""Backward-compatible facade for serving generation policies."""

from .serving.generation import (
    GenerationResult,
    GreedySampler,
    SamplingStrategy,
    TemperatureSampler,
    TextGenerator,
)


__all__ = [
    "GenerationResult",
    "GreedySampler",
    "SamplingStrategy",
    "TemperatureSampler",
    "TextGenerator",
]
