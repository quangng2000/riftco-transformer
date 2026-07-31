"""Backward-compatible facade for :mod:`transformer_lab.artifacts`."""

from .artifacts.bundle import (
    FORMAT_NAME,
    FORMAT_VERSION,
    MANIFEST_NAME,
    MAXIMUM_MANIFEST_BYTES,
    WEIGHTS_NAME,
    ModelBundle,
    ModelRuntime,
    ParameterSpec,
    TokenizerSpec,
)


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "ModelBundle",
    "ModelRuntime",
    "ParameterSpec",
    "TokenizerSpec",
]
