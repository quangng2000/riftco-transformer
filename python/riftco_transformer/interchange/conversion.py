"""Explicit high-level conversion between complete model artifacts."""

from __future__ import annotations

import os
from pathlib import Path

from ..artifacts import ModelBundle
from .gguf import export_gguf, load_gguf
from .huggingface import (
    export_huggingface_directory,
    load_huggingface_directory,
)
from .onnx import export_onnx, load_onnx


IMPORT_FORMATS = ("rift", "huggingface", "gguf", "onnx")
EXPORT_FORMATS = ("rift", "huggingface", "gguf", "onnx")


def load_model(
    source: str | os.PathLike[str],
    *,
    format: str,
) -> ModelBundle:
    """Load one complete model through an explicitly selected adapter.

    The format is never inferred from a suffix. This keeps a renamed file or
    a directory with ambiguous contents from silently selecting a parser.
    Standalone SafeTensors files are intentionally absent: weights alone do
    not define the model architecture or tokenizer.
    """

    checked_format = _format(format, "source", IMPORT_FORMATS)
    if checked_format == "rift":
        return ModelBundle.load(source)
    if checked_format == "huggingface":
        return load_huggingface_directory(source)
    if checked_format == "gguf":
        return load_gguf(source)
    return load_onnx(source)


def export_model(
    bundle: ModelBundle,
    destination: str | os.PathLike[str],
    *,
    format: str,
    gguf_model_name: str = "Riftco Decoder",
) -> Path:
    """Export a complete native bundle through one selected adapter."""

    if not isinstance(bundle, ModelBundle):
        raise TypeError("bundle must be a ModelBundle")
    checked_format = _format(format, "destination", EXPORT_FORMATS)
    if checked_format == "rift":
        return bundle.save(destination)
    if checked_format == "huggingface":
        return export_huggingface_directory(bundle, destination)
    if checked_format == "gguf":
        return export_gguf(
            bundle,
            destination,
            model_name=gguf_model_name,
        )
    return export_onnx(bundle, destination)


def convert_model(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    source_format: str,
    destination_format: str,
    gguf_model_name: str = "Riftco Decoder",
) -> Path:
    """Load a complete model and export it through another adapter."""

    bundle = load_model(source, format=source_format)
    return export_model(
        bundle,
        destination,
        format=destination_format,
        gguf_model_name=gguf_model_name,
    )


def _format(
    value: object,
    role: str,
    choices: tuple[str, ...],
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{role} format must be a string")
    if value not in choices:
        quoted = ", ".join(repr(choice) for choice in choices)
        raise ValueError(f"{role} format must be one of {quoted}")
    return value


__all__ = [
    "EXPORT_FORMATS",
    "IMPORT_FORMATS",
    "convert_model",
    "export_model",
    "load_model",
]
