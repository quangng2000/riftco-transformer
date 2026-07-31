"""Deterministic serializers for prepared training records."""

from __future__ import annotations

import json
from typing import Mapping, Protocol

from .client import DatasetSchemaError


class RecordSerializer(Protocol):
    """Composition point for stage-specific output representations."""

    @property
    def serializer_id(self) -> str:
        """Stable identifier for the provenance manifest."""

    @property
    def file_extension(self) -> str:
        """Extension, including the leading dot."""

    @property
    def media_type(self) -> str:
        """Media type recorded alongside output hashes."""

    def serialize(self, record: Mapping[str, str]) -> bytes:
        """Return the complete byte representation of one record."""


class CanonicalJsonlSerializer:
    """Serialize mappings as stable, compact UTF-8 JSON lines."""

    __slots__ = ()

    @property
    def serializer_id(self) -> str:
        return "canonical_jsonl_v1"

    @property
    def file_extension(self) -> str:
        return ".jsonl"

    @property
    def media_type(self) -> str:
        return "application/x-ndjson"

    def serialize(self, record: Mapping[str, str]) -> bytes:
        _validate_record(record)
        value = json.dumps(
            dict(record),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"{value}\n".encode("utf-8")


class TextCorpusSerializer:
    """Serialize one text field with a blank line between documents."""

    __slots__ = ("_field",)

    def __init__(self, field: str = "text") -> None:
        if not isinstance(field, str):
            raise TypeError("field must be a str")
        if not field.strip():
            raise ValueError("field must not be blank")
        self._field = field

    @property
    def serializer_id(self) -> str:
        return f"text_corpus_{self._field}_v1"

    @property
    def file_extension(self) -> str:
        return ".txt"

    @property
    def media_type(self) -> str:
        return "text/plain; charset=utf-8"

    def serialize(self, record: Mapping[str, str]) -> bytes:
        _validate_record(record)
        if set(record) != {self._field}:
            raise DatasetSchemaError(
                "text corpus records must contain only "
                f"the {self._field!r} field"
            )
        text = record[self._field].strip()
        if not text:
            raise DatasetSchemaError(
                f"text corpus field {self._field!r} must not be blank"
            )
        return f"{text}\n\n".encode("utf-8")


def _validate_record(record: Mapping[str, str]) -> None:
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if not record:
        raise DatasetSchemaError("prepared record must not be empty")
    for name, value in record.items():
        if not isinstance(name, str) or not name:
            raise DatasetSchemaError(
                "prepared record field names must be nonempty strings"
            )
        if not isinstance(value, str):
            raise DatasetSchemaError(
                f"prepared record field {name!r} must be a string"
            )


__all__ = [
    "CanonicalJsonlSerializer",
    "RecordSerializer",
    "TextCorpusSerializer",
]
