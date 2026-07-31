"""Dataset-specific adapters for the lab's stage-oriented record formats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

from .client import DatasetRow, DatasetSchemaError
from .serialization import (
    CanonicalJsonlSerializer,
    RecordSerializer,
    TextCorpusSerializer,
)


class DatasetAdapter(Protocol):
    """Composition point for converting source rows into training records."""

    @property
    def adapter_id(self) -> str:
        """Stable identifier stored in provenance manifests."""

    def adapt(self, row: DatasetRow) -> Mapping[str, str]:
        """Validate and convert one source row."""


class DollyInstructionAdapter:
    """Convert Dolly rows to ``prompt``/``response`` SFT records."""

    __slots__ = ()

    @property
    def adapter_id(self) -> str:
        return "dolly_instruction_v1"

    def adapt(self, row: DatasetRow) -> Mapping[str, str]:
        _require_dataset_row(row)
        instruction = _required_text(
            row,
            "instruction",
            allow_blank=False,
        ).strip()
        context = _required_text(row, "context", allow_blank=True).strip()
        response = _required_text(
            row,
            "response",
            allow_blank=False,
        ).strip()
        category = _required_text(
            row,
            "category",
            allow_blank=False,
        ).strip()
        prompt = instruction
        if context:
            prompt = f"{instruction}\n\nContext:\n{context}"
        return {
            "prompt": prompt,
            "response": response,
            "category": category,
        }


class TinyStoriesTextAdapter:
    """Convert TinyStories rows to plain pretraining text records."""

    __slots__ = ()

    @property
    def adapter_id(self) -> str:
        return "tinystories_text_v1"

    def adapt(self, row: DatasetRow) -> Mapping[str, str]:
        _require_dataset_row(row)
        text = _required_text(row, "text", allow_blank=False).strip()
        return {"text": text}


class HhRlhfPreferenceAdapter:
    """Convert HH-RLHF rows to chosen/rejected preference records."""

    __slots__ = ()

    @property
    def adapter_id(self) -> str:
        return "hh_rlhf_preference_v1"

    def adapt(self, row: DatasetRow) -> Mapping[str, str]:
        _require_dataset_row(row)
        chosen = _required_text(row, "chosen", allow_blank=False).strip()
        rejected = _required_text(row, "rejected", allow_blank=False).strip()
        if chosen == rejected:
            raise DatasetSchemaError(
                f"source row {row.index} has identical chosen and rejected text"
            )
        return {"chosen": chosen, "rejected": rejected}


@dataclass(frozen=True, slots=True)
class DatasetPreset:
    """Audited source coordinates, license, and row adapter."""

    name: str
    dataset: str
    config: str
    split: str
    license_id: str
    record_kind: str
    lab_training_support: str
    usage_note: str
    adapter: DatasetAdapter
    serializer: RecordSerializer
    identity_fields: tuple[str, ...]
    stratification_field: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "dataset",
            "config",
            "split",
            "license_id",
            "record_kind",
            "lab_training_support",
            "usage_note",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a str")
            if not value.strip():
                raise ValueError(f"{field_name} must not be blank")
        if not isinstance(getattr(self.adapter, "adapter_id", None), str):
            raise TypeError("adapter must provide a string adapter_id")
        if not callable(getattr(self.adapter, "adapt", None)):
            raise TypeError("adapter must provide a callable adapt() method")
        for attribute in (
            "serializer_id",
            "file_extension",
            "media_type",
        ):
            if not isinstance(getattr(self.serializer, attribute, None), str):
                raise TypeError(
                    f"serializer must provide a string {attribute}"
                )
        if not callable(getattr(self.serializer, "serialize", None)):
            raise TypeError(
                "serializer must provide a callable serialize() method"
            )
        if (
            not isinstance(self.identity_fields, tuple)
            or not self.identity_fields
            or any(
                not isinstance(name, str) or not name
                for name in self.identity_fields
            )
        ):
            raise TypeError(
                "identity_fields must be a nonempty tuple of field names"
            )
        if len(set(self.identity_fields)) != len(self.identity_fields):
            raise ValueError("identity_fields must not contain duplicates")
        if self.stratification_field is not None:
            if not isinstance(self.stratification_field, str):
                raise TypeError("stratification_field must be a str or None")
            if not self.stratification_field:
                raise ValueError("stratification_field must not be empty")

    @property
    def source_url(self) -> str:
        return f"https://huggingface.co/datasets/{self.dataset}"

    def identity_record(
        self,
        record: Mapping[str, str],
    ) -> Mapping[str, str]:
        """Project metadata away before deduplication and partitioning."""

        missing = [
            name for name in self.identity_fields if name not in record
        ]
        if missing:
            raise DatasetSchemaError(
                "adapted record is missing identity field(s): "
                + ", ".join(missing)
            )
        identity = {
            name: record[name]
            for name in self.identity_fields
        }
        if any(not isinstance(value, str) for value in identity.values()):
            raise DatasetSchemaError(
                "adapted identity fields must contain strings"
            )
        return identity


DOLLY_15K = DatasetPreset(
    name="dolly",
    dataset="databricks/databricks-dolly-15k",
    config="default",
    split="train",
    license_id="cc-by-sa-3.0",
    record_kind="instruction_sft",
    lab_training_support="supported",
    usage_note="Use for supervised instruction post-training.",
    adapter=DollyInstructionAdapter(),
    serializer=CanonicalJsonlSerializer(),
    identity_fields=("prompt", "response"),
    stratification_field="category",
)

TINY_STORIES = DatasetPreset(
    name="tinystories",
    dataset="roneneldan/TinyStories",
    config="default",
    split="train",
    license_id="cdla-sharing-1.0",
    record_kind="pretraining_text",
    lab_training_support="supported",
    usage_note="Use as causal language-model pretraining text.",
    adapter=TinyStoriesTextAdapter(),
    serializer=TextCorpusSerializer(),
    identity_fields=("text",),
)

HH_RLHF = DatasetPreset(
    name="hh-rlhf",
    dataset="Anthropic/hh-rlhf",
    config="default",
    split="train",
    license_id="mit",
    record_kind="preference_pair",
    lab_training_support="data_preparation_only",
    usage_note=(
        "Preference-ranking data only; the lab does not yet implement a "
        "preference objective, and this preset must not be used as SFT data."
    ),
    adapter=HhRlhfPreferenceAdapter(),
    serializer=CanonicalJsonlSerializer(),
    identity_fields=("chosen", "rejected"),
)

DATASET_PRESETS = {
    preset.name: preset
    for preset in (
        DOLLY_15K,
        TINY_STORIES,
        HH_RLHF,
    )
}


def _require_dataset_row(value: object) -> DatasetRow:
    if not isinstance(value, DatasetRow):
        raise TypeError("row must be a DatasetRow")
    return value


def _required_text(
    row: DatasetRow,
    name: str,
    *,
    allow_blank: bool,
) -> str:
    if name not in row.values:
        raise DatasetSchemaError(
            f"source row {row.index} requires a {name} field"
        )
    value = row.values[name]
    if not isinstance(value, str):
        raise DatasetSchemaError(
            f"source row {row.index}.{name} must be a string"
        )
    if not allow_blank and not value.strip():
        raise DatasetSchemaError(
            f"source row {row.index}.{name} must not be blank"
        )
    return value


__all__ = [
    "DATASET_PRESETS",
    "DOLLY_15K",
    "HH_RLHF",
    "TINY_STORIES",
    "DatasetAdapter",
    "DatasetPreset",
    "DollyInstructionAdapter",
    "HhRlhfPreferenceAdapter",
    "TinyStoriesTextAdapter",
]
