"""Leakage-resistant deterministic partitioning for prepared records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping


PARTITION_NAMES = ("train", "validation", "test")
_HASH_SPACE = 1 << 256


@dataclass(frozen=True, slots=True)
class SplitFractions:
    """Relative sizes for train, validation, and held-out test data."""

    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        for name, value in zip(PARTITION_NAMES, values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} fraction must be a real number")
            if not math.isfinite(float(value)) or value < 0.0:
                raise ValueError(
                    f"{name} fraction must be finite and nonnegative"
                )
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("split fractions must sum to one")

    def as_dict(self) -> dict[str, float]:
        return {
            "train": float(self.train),
            "validation": float(self.validation),
            "test": float(self.test),
        }


class StableHashSplitter:
    """Assign canonical records by seeded SHA-256.

    Assignment does not depend on source ordering. Exact duplicate records
    produce the same digest and therefore cannot leak across partitions.
    """

    __slots__ = ("_fractions", "_seed_bytes", "_seed_label", "_thresholds")

    def __init__(
        self,
        fractions: SplitFractions | None = None,
        *,
        seed: int | str = 7,
    ) -> None:
        configured = SplitFractions() if fractions is None else fractions
        if not isinstance(configured, SplitFractions):
            raise TypeError("fractions must be a SplitFractions")
        if isinstance(seed, bool) or not isinstance(seed, (int, str)):
            raise TypeError("seed must be an int or str")
        if isinstance(seed, str):
            if not seed:
                raise ValueError("seed must not be empty")
            self._seed_label = f"str:{seed}"
            self._seed_bytes = f"str:{seed}".encode("utf-8")
        else:
            self._seed_label = f"int:{seed}"
            self._seed_bytes = f"int:{seed}".encode("ascii")
        self._fractions = configured
        train_end = int(float(configured.train) * _HASH_SPACE)
        validation_end = int(
            (float(configured.train) + float(configured.validation))
            * _HASH_SPACE
        )
        self._thresholds = (train_end, validation_end)

    @property
    def fractions(self) -> SplitFractions:
        return self._fractions

    @property
    def seed(self) -> str:
        return self._seed_label

    def fingerprint(self, record: Mapping[str, str]) -> str:
        canonical = canonical_record_bytes(record)
        digest = hashlib.sha256()
        digest.update(b"riftco_transformer.stable_split.v1\0")
        digest.update(self._seed_bytes)
        digest.update(b"\0")
        digest.update(canonical)
        return digest.hexdigest()

    def assign(self, record: Mapping[str, str]) -> str:
        value = int(self.fingerprint(record), 16)
        train_end, validation_end = self._thresholds
        if value < train_end:
            return "train"
        if value < validation_end:
            return "validation"
        return "test"


def canonical_record_bytes(record: Mapping[str, str]) -> bytes:
    """Return the representation used solely for split identity."""

    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    if not record:
        raise ValueError("record must not be empty")
    result: dict[str, str] = {}
    for name, value in record.items():
        if not isinstance(name, str) or not name:
            raise ValueError("record field names must be nonempty strings")
        if not isinstance(value, str):
            raise TypeError(f"record field {name!r} must be a str")
        result[name] = value
    return json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


__all__ = [
    "PARTITION_NAMES",
    "SplitFractions",
    "StableHashSplitter",
    "canonical_record_bytes",
]
