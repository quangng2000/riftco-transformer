"""Shared numeric boundary checks for dependency-free Python surfaces."""

from __future__ import annotations

import math
import struct


def positive_float32(value: object, name: str) -> float:
    """Return the FP32 conversion of a finite, strictly positive number.

    Python floats are binary64 values.  A value can therefore look finite and
    positive in Python while overflowing to infinity or underflowing to zero
    when it crosses the native FP32 boundary.  Validate the converted value,
    not only the original Python value.
    """

    if isinstance(value, (bool, str, bytes, bytearray)):
        raise TypeError(f"{name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a number") from error
    except OverflowError as error:
        raise ValueError(
            f"{name} must be representable as a finite, strictly positive "
            "float32"
        ) from error

    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(
            f"{name} must be representable as a finite, strictly positive "
            "float32"
        )
    try:
        encoded = struct.pack("<f", number)
    except (OverflowError, struct.error) as error:
        raise ValueError(
            f"{name} must be representable as a finite, strictly positive "
            "float32"
        ) from error
    converted = struct.unpack("<f", encoded)[0]
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(
            f"{name} must be representable as a finite, strictly positive "
            "float32"
        )
    return converted


__all__ = ["positive_float32"]
