"""Sparse, task-owned F/P/T/I program specifications.

The runtime eventually consumes dense multilinear coefficients, but this lab
stores only the exact nonzero reference indices.  In particular, constructing
F or T never allocates a Python list proportional to ``(L * K) ** 3``.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterator, Sequence

from .config import Variant


class CoefficientInitialization(str, Enum):
    COMPILED = "compiled"
    RANDOM_UNIFORM = "random_uniform"


class ProgramInputSource(str, Enum):
    WHOLE_SOURCE = "whole_source"
    SOURCE_POSITION = "source_position"


@dataclass(frozen=True, slots=True)
class ProgramInputLayout:
    source: ProgramInputSource = ProgramInputSource.WHOLE_SOURCE
    position: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.source, ProgramInputSource):
            raise TypeError("source must be a ProgramInputSource")
        _require_nonnegative_int("position", self.position)


@dataclass(frozen=True, slots=True)
class SparseMultilinearMap:
    """Unit-valued nonzeros in output-major flattened tensor order."""

    input_dimensions: tuple[int, ...]
    output_dimension: int
    nonzero_flat_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.input_dimensions, tuple):
            raise TypeError("input_dimensions must be a tuple")
        for index, dimension in enumerate(self.input_dimensions):
            _require_positive_int(f"input_dimensions[{index}]", dimension)
        _require_positive_int("output_dimension", self.output_dimension)
        if not isinstance(self.nonzero_flat_indices, tuple):
            raise TypeError("nonzero_flat_indices must be a tuple")
        previous = -1
        for index, flat_index in enumerate(self.nonzero_flat_indices):
            _require_nonnegative_int(
                f"nonzero_flat_indices[{index}]", flat_index
            )
            if flat_index <= previous:
                raise ValueError(
                    "nonzero_flat_indices must be strictly increasing"
                )
            if flat_index >= self.logical_coefficient_count:
                raise ValueError("nonzero coefficient index is out of range")
            previous = flat_index

    @property
    def arity(self) -> int:
        return len(self.input_dimensions)

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return (self.output_dimension,) + self.input_dimensions

    @property
    def logical_coefficient_count(self) -> int:
        result = self.output_dimension
        for dimension in self.input_dimensions:
            result *= dimension
        return result

    @property
    def nonzero_count(self) -> int:
        return len(self.nonzero_flat_indices)

    def coefficient_at(
        self,
        output_index: int,
        input_indices: Sequence[int],
    ) -> float:
        flat_index = output_major_flat_index(
            output_index,
            input_indices,
            self.output_dimension,
            self.input_dimensions,
        )
        position = bisect_left(self.nonzero_flat_indices, flat_index)
        return (
            1.0
            if position < self.nonzero_count
            and self.nonzero_flat_indices[position] == flat_index
            else 0.0
        )

    def iter_nonzero(self) -> Iterator[tuple[int, float]]:
        return ((index, 1.0) for index in self.nonzero_flat_indices)


@dataclass(frozen=True, slots=True)
class ProgramSpec:
    """Lab policy translated later into a generic native program spec."""

    variant: Variant
    reference_map: SparseMultilinearMap
    sequence_length: int
    program_width: int
    input_layouts: tuple[ProgramInputLayout, ...]
    input_projection_groups: tuple[int, ...]
    initialization: CoefficientInitialization
    trainable: bool
    attention_query_axis: int | None
    random_seed: int
    random_scale: float
    source_offset: int = 0
    target_offset: int = 0
    input_projection_bias: bool = False
    program_merge_bias: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.variant, Variant):
            raise TypeError("variant must be a Variant")
        if self.variant is Variant.I:
            raise ValueError("variant I has no ProgramSpec")
        if not isinstance(self.reference_map, SparseMultilinearMap):
            raise TypeError("reference_map must be a SparseMultilinearMap")
        _require_positive_int("sequence_length", self.sequence_length)
        _require_positive_int("program_width", self.program_width)
        _require_nonnegative_int("source_offset", self.source_offset)
        _require_nonnegative_int("target_offset", self.target_offset)
        if not isinstance(self.input_layouts, tuple) or any(
            not isinstance(layout, ProgramInputLayout)
            for layout in self.input_layouts
        ):
            raise TypeError("input_layouts must contain ProgramInputLayout values")
        if not isinstance(self.input_projection_groups, tuple):
            raise TypeError("input_projection_groups must be a tuple")
        if len(self.input_layouts) != self.reference_map.arity:
            raise ValueError("input layouts must match map arity")
        if len(self.input_projection_groups) != self.reference_map.arity:
            raise ValueError("projection groups must match map arity")
        for index, group in enumerate(self.input_projection_groups):
            _require_nonnegative_int(
                f"input_projection_groups[{index}]", group
            )
        if not isinstance(self.initialization, CoefficientInitialization):
            raise TypeError("initialization must be a CoefficientInitialization")
        if not isinstance(self.trainable, bool):
            raise TypeError("trainable must be a bool")
        if self.attention_query_axis is not None:
            _require_nonnegative_int(
                "attention_query_axis", self.attention_query_axis
            )
            if self.attention_query_axis >= self.reference_map.arity:
                raise ValueError("attention_query_axis is out of range")
        _require_int("random_seed", self.random_seed)
        _require_positive_finite("random_scale", self.random_scale)
        if not isinstance(self.input_projection_bias, bool):
            raise TypeError("input_projection_bias must be a bool")
        if not isinstance(self.program_merge_bias, bool):
            raise TypeError("program_merge_bias must be a bool")

        dimension = self.sequence_length * self.program_width
        expected_arity = 1 if self.variant is Variant.P else 2
        expected_groups = (0,) if self.variant is Variant.P else (0, 0)
        if self.reference_map.input_dimensions != (dimension,) * expected_arity:
            raise ValueError("reference map input dimensions are incompatible")
        if self.reference_map.output_dimension != dimension:
            raise ValueError("reference map output dimension is incompatible")
        if self.input_projection_groups != expected_groups:
            raise ValueError("projection groups violate the variant contract")
        if self.target_offset != self.sequence_length:
            raise ValueError("program output must be placed on the target half")

        if self.variant in {Variant.F, Variant.P}:
            if self.initialization is not CoefficientInitialization.COMPILED:
                raise ValueError("F and P require compiled coefficients")
            if self.trainable:
                raise ValueError("F and P coefficients must be frozen")
        elif self.variant is Variant.T:
            if self.initialization is not CoefficientInitialization.RANDOM_UNIFORM:
                raise ValueError("T requires random-uniform initialization")
            if not self.trainable:
                raise ValueError("T coefficients must be trainable")

    @property
    def dimension(self) -> int:
        return self.sequence_length * self.program_width

    @property
    def output_length(self) -> int:
        return self.sequence_length

    @property
    def source_length(self) -> int:
        return self.sequence_length

    @property
    def shares_input_projection(self) -> bool:
        return len(set(self.input_projection_groups)) < len(
            self.input_projection_groups
        )


def sequence_coordinate(
    position: int,
    symbol: int,
    sequence_length: int,
    symbol_count: int,
) -> int:
    """Return the direct-sum coordinate ``position * K + symbol``."""

    _validate_dimensions(sequence_length, symbol_count)
    _require_nonnegative_int("position", position)
    _require_nonnegative_int("symbol", symbol)
    if position >= sequence_length:
        raise ValueError("position is out of range")
    if symbol >= symbol_count:
        raise ValueError("symbol is out of range")
    return position * symbol_count + symbol


def output_major_flat_index(
    output_index: int,
    input_indices: Sequence[int],
    output_dimension: int,
    input_dimensions: Sequence[int],
) -> int:
    """Flatten ``[output, input_0, ...]`` in C/output-major order."""

    _require_positive_int("output_dimension", output_dimension)
    _require_nonnegative_int("output_index", output_index)
    if output_index >= output_dimension:
        raise ValueError("output_index is out of range")
    dimensions = tuple(input_dimensions)
    indices = tuple(input_indices)
    if len(indices) != len(dimensions):
        raise ValueError("input_indices must match input_dimensions")
    flat_index = output_index
    for axis, (index, dimension) in enumerate(zip(indices, dimensions)):
        _require_positive_int(f"input_dimensions[{axis}]", dimension)
        _require_nonnegative_int(f"input_indices[{axis}]", index)
        if index >= dimension:
            raise ValueError(f"input_indices[{axis}] is out of range")
        flat_index = flat_index * dimension + index
    return flat_index


def p_coefficient_index(
    output_position: int,
    symbol: int,
    sequence_length: int,
    symbol_count: int,
) -> int:
    """Flat index of P's unit reversal coefficient.

    With ``D = L*K``, the formula is ``out * D + reversed_input``.
    """

    output_coordinate = sequence_coordinate(
        output_position, symbol, sequence_length, symbol_count
    )
    input_coordinate = sequence_coordinate(
        sequence_length - 1 - output_position,
        symbol,
        sequence_length,
        symbol_count,
    )
    dimension = sequence_length * symbol_count
    return output_coordinate * dimension + input_coordinate


def f_coefficient_index(
    output_position: int,
    symbol: int,
    selector_symbol: int,
    sequence_length: int,
    symbol_count: int,
) -> int:
    """Flat index of F's unit conditional coefficient.

    The selector is coordinate ``selector_symbol`` at the first position.
    Coordinate zero selects reversal and every other coordinate selects copy.
    For ``D = L*K`` the output-major formula is
    ``(out * D + selector) * D + selected_input``.
    """

    output_coordinate = sequence_coordinate(
        output_position, symbol, sequence_length, symbol_count
    )
    selector_coordinate = sequence_coordinate(
        0, selector_symbol, sequence_length, symbol_count
    )
    selected_position = (
        sequence_length - 1 - output_position
        if selector_symbol == 0
        else output_position
    )
    selected_input = sequence_coordinate(
        selected_position, symbol, sequence_length, symbol_count
    )
    dimension = sequence_length * symbol_count
    return (
        (output_coordinate * dimension + selector_coordinate) * dimension
        + selected_input
    )


def iter_p_coefficient_indices(
    sequence_length: int,
    symbol_count: int,
) -> Iterator[int]:
    """Yield P's exactly ``D`` unit coefficients in flat sorted order."""

    _validate_dimensions(sequence_length, symbol_count)
    for output_position in range(sequence_length):
        for symbol in range(symbol_count):
            yield p_coefficient_index(
                output_position, symbol, sequence_length, symbol_count
            )


def iter_f_coefficient_indices(
    sequence_length: int,
    symbol_count: int,
) -> Iterator[int]:
    """Yield F's exactly ``K*D`` unit coefficients in flat sorted order."""

    _validate_dimensions(sequence_length, symbol_count)
    for output_position in range(sequence_length):
        for symbol in range(symbol_count):
            for selector_symbol in range(symbol_count):
                yield f_coefficient_index(
                    output_position,
                    symbol,
                    selector_symbol,
                    sequence_length,
                    symbol_count,
                )


def make_p_reference_map(
    sequence_length: int,
    program_width: int,
) -> SparseMultilinearMap:
    _validate_dimensions(sequence_length, program_width)
    dimension = sequence_length * program_width
    result = SparseMultilinearMap(
        input_dimensions=(dimension,),
        output_dimension=dimension,
        nonzero_flat_indices=tuple(
            iter_p_coefficient_indices(sequence_length, program_width)
        ),
    )
    if result.nonzero_count != dimension:
        raise AssertionError("P reference map has the wrong nonzero count")
    return result


def make_f_reference_map(
    sequence_length: int,
    program_width: int,
) -> SparseMultilinearMap:
    _validate_dimensions(sequence_length, program_width)
    dimension = sequence_length * program_width
    result = SparseMultilinearMap(
        input_dimensions=(dimension, dimension),
        output_dimension=dimension,
        nonzero_flat_indices=tuple(
            iter_f_coefficient_indices(sequence_length, program_width)
        ),
    )
    if result.nonzero_count != program_width * dimension:
        raise AssertionError("F reference map has the wrong nonzero count")
    return result


def build_program_spec(
    variant: Variant,
    sequence_length: int,
    program_width: int,
    *,
    seed: int = 42,
    random_scale: float = 0.02,
) -> ProgramSpec | None:
    """Build lab metadata; I returns ``None`` and owns no program state."""

    if not isinstance(variant, Variant):
        raise TypeError("variant must be a Variant")
    _validate_dimensions(sequence_length, program_width)
    _require_int("seed", seed)
    _require_positive_finite("random_scale", random_scale)
    if variant is Variant.I:
        return None

    layout = ProgramInputLayout()
    if variant is Variant.P:
        return ProgramSpec(
            variant=variant,
            reference_map=make_p_reference_map(sequence_length, program_width),
            sequence_length=sequence_length,
            program_width=program_width,
            input_layouts=(layout,),
            input_projection_groups=(0,),
            initialization=CoefficientInitialization.COMPILED,
            trainable=False,
            attention_query_axis=None,
            random_seed=seed,
            random_scale=random_scale,
            target_offset=sequence_length,
        )

    return ProgramSpec(
        variant=variant,
        reference_map=make_f_reference_map(sequence_length, program_width),
        sequence_length=sequence_length,
        program_width=program_width,
        input_layouts=(layout, layout),
        input_projection_groups=(0, 0),
        initialization=(
            CoefficientInitialization.RANDOM_UNIFORM
            if variant is Variant.T
            else CoefficientInitialization.COMPILED
        ),
        trainable=variant is Variant.T,
        attention_query_axis=1,
        random_seed=seed,
        random_scale=random_scale,
        target_offset=sequence_length,
    )


def _validate_dimensions(sequence_length: object, symbol_count: object) -> None:
    _require_positive_int("sequence_length", sequence_length)
    _require_positive_int("symbol_count", symbol_count)


def _require_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")


def _require_nonnegative_int(name: str, value: object) -> None:
    _require_int(name, value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_positive_int(name: str, value: object) -> None:
    _require_int(name, value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _require_positive_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{name} must be positive and finite")


__all__ = [
    "CoefficientInitialization",
    "ProgramInputLayout",
    "ProgramInputSource",
    "ProgramSpec",
    "SparseMultilinearMap",
    "build_program_spec",
    "f_coefficient_index",
    "iter_f_coefficient_indices",
    "iter_p_coefficient_indices",
    "make_f_reference_map",
    "make_p_reference_map",
    "output_major_flat_index",
    "p_coefficient_index",
    "sequence_coordinate",
]
