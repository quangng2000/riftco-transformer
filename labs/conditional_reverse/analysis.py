"""Standard-library analysis helpers for conditional-reverse representations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Mapping, Sequence

from .config import Variant


@dataclass(frozen=True, slots=True)
class PcaResult:
    """A fitted PCA basis with components stored as row vectors."""

    observation_count: int
    feature_count: int
    mean: tuple[float, ...]
    components: tuple[tuple[float, ...], ...]
    eigenvalues: tuple[float, ...]
    explained_variance_ratio: tuple[float, ...]
    converged: bool
    sweeps: int

    @property
    def component_count(self) -> int:
        return len(self.components)


class PcaAccumulator:
    """Online covariance for probe PCA with memory independent of row count."""

    __slots__ = ("_count", "_feature_count", "_mean", "_second_moment")

    def __init__(self, feature_count: int | None = None) -> None:
        if feature_count is not None:
            _require_positive_int("feature_count", feature_count)
        self._count = 0
        self._feature_count = feature_count
        self._mean = (
            [] if feature_count is None else [0.0] * feature_count
        )
        self._second_moment = (
            []
            if feature_count is None
            else [[0.0] * feature_count for _ in range(feature_count)]
        )

    @property
    def observation_count(self) -> int:
        return self._count

    @property
    def feature_count(self) -> int | None:
        return self._feature_count

    def update(self, rows: Iterable[Sequence[float]]) -> None:
        """Consume one capture batch while retaining only O(width squared)."""

        try:
            iterator = iter(rows)
        except TypeError as error:
            raise TypeError("rows must be iterable") from error
        for row_index, row in enumerate(iterator):
            values = _finite_row(row, f"rows[{row_index}]")
            if self._feature_count is None:
                self._feature_count = len(values)
                self._mean = [0.0] * len(values)
                self._second_moment = [
                    [0.0] * len(values) for _ in range(len(values))
                ]
            if len(values) != self._feature_count:
                raise ValueError("rows must be rectangular across updates")

            self._count += 1
            delta = tuple(
                values[index] - self._mean[index]
                for index in range(self._feature_count)
            )
            for index in range(self._feature_count):
                self._mean[index] += delta[index] / self._count
            adjusted = tuple(
                values[index] - self._mean[index]
                for index in range(self._feature_count)
            )
            for left in range(self._feature_count):
                for right in range(self._feature_count):
                    self._second_moment[left][right] += (
                        delta[left] * adjusted[right]
                    )

    def finish(
        self,
        component_count: int = 2,
        *,
        tolerance: float = 1.0e-10,
        max_sweeps: int = 64,
    ) -> PcaResult:
        if self._count < 2 or self._feature_count is None:
            raise ValueError("PCA requires at least two observations")
        covariance = [
            [0.0] * self._feature_count for _ in range(self._feature_count)
        ]
        denominator = self._count - 1
        for left in range(self._feature_count):
            for right in range(left, self._feature_count):
                value = 0.5 * (
                    self._second_moment[left][right]
                    + self._second_moment[right][left]
                ) / denominator
                covariance[left][right] = value
                covariance[right][left] = value
        return _finish_pca(
            observation_count=self._count,
            mean=tuple(self._mean),
            covariance=covariance,
            component_count=component_count,
            tolerance=tolerance,
            max_sweeps=max_sweeps,
        )


@dataclass(frozen=True, slots=True)
class PairedEffect:
    """Per-example intervention deltas and their paired aggregate."""

    example_count: int
    baseline_mean: float
    intervention_mean: float
    mean_delta: float
    mean_effect: float
    standard_error: float
    improved_count: int
    worsened_count: int
    tied_count: int
    deltas: tuple[float, ...]
    higher_is_better: bool


class SelectorTarget(str, Enum):
    REVERSE = "reverse"
    COPY = "copy"


@dataclass(frozen=True, slots=True)
class SteeringSpec:
    """Lab-local affine steering request for the F selector basis."""

    label: str
    target: SelectorTarget
    input_index: int
    positions: tuple[int, ...]
    scales: tuple[float, ...]
    offsets: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label:
            raise ValueError("label must be a nonempty str")
        if not isinstance(self.target, SelectorTarget):
            raise TypeError("target must be a SelectorTarget")
        _require_nonnegative_int("input_index", self.input_index)
        if not self.positions:
            raise ValueError("positions must not be empty")
        for index, position in enumerate(self.positions):
            _require_nonnegative_int(f"positions[{index}]", position)
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("positions must be unique")
        if len(self.scales) < 2:
            raise ValueError("selector steering needs at least two scales")
        _require_finite_vector("scales", self.scales)
        if self.offsets:
            if len(self.offsets) != len(self.scales):
                raise ValueError("offsets must match scales")
            _require_finite_vector("offsets", self.offsets)


def fit_pca(
    rows: Iterable[Sequence[float]],
    component_count: int = 2,
    *,
    tolerance: float = 1.0e-10,
    max_sweeps: int = 64,
) -> PcaResult:
    """Fit PCA to captured rows using a cyclic symmetric Jacobi solver."""

    observations = _materialize_rows(rows, minimum_rows=2)
    feature_count = len(observations[0])

    observation_count = len(observations)
    mean = tuple(
        math.fsum(row[column] for row in observations) / observation_count
        for column in range(feature_count)
    )
    centered = tuple(
        tuple(value - mean[column] for column, value in enumerate(row))
        for row in observations
    )
    denominator = observation_count - 1
    covariance = [[0.0] * feature_count for _ in range(feature_count)]
    for left in range(feature_count):
        for right in range(left, feature_count):
            value = math.fsum(
                row[left] * row[right] for row in centered
            ) / denominator
            covariance[left][right] = value
            covariance[right][left] = value

    return _finish_pca(
        observation_count=observation_count,
        mean=mean,
        covariance=covariance,
        component_count=component_count,
        tolerance=tolerance,
        max_sweeps=max_sweeps,
    )


def project_rows(
    rows: Iterable[Sequence[float]],
    pca: PcaResult,
) -> tuple[tuple[float, ...], ...]:
    """Project captured rows through a previously fitted PCA basis."""

    if not isinstance(pca, PcaResult):
        raise TypeError("pca must be a PcaResult")
    observations = _materialize_rows(
        rows,
        minimum_rows=0,
        empty_width=pca.feature_count,
    )
    if observations and len(observations[0]) != pca.feature_count:
        raise ValueError("row width does not match the PCA feature count")
    return tuple(
        tuple(
            math.fsum(
                (row[index] - pca.mean[index]) * component[index]
                for index in range(pca.feature_count)
            )
            for component in pca.components
        )
        for row in observations
    )


def paired_effect(
    baseline: Iterable[float],
    intervention: Iterable[float],
    *,
    higher_is_better: bool = True,
    tie_tolerance: float = 0.0,
) -> PairedEffect:
    """Compare aligned per-example scores without discarding pairing."""

    baseline_values = _finite_values("baseline", baseline)
    intervention_values = _finite_values("intervention", intervention)
    if not baseline_values:
        raise ValueError("paired effects require at least one example")
    if len(baseline_values) != len(intervention_values):
        raise ValueError("baseline and intervention must have equal length")
    if not isinstance(higher_is_better, bool):
        raise TypeError("higher_is_better must be a bool")
    _require_nonnegative_finite("tie_tolerance", tie_tolerance)

    deltas = tuple(
        changed - original
        for original, changed in zip(baseline_values, intervention_values)
    )
    oriented = deltas if higher_is_better else tuple(-value for value in deltas)
    count = len(deltas)
    mean_delta = math.fsum(deltas) / count
    mean_effect = math.fsum(oriented) / count
    if count == 1:
        standard_error = 0.0
    else:
        squared_deviations = math.fsum(
            (value - mean_effect) ** 2 for value in oriented
        )
        standard_error = math.sqrt(squared_deviations / (count * (count - 1)))
    tolerance_value = float(tie_tolerance)
    improved = sum(value > tolerance_value for value in oriented)
    worsened = sum(value < -tolerance_value for value in oriented)
    tied = count - improved - worsened
    return PairedEffect(
        example_count=count,
        baseline_mean=math.fsum(baseline_values) / count,
        intervention_mean=math.fsum(intervention_values) / count,
        mean_delta=mean_delta,
        mean_effect=mean_effect,
        standard_error=standard_error,
        improved_count=improved,
        worsened_count=worsened,
        tied_count=tied,
        deltas=deltas,
        higher_is_better=higher_is_better,
    )


def compare_paired_conditions(
    baseline: Iterable[float],
    interventions: Mapping[str, Iterable[float]],
    *,
    higher_is_better: bool = True,
    tie_tolerance: float = 0.0,
) -> dict[str, PairedEffect]:
    """Apply the same paired baseline to named ablation/intervention scores."""

    if not isinstance(interventions, Mapping):
        raise TypeError("interventions must be a mapping")
    baseline_values = _finite_values("baseline", baseline)
    result: dict[str, PairedEffect] = {}
    for name, scores in interventions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("intervention names must be nonempty strings")
        result[name] = paired_effect(
            baseline_values,
            scores,
            higher_is_better=higher_is_better,
            tie_tolerance=tie_tolerance,
        )
    return result


def make_selector_steering(
    target: SelectorTarget,
    program_width: int,
    *,
    position: int = 0,
    strength: float = 100.0,
) -> SteeringSpec:
    """Construct the F-selector scale vector for one semantic direction."""

    if not isinstance(target, SelectorTarget):
        raise TypeError("target must be a SelectorTarget")
    _require_positive_int("program_width", program_width)
    if program_width < 2:
        raise ValueError("program_width must provide reverse and copy coordinates")
    _require_nonnegative_int("position", position)
    _require_positive_finite("strength", strength)
    strength_value = float(strength)
    if target is SelectorTarget.REVERSE:
        scales = (strength_value,) + (0.0,) * (program_width - 1)
    else:
        scales = (0.0,) + (strength_value,) * (program_width - 1)
    return SteeringSpec(
        label=f"only_{target.value}_basis",
        target=target,
        input_index=0,
        positions=(position,),
        scales=scales,
    )


def steering_specs_for_variant(
    variant: Variant,
    program_width: int,
    *,
    position: int = 0,
    strength: float = 100.0,
) -> tuple[SteeringSpec, ...]:
    """Return semantic selector interventions only where they are defined."""

    if not isinstance(variant, Variant):
        raise TypeError("variant must be a Variant")
    _require_positive_int("program_width", program_width)
    if variant is not Variant.F:
        return ()
    return tuple(
        make_selector_steering(
            target,
            program_width,
            position=position,
            strength=strength,
        )
        for target in (SelectorTarget.REVERSE, SelectorTarget.COPY)
    )


def _jacobi_eigensystem(
    matrix: list[list[float]],
    *,
    tolerance: float,
    max_sweeps: int,
) -> tuple[list[float], list[list[float]], bool, int]:
    size = len(matrix)
    eigenvectors = [
        [1.0 if row == column else 0.0 for column in range(size)]
        for row in range(size)
    ]
    if size == 1:
        return [matrix[0][0]], eigenvectors, True, 0

    converged = False
    completed_sweeps = 0
    for sweep in range(1, max_sweeps + 1):
        completed_sweeps = sweep
        diagonal_scale = max(
            1.0, max(abs(matrix[index][index]) for index in range(size))
        )
        largest_off_diagonal = max(
            abs(matrix[left][right])
            for left in range(size)
            for right in range(left + 1, size)
        )
        if largest_off_diagonal <= tolerance * diagonal_scale:
            converged = True
            break

        for left in range(size - 1):
            for right in range(left + 1, size):
                cross = matrix[left][right]
                if abs(cross) <= tolerance * diagonal_scale:
                    continue
                left_value = matrix[left][left]
                right_value = matrix[right][right]
                tau = (right_value - left_value) / (2.0 * cross)
                tangent = math.copysign(
                    1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau)),
                    tau,
                )
                cosine = 1.0 / math.sqrt(1.0 + tangent * tangent)
                sine = tangent * cosine

                for index in range(size):
                    if index in (left, right):
                        continue
                    old_left = matrix[index][left]
                    old_right = matrix[index][right]
                    new_left = cosine * old_left - sine * old_right
                    new_right = sine * old_left + cosine * old_right
                    matrix[index][left] = new_left
                    matrix[left][index] = new_left
                    matrix[index][right] = new_right
                    matrix[right][index] = new_right

                matrix[left][left] = (
                    cosine * cosine * left_value
                    - 2.0 * sine * cosine * cross
                    + sine * sine * right_value
                )
                matrix[right][right] = (
                    sine * sine * left_value
                    + 2.0 * sine * cosine * cross
                    + cosine * cosine * right_value
                )
                matrix[left][right] = 0.0
                matrix[right][left] = 0.0

                for row in range(size):
                    old_left = eigenvectors[row][left]
                    old_right = eigenvectors[row][right]
                    eigenvectors[row][left] = (
                        cosine * old_left - sine * old_right
                    )
                    eigenvectors[row][right] = (
                        sine * old_left + cosine * old_right
                    )

    if not converged:
        diagonal_scale = max(
            1.0, max(abs(matrix[index][index]) for index in range(size))
        )
        largest_off_diagonal = max(
            abs(matrix[left][right])
            for left in range(size)
            for right in range(left + 1, size)
        )
        converged = largest_off_diagonal <= tolerance * diagonal_scale
    return (
        [matrix[index][index] for index in range(size)],
        eigenvectors,
        converged,
        completed_sweeps,
    )


def _finish_pca(
    *,
    observation_count: int,
    mean: tuple[float, ...],
    covariance: list[list[float]],
    component_count: int,
    tolerance: float,
    max_sweeps: int,
) -> PcaResult:
    _require_positive_int("component_count", component_count)
    _require_positive_finite("tolerance", tolerance)
    _require_positive_int("max_sweeps", max_sweeps)
    feature_count = len(mean)
    if component_count > feature_count:
        raise ValueError("component_count exceeds the feature count")
    eigenvalues, eigenvectors, converged, sweeps = _jacobi_eigensystem(
        covariance,
        tolerance=float(tolerance),
        max_sweeps=max_sweeps,
    )
    ordered = sorted(
        range(feature_count),
        key=lambda index: eigenvalues[index],
        reverse=True,
    )
    nonnegative_eigenvalues = tuple(
        max(0.0, eigenvalues[index]) for index in ordered
    )
    total_variance = math.fsum(nonnegative_eigenvalues)
    components: list[tuple[float, ...]] = []
    for eigen_index in ordered[:component_count]:
        component = tuple(
            eigenvectors[row][eigen_index] for row in range(feature_count)
        )
        components.append(_canonicalize_component_sign(component))
    selected_eigenvalues = nonnegative_eigenvalues[:component_count]
    explained = tuple(
        value / total_variance if total_variance > 0.0 else 0.0
        for value in selected_eigenvalues
    )
    return PcaResult(
        observation_count=observation_count,
        feature_count=feature_count,
        mean=mean,
        components=tuple(components),
        eigenvalues=selected_eigenvalues,
        explained_variance_ratio=explained,
        converged=converged,
        sweeps=sweeps,
    )


def _canonicalize_component_sign(
    component: tuple[float, ...],
) -> tuple[float, ...]:
    pivot = max(range(len(component)), key=lambda index: abs(component[index]))
    if component[pivot] < 0.0:
        return tuple(-value for value in component)
    return component


def _materialize_rows(
    rows: Iterable[Sequence[float]],
    *,
    minimum_rows: int,
    empty_width: int | None = None,
) -> tuple[tuple[float, ...], ...]:
    try:
        raw_rows = tuple(rows)
    except TypeError as error:
        raise TypeError("rows must be iterable") from error
    if len(raw_rows) < minimum_rows:
        raise ValueError(f"rows must contain at least {minimum_rows} observations")
    if not raw_rows:
        return ()

    result: list[tuple[float, ...]] = []
    width: int | None = empty_width
    for row_index, row in enumerate(raw_rows):
        try:
            values = tuple(row)
        except TypeError as error:
            raise TypeError(f"rows[{row_index}] must be iterable") from error
        if not values:
            raise ValueError("rows must have at least one feature")
        if width is None:
            width = len(values)
        if len(values) != width:
            raise ValueError("rows must be rectangular")
        converted: list[float] = []
        for column, value in enumerate(values):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    f"rows[{row_index}][{column}] must be a number"
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(
                    f"rows[{row_index}][{column}] must be finite"
                )
            converted.append(number)
        result.append(tuple(converted))
    return tuple(result)


def _finite_row(row: Sequence[float], name: str) -> tuple[float, ...]:
    try:
        values = tuple(row)
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error
    if not values:
        raise ValueError("rows must have at least one feature")
    converted: list[float] = []
    for column, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{column}] must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{column}] must be finite")
        converted.append(number)
    return tuple(converted)


def _finite_values(name: str, values: Iterable[float]) -> tuple[float, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as error:
        raise TypeError(f"{name} must be iterable") from error
    result: list[float] = []
    for index, value in enumerate(raw_values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{index}] must be a number")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be finite")
        result.append(number)
    return tuple(result)


def _require_finite_vector(name: str, values: Sequence[float]) -> None:
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name}[{index}] must be a number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name}[{index}] must be finite")


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


def _require_nonnegative_finite(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{name} must be nonnegative and finite")


def _require_positive_finite(name: str, value: object) -> None:
    _require_nonnegative_finite(name, value)
    if float(value) == 0.0:
        raise ValueError(f"{name} must be greater than zero")


__all__ = [
    "PairedEffect",
    "PcaAccumulator",
    "PcaResult",
    "SelectorTarget",
    "SteeringSpec",
    "compare_paired_conditions",
    "fit_pca",
    "make_selector_steering",
    "paired_effect",
    "project_rows",
    "steering_specs_for_variant",
]
