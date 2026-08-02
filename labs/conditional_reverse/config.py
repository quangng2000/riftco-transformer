"""Validated, Python-owned configuration for the conditional-reverse lab."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math

from .protocol import ProtocolConfig, SplitSizes


class Variant(str, Enum):
    """The four learned controls defined by the experiment, not the runtime."""

    F = "F"
    P = "P"
    T = "T"
    I = "I"

    @property
    def has_program(self) -> bool:
        return self is not Variant.I


class Profile(str, Enum):
    """Named run sizes with distinct smoke-test and full-study budgets."""

    QUICK = "quick"
    PAPER = "paper"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Task-side choices for the generic programmed-hybrid model."""

    variant: Variant = Variant.F
    model_width: int = 20
    head_count: int = 2
    parallel_attention_count: int = 2
    feed_forward_width: int = 80
    backend: str = "cpu"
    seed: int = 42
    random_program_scale: float = 0.02

    def __post_init__(self) -> None:
        if not isinstance(self.variant, Variant):
            raise TypeError("variant must be a Variant")
        for name in (
            "model_width",
            "head_count",
            "parallel_attention_count",
            "feed_forward_width",
        ):
            _require_positive_int(name, getattr(self, name))
        if self.model_width % self.head_count != 0:
            raise ValueError("model_width must be divisible by head_count")
        if self.program_width < 2:
            raise ValueError("each attention head needs at least two coordinates")
        if not isinstance(self.backend, str):
            raise TypeError("backend must be a str")
        normalized_backend = self.backend.strip().lower()
        if normalized_backend not in {"cpu", "metal", "cuda", "tpu"}:
            raise ValueError("backend must be cpu, metal, cuda, or tpu")
        object.__setattr__(self, "backend", normalized_backend)
        _require_int("seed", self.seed)
        _require_positive_finite(
            "random_program_scale", self.random_program_scale
        )

    @property
    def program_width(self) -> int:
        return self.model_width // self.head_count

    def with_variant(self, variant: Variant) -> ModelConfig:
        if not isinstance(variant, Variant):
            raise TypeError("variant must be a Variant")
        return replace(self, variant=variant)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Python-loop training controls shared by every variant."""

    epochs: int = 10
    batch_size: int = 128
    evaluation_batch_size: int = 256
    shuffle: bool = True
    seed: int = 42
    maximum_steps: int | None = None
    learning_rate: float = 1.0e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    # The study uses ordinary Adam without gradient clipping. The native Adam
    # contract requires a finite limit, so float32 max is the faithful value.
    maximum_gradient_norm: float = 3.4028234663852886e38

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "evaluation_batch_size"):
            _require_positive_int(name, getattr(self, name))
        if not isinstance(self.shuffle, bool):
            raise TypeError("shuffle must be a bool")
        _require_int("seed", self.seed)
        if self.maximum_steps is not None:
            _require_positive_int("maximum_steps", self.maximum_steps)
        _require_positive_finite("learning_rate", self.learning_rate)
        for name in ("beta1", "beta2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
                raise ValueError(f"{name} must be finite and between zero and one")
        _require_positive_finite("epsilon", self.epsilon)
        _require_positive_finite(
            "maximum_gradient_norm", self.maximum_gradient_norm
        )


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Numerical and intervention settings owned by the lab."""

    pca_components: int = 3
    pca_tolerance: float = 1.0e-10
    pca_max_sweeps: int = 64
    ablation_shift: int = 1
    steering_strength: float = 100.0

    def __post_init__(self) -> None:
        _require_positive_int("pca_components", self.pca_components)
        _require_positive_finite("pca_tolerance", self.pca_tolerance)
        _require_positive_int("pca_max_sweeps", self.pca_max_sweeps)
        _require_positive_int("ablation_shift", self.ablation_shift)
        _require_positive_finite("steering_strength", self.steering_strength)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete reproducible configuration for one lab variant."""

    profile: Profile
    protocol: ProtocolConfig
    split_sizes: SplitSizes
    model: ModelConfig
    training: TrainingConfig
    analysis: AnalysisConfig

    def __post_init__(self) -> None:
        if not isinstance(self.profile, Profile):
            raise TypeError("profile must be a Profile")
        if not isinstance(self.protocol, ProtocolConfig):
            raise TypeError("protocol must be a ProtocolConfig")
        if not isinstance(self.split_sizes, SplitSizes):
            raise TypeError("split_sizes must be SplitSizes")
        if not isinstance(self.model, ModelConfig):
            raise TypeError("model must be a ModelConfig")
        if not isinstance(self.training, TrainingConfig):
            raise TypeError("training must be a TrainingConfig")
        if not isinstance(self.analysis, AnalysisConfig):
            raise TypeError("analysis must be an AnalysisConfig")


def make_profile(
    profile: Profile | str,
    *,
    variant: Variant = Variant.F,
    seed: int = 42,
    backend: str = "cpu",
) -> ExperimentConfig:
    """Construct a named profile without importing the native framework."""

    if isinstance(profile, str):
        try:
            profile = Profile(profile.strip().lower())
        except ValueError as error:
            raise ValueError("profile must be quick or paper") from error
    if not isinstance(profile, Profile):
        raise TypeError("profile must be a Profile or str")
    if not isinstance(variant, Variant):
        raise TypeError("variant must be a Variant")
    _require_int("seed", seed)

    if profile is Profile.QUICK:
        protocol = ProtocolConfig(
            sequence_length=3,
            alphabet="abcdefghij",
            reverse_when_first_is="ae",
            seed=seed,
        )
        return ExperimentConfig(
            profile=profile,
            protocol=protocol,
            split_sizes=SplitSizes(
                train=128,
                probe=64,
                validation=64,
                test=64,
            ),
            model=ModelConfig(
                variant=variant,
                model_width=8,
                head_count=2,
                parallel_attention_count=2,
                feed_forward_width=24,
                backend=backend,
                seed=seed,
            ),
            training=TrainingConfig(
                epochs=8,
                batch_size=16,
                evaluation_batch_size=64,
                seed=seed,
                maximum_steps=64,
                learning_rate=1.0e-2,
            ),
            analysis=AnalysisConfig(pca_components=2),
        )

    return ExperimentConfig(
        profile=profile,
        protocol=ProtocolConfig(seed=seed),
        split_sizes=SplitSizes(),
        model=ModelConfig(variant=variant, backend=backend, seed=seed),
        training=TrainingConfig(seed=seed),
        analysis=AnalysisConfig(),
    )


def _require_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")


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
    "AnalysisConfig",
    "ExperimentConfig",
    "ModelConfig",
    "Profile",
    "TrainingConfig",
    "Variant",
    "make_profile",
]
