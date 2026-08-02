"""Full fine-tuning versus LoRA experiment protocol."""

from .protocol import (
    FineTuningCandidate,
    FineTuningComparison,
    FineTuningExperimentConfig,
    FineTuningTrial,
    compare_fine_tuning,
)
from riftco_transformer.post_training import load_prepared_instruction_splits


__all__ = [
    "FineTuningCandidate",
    "FineTuningComparison",
    "FineTuningExperimentConfig",
    "FineTuningTrial",
    "compare_fine_tuning",
    "load_prepared_instruction_splits",
]
