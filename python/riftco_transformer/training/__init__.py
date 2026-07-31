"""Shared causal language-model training engine."""

from .engine import (
    BatchSource,
    CausalLanguageModelTrainer,
    ExampleWindowBatchSource,
    MetricSink,
    RandomWindowBatchSource,
    SequenceWindowBatchSource,
    TrainingBatch,
    TrainingLoopConfig,
    TrainingMetric,
    fixed_batches,
    loss_perplexity,
    selected_backend,
)


__all__ = [
    "BatchSource",
    "CausalLanguageModelTrainer",
    "ExampleWindowBatchSource",
    "MetricSink",
    "RandomWindowBatchSource",
    "SequenceWindowBatchSource",
    "TrainingBatch",
    "TrainingLoopConfig",
    "TrainingMetric",
    "fixed_batches",
    "loss_perplexity",
    "selected_backend",
]
