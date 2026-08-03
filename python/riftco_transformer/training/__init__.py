"""Shared causal language-model training engine."""

from .engine import (
    BatchSource,
    BatchSourceState,
    CausalLanguageModelTrainer,
    CheckpointableBatchSource,
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
    "BatchSourceState",
    "CausalLanguageModelTrainer",
    "CheckpointableBatchSource",
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
