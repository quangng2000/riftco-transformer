"""Self-supervised next-token pretraining pipeline."""

from .pipeline import (
    PretrainingConfig,
    PretrainingResult,
    pretrain_file,
    pretrain_files,
    pretrain_splits,
    pretrain_text,
    split_pretraining_text,
)


__all__ = [
    "PretrainingConfig",
    "PretrainingResult",
    "pretrain_file",
    "pretrain_files",
    "pretrain_splits",
    "pretrain_text",
    "split_pretraining_text",
]
