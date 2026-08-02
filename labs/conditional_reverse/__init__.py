"""Conditional string-reversal research protocol."""

from .protocol import (
    Evaluation,
    Example,
    ProtocolConfig,
    SplitSizes,
    evaluate,
    generate_disjoint_splits,
    predict_copy,
    predict_oracle,
    predict_reverse,
)


__all__ = [
    "Evaluation",
    "Example",
    "ProtocolConfig",
    "SplitSizes",
    "evaluate",
    "generate_disjoint_splits",
    "predict_copy",
    "predict_oracle",
    "predict_reverse",
]
