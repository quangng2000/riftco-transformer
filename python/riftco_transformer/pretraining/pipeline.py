"""Self-supervised next-token pretraining as a reusable pipeline stage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable

from ..native import Adam, DecoderOnlyTransformer, Tokenizer, TransformerConfig
from ..artifacts import ModelBundle
from ..training import (
    CausalLanguageModelTrainer,
    RandomWindowBatchSource,
    TrainingLoopConfig,
    TrainingMetric,
    fixed_batches,
    selected_backend,
)


@dataclass(frozen=True, slots=True)
class PretrainingConfig:
    """Configuration for a small auditable pretraining run."""

    steps: int = 100
    context_size: int = 32
    batch_size: int = 4
    validation_fraction: float = 0.1
    validation_batch_count: int = 4
    evaluation_interval: int = 10
    loss_average_window: int = 10
    tokenizer_method: str = "bpe"
    vocabulary_size: int = 272
    minimum_pair_frequency: int = 2
    model_width: int = 16
    head_count: int = 4
    block_count: int = 1
    feed_forward_width: int = 32
    learning_rate: float = 1.0e-2
    random_seed: int = 7
    validation_random_seed: int = 17
    backend: str = "auto"
    attention: str = "materialized"
    activation_checkpointing: str = "disabled"

    def __post_init__(self) -> None:
        for name in (
            "steps",
            "context_size",
            "batch_size",
            "validation_batch_count",
            "evaluation_interval",
            "loss_average_window",
            "model_width",
            "head_count",
            "block_count",
            "feed_forward_width",
            "minimum_pair_frequency",
        ):
            _positive_integer(getattr(self, name), name)
        _nonnegative_integer(self.random_seed, "random_seed")
        _nonnegative_integer(
            self.validation_random_seed,
            "validation_random_seed",
        )
        if self.random_seed > (1 << 32) - 1:
            raise ValueError("random_seed must be at most 2**32 - 1")
        if self.model_width % self.head_count != 0:
            raise ValueError("model_width must be divisible by head_count")
        if self.tokenizer_method not in {"byte", "bpe"}:
            raise ValueError("tokenizer_method must be 'byte' or 'bpe'")
        if self.tokenizer_method == "bpe":
            if (
                isinstance(self.vocabulary_size, bool)
                or not isinstance(self.vocabulary_size, int)
            ):
                raise TypeError("vocabulary_size must be an int")
            if not 256 <= self.vocabulary_size <= (1 << 32) - 1:
                raise ValueError(
                    "vocabulary_size must be between 256 and 2**32 - 1"
                )
        fraction = _finite_real(
            self.validation_fraction,
            "validation_fraction",
        )
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                "validation_fraction must be between zero and one"
            )
        learning_rate = _finite_real(
            self.learning_rate,
            "learning_rate",
        )
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero")
        if self.backend not in {"auto", "cpu", "metal"}:
            raise ValueError("backend must be 'auto', 'cpu', or 'metal'")
        if self.attention not in {"materialized", "flash"}:
            raise ValueError(
                "attention must be 'materialized' or 'flash'"
            )
        if self.activation_checkpointing not in {"disabled", "block"}:
            raise ValueError(
                "activation_checkpointing must be 'disabled' or 'block'"
            )


@dataclass(frozen=True, slots=True)
class PretrainingResult:
    """The immutable base artifact and observations from pretraining."""

    bundle: ModelBundle
    metrics: tuple[TrainingMetric, ...]
    training_token_count: int
    validation_token_count: int


def split_pretraining_text(
    text: str,
    validation_fraction: float,
) -> tuple[str, str]:
    """Split raw text before tokenizer fitting to avoid vocabulary leakage."""

    if not isinstance(text, str):
        raise TypeError("text must be a str")
    if len(text) < 2:
        raise ValueError("pretraining text must contain at least two characters")
    fraction = _finite_real(
        validation_fraction,
        "validation_fraction",
    )
    if not 0.0 < fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    validation_characters = max(1, math.ceil(len(text) * fraction))
    training_characters = len(text) - validation_characters
    if training_characters <= 0:
        raise ValueError(
            "validation_fraction leaves no pretraining characters"
        )
    return text[:training_characters], text[training_characters:]


def pretrain_text(
    text: str,
    config: PretrainingConfig | None = None,
    *,
    metric_sink: Callable[[TrainingMetric], None] | None = None,
) -> PretrainingResult:
    """Train from raw text and return a portable base-model artifact."""

    config = PretrainingConfig() if config is None else config
    if not isinstance(config, PretrainingConfig):
        raise TypeError("config must be a PretrainingConfig")
    training_text, validation_text = split_pretraining_text(
        text,
        config.validation_fraction,
    )
    return _pretrain_splits(
        training_text,
        validation_text,
        config,
        validation_source="derived_fraction",
        metric_sink=metric_sink,
    )


def pretrain_splits(
    training_text: str,
    validation_text: str,
    config: PretrainingConfig | None = None,
    *,
    metric_sink: Callable[[TrainingMetric], None] | None = None,
) -> PretrainingResult:
    """Train with an explicit held-out validation corpus.

    The tokenizer is fitted only on ``training_text``. This entry point is
    useful for datasets, such as TinyStories, that already publish a
    validation split.
    """

    config = PretrainingConfig() if config is None else config
    if not isinstance(config, PretrainingConfig):
        raise TypeError("config must be a PretrainingConfig")
    if not isinstance(training_text, str):
        raise TypeError("training_text must be a str")
    if not isinstance(validation_text, str):
        raise TypeError("validation_text must be a str")
    if not training_text:
        raise ValueError("training_text must not be empty")
    if not validation_text:
        raise ValueError("validation_text must not be empty")
    if _text_sha256(training_text) == _text_sha256(validation_text):
        raise ValueError(
            "training_text and validation_text must be distinct held-out "
            "corpora"
        )
    return _pretrain_splits(
        training_text,
        validation_text,
        config,
        validation_source="explicit",
        metric_sink=metric_sink,
    )


def _pretrain_splits(
    training_text: str,
    validation_text: str,
    config: PretrainingConfig,
    *,
    validation_source: str,
    metric_sink: Callable[[TrainingMetric], None] | None,
) -> PretrainingResult:
    tokenizer_arguments: dict[str, object] = {
        "method": config.tokenizer_method,
    }
    if config.tokenizer_method == "bpe":
        tokenizer_arguments.update(
            vocabulary_size=config.vocabulary_size,
            minimum_pair_frequency=config.minimum_pair_frequency,
        )

    with Tokenizer(training_text, **tokenizer_arguments) as tokenizer:
        training_tokens = tokenizer.encode(training_text)
        try:
            validation_tokens = tokenizer.encode(validation_text)
        except Exception as error:
            if config.tokenizer_method == "byte":
                raise ValueError(
                    "the training-derived byte vocabulary cannot encode the "
                    "validation split; use tokenizer_method='bpe'"
                ) from error
            raise
        if len(training_tokens) <= config.context_size:
            raise ValueError(
                "training split must encode to at least context_size + 1 "
                "tokens"
            )
        if len(validation_tokens) <= config.context_size:
            raise ValueError(
                "validation split must encode to at least context_size + 1 "
                "tokens"
            )

        model_config = TransformerConfig(
            vocabulary_size=tokenizer.vocab_size,
            maximum_context=config.context_size,
            model_width=config.model_width,
            head_count=config.head_count,
            block_count=config.block_count,
            feed_forward_width=config.feed_forward_width,
            random_seed=config.random_seed,
        )
        backend = selected_backend(config.backend)
        training_source = RandomWindowBatchSource(
            training_tokens,
            batch_size=config.batch_size,
            context_size=config.context_size,
            random_seed=config.random_seed,
        )
        validation_batch_source = RandomWindowBatchSource(
            validation_tokens,
            batch_size=config.batch_size,
            context_size=config.context_size,
            random_seed=config.validation_random_seed,
        )
        validation_batches = fixed_batches(
            validation_batch_source,
            config.validation_batch_count,
        )
        loop_config = TrainingLoopConfig(
            steps=config.steps,
            evaluation_interval=config.evaluation_interval,
            loss_average_window=config.loss_average_window,
        )

        with DecoderOnlyTransformer(
            model_config,
            attention=config.attention,
            activation_checkpointing=config.activation_checkpointing,
        ).to(backend) as model:
            with model.parameters() as parameters:
                with Adam(
                    parameters,
                    learning_rate=config.learning_rate,
                ) as optimizer:
                    trainer = CausalLanguageModelTrainer(model, optimizer)
                    metrics = trainer.run(
                        training_source,
                        loop_config,
                        validation_batches=validation_batches,
                        metric_sink=metric_sink,
                    )
            bundle = ModelBundle.capture(
                model,
                tokenizer,
                stage="pretraining",
                metadata={
                    "objective": "next_token_prediction",
                    "training_token_count": len(training_tokens),
                    "validation_token_count": len(validation_tokens),
                    "tokenizer_fit_scope": "training_split_only",
                    "validation_source": validation_source,
                    "training_text_sha256": _text_sha256(training_text),
                    "validation_text_sha256": _text_sha256(validation_text),
                    "applied_validation_fraction": (
                        config.validation_fraction
                        if validation_source == "derived_fraction"
                        else None
                    ),
                    "training_config": asdict(config),
                    "resolved_backend": backend,
                    "optimizer": "adam",
                    "optimizer_state_included": False,
                },
            )
    return PretrainingResult(
        bundle=bundle,
        metrics=metrics,
        training_token_count=len(training_tokens),
        validation_token_count=len(validation_tokens),
    )


def pretrain_file(
    path: str | Path,
    config: PretrainingConfig | None = None,
    *,
    metric_sink: Callable[[TrainingMetric], None] | None = None,
) -> PretrainingResult:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{source} is not valid UTF-8") from error
    return pretrain_text(text, config, metric_sink=metric_sink)


def pretrain_files(
    training_path: str | Path,
    validation_path: str | Path,
    config: PretrainingConfig | None = None,
    *,
    metric_sink: Callable[[TrainingMetric], None] | None = None,
) -> PretrainingResult:
    """Train from separate UTF-8 training and validation files."""

    training_source = Path(training_path)
    validation_source = Path(validation_path)
    if (
        training_source.exists()
        and validation_source.exists()
        and training_source.samefile(validation_source)
    ):
        raise ValueError(
            "training and validation paths must identify distinct held-out "
            "files"
        )
    training_text = _read_utf8_text(training_source, "training")
    validation_text = _read_utf8_text(validation_source, "validation")
    return pretrain_splits(
        training_text,
        validation_text,
        config,
        metric_sink=metric_sink,
    )


def _read_utf8_text(path: str | Path, role: str) -> str:
    source = Path(path)
    try:
        return source.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(
            f"{role} file {source} is not valid UTF-8"
        ) from error


def _text_sha256(text: str) -> str:
    """Fingerprint the exact UTF-8 corpus consumed by the tokenizer."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _positive_integer(value: object, name: str) -> int:
    checked = _nonnegative_integer(value, name)
    if checked == 0:
        raise ValueError(f"{name} must be greater than zero")
    return checked


def _nonnegative_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


__all__ = [
    "PretrainingConfig",
    "PretrainingResult",
    "pretrain_file",
    "pretrain_files",
    "pretrain_splits",
    "pretrain_text",
    "split_pretraining_text",
]
