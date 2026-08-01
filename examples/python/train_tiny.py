"""Train a tiny decoder-only Transformer directly from UTF-8 text."""

from __future__ import annotations

import argparse
from collections import deque
import math
import random
from pathlib import Path

from riftco_transformer import (
    Adam,
    DecoderOnlyTransformer,
    Tokenizer,
    TransformerConfig,
    backend_available,
    cross_entropy,
)

DEFAULT_TEXT = (
    "small models learn from text.\n"
    "text becomes bytes, and bytes become tokens.\n"
) * 3
TRAINING_RANDOM_SEED = 7
VALIDATION_RANDOM_SEED = 17

TokenRows = list[list[int]]
NextTokenBatch = tuple[TokenRows, TokenRows]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a dependency-free Transformer on UTF-8 text."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--text",
        help="Literal UTF-8 training text.",
    )
    source.add_argument(
        "--file",
        type=Path,
        help="Path to a UTF-8 training-text file.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "metal", "cuda", "tpu"),
        default="auto",
        help=(
            "Execution backend; auto prefers TPU, then CUDA, Metal, and CPU."
        ),
    )
    parser.add_argument(
        "--attention",
        choices=("materialized", "flash"),
        default="materialized",
        help="Full-sequence attention implementation.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        choices=("disabled", "block"),
        default="disabled",
        help="Retain every activation or recompute each block in backward.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of training steps.",
    )
    parser.add_argument(
        "--context",
        type=int,
        default=16,
        help="Tokens in each next-token training window.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of sampled windows per training step.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.1,
        help=(
            "Target fraction of encoded tokens held out from model "
            "training; at least context + 1 are reserved (default: 0.1)."
        ),
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=4,
        help=(
            "Number of fixed held-out batches averaged per evaluation "
            "(default: 4)."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=10,
        help=(
            "Evaluate every N optimizer steps; the final step is always "
            "evaluated (default: 10)."
        ),
    )
    parser.add_argument(
        "--loss-average-window",
        type=int,
        default=10,
        help=(
            "Number of recent training losses in the moving average "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        choices=("byte", "bpe"),
        default="bpe",
        help="Tokenization strategy; BPE is the tutorial default.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=272,
        help="Maximum BPE vocabulary size (256 base bytes plus merges).",
    )
    parser.add_argument(
        "--min-pair-frequency",
        type=int,
        default=2,
        help="Minimum corpus frequency required for a BPE merge.",
    )
    arguments = parser.parse_args()
    if arguments.steps <= 0:
        parser.error("--steps must be greater than zero")
    if arguments.context <= 0:
        parser.error("--context must be greater than zero")
    if arguments.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    if not 0.0 < arguments.validation_fraction < 1.0:
        parser.error("--validation-fraction must be between zero and one")
    if arguments.validation_batches <= 0:
        parser.error("--validation-batches must be greater than zero")
    if arguments.eval_every <= 0:
        parser.error("--eval-every must be greater than zero")
    if arguments.loss_average_window <= 0:
        parser.error("--loss-average-window must be greater than zero")
    if arguments.tokenizer == "bpe":
        if not 256 <= arguments.vocab_size <= (1 << 32) - 1:
            parser.error(
                "--vocab-size must be between 256 and 4294967295"
            )
        if arguments.min_pair_frequency < 1:
            parser.error("--min-pair-frequency must be at least 1")
    return arguments


def selected_backend(requested: str) -> str:
    if requested == "auto":
        for accelerated in ("tpu", "cuda", "metal"):
            if backend_available(accelerated):
                return accelerated
        return "cpu"
    if requested in {"metal", "cuda", "tpu"} and not backend_available(
        requested
    ):
        display_name = {
            "metal": "Metal",
            "cuda": "CUDA",
            "tpu": "TPU",
        }[requested]
        raise RuntimeError(
            f"{display_name} was requested but is unavailable"
        )
    return requested


def training_text(arguments: argparse.Namespace) -> str:
    if arguments.file is not None:
        try:
            return arguments.file.read_bytes().decode("utf-8")
        except OSError as error:
            raise RuntimeError(
                f"could not read training file {arguments.file}: {error}"
            ) from error
        except UnicodeDecodeError as error:
            raise RuntimeError(
                f"training file {arguments.file} is not valid UTF-8"
            ) from error
    if arguments.text is not None:
        return arguments.text
    return DEFAULT_TEXT


def next_token_batch(
    corpus_tokens: list[int],
    batch_size: int,
    context_size: int,
    random_generator: random.Random,
) -> NextTokenBatch:
    if batch_size <= 0:
        raise ValueError("batch size must be greater than zero")
    if context_size <= 0:
        raise ValueError("context size must be greater than zero")
    if len(corpus_tokens) <= context_size:
        raise ValueError(
            "a token region must contain at least context size + 1 tokens"
        )

    start_count = len(corpus_tokens) - context_size
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for _ in range(batch_size):
        start = random_generator.randrange(start_count)
        inputs.append(corpus_tokens[start : start + context_size])
        targets.append(corpus_tokens[start + 1 : start + context_size + 1])
    return inputs, targets


def split_token_corpus(
    corpus_tokens: list[int],
    context_size: int,
    validation_fraction: float,
) -> tuple[list[int], list[int]]:
    """Return disjoint training and held-out next-token regions."""

    if context_size <= 0:
        raise ValueError("context size must be greater than zero")
    if (
        not math.isfinite(validation_fraction)
        or not 0.0 < validation_fraction < 1.0
    ):
        raise ValueError("validation fraction must be between zero and one")

    minimum_region_size = context_size + 1
    validation_size = max(
        minimum_region_size,
        math.ceil(len(corpus_tokens) * validation_fraction),
    )
    training_size = len(corpus_tokens) - validation_size
    if training_size < minimum_region_size:
        raise ValueError(
            "training and validation each need at least --context + 1 "
            "encoded tokens; provide more text, reduce --context, or "
            "reduce --validation-fraction"
        )
    return (
        corpus_tokens[:training_size],
        corpus_tokens[training_size:],
    )


def fixed_validation_batches(
    validation_tokens: list[int],
    batch_size: int,
    context_size: int,
    batch_count: int,
) -> tuple[NextTokenBatch, ...]:
    """Sample a deterministic validation set once for the entire run."""

    if batch_count <= 0:
        raise ValueError("validation batch count must be greater than zero")
    validation_random = random.Random(VALIDATION_RANDOM_SEED)
    return tuple(
        next_token_batch(
            validation_tokens,
            batch_size,
            context_size,
            validation_random,
        )
        for _ in range(batch_count)
    )


def average_validation_loss(
    model: DecoderOnlyTransformer,
    batches: tuple[NextTokenBatch, ...],
) -> float:
    """Evaluate fixed batches without backward passes or parameter updates."""

    if not batches:
        raise ValueError("validation batches must not be empty")
    loss_total = 0.0
    for tokens, targets in batches:
        with model(tokens) as logits:
            with cross_entropy(logits, targets) as loss:
                loss_total += loss.item()
    return loss_total / len(batches)


def loss_perplexity(loss_value: float) -> float:
    try:
        return math.exp(loss_value)
    except OverflowError:
        return math.inf


def print_validation(step: int, loss_value: float) -> None:
    print(
        f"validation step={step} "
        f"validation_loss={loss_value:.6f} "
        f"perplexity={loss_perplexity(loss_value):.6f}"
    )


def main() -> None:
    arguments = parse_arguments()
    backend = selected_backend(arguments.backend)
    corpus = training_text(arguments)

    if arguments.tokenizer == "bpe":
        tokenizer_context = Tokenizer(
            corpus,
            method="bpe",
            vocabulary_size=arguments.vocab_size,
            minimum_pair_frequency=arguments.min_pair_frequency,
        )
    else:
        tokenizer_context = Tokenizer(corpus, method="byte")

    with tokenizer_context as tokenizer:
        corpus_tokens = tokenizer.encode(corpus)
        if tokenizer.decode(corpus_tokens) != corpus:
            raise RuntimeError("tokenizer round trip changed the corpus")
        training_tokens, validation_tokens = split_token_corpus(
            corpus_tokens,
            arguments.context,
            arguments.validation_fraction,
        )
        validation_batches = fixed_validation_batches(
            validation_tokens,
            arguments.batch_size,
            arguments.context,
            arguments.validation_batches,
        )

        config = TransformerConfig(
            vocabulary_size=tokenizer.vocab_size,
            maximum_context=arguments.context,
            model_width=16,
            head_count=4,
            block_count=1,
            feed_forward_width=32,
            random_seed=TRAINING_RANDOM_SEED,
        )
        batch_random = random.Random(TRAINING_RANDOM_SEED)
        recent_losses: deque[float] = deque(
            maxlen=arguments.loss_average_window
        )

        # Move the model before creating its parameter view and optimizer.
        with DecoderOnlyTransformer(
            config,
            attention=arguments.attention,
            activation_checkpointing=(
                arguments.activation_checkpointing
            ),
        ).to(backend) as model:
            with model.parameters() as parameters:
                with Adam(parameters, learning_rate=1.0e-2) as optimizer:
                    print(
                        f"backend={model.backend} "
                        f"attention={model.full_sequence_attention} "
                        "activation_checkpointing="
                        f"{model.activation_checkpointing} "
                        f"tokenizer={tokenizer.method} "
                        f"corpus_bytes={len(corpus.encode('utf-8'))} "
                        f"corpus_tokens={len(corpus_tokens)} "
                        f"training_tokens={len(training_tokens)} "
                        f"validation_tokens={len(validation_tokens)} "
                        f"vocabulary={tokenizer.vocab_size} "
                        f"parameter_tensors={len(parameters)}"
                    )
                    print_validation(
                        0,
                        average_validation_loss(model, validation_batches),
                    )

                    for _ in range(arguments.steps):
                        tokens, targets = next_token_batch(
                            training_tokens,
                            arguments.batch_size,
                            arguments.context,
                            batch_random,
                        )

                        # A parameter update invalidates the previous graph,
                        # so each step constructs a fresh forward and loss.
                        with model(tokens) as logits:
                            with cross_entropy(logits, targets) as loss:
                                loss_value = loss.item()
                                loss.backward()
                                statistics = optimizer.step()

                        recent_losses.append(loss_value)
                        print(
                            f"step={statistics.step} "
                            f"train_loss={loss_value:.6f} "
                            f"train_loss_average="
                            f"{sum(recent_losses) / len(recent_losses):.6f} "
                            f"gradient_norm="
                            f"{statistics.gradient_norm:.6f} "
                            f"clip_scale={statistics.clip_scale:.6f}"
                        )
                        if (
                            statistics.step % arguments.eval_every == 0
                            or statistics.step == arguments.steps
                        ):
                            print_validation(
                                statistics.step,
                                average_validation_loss(
                                    model,
                                    validation_batches,
                                ),
                            )


if __name__ == "__main__":
    main()
