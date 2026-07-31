# transformer-lab Python client

This package is the typed, runtime-dependency-free `ctypes` interface to
`libtransformer_lab_c`, plus explicit data preparation, pretraining,
post-training, experiment, artifact, generation, and local-serving modules.
A platform wheel carries both the Python modules and its native C ABI library;
users do not install the native framework separately.

## Install

After a release has been published to PyPI:

```bash
python3 -m pip install transformer-lab
python3 -c "from transformer_lab import Context; print(Context().backend)"
```

The wheel stores `libtransformer_lab_c.so`,
`libtransformer_lab_c.dylib`, or `transformer_lab_c.dll` under
`transformer_lab/.libs`. It has no third-party runtime dependencies and needs
no compiler or environment variable after installation. Initial binary wheels
cover Linux `x86_64` and `aarch64` for both glibc (`manylinux`) and musl
(`musllinux`), macOS `x86_64` and `arm64`, and Windows `AMD64`. CPU is
available on every supported platform; the macOS wheels also include Metal.

From a source checkout, install at the repository root with:

```bash
python3 -m pip install .
```

That source build compiles the C++20 implementation, so it needs a supported
native compiler and platform SDK. `TRANSFORMER_LAB_LIBRARY` remains an advanced
development override for selecting a particular local native build; released
wheels do not require it.

The Python package follows the framework release version (`0.1.0` here), while
the native C ABI has its own compatibility version (`1.8`). The client accepts
the same ABI major and an equal or newer additive minor, and rejects older or
breaking ABIs before use.

## Release automation

`.github/workflows/release.yml` builds and verifies the source distribution and
self-contained platform wheels. `workflow_dispatch` is verification-only.
Pushing a `v<version>` tag creates a GitHub Release after the artifacts pass.
It also publishes to PyPI only when the repository variable
`PUBLISH_TO_PYPI` is `true`.

PyPI publication uses Trusted Publishing rather than a stored API token. The
publisher configuration is project `transformer-lab`, owner `quangng2000`,
repository `transformer-lab`, workflow `release.yml`, and environment `pypi`.
The project is licensed under Apache-2.0. Do not enable publication until the
Trusted Publisher is configured.

## Selectable tokenizers

`Tokenizer` offers interchangeable byte and byte-pair-encoding strategies:

```python
from transformer_lab import Tokenizer

with Tokenizer(
    "Hello, café 🙂 Hello again.",
    method="bpe",
    vocabulary_size=272,
    minimum_pair_frequency=2,
) as tokenizer:
    token_ids = tokenizer.encode("café 🙂")
    assert tokenizer.decode(token_ids) == "café 🙂"
    print(tokenizer.method, tokenizer.vocab_size)
    print(tokenizer.vocabulary)
```

The BPE vocabulary begins with all 256 single-byte tokens and appends learned
pair pieces, so unseen bytes remain encodable. `vocabulary_size` is a maximum:
learning can stop earlier when no pair reaches `minimum_pair_frequency`.
Repeated pair counts are resolved deterministically.

For backward compatibility, `Tokenizer(corpus)` selects the corpus-derived
byte method. It assigns IDs by sorted unsigned byte value:

```python
with Tokenizer(b"cab\ncab") as tokenizer:
    assert tokenizer.method == "byte"
    assert tokenizer.vocabulary_bytes == b"\nabc"
```

`Tokenizer` accepts a `str` corpus, encoded as UTF-8, or a
`bytes`/`bytearray`/`memoryview` corpus. `encode()` and `decode()` are strict
UTF-8 conveniences. Use `encode_bytes()` and `decode_bytes()` for arbitrary
binary data, including embedded NUL bytes. `vocabulary` returns a tuple of
byte pieces for either method; `vocabulary_bytes` is the byte-only
compatibility property.

## End-to-end training

An end-to-end BPE training step uses only public Python objects:

```python
from transformer_lab import (
    Adam,
    DecoderOnlyTransformer,
    Tokenizer,
    TransformerConfig,
    cross_entropy,
)

corpus = "hello hello hello"
with Tokenizer(
    corpus,
    method="bpe",
    vocabulary_size=272,
) as tokenizer:
    encoded = tokenizer.encode(corpus)
    tokens = [encoded[:-1]]
    targets = [encoded[1:]]

    config = TransformerConfig(
        vocabulary_size=tokenizer.vocab_size,
        maximum_context=len(tokens[0]),
        model_width=16,
        head_count=4,
        block_count=1,
        feed_forward_width=32,
    )
    with DecoderOnlyTransformer(
        config,
        attention="flash",
        activation_checkpointing="block",
    ).to("cpu") as model:
        with model.parameters() as parameters:
            with Adam(parameters) as optimizer:
                with cross_entropy(model(tokens), targets) as loss:
                    loss_value = loss.item()
                    loss.backward()
                    stats = optimizer.step()
                print(loss_value, stats.gradient_norm)
```

Change `.to("cpu")` to `.to("metal")` on systems with the Metal backend.
Computation graphs are single-use: build a fresh forward/loss graph for each
training step. `attention="flash"` selects the dependency-free exact tile-8
full-sequence forward/backward implementation; omit it to keep the
`"materialized"` default. The Flash path saves `[batch, heads, time]` row
maxima and exponential sums and reconstructs probabilities during backward,
rather than saving `[batch, heads, time, time]` probabilities. The explicit
probability-returning diagnostic remains materialized. This selector does not
change incremental serving, and no speedup is assumed without measuring the
target workload.

`activation_checkpointing="block"` retains only transformer-block boundaries
and replays each block during backward. Omit it for the `"disabled"` default.
This reduces retained activation graph state at the cost of another block
forward calculation during backward. It composes with FlashAttention and
LoRA, but does not affect model artifacts or incremental decode.

On Metal, Flash working storage is proportional to the per-head width and
must fit the device's threadgroup-memory limit. The native runtime preflights
the complete forward/backward path before starting the forward pass. If the
device rejects a very wide head, use more heads or select
`attention="materialized"`.

## Hugging Face data and rank experiments

The `transformer_lab.data` package is also dependency-free. Its default
transport uses `urllib` to read bounded pages from the official Hugging Face
Dataset Viewer API, while adapters convert TinyStories, Dolly 15K, and
HH-RLHF into stage-specific files. Preparation removes exact duplicates,
assigns records to deterministic content-hash splits, and writes an atomic
directory with a provenance manifest and SHA-256 file digests.

From the framework directory:

```bash
PYTHONPATH="$PWD/python" \
python3 examples/python/prepare_huggingface_data.py \
  --preset dolly \
  --output data/external/huggingface/dolly-lora-v1 \
  --limit 2000 \
  --seed lora-v1
```

`HF_TOKEN` is an optional environment variable, never a CLI argument.
Prepared downloads under `data/external/` are ignored by Git. The Dolly
adapter maps `instruction` plus optional `context` into `prompt`, preserves
`response`, and retains `category`. TinyStories becomes plain text. HH-RLHF
remains chosen/rejected preference data and is not accepted by the current SFT
pipeline.

The `transformer_lab.experiments` package compares LoRA ranks from the same
immutable base:

```python
from transformer_lab.artifacts import ModelBundle
from transformer_lab.experiments import (
    LoraRankExperimentConfig,
    compare_lora_ranks,
    load_prepared_instruction_splits,
)

base = ModelBundle.load("results/stages/tinystories_pretrained.tlab")
splits = load_prepared_instruction_splits(
    "data/external/huggingface/dolly-lora-v1"
)
comparison = compare_lora_ranks(
    base,
    splits,
    LoraRankExperimentConfig(
        ranks=(1, 2, 4, 8),
        alpha_over_rank=2.0,
        steps=20,
        backend="cpu",
    ),
)
print(comparison.best_rank, comparison.selected_test.loss)
```

Every rank shares data fingerprints, seeds, sampler, optimizer controls,
LoRA targets, and `alpha / rank`. Validation selects the winner; held-out test
evaluation begins only after selection. The objective is still
full-sequence causal SFT, not response-only loss. All adapters are merged
before persistence, so ranks have the same serving topology and inference
timings are only smoke measurements. The CLI atomically publishes a new,
complete output directory and embeds the verified prepared-data manifest plus
its SHA-256 in `comparison.json`.

See
`docs/DATASETS_AND_LORA_EXPERIMENTS.md` in the framework repository for
license links, exact TinyStories train/validation commands, sample-size
guidance, provenance details, and the CLI rank workflow.

## Staged pipeline

The high-level modules make the stage boundaries explicit:

```python
from transformer_lab.artifacts import ModelBundle
from transformer_lab import LoraConfig
from transformer_lab.post_training import (
    PostTrainingConfig,
    post_train_jsonl,
)
from transformer_lab.pretraining import PretrainingConfig, pretrain_file
from transformer_lab.serving import ServingConfig, serve_model

base = pretrain_file(
    "data/pretraining/tiny_corpus.txt",
    PretrainingConfig(
        steps=20,
        backend="cpu",
        attention="flash",
        activation_checkpointing="block",
    ),
)
base.bundle.save("results/stages/tiny_pretrained.tlab")

restored = ModelBundle.load("results/stages/tiny_pretrained.tlab")
assistant = post_train_jsonl(
    restored,
    "data/post_training/tiny_instructions.jsonl",
    PostTrainingConfig(
        steps=10,
        backend="cpu",
        attention="flash",
        activation_checkpointing="block",
        fine_tuning_method="lora",
        lora=LoraConfig(rank=4, alpha=8.0),
    ),
)
assistant.bundle.save("results/stages/tiny_post_trained.tlab")

serve_model(
    "results/stages/tiny_post_trained.tlab",
    host="127.0.0.1",
    port=8000,
    config=ServingConfig(
        backend="cpu",
        kv_cache="paged",
        kv_cache_block_size=16,
    ),
)
```

The HTTP adapter serves a dependency-free browser chat at `/` and keeps
`POST /v1/generate` as the stable JSON generation endpoint. Each chat message
is formatted with `PlainChatFormatter` as one independent single-turn SFT
prompt; the visual transcript is not added to the model context. Custom
formatter templates are not persisted in the current artifact format.
`GET /health` reports the selected backend, context and vocabulary sizes, and
active KV-cache strategy.

`ModelBundle` persists the exact byte/BPE tokenizer definition, model
configuration, named parameter shapes, float32 weights, checksums, stage
metadata, and parent artifact ID. It is an immutable inference or warm-start
artifact, not a resumable training checkpoint: Adam moments, optimizer step,
data position, and random-generator state are deliberately not included yet.
LoRA post-training optimizes only adapter factors, then merges them before
capturing this ordinary serving-ready bundle. Full-parameter post-training
remains the default; adapter-only persistence is not part of `ModelBundle`.

The first post-training objective is explicitly
`full_sequence_causal_sft`: it applies causal cross-entropy to the complete
formatted prompt/response sequence. Response-only loss masking is a future
extension.

## Incremental generation

Native models use the current ABI 1.8 `DecodeSession` surface (introduced in
ABI 1.6) instead of rerunning the full-sequence training forward for every
generated token. `TextGenerator` creates a request-local session, prefills the
prompt one token at a time, and then performs one-token decode:

```python
from transformer_lab.artifacts import ModelBundle
from transformer_lab.serving import TextGenerator

bundle = ModelBundle.load("results/stages/tiny_post_trained.tlab")
with bundle.instantiate("cpu") as runtime:
    result = TextGenerator(
        runtime.model,
        runtime.tokenizer,
        kv_cache="paged",
        kv_cache_block_size=16,
    ).generate("Tensor:", max_new_tokens=32)
    print(result.text)
```

Paged caching is the default; use `kv_cache="contiguous"` for the reference
strategy. Both run direct CPU or Metal paged decode attention. When the learned
absolute-position context fills, `TextGenerator` resets the cache and replays
the retained suffix from position zero. A raw session exposes the lower-level
step/reset contract:

```python
with bundle.instantiate("cpu") as runtime:
    with runtime.model.decode_session(
        cache="paged",
        block_size=16,
    ) as session:
        for token in runtime.tokenizer.encode("Tensor:"):
            logits = session.step(token)
```

The raw session does not tokenize, sample, or implement rollover. Paged
storage also does not yet imply continuous batching, a request scheduler, or
prefix sharing. Full-sequence `attention="flash"` does not alter this path:
serving prefill is still token-at-a-time and decode remains paged.

See `docs/PIPELINE.md`, `docs/SERVING.md`, `docs/TOKENIZATION.md`, and
`docs/BACKENDS_AND_PYTHON.md` in the framework repository for the complete
workflow, lifecycle, backend, and error-handling contracts.

## Package layout

The physical package mirrors the runtime boundaries:

```text
transformer_lab/
├── native/          # stable C ABI bindings
├── artifacts/       # ModelBundle persistence
├── data/            # external dataset adapters and preparation
├── experiments/     # controlled LoRA-rank comparisons
├── training/        # shared batches, metrics, and trainer
├── pretraining/     # next-token pretraining stage
├── post_training/   # supervised continuation stage
└── serving/         # generation, model service, and HTTP
```

The package root still re-exports the original low-level API.
`transformer_lab.artifact` and `transformer_lab.generation` remain
compatibility facades; new code should prefer `transformer_lab.artifacts` and
`transformer_lab.serving`.

## License

Copyright 2026 Quang T Nguyen. Licensed under the Apache License 2.0. The full
license text is included in the source distribution and every wheel.
