# riftco-transformer Python distribution

This package is the typed, runtime-dependency-free-by-default `ctypes` interface to
`libriftco_transformer_c`, plus explicit data preparation, Python-owned
training orchestration, pretraining, post-training, artifact, generation, and
local-serving modules.
A platform wheel carries both the Python modules and its native C ABI library;
users do not install the native framework separately.

The ownership boundary is intentional: Python owns datasets, high-level
training loops, evaluation, and workflow policy; C++ owns tensors, autograd,
models, losses, Adam, artifacts, serving primitives, and hardware kernels.
Repository research protocols live in top-level `labs/` and are not included
in the installed distribution.

## Install

After a release has been published to PyPI:

```bash
python3 -m pip install riftco-transformer
python3 -c "from riftco_transformer import Context; print(Context().backend)"
```

`riftco-transformer` is the installable distribution name; Python code imports
the stable `riftco_transformer` package.

The wheel stores `libriftco_transformer_c.so`,
`libriftco_transformer_c.dylib`, or `riftco_transformer_c.dll` under
`riftco_transformer/.libs`. It has no third-party runtime dependencies and needs
no compiler or environment variable after installation. Initial binary wheels
cover Linux `x86_64` and `aarch64` for both glibc (`manylinux`) and musl
(`musllinux`), macOS `x86_64` and `arm64`, and Windows `AMD64`. CPU is
available on every supported platform; the macOS wheels also include Metal.
Standard wheels recognize the stable `cuda` and `tpu` backend names but compile
their unavailable stubs, so installing one does not add CUDA or `libtpu`
runtime dependencies.

From a source checkout, install at the repository root with:

```bash
python3 -m pip install .
```

That source build compiles the C++20 implementation, so it needs a supported
native compiler and platform SDK. `RIFTCO_TRANSFORMER_LIBRARY` remains an advanced
development override for selecting a particular local native build; released
wheels do not require it.

CUDA is an explicit source-build option. It needs CUDA Toolkit 12 or newer, a
compatible NVIDIA driver, and an NVIDIA GPU:

```bash
CMAKE_ARGS="-DRIFTCO_TRANSFORMER_ENABLE_CUDA=ON" \
  python3 -m pip install .
```

The CUDA backend provides managed tensor and packed-weight storage plus native
GPU NN, matmul, packed NF4 linear, materialized/Flash attention, attention
gradient, paged-decode, and Adam-update kernels. Full fine-tuning, LoRA, QLoRA,
held-out generalization evaluation, artifact creation, and serving can all use
`backend="cuda"`, but host control flow and managed-memory migration remain.
The actual NVIDIA hardware path was not validated on the macOS development
host used for this milestone.

TPU is a separate experimental source-build option for Linux x86-64 Cloud TPU
VMs. It requires Google's external `libtpu.so` at runtime:

```bash
export RIFTCO_TRANSFORMER_TPU_LIBRARY=/absolute/path/to/libtpu.so
CMAKE_ARGS="-DRIFTCO_TRANSFORMER_ENABLE_TPU=ON" \
  python3 -m pip install .
```

The TPU backend compiles packed NF4 linear forward/input backward, batched
matmul, materialized attention and its gradients, and paged decode through the
PJRT C API and StableHLO. Flash attention and the other capabilities use
synchronous audited reference paths over host-mirrored storage, so full
fine-tuning, LoRA, QLoRA, evaluation, and serving are functionally wired
without implying an end-to-end TPU speedup. Real Cloud TPU validation is still
pending. Default installs never load `libtpu`.

The Python package follows the framework release version (`0.5.0` here), while
the native C ABI has its own compatibility version (`2.5`). The client accepts
the same ABI major and an equal or newer additive minor, and rejects older or
breaking ABIs before use.

Backend names are `"cpu"`, `"metal"`, `"cuda"`, and `"tpu"`. High-level
configurations also accept `"auto"`, which prefers TPU when available, then
CUDA, Metal, and CPU. An explicitly requested unavailable backend raises an
error instead of silently changing the experiment backend.
QLoRA uses the same auto-selection order and accepts every available backend.

## Release automation

`.github/workflows/release.yml` builds and verifies the source distribution and
self-contained platform wheels. `workflow_dispatch` is verification-only.
Pushing a `v<version>` tag creates a GitHub Release after the artifacts pass.
It also publishes to PyPI only when the repository variable
`PUBLISH_TO_PYPI` is `true`.

PyPI publication uses Trusted Publishing rather than a stored API token. The
publisher configuration is project `riftco-transformer`, owner `quangng2000`,
repository `riftco-transformer`, workflow `release.yml`, and environment `pypi`.
The project is licensed under Apache-2.0. Do not enable publication until the
Trusted Publisher is configured.

## Selectable tokenizers

`Tokenizer` offers interchangeable byte and byte-pair-encoding strategies:

```python
from riftco_transformer import Tokenizer

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
from riftco_transformer import (
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

Change `.to("cpu")` to `.to("metal")` on systems with the Metal backend,
`.to("cuda")` in a CUDA-enabled source build, or `.to("tpu")` in a TPU-enabled
Cloud TPU build. Computation graphs are single-use: build a fresh forward/loss
graph for each training step.
`attention="flash"` selects the dependency-free exact memory-linear
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

## Program-augmented models

`riftco_transformer.programmed` is the installed, task-neutral Python surface
for composing learned sequence paths with an optional lowered multilinear
program. It exports `MultilinearMap`, `NeuralLoweringConfig`,
`ProgramInputLayout`, `ProgramBranch`, `ProgramAugmentedModelConfig`,
`ProgramAugmentedModel`, forward options, and owning representation-trace
values. `MultilinearMap.from_sparse(...)` accepts output-major flat nonzero
indices and values, avoiding a dense Python coefficient list; the current
native lowerer still materializes its configured dense representation.

The model has a fixed context length, a residual ReLU feed-forward path, a
configurable number of independent causal-attention branches, and an optional
program core whose raw output is placed at a configured target offset before a
learned residual merge. Forward options can capture stable named
representations, batch-roll learned attention, batch-roll selected program
inputs or output, and apply affine input steering. `cross_entropy_time_range`
trains only one contiguous time range per batch, which lets a lab supervise a
target half without treating source positions as loss targets.

The installed module owns no F/P/T/I enum, dataset, training loop, metric,
PCA policy, or report. The source-only conditional-reversal lab constructs
those controls and drives the generic model:

```bash
PYTHONPATH=python:. python3 -m labs.conditional_reverse.run --help
PYTHONPATH=python:. python3 -m labs.conditional_reverse.run \
  --profile quick --variants F --backend cpu \
  --output runs/conditional-reverse/quick.json
```

Check `--help` for the exact current CLI before starting either the `quick` or
long-running `paper` profile. The generic path is implemented; fresh reviewed
quick/paper results are intentionally not asserted in this package README.
The archived seed-42 F result in the repository came from the retired
task-specific C++ prototype and is not a multi-seed reproduction.

## Hugging Face data and research labs

The `riftco_transformer.data` package is also dependency-free. Its default
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

Controlled comparisons are repository labs rather than installed framework
API. From a source checkout, compare LoRA ranks from one immutable base with:

```bash
PYTHONPATH=python:. python3 -m labs.lora_rank.run \
  --base results/stages/tinystories_pretrained.rift \
  --data data/external/huggingface/dolly-lora-v1 \
  --output runs/lora-rank \
  --ranks 1,2,4,8 \
  --alpha-over-rank 2 \
  --steps 20 \
  --backend cpu
```

Every rank shares data fingerprints, seeds, sampler, optimizer controls,
LoRA targets, and `alpha / rank`. Validation selects the winner; held-out test
evaluation begins only after selection. The objective is still
full-sequence causal SFT, not response-only loss. All adapters are merged
before persistence, so ranks have the same serving topology and inference
timings are only smoke measurements. The CLI atomically publishes a new,
complete output directory and embeds the verified prepared-data manifest plus
its SHA-256 in `comparison.json`.

The fine-tuning lab generalizes the same held-out protocol to fixed full
fine-tuning recipes and LoRA rank groups. It exhaustively evaluates train and
validation to calculate a comparable generalization gap, then evaluates test
only for the fixed full recipe and validation-selected LoRA rank:

```bash
PYTHONPATH=python:. python3 -m labs.fine_tuning.run \
  --base results/stages/tinystories_pretrained.rift \
  --data data/external/huggingface/dolly-lora-v1 \
  --output runs/fine-tuning \
  --full-learning-rate 0.001 \
  --lora-learning-rate 0.005 \
  --backend cpu
```

The resulting test split is consumed for a final method comparison and must be
retired before further tuning.

See
`docs/DATASETS_AND_LORA_EXPERIMENTS.md` in the framework repository for
license links, exact TinyStories train/validation commands, sample-size
guidance, provenance details, and the CLI rank workflow.

## Staged pipeline

The high-level modules make the stage boundaries explicit:

```python
from riftco_transformer.artifacts import ModelBundle
from riftco_transformer import LoraConfig
from riftco_transformer.post_training import (
    PostTrainingConfig,
    post_train_jsonl,
)
from riftco_transformer.pretraining import PretrainingConfig, pretrain_file
from riftco_transformer.serving import ServingConfig, serve_model

base = pretrain_file(
    "data/pretraining/tiny_corpus.txt",
    PretrainingConfig(
        steps=20,
        backend="cpu",
        attention="flash",
        activation_checkpointing="block",
    ),
)
base.bundle.save("results/stages/tiny_pretrained.rift")

restored = ModelBundle.load("results/stages/tiny_pretrained.rift")
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
assistant.bundle.save("results/stages/tiny_post_trained.rift")

serve_model(
    "results/stages/tiny_post_trained.rift",
    host="127.0.0.1",
    port=8000,
    config=ServingConfig(
        backend="cpu",
        kv_cache="paged",
        kv_cache_block_size=16,
    ),
)
```

For memory-constrained adapter training, select QLoRA instead:

```python
assistant = post_train_jsonl(
    restored,
    "data/post_training/tiny_instructions.jsonl",
    PostTrainingConfig(
        steps=10,
        backend="auto",
        fine_tuning_method="qlora",
        nf4_block_size=64,
        double_quantization=True,
        nf4_scale_block_size=256,
        optimizer_state="auto",
        optimizer_page_size=4096,
        lora=LoraConfig(rank=4, alpha=8.0),
    ),
)
print(assistant.bundle.metadata["quantization"]["training_memory"])
```

During QLoRA training, every eligible frozen `Linear` base weight remains in
blockwise packed NF4 storage while Adam updates only floating-point LoRA
adapters. First-level NF4 scales are double-quantized by default; set
`double_quantization=False` to retain legacy FP32 block scales. CPU decodes in
the readable reference loop, Metal and CUDA decode inside accelerator kernels,
and TPU dequantizes packed inputs inside its StableHLO program. No backend
retains a full FP32 base-weight matrix during training.

`optimizer_state="auto"` selects paged Adam for QLoRA and contiguous Adam for
the other methods. Paged Adam stores each first/second-moment vector in bounded
`optimizer_page_size` chunks and updates one chunk at a time; its total moment
payload is still two FP32 values per trainable parameter. CUDA pages use
managed allocations, but this implementation has no general OS spill budget,
explicit eviction, disk paging, or page-fault manager. Use `"contiguous"` or
`"paged"` to override the automatic choice.

The lower-level model API exposes `model.quantize_nf4()` and
`model.quantized_memory` for direct lifecycle control and memory accounting.
Packed `ModelBundle` artifacts are deliberately unsupported in this release:
`post_train_jsonl()` merges the adapter and materializes an ordinary FP32
serving bundle, and direct `ModelBundle.capture()` rejects a still-packed
model.

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

Native models use the current ABI 2.5 `DecodeSession` surface instead of
rerunning the full-sequence training forward for every generated token.
`TextGenerator` creates a request-local session, prefills the
prompt one token at a time, and then performs one-token decode:

```python
from riftco_transformer.artifacts import ModelBundle
from riftco_transformer.serving import TextGenerator

bundle = ModelBundle.load("results/stages/tiny_post_trained.rift")
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
strategy. CPU, Metal, CUDA, and TPU have backend-owned paged-decode
implementations; TPU stages the request through PJRT from host-mirrored
storage. When the learned
absolute-position context fills, `TextGenerator`
resets the cache and replays the retained suffix from position zero. A raw
session exposes the lower-level step/reset contract:

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
riftco_transformer/
├── native/          # stable C ABI bindings
├── programmed/      # generic learned/programmed composition
├── artifacts/       # ModelBundle persistence
├── data/            # external dataset adapters and preparation
├── training/        # shared batches, metrics, and trainer
├── pretraining/     # next-token pretraining stage
├── post_training/   # supervised continuation stage
└── serving/         # generation, model service, and HTTP
```

The package root re-exports the public low-level API. The breaking rename
installs only `riftco_transformer`; no legacy package-name alias is provided.
Top-level `labs/` composes these public APIs into controlled experiments, but
is source-checkout-only and is not packaged in the wheel.

## License

Copyright 2026 Quang T Nguyen. Licensed under the Apache License 2.0. The full
license text is included in the source distribution and every wheel.
