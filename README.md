# Riftco Transformer in C++

This is a small, auditable transformer framework built in dependency-free
C++20. It began as a learning project, so the model, gradients, and Adam remain
implemented directly rather than hidden behind another machine-learning
library.

The goal is a framework people can understand, test, install, and extend.
Correctness and stable boundaries take priority over GPT-scale performance.

## Current status

**Milestone 9: native pretraining, post-training, and serving composition
roots are active, alongside the Python persisted-artifact, incremental
generation, and local-serving workflow. Resumable training checkpoints remain
future work.**

The project currently provides:

- a strict C++20 build with warnings enabled;
- a small validated configuration format;
- a tiny UTF-8 training corpus;
- an end-to-end training executable;
- dependency-free correctness test executables;
- an architecture guide and incremental roadmap;
- row-major tensor storage with checked indexing and reshape;
- a reusable raw tensor-operation layer for arithmetic, reductions,
  broadcasting, permutation, elementary functions, and batched matmul;
- hand-calculated storage and numerical-operation tests;
- a reverse-mode operation graph with gradient accumulation and a validated
  public custom-VJP operation seam;
- exception-atomic per-pass gradient contexts and reusable activation
  checkpoint replay;
- differentiable arithmetic, matmul, reductions, reshape, and transpose;
- centered finite-difference gradient verification;
- interchangeable deterministic byte and byte-pair-encoding tokenizers with
  exact encode/decode;
- checked integer next-token batches with shape `[batch, context]`;
- a nonmoving, sealed registered `Module` lifecycle and shared-owning
  `ModuleList`, with recursive stable names and transactional backend transfer;
- handle-backed named parameters whose raw-pointer compatibility view remains
  safe inside copied parameter lists and optimizers;
- embedding lookup and arbitrary-rank linear projection;
- attachable low-rank adapters with configurable rank, scale, seed, and target
  projections, plus one-way merge into ordinary linear weights;
- layer normalization, erf-form GELU, stable softmax, and fused cross-entropy;
- a tested `Linear → GELU → Linear` feed-forward layer;
- query, key, value, and output projections for multi-head self-attention;
- head splitting/merging, scaled dot products, and an exact causal mask;
- learned token and positional embedding tables;
- pre-normalized attention and feed-forward residual blocks;
- a configurable block stack, final normalization, and language-model head;
- persistent Adam first/second moments and bias correction;
- one global gradient-norm clip applied before each optimizer update;
- deterministic seeded mini-batch sampling with replacement;
- a fresh forward and autograd graph for every training step;
- CSV loss, gradient-norm, and clipping metrics;
- a tiny repeated-batch overfitting acceptance test;
- backend-owned CPU and Metal tensor storage with explicit transfer;
- backend-preserving autograd and `to(...)` support across NN/model modules;
- one validated Adapter path for CPU reference math and Metal execution;
- real Metal kernels for layout transforms, elementwise math, reductions,
  erf-form GELU, layer normalization, softmax and causal softmax, embedding
  gather/scatter, fused cross-entropy, batched matmul, and their routed
  backward calculations;
- selectable exact full-sequence causal attention: the materialized reference
  path and dependency-free tile-8 FlashAttention forward/backward kernels on
  CPU and Metal;
- selectable transformer-block activation checkpointing on CPU and Metal,
  including active LoRA parameters and reduced retained graph diagnostics;
- one-submission, out-of-place fused Metal Adam updates for safe,
  well-conditioned arithmetic, with a wide-reference retry otherwise;
- a private backend Adapter contract with deterministic CPU/Metal providers;
- a stable opaque-handle C ABI 1.8 shared library with exact byte/BPE
  tokenizer reconstruction, named parameter shape/flat-state access,
  size-versioned option structures, model-derived ownership, LoRA lifecycle
  controls, full-sequence attention selection, and incremental-decode
  sessions, plus activation-checkpointing selection;
- a runtime-dependency-free Python `ctypes` client, distributed with the native
  C ABI in self-contained platform wheels, for tensors and complete
  model/loss/backward/Adam training;
- native stage-neutral `BatchSource`, `OptimizerStrategy`, and
  `CausalLanguageModelTrainer` contracts;
- native `PretrainingStack` and `PostTrainingStack` composition roots that
  share the trainer and artifact-state boundary;
- a native inference-only `ServingStack` that restores a model and tokenizer
  without depending on training or optimizer code;
- one-token prefill/decode with request-local logical page tables over
  backend-resident per-layer K/V pools, using direct CPU or Metal paged
  attention;
- paged KV caching by default, with a contiguous reference factory behind the
  same swappable cache interface;
- an in-memory native `ModelSnapshot` for exact model/tokenizer handoff between
  composition roots;
- a shared Python causal-language-model training engine with explicit
  pretraining and full-sequence supervised post-training stages, selectable
  between full-parameter and LoRA updates;
- dependency-free Hugging Face Dataset Viewer adapters for bounded
  TinyStories, Dolly 15K, and HH-RLHF preparation, with deterministic
  content-hash splits, deduplication, provenance manifests, and file digests;
- a controlled LoRA-rank experiment that starts every rank from the same base
  artifact, selects on validation loss, and evaluates held-out test data only
  after selection;
- immutable, versioned Python `ModelBundle` artifacts containing model
  configuration, restorable tokenizer state, named weights, checksums,
  metadata, and lineage;
- byte-safe autoregressive generation with greedy and seeded
  temperature/top-k sampling through stable C/Python decode sessions;
- a Python in-process model service and dependency-free local JSON HTTP
  adapter;
- installable CMake targets for C++ and C consumers.

The training executable now delegates the complete transaction—sample a
batch, forward, cross-entropy, backward, global clipping, and one Adam
update—to the native `PretrainingStack`. Its returned `ModelSnapshot` is an
in-memory value containing model weights/configuration and tokenizer state. It
does not define persistence, checksums, artifact identity, lineage, optimizer
state, or random-generator state.

The separate Python pipeline can persist a trained model as an immutable
`ModelBundle`, derive a lineage-linked post-trained child bundle, and serve
that bundle locally. A Python `ModelBundle` adds a versioned file format,
checksum, artifact ID, metadata, and lineage, but deliberately excludes Adam
and data-sampling state. LoRA post-training merges its adapter into the copied
base weights before capture, so the child is an ordinary serving artifact and
does not contain adapter-only state. Neither handoff is a resumable
checkpoint; exact training resumption requires the future
`TrainingCheckpoint` contract.

The backend work now routes the numerical operations used by the complete
training graph through the same capability-segregated Adapter. CPU executes the
readable reference implementations. Metal tensors own persistent shared
`MTLBuffer` storage, and the routed layout, elementwise, reduction, neural,
attention, matmul, and Adam operations execute as Metal kernels. Autograd graph
construction and traversal remain host control flow, while its local numerical
rules dispatch to the tensor backend.

This is still a deliberately synchronous learning runtime: each Metal call
waits before returning and buffers use host-visible shared memory.
Full-sequence model forwards use materialized causal attention by default, or
the opt-in exact Flash path. The tile-8 Flash implementation saves row maxima
and exponential sums of shape `[batch, heads, time]` for backward and
reconstructs probabilities instead of retaining a
`[batch, heads, time, time]` tensor. The public probability-returning
diagnostic remains materialized.

Activation checkpointing is a separate opt-in execution policy. It retains
transformer-block boundaries and parameter leaves, discards each block's
internal forward graph, and reconstructs that graph inside an isolated nested
VJP during backward. It composes with materialized or Flash attention and
full-parameter or LoRA training. It does not change parameter count,
artifacts, or incremental serving.

Incremental serving is a separate path: prompt prefill currently advances one
token at a time, and both prefill and decode read paged K/V storage directly on
CPU or Metal. The full-sequence Flash selector does not change that serving
path. CPU and Metal are checked with numerical tolerances, not bitwise
equality. No performance speedup is claimed without workload-specific
benchmarks; private GPU memory, asynchronous streams, command batching across
graph operations, and batched serving prefill remain future work.

Metal Flash working storage grows with the per-head width and must fit the
device's threadgroup-memory limit. The runtime checks every forward and
backward kernel before launching the forward pass, so an unsupported width
fails immediately with a device-limit error. Use more heads to reduce the head
width, or select materialized attention, when that limit is reached.

## Visualize feature superposition

The beginner-facing
[Three.js vector-angle lab](visualizations/vector-distribution.html) starts
with literal 3D geometry, highlights vectors between 89° and 91°, and then
shows how the full angle distribution changes from 3D through 4096D. It
distinguishes angles to one query from angles between every feature pair.

From the repository root:

```bash
.venv/bin/python -m http.server 8000 \
  --directory visualizations
```

Open `http://localhost:8000/vector-distribution.html`.

The interactive Python visualization in
[visualizations/superposition.py](visualizations/superposition.py) remains the
mathematical reference. It shows the feature Gram matrix, encodes several
active features into one vector, decodes them with dot products, and
illustrates why high-dimensional spaces can hold an exponentially large
codebook of nearly orthogonal feature directions.

From the repository root:

```bash
uv pip install --python .venv/bin/python 'matplotlib>=3.9' 'numpy>=2.0'
.venv/bin/python \
  visualizations/superposition.py
```

See [the visualization guide](visualizations/README.md) for the equations,
controls, headless image export, and the limits of the exponential-capacity
claim.

## Build and test

From this directory:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
./build/debug/transformer_lab --config configs/tiny.conf \
  --steps 20 --metrics results/metrics.csv --backend cpu \
  --attention flash --activation-checkpointing block
```

Use the `release` preset for optimized builds. On Clang, AppleClang, or GCC,
the non-Windows `sanitize` preset enables AddressSanitizer and
UndefinedBehaviorSanitizer through the project's compiler-checked CMake
option. The native pretraining CLI accepts
`--attention materialized|flash` and
`--activation-checkpointing disabled|block`; both use the first choice as
their default.

On a Metal-capable Mac, use `--backend metal`. The same build also creates
`libtransformer_lab_c.dylib`; the Python sources are under `python/`.

## Install and consume

Install the native CLI, headers, libraries, and CMake package with Homebrew:

```bash
brew install quangng2000/tap/riftco-transformer
```

Or build and install those native components from source:

```bash
cmake --preset release
cmake --build --preset release
cmake --install build/release --prefix "$PWD/install"
```

A downstream CMake project can then use either supported surface:

```cmake
find_package(transformer_lab 0.1 CONFIG REQUIRED)

target_link_libraries(cpp_app PRIVATE transformer_lab::library)
target_link_libraries(c_app PRIVATE transformer_lab::c_api)
```

Configure the downstream project with
`-DCMAKE_PREFIX_PATH=/path/to/transformer_lab/install`.
The backend adapter vtable is intentionally not installed; users depend on the
public C++, C, or Python contract rather than a compiler-specific private ABI.

The framework can also be embedded directly:

```cmake
add_subdirectory(path/to/transformer_lab)
target_link_libraries(my_app PRIVATE transformer_lab::library)
```

When embedded, the CLI, framework tests, and install rules default to `OFF` so
the parent project keeps control. They can be enabled independently with
`TRANSFORMER_LAB_BUILD_CLI`, `TRANSFORMER_LAB_BUILD_TESTS`, and
`TRANSFORMER_LAB_ENABLE_INSTALL`. `TRANSFORMER_LAB_ENABLE_METAL` controls the
Apple adapter, and `TRANSFORMER_LAB_ENABLE_SANITIZERS` enables compiler-checked
ASan/UBSan instrumentation on supported GNU-like toolchains.

Install the self-contained Python package from PyPI with:

```bash
python3 -m pip install riftco-transformer
python3 -c "from transformer_lab import Context; print(Context().backend)"
```

`riftco-transformer` is the PyPI distribution name. The stable Python import
remains `transformer_lab`, matching the native framework namespace.

The platform wheel includes the matching C ABI library under
`transformer_lab/.libs`, so normal users do not need CMake, a C++ compiler, a
separate native installation, or `TRANSFORMER_LAB_LIBRARY`. The installed
package has no third-party runtime dependencies. The initial wheel matrix is:

- Linux `x86_64` and `aarch64` for glibc (`manylinux`) and musl
  (`musllinux`), with the CPU backend;
- macOS `x86_64` and `arm64`, with CPU and Metal backends; and
- Windows `AMD64`, with the CPU backend.

Python 3.10 or newer is required. The Python code does not use CPython's
extension ABI, so each architecture receives one `py3-none-<platform>` wheel.

To build and install from a source checkout instead, run this at the repository
root:

```bash
python3 -m pip install .
```

A source install compiles the bundled C++20 implementation and therefore needs
a supported native toolchain and platform SDK. For native-library development,
`TRANSFORMER_LAB_LIBRARY` remains an explicit override that can point the
Python client at a particular local `.so`, `.dylib`, or `.dll`; it is not part
of ordinary wheel installation.

The high-level binding uses the same native model and optimizer:

```python
from transformer_lab import (
    Adam,
    DecoderOnlyTransformer,
    Tokenizer,
    TransformerConfig,
    cross_entropy,
)

corpus = "hello hello hello"
tokenizer = Tokenizer(
    corpus,
    method="bpe",
    vocabulary_size=272,
    minimum_pair_frequency=2,
)
encoded = tokenizer.encode(corpus)
tokens = [encoded[:-1]]
targets = [encoded[1:]]

config = TransformerConfig(
    vocabulary_size=tokenizer.vocab_size,
    maximum_context=len(tokens[0]),
    model_width=64,
    head_count=4,
    block_count=2,
    feed_forward_width=256,
)
model = DecoderOnlyTransformer(
    config,
    attention="flash",
    activation_checkpointing="block",
).to("metal")
optimizer = Adam(model.parameters())

loss = cross_entropy(model(tokens), targets)
loss.backward()
optimizer.step()
```

Choose `"cpu"` when Metal is unavailable. Build a fresh forward/loss graph for
each update; the binding intentionally consumes a graph after `backward()` and
invalidates pre-update graphs after `optimizer.step()`. Omit
`attention="flash"` to retain the materialized default. The selector affects
full-sequence training/evaluation forwards, not incremental paged decode.
Omit `activation_checkpointing="block"` to retain every ordinary graph node.
Block checkpointing recomputes each transformer block during backward to
reduce retained activations.

The runnable
[`examples/python/train_tiny.py`](examples/python/train_tiny.py) program shows
the same workflow on literal text or a UTF-8 file, with sampled next-token
windows, selectable byte/BPE tokenization, automatic CPU/Metal selection, and
explicit resource lifetimes. It also reports a rolling training-loss average
and deterministic loss on fixed samples from a token tail held out from model
updates.

For a larger learning-data experiment, the standard-library-only preparation
CLI can sample Hugging Face through its official Dataset Viewer API:

```bash
PYTHONPATH="$PWD/python" \
python3 examples/python/prepare_huggingface_data.py \
  --preset dolly \
  --output data/external/huggingface/dolly-lora-v1 \
  --limit 2000 \
  --seed lora-v1
```

After pretraining a base artifact on separate TinyStories train and validation
files, compare ranks without using test data for model selection:

```bash
PYTHONPATH="$PWD/python" \
python3 examples/python/compare_lora_ranks.py \
  --base results/stages/tinystories_pretrained.tlab \
  --data data/external/huggingface/dolly-lora-v1 \
  --ranks 1,2,4,8 \
  --alpha-over-rank 2 \
  --backend cpu
```

Downloaded data and generated experiment results are ignored by Git. See the
[dataset and LoRA experiment guide](docs/DATASETS_AND_LORA_EXPERIMENTS.md)
before running the commands; it covers source licenses, provenance, explicit
TinyStories splits, and the full-sequence SFT limitation.

The repository's [release workflow](.github/workflows/release.yml) builds and
tests the source distribution and self-contained wheels on the supported
platforms. A manual run performs build and verification only. Pushing a
`v<version>` tag builds the same artifacts and creates a GitHub Release after
they pass. PyPI publication is additionally gated by the repository variable
`PUBLISH_TO_PYPI=true` and the `pypi` environment's Trusted Publisher.

The PyPI Trusted Publisher must name project `riftco-transformer`, owner
`quangng2000`, repository `riftco-transformer`, workflow `release.yml`, and
environment `pypi`. This OIDC path avoids a long-lived PyPI API token.

Riftco Transformer is licensed under Apache-2.0. PyPI Trusted Publishing must
be configured before enabling `PUBLISH_TO_PYPI`.

## Project layout

```text
.github/                    Release build and publication automation
apps/
  pretraining/train.cpp    Native pretraining CLI source
configs/                    Human-readable experiment settings
cmake/                      Installed-package configuration
data/
  pretraining/              Raw language-model corpora
  post_training/            Prompt/response JSONL examples
  external/                 Ignored prepared/downloaded datasets
docs/                       Architecture decisions and milestone plan
include/transformer_lab/
  core/                     Tensor storage, operations, and autograd
  data/                     Tokenization and next-token batches
  nn/                       Reusable neural-network primitives
  model/                    Transformer-specific compositions
  optim/                    Optimizers and update rules
  artifacts/                In-memory model/tokenizer state handoff
  training/                 Stage-neutral trainer, batches, optimizer adapter
  stages/
    pretraining/            Native pretraining composition root
    post_training/          Native supervised-training composition root
    serving/                Native inference-only composition root
src/                        Implementations mirroring public headers
  core/backend/             Private adapter contract and backend providers
  artifacts/                Native state capture and restoration
  training/                 Shared native training implementations
  stages/                   Native stage implementations
tests/                      Tests grouped by the same domains
  stages/                   Native composition and dependency-contract tests
python/transformer_lab/
  native/                   Python ctypes client for the bundled C ABI
  artifacts/                Immutable model-bundle contract
  data/                     Dataset clients, adapters, splitting, provenance
  experiments/              Reproducible LoRA-rank comparisons
  training/                 Shared stage-neutral training engine
  pretraining/              Stage 1 orchestration
  post_training/            Stage 2 orchestration
  serving/                  Generation, service, and HTTP adapter
results/                    Generated metrics, samples, and Python artifacts
visualizations/             Interactive browser and Python concept demos
```

This separation keeps low-level tensor code independent of model code and gives
the attention, transformer-block, complete-model, and optimizer components
clear homes. See
[PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for the dependency rules.

## Learning rules

1. Add one concept at a time.
2. Write its test before connecting it to the transformer.
3. Check every tensor shape at runtime while the project is small.
4. Compare analytical gradients with finite differences.
5. Prefer obvious loops over clever optimizations.
6. Optimize only after the tiny model can overfit a tiny batch.

The detailed order is in [ROADMAP.md](docs/ROADMAP.md), and the intended model
dataflow is in [ARCHITECTURE.md](docs/ARCHITECTURE.md).
The completed tensor design is explained in [TENSOR.md](docs/TENSOR.md).
The reusable numerical layer is explained in
[TENSOR_OPS.md](docs/TENSOR_OPS.md).
The gradient engine is explained in [AUTOGRAD.md](docs/AUTOGRAD.md).
The block replay algorithm, safety contract, memory tradeoff, and APIs are
explained in
[ACTIVATION_CHECKPOINTING.md](docs/ACTIVATION_CHECKPOINTING.md).
The text-to-training-data path is explained in
[TOKENIZATION.md](docs/TOKENIZATION.md).
The trainable layers and feed-forward path are explained in
[NEURAL_NETWORK.md](docs/NEURAL_NETWORK.md).
The registered module tree, parameter ownership, transfer, and custom-module
extension rules are explained in [MODULES.md](docs/MODULES.md).
The causal multi-head attention dataflow is explained in
[ATTENTION.md](docs/ATTENTION.md).
The complete model and residual-block composition are explained in
[TRANSFORMER.md](docs/TRANSFORMER.md).
The optimizer equations and clipping behavior are explained in
[ADAM.md](docs/ADAM.md).
The repeated training transaction, deterministic batch sampling, and CSV
metrics are explained in [TRAINING.md](docs/TRAINING.md).
The CPU/Metal dispatch boundary, stable C ABI, and Python client are explained
in [BACKENDS_AND_PYTHON.md](docs/BACKENDS_AND_PYTHON.md).
The native stage composition roots, Python persisted-artifact workflow, and
their current limits are explained in [PIPELINE.md](docs/PIPELINE.md).
The dependency-free Hugging Face preparation recipes, dataset licenses,
provenance contract, and fair LoRA-rank protocol are explained in
[DATASETS_AND_LORA_EXPERIMENTS.md](docs/DATASETS_AND_LORA_EXPERIMENTS.md).
The low-rank adapter equations, targets, lifecycle, and merged-artifact
boundary are explained in [LORA.md](docs/LORA.md).
The one-token decode path, paged KV-cache layout, rollover behavior, and
C/Python session contracts are explained in [SERVING.md](docs/SERVING.md).

The learning documents use GitHub-compatible LaTeX for equations: `$...$` for
inline notation and fenced `math` blocks for larger derivations.

## What we will build

The finished learning model will contain:

- interchangeable byte and BPE tokenizers;
- token and positional embeddings;
- pre-normalization transformer blocks;
- causal multi-head self-attention;
- a GELU feed-forward network;
- residual connections;
- a language-model output head;
- cross-entropy loss and reverse-mode gradients;
- transformer-block activation checkpointing;
- gradient clipping and Adam;
- full-parameter and low-rank-adapter post-training;
- incremental autoregressive sampling with paged KV caching, native in-memory
  snapshots, and immutable Python model bundles;
- resumable training checkpoints.

Milestone 9 now has native stage composition, generation, and immutable Python
model artifacts. A persistent native artifact and exact resumable training
checkpoints remain future persistence work.

This lab remains separate from the repository's MLX-LM experiments. Its native
LoRA path is the small, auditable implementation used to reconstruct the
machinery; the MLX-LM experiments remain the practical large-model workflow.

## License

Copyright 2026 Quang T Nguyen. Licensed under the
[Apache License 2.0](LICENSE).
