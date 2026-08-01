# Riftco Transformer

[![PyPI](https://img.shields.io/pypi/v/riftco-transformer.svg)](https://pypi.org/project/riftco-transformer/)
[![Python](https://img.shields.io/pypi/pyversions/riftco-transformer.svg)](https://pypi.org/project/riftco-transformer/)
[![Release](https://github.com/quangng2000/riftco-transformer/actions/workflows/release.yml/badge.svg)](https://github.com/quangng2000/riftco-transformer/actions/workflows/release.yml)
[![License](https://img.shields.io/github/license/quangng2000/riftco-transformer.svg)](LICENSE)

A small, auditable decoder-only Transformer built directly in C++20, with a
zero-third-party-dependency default Python API. Train on CPU, Apple Metal, or
optional source-built NVIDIA CUDA and Google Cloud TPU backends; post-train
with full fine-tuning or LoRA, measure held-out generalization, save portable
artifacts, and serve through paged attention.

| Install | Import | Third-party Python dependencies |
| --- | --- | --- |
| `riftco-transformer` | `riftco_transformer` | None |

This breaking rename exposes only the `riftco_transformer` Python package;
no legacy package-name alias is installed.

## Backend support

| Backend | Status | Current execution path |
| --- | --- | --- |
| CPU | Supported in every build | Complete readable reference implementation |
| Apple Metal | Supported on compatible Macs | Persistent shared buffers and native Metal kernels |
| NVIDIA CUDA | Optional source build | Managed CUDA storage with native tensor/NN, matmul, attention, and Adam-update kernels |
| Google Cloud TPU | Experimental Linux x86-64 source build | Host-mirrored storage with PJRT/StableHLO matmul, materialized attention, and paged decode |

The TPU adapter is opt-in and dynamically loads Google's external `libtpu.so`;
an absolute zero-library TPU build is therefore not possible. Default builds
and standard wheels retain the dependency-free behavior and contain a clean
unavailable TPU stub. The TPU slice targets one addressable device in one
process. It runs batched matmul, materialized attention and its gradients, and
paged decode through PJRT; Flash attention and the remaining capabilities stay
on audited host reference paths. CI covers compilation, no-device behavior,
and the loader/compile/transfer/execute/download sequence with a tests-only fake
PJRT plugin. Real `libtpu` and Cloud TPU hardware validation is still required
before treating it as production support.

## Architecture

```mermaid
flowchart LR
    subgraph P["1 · Pretraining"]
        Text["UTF-8 text"] --> Tokens["Byte / BPE<br/>tokenizer"]
        Tokens --> Model["Decoder-only<br/>Transformer"]
        Model --> Loss["Fused<br/>cross-entropy"]
        Loss --> Grad["Reverse-mode<br/>autograd"]
        Grad --> Adam["Adam + gradient<br/>clipping"]
        Adam -->|update| Model
    end

    Model --> Base[("Base .rift<br/>ModelBundle")]
    Tokens -->|tokenizer state| Base

    subgraph T["2 · Post-training"]
        Base --> Tune["Full fine-tuning<br/>or LoRA"]
        Instructions["Prompt / response<br/>JSONL"] --> Tune
        Tune -->|capture; merge LoRA first| Child[("Child .rift<br/>ModelBundle")]
    end

    subgraph S["3 · Serving"]
        Child --> API["Serving<br/>chat + JSON API"]
        API --> Decode["Autoregressive<br/>generation"]
        Decode <--> Cache["Paged KV<br/>cache"]
    end

    Runtime["CPU · Apple Metal · optional NVIDIA CUDA / Cloud TPU"] -.-> Model
    Runtime -.-> Tune
    Runtime -.-> Decode
```

| Core | Training | Serving |
| --- | --- | --- |
| Tensors · modules · autograd · C ABI | Flash attention · activation checkpointing · fused Metal Adam · LoRA | Immutable bundles · paged KV cache · sampling · local chat/API |

## From text to chat

```mermaid
sequenceDiagram
    participant D as Dataset
    participant I as Instructions
    participant T as Tokenizer
    participant M as Transformer
    participant O as Autograd + Adam
    participant A as .rift artifact
    participant S as Serving
    participant U as User

    D->>T: UTF-8 text
    T->>M: next-token batches
    loop Pretraining steps
        M->>M: forward + cross-entropy
        M->>O: backward gradients
        O-->>M: update weights
    end
    M->>A: save base weights
    T->>A: save tokenizer state
    A->>M: restore base
    loop Post-training steps
        I->>M: prompt/response batches
        M->>O: full or LoRA gradients
        O-->>M: update selected weights
    end
    M->>A: save child after LoRA merge
    A->>S: load once
    U->>S: prompt
    loop Each new token
        S->>S: paged KV attention
    end
    S-->>U: generated text
```

> Implemented: pretraining, full/LoRA post-training, immutable model bundles,
> and local serving. Exact resumable optimizer/data checkpoints are future work.

## Setup

### Python

```bash
python3 -m pip install riftco-transformer
python3 -c "from riftco_transformer import Context; print(Context().backend)"
```

Python 3.10+ is supported on Linux, macOS, and Windows. Standard released
wheels include CPU, and macOS wheels also include Metal. They recognize the
stable `cuda` and `tpu` backend names but contain unavailable stubs; both
accelerators require explicit source builds below.

### Homebrew

```bash
brew install quangng2000/tap/riftco-transformer
```

### Source

```bash
git clone https://github.com/quangng2000/riftco-transformer.git
cd riftco-transformer
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
cmake --install build/debug --prefix "$PWD/install"
```

Source builds require CMake 3.24+, Ninja, and a C++20 compiler.

For CUDA, use an NVIDIA GPU and compatible driver plus CUDA Toolkit 12 or
newer:

```bash
cmake -S . -B build/cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRIFTCO_TRANSFORMER_ENABLE_CUDA=ON
cmake --build build/cuda
ctest --test-dir build/cuda --output-on-failure
```

CUDA is functionally available to tensors, autograd, pretraining, Full and LoRA
post-training, held-out evaluation, and serving. CUDA tensors use managed
memory; layout, elementwise, reduction, indexing, normalization, loss, matmul,
materialized and memory-linear Flash attention, their gradient kernels, paged
decode, and Adam's candidate-state update run on the GPU. Autograd traversal
and Adam's overflow-safe global gradient norm remain host control flow over
host-visible storage, so selecting CUDA is not a claim that every part of the
workload is device-resident or faster.

For the experimental Cloud TPU path, use a Linux x86-64 Cloud TPU VM and make
Google's `libtpu.so` available at runtime:

```bash
export RIFTCO_TRANSFORMER_TPU_LIBRARY=/absolute/path/to/libtpu.so
cmake --preset tpu-release
cmake --build --preset tpu-release
ctest --preset tpu-release
```

The loader also checks `TPU_LIBRARY_PATH` and then the system loader path for
`libtpu.so`. On a real TPU host, reconfigure with
`-DRIFTCO_TRANSFORMER_TEST_REQUIRE_TPU=ON` to make device absence a test
failure. The TPU option is off by default and standard wheels do not bundle or
load `libtpu`.

## One training step

```python
from riftco_transformer import (
    Adam,
    DecoderOnlyTransformer,
    Tokenizer,
    TransformerConfig,
    cross_entropy,
)

text = "tiny models learn from text. " * 4

with Tokenizer(text, method="byte") as tokenizer:
    ids = tokenizer.encode(text)
    tokens, targets = [ids[:8]], [ids[1:9]]
    config = TransformerConfig(
        vocabulary_size=tokenizer.vocab_size,
        maximum_context=8,
        model_width=16,
        head_count=4,
        block_count=1,
        feed_forward_width=32,
    )

    with DecoderOnlyTransformer(config).to("cpu") as model:
        with model.parameters() as parameters:
            with Adam(parameters, learning_rate=1e-2) as optimizer:
                with model(tokens) as logits:
                    with cross_entropy(logits, targets) as loss:
                        print(f"loss={loss.item():.4f}")
                        loss.backward()
                        optimizer.step()
```

Use `.to("metal")` on a supported Mac, `.to("cuda")` in a CUDA-enabled source
build, or `.to("tpu")` in a TPU-enabled Cloud TPU build. Construct a fresh
forward/loss graph for every optimizer step.

## Train → LoRA → chat

From a source checkout:

```bash
python3 -m pip install .

python3 examples/python/pretrain_stage.py \
  --backend cpu --steps 10

python3 examples/python/post_train_stage.py \
  --backend cpu --fine-tuning-method lora --steps 5

python3 examples/python/serve_stage.py --backend cpu
```

Open `http://127.0.0.1:8000/`. The stages exchange immutable `.rift` bundles in
`results/stages/`; serving uses a paged KV cache by default.

## Native trainer

```bash
./build/debug/riftco-transformer \
  --config configs/tiny.conf \
  --steps 20 \
  --backend cpu \
  --attention flash \
  --activation-checkpointing block
```

## Explore

| Goal | Start here |
| --- | --- |
| Understand the full model | [Architecture](docs/ARCHITECTURE.md) · [Transformer](docs/TRANSFORMER.md) |
| Learn tensors and gradients | [Tensor](docs/TENSOR.md) · [Tensor operations](docs/TENSOR_OPS.md) · [Autograd](docs/AUTOGRAD.md) |
| Extend layers and modules | [Neural network](docs/NEURAL_NETWORK.md) · [Modules](docs/MODULES.md) |
| Understand training | [Training](docs/TRAINING.md) · [Adam](docs/ADAM.md) · [Activation checkpointing](docs/ACTIVATION_CHECKPOINTING.md) |
| Compare attention paths | [Attention](docs/ATTENTION.md) · [Execution backends and Python](docs/BACKENDS_AND_PYTHON.md) |
| Run all three stages | [Pipeline](docs/PIPELINE.md) · [LoRA](docs/LORA.md) · [Serving](docs/SERVING.md) |
| Compare full tuning and LoRA | [Post-training generalization](docs/GENERALIZATION.md) |
| Prepare Hugging Face data | [Datasets and LoRA experiments](docs/DATASETS_AND_LORA_EXPERIMENTS.md) |
| Navigate or contribute | [Project structure](docs/PROJECT_STRUCTURE.md) · [Roadmap](docs/ROADMAP.md) · [Release automation](python/README.md#release-automation) |
| See feature superposition | [3D vector lab source](visualizations/vector-distribution.html) · [Run the visualization](visualizations/README.md) |

## CMake consumers

```cmake
find_package(riftco_transformer 0.2 CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE riftco_transformer::library)
```

Install Riftco Transformer first, then configure the consuming project with
`-DCMAKE_PREFIX_PATH=/path/to/riftco-transformer/install`. The public C API is
`riftco_transformer::c_api`. See
[Execution backends and Python](docs/BACKENDS_AND_PYTHON.md) for the stable ABI and
backend boundary.

## License

Copyright 2026 Quang T Nguyen. Licensed under the
[Apache License 2.0](LICENSE).
