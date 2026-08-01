# Project Structure

The lab is organized by responsibility rather than by implementation
milestone. Milestones remain in `ROADMAP.md`; source files live where future
components can extend them without reorganizing the project again.

## Dependency direction

```text
apps/pretraining ──→ stages/pretraining

stages/pretraining ──────→ artifacts + shared training
stages/post_training ────→ artifacts + shared training
stages/serving ──────────→ artifacts + model + data
                            (no training or optim dependency)

shared training ──→ model + data + optimizer strategy
Adam adapter ─────→ optim ──→ nn
artifacts ────────→ model + data
model ────────────→ nn ─────→ core

Python pretraining/post-training ──→ Python shared training ──→ C ABI
Python dataset preparation ────────→ stdlib HTTP/JSON + prepared files
Python tuning experiments ─────────→ artifacts + post_training + evaluation
Python serving ──→ Python ModelBundle + generation ──────────→ C ABI
```

- `core` owns tensor storage, numerical tensor operations, and autograd. Its
  public `custom_gradient` seam connects an externally computed tensor result
  to validated positional VJPs without depending on a neural-network layer or
  model.
- `data` owns token IDs, byte tokenization, and next-token batches.
- `nn` owns reusable parameters, initialization, activations, layers, and
  losses, including the registered `Module`/`ModuleList` lifecycle,
  handle-backed named parameters, and the reusable low-rank linear adapter. It
  depends on `core`; embedding and cross-entropy also consume token IDs from
  `data`.
- `model` composes reusable layers into transformer-specific components,
  owns transformer-wide LoRA target selection and merge coordination, and
  defines the inference-time `DecoderKeyValueCache` contract consumed by its
  detached one-token path.
- `optim` owns parameter-update rules and depends on `nn` parameter
  registration without depending on `model`.
- `artifacts` captures and restores a native model configuration, named
  parameter values, and exact byte/BPE tokenizer state. Its `ModelSnapshot`
  is an in-memory handoff value, not a persisted artifact or training
  checkpoint. Active, unmerged adapters are rejected so snapshots always use
  the ordinary base-model parameter schema.
- `training` owns stage-neutral batch sources, the optimizer strategy,
  `CausalLanguageModelTrainer`, and the adapter that presents native `Adam`
  through that strategy. Training policy does not own a stage.
- `stages/pretraining` and `stages/post_training` are composition roots. They
  wire model, tokenizer, shared training, artifact capture, and optimizer
  choices for one stage run. Post-training selects either all base parameters
  or only LoRA adapter parameters, then merges adapters before handoff.
- `stages/serving` restores a native snapshot and owns autoregressive
  generation, KV-cache allocation strategies, and the shared paged pool. It
  depends on `artifacts`, `model`, and `data`, with no dependency on
  `training` or `optim`.
- `apps` owns command-line concerns.
  `apps/pretraining/train.cpp` delegates the native training run to
  `PretrainingStack`.
- `c_api.h` and `src/c_api.cpp` expose opaque C handles over selected tensor,
  model, LoRA, incremental decode, autograd, loss, and optimizer operations
  without exporting C++ layouts.
- `python/riftco_transformer/native` wraps only the C ABI through the Python
  standard library's `ctypes`; it does not bind the C++ ABI. The package root
  re-exports this public API without installing a legacy package-name alias.
- `python/riftco_transformer/artifacts` is the immutable handoff contract between
  stages.
- `python/riftco_transformer/data` owns the dependency-free Hugging Face client,
  dataset-specific row adapters, deterministic content-hash splitting,
  serialization, provenance manifests, and prepared-file verification. It
  does not own a training objective.
- `python/riftco_transformer/experiments` composes existing artifact,
  post-training, and evaluation APIs into controlled comparisons. The generic
  fine-tuning experiment compares full and LoRA recipes, while the specialized
  rank experiment preserves rank-only selection. Both verify prepared splits,
  select on validation, and defer test evaluation until after selection.
- `python/riftco_transformer/training` owns stage-neutral batching, evaluation,
  and optimizer-loop policy. The `pretraining` and `post_training` packages
  configure that engine rather than duplicating it.
- `python/riftco_transformer/serving` owns generation, the in-process model
  service, and the HTTP adapter. It depends on artifacts and the native model,
  not on the training engine.

Dependencies should point downward in this list. In particular, `core` must
not include anything from `nn` or `model`, `nn` must not include `model`, and
the native serving stage must not acquire a training or optimizer dependency.

## Two handoff contracts

The native and Python stage surfaces solve related but different problems:

| Contract | Lifetime and storage | Integrity and lineage | Deliberately excluded |
| --- | --- | --- | --- |
| Native `ModelSnapshot` | Value-like, in-memory handoff of model and tokenizer state | No checksum, artifact ID, metadata, or lineage | Persistence, optimizer state, random state, and training progress |
| Python `ModelBundle` | Immutable, versioned persisted ZIP artifact | Weight checksum, content-derived artifact ID, metadata, and parent lineage | Optimizer state, random state, and training progress |

Neither contract supports exact training resumption. That requires the future
`TrainingCheckpoint` contract. For LoRA, both contracts currently carry only
the merged base weights; separate adapter-factor persistence is deliberately
outside these handoffs.

## Interface and implementation pairing

Every public interface lives below `include/riftco_transformer/`. Its
implementation uses the corresponding path below `src/`:

```text
include/riftco_transformer/core/tensor.hpp
src/core/tensor/
  storage.cpp
  layout.cpp

include/riftco_transformer/core/tensor_ops.hpp
src/core/tensor/
  elementwise.cpp
  matmul.cpp
  layout_ops.cpp
  indexing.cpp
  reductions.cpp
  softmax.cpp

include/riftco_transformer/core/autograd.hpp
src/core/autograd/
  graph.cpp
  operations.cpp
  custom_gradient.cpp
  checkpoint.cpp

include/riftco_transformer/model/feed_forward.hpp
src/model/feed_forward.cpp

include/riftco_transformer/artifacts/state.hpp
src/artifacts/state.cpp

include/riftco_transformer/stages/serving/stack.hpp
src/stages/serving/stack.cpp
```

Tests use the same domain names below `tests/`. This makes a component's
contract, implementation, and verification easy to locate.

## Component map

```text
module and parameter lifecycle
  include/riftco_transformer/nn/module.hpp
  include/riftco_transformer/nn/parameter.hpp
  src/nn/module.cpp
  src/nn/parameter.cpp
  tests/nn/test_module.cpp
  docs/MODULES.md

public custom-gradient operation
  include/riftco_transformer/core/autograd.hpp
  src/core/autograd/custom_gradient.cpp
  tests/core/test_autograd.cpp
  docs/AUTOGRAD.md

causal attention
  include/riftco_transformer/model/causal_self_attention.hpp
  src/model/causal_self_attention.cpp
  tests/model/test_causal_self_attention.cpp

transformer block
  include/riftco_transformer/model/activation_checkpointing.hpp
  include/riftco_transformer/model/transformer_block.hpp
  src/model/transformer_block.cpp
  tests/model/test_transformer_block.cpp

decoder-only transformer
  include/riftco_transformer/model/decoder_kv_cache.hpp
  include/riftco_transformer/model/decoder_only_transformer.hpp
  src/model/decoder_only_transformer.cpp
  tests/model/test_decoder_only_transformer.cpp

activation checkpoint primitive
  include/riftco_transformer/core/autograd.hpp
  src/core/autograd/checkpoint.cpp
  tests/core/test_autograd.cpp
  docs/ACTIVATION_CHECKPOINTING.md

serving KV cache
  include/riftco_transformer/stages/serving/kv_cache.hpp
  src/stages/serving/cache/
    detail/
      validation.hpp
      page_storage.hpp
      page_table_cache.hpp
    validation.cpp
    page_storage.cpp
    page_table_cache.cpp
    contiguous_kv_cache.cpp
    paged_kv_cache.cpp
  tests/stages/test_native_serving_generation.cpp

low-rank adaptation
  include/riftco_transformer/nn/low_rank_adapter.hpp
  src/nn/low_rank_adapter.cpp
  include/riftco_transformer/model/lora.hpp

Adam
  include/riftco_transformer/optim/adam.hpp
  src/optim/adam.cpp
  tests/optim/test_adam.cpp

native artifact state
  include/riftco_transformer/artifacts/state.hpp
  src/artifacts/state.cpp

shared native training
  include/riftco_transformer/training/
    optimizer.hpp
    batch_source.hpp
    causal_language_model_trainer.hpp
    adam_optimizer_adapter.hpp
  src/training/
    batch_source.cpp
    causal_language_model_trainer.cpp
    adam_optimizer_adapter.cpp

native stage composition roots
  include/riftco_transformer/stages/
    pretraining/
    post_training/
    serving/
  src/stages/
    pretraining/
    post_training/
    serving/
  tests/stages/
    test_native_stage_stacks.cpp
    test_native_serving_generation.cpp
    test_stage_contracts.cpp

native pretraining CLI
  apps/pretraining/train.cpp

execution backends
  include/riftco_transformer/core/backend.hpp
  src/core/backend/
    storage.hpp                       shared storage contract
    adapter.hpp                       composition facade for capabilities
    registry.cpp                      closed backend selection
    unavailable_adapter.hpp           shared optional-backend stub
    adapters/
      cpu/adapter.cpp
      cuda/adapter.cu
      cuda/stub.cpp
      metal/adapter.mm
      metal/stub.cpp
      metal/runtime.mm
      tpu/adapter.cpp
      tpu/stub.cpp
      tpu/compile_options.hpp
      tpu/runtime.hpp
      tpu/runtime.cpp
    attention/
      contracts.hpp
      capability.hpp
      dispatch.hpp
      dispatch.cpp
      reference/
        materialized_causal.hpp
        materialized_causal.cpp
        flash_causal.hpp
        flash_causal.cpp
        paged_decode.hpp
        paged_decode.cpp
      cuda/
        launch.hpp
        common.cuh
        materialized_causal.cu
        flash_causal.cu
        paged_decode.cu
      metal/
        launch.hpp
        materialized_causal_kernels.hpp
        flash_causal_kernels.hpp
        paged_decode_kernels.hpp
      tpu/
        common.hpp
        materialized_causal.hpp
        materialized_causal.cpp
        paged_decode.hpp
        paged_decode.cpp
    nn/
      contracts.hpp
      capability.hpp
      dispatch.hpp
      dispatch.cpp
      reference/
        operations.hpp
        operations.cpp
      cuda/
        launch.hpp
        common.cuh
        elementwise.cu
        reduction.cu
        layout.cu
        softmax.cu
        indexing.cu
        normalization.cu
        loss.cu
      metal/
        kernels.hpp
        launch.hpp
    optim/adam/
      contracts.hpp
      capability.hpp
      dispatch.hpp
      dispatch.cpp
      reference/
        update.hpp
        update.cpp
      cuda/
        launch.hpp
        update.cu
      metal/
        kernels.hpp
        launch.hpp
        diagnostics.hpp
  third_party/pjrt/                   pinned PJRT C ABI header and license
  tests/core/backend/test_backend.cpp
  tests/core/backend/test_nn_backend.cpp
  tests/fakes/fake_pjrt_tpu.cpp       tests-only PJRT boundary emulator

C ABI and Python
  include/riftco_transformer/c_api.h
  src/c_api.cpp
  tests/abi/test_c_api.c
  python/riftco_transformer/
    __init__.py                    public native API exports
    native/
      bindings.py                 typed ctypes C ABI client
    artifacts/
      bundle.py                   immutable stage handoff
    data/
      client.py                   stdlib Hugging Face API adapter
      adapters.py                 audited source-to-record presets
      splitting.py                seeded content-hash partitions
      serialization.py            canonical JSONL/plain-text writers
      preparation.py              atomic output and provenance verification
    experiments/
      fine_tuning.py               full/LoRA generalization comparison
      lora_rank.py                validation-selected rank comparison
    training/
      engine.py                   shared batches and optimizer loop
    pretraining/
      pipeline.py                 stage 1 orchestration
    post_training/
      evaluation.py               split integrity and exhaustive scoring
      pipeline.py                 stage 2 orchestration
    serving/
      generation.py               sampling and ABI decode-session orchestration
      service.py                  synchronized model runtime
      http.py                     local JSON HTTP adapter
  examples/python/
    prepare_huggingface_data.py   bounded dataset preparation CLI
    compare_fine_tuning.py        full-versus-LoRA generalization CLI
    compare_lora_ranks.py         reproducible LoRA-rank CLI
  tests/python/test_python_binding.py
  tests/python/test_huggingface_data.py
  tests/python/test_fine_tuning_experiment.py
  tests/python/test_lora_rank_experiment.py
  tests/python/test_generation.py
  tests/python/test_stage_stack.py
  tests/python/test_package_structure.py
```

The `model` test directory mirrors all three model components. The `optim`
test directory contains optimizer verification. The native artifact,
training, and stage tests cover state handoff, configuration contracts,
composition, paged KV-cache behavior, and serving generation. The serving
design and its deliberate scheduler limits are documented in `docs/SERVING.md`.
Downloaded corpora are generated inputs below ignored `data/external/`;
experiment bundles and summaries are generated outputs below ignored
`results/`. Their manifests and fingerprints should be retained with
experimental records rather than committed as framework source. The data
preparation and selection contracts are documented in
`docs/DATASETS_AND_LORA_EXPERIMENTS.md`; metric definitions and the native
split-aware stage are documented in `docs/GENERALIZATION.md`.
Avoid generic helper directories: a helper should live with the domain that
owns its behavior.

## Framework extension seams

The public C++ framework separates three forms of extension:

- derive a concrete `Module`, keep direct children at stable addresses, and
  register parameters or child modules once during construction before the
  subtree is attached and sealed;
- implement a typed forward method by composing existing `Variable`
  operations, or attach an externally computed tensor with
  `custom_gradient(output, inputs, vjp)`;
- pass the resulting `ParameterList` to generic consumers such as Adam,
  backend transfer, or `global_gradient_norm`.

`parameters()` traverses the registered tree in stable depth-first insertion
order. `ModuleList` shared-owns repeated children and supplies numeric
registration names. Parameter entries retain canonical state through
`ParameterHandle`;
their public raw pointers are compatibility views rather than a separate
ownership contract.

The module base intentionally does not define one virtual `forward()` and does
not own backward traversal. Input types remain concrete, while core autograd
owns the chain rule and validates public custom VJPs. This is source-level
framework extensibility, not a runtime binary-plugin system. Dynamically
attached LoRA factors also remain an explicit separate list rather than
changing the stable base-parameter tree.

## CMake organization

The root `CMakeLists.txt` defines the project targets and delegates focused
policy to:

```text
cmake/RiftcoTransformerBackends.cmake
cmake/RiftcoTransformerSanitizers.cmake
cmake/RiftcoTransformerWarnings.cmake
cmake/RiftcoTransformerInstall.cmake
```

The install module exports `riftco_transformer::library` and
`riftco_transformer::c_api`. The private adapter header stays under
`src/core/backend` and is not installed. `tests/CMakeLists.txt` owns test
registration through one helper function, while `tests/package` verifies that
fresh C and C++ projects can consume only the installed package.

At repository level, `.github/workflows/release.yml` builds and verifies the
source distribution and the supported Linux, macOS, and Windows wheels before
creating a tagged release. Platform claims should be based on a green wheel
job, not inferred from a local stub build.

## SOLID-oriented backend module

The `src/core/backend/` folder mirrors the design roles:

| File | Responsibility |
| --- | --- |
| `storage.hpp` | Backend-owned contiguous storage contract |
| `adapter.hpp` | Composition facade for the segregated backend capabilities |
| `attention/contracts.hpp` | Separate materialized-causal, Flash-causal, and paged-decode request shapes |
| `attention/capability.hpp` | Attention-only Adapter interface |
| `attention/dispatch.*` | Backend selection plus attention contract and alias validation |
| `attention/reference/*` | Readable CPU materialized-causal, tile-8 Flash-causal, and paged-decode algorithms |
| `attention/cuda/*` | Native CUDA materialized-causal, memory-linear Flash-causal, and paged-decode launchers/kernels |
| `attention/metal/*` | Focused materialized, tile-8 Flash, and paged-decode Metal shader-source families |
| `attention/tpu/*` | Shape-specialized StableHLO materialized-causal and paged-decode programs; Flash intentionally remains reference-backed |
| `nn/contracts.hpp` | Elementwise, reduction, layout, softmax, indexing, normalization, and loss request shapes |
| `nn/capability.hpp` | Segregated NN Adapter interfaces |
| `nn/dispatch.*` | Backend selection plus NN shape, storage, and alias validation |
| `nn/reference/*` | Shared readable host math and the CPU/TPU semantic oracle |
| `nn/cuda/*` | Native synchronous CUDA launchers/kernels for every NN request |
| `nn/metal/*` | Native Metal NN shader source and launch boundary into the shared runtime |
| `optim/adam/contracts.hpp` | Transactional out-of-place Adam batch request |
| `optim/adam/capability.hpp` | Adam-only Adapter interface |
| `optim/adam/dispatch.*` | Backend selection plus Adam scalar, storage, and candidate-alias validation |
| `optim/adam/reference/*` | Shared double-intermediate Adam semantics used by CPU, TPU, and Metal safety retry |
| `optim/adam/cuda/*` | Native double-intermediate CUDA candidate-state update |
| `optim/adam/metal/*` | Fused Metal shader source, launch boundary, and path diagnostics |
| `registry.cpp` | Backend identity lookup, availability, and selection |
| `adapters/cpu/adapter.cpp` | CPU storage, matmul, and reference-operation delegation |
| `adapters/metal/*` | Apple Metal adapter/stub and shared lazy kernel runtime |
| `adapters/cuda/*` | Optional managed CUDA adapter, native capability delegation, and default-build stub |
| `adapters/tpu/*` | Optional host-mirrored TPU adapter, generic dynamic PJRT runtime, and default-build stub |
| `unavailable_adapter.hpp` | Shared closed-registry stub implementation for optional backends |

This structure applies the useful parts of SOLID without hiding control flow:

- **Single responsibility:** shared contracts, portable reference math, backend
  selection, and each device integration live in separate subtrees.
- **Open/closed:** a built-in backend adds one adapter and one explicit registry
  entry; tensor, autograd, model, C ABI, and Python code do not change unless
  that backend is exposed through those public surfaces.
- **Liskov substitution:** every available adapter must honor the same
  validated, synchronous storage and operation contracts.
- **Interface segregation:** storage, layout, elementwise, reduction, matmul,
  softmax, indexing, normalization, loss, attention, and Adam remain focused
  capability interfaces rather than one unstructured device API.
- **Dependency inversion:** `tensor_ops` dispatches through `BackendAdapter`;
  it never includes CPU, Objective-C, Metal, CUDA, or PJRT implementation
  details.

The registry remains explicit rather than using static self-registration.
That makes startup order, duplicate identity, and adapter lifetime behavior
deterministic. Future kernels should extend or add focused capability
interfaces instead of accumulating unrelated methods in one contract.

There are deliberately no empty `nn/tpu/` or `optim/adam/tpu/` directories.
The TPU adapter names those two fallbacks by calling the shared reference
modules directly; a TPU-specific subtree should appear only when it owns a
real PJRT/StableHLO implementation.
