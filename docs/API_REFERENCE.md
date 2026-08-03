# API reference

This page indexes the supported public surfaces of Riftco Transformer 0.6.0.
It is a navigation reference, not generated Doxygen: signatures are shortened
where that improves scanning, and the linked headers remain authoritative.
See [Architecture](ARCHITECTURE.md) for subsystem boundaries and
[Project structure](PROJECT_STRUCTURE.md) for implementation ownership.

## Package targets

Installed consumers use CMake 3.24 or newer and C++20:

```cmake
find_package(riftco_transformer 0.6 CONFIG REQUIRED)
target_link_libraries(app PRIVATE riftco_transformer::library)
```

| Imported target | Public surface | Dependency boundary |
| --- | --- | --- |
| `riftco_transformer::library` | Tensor, autograd, neural modules, model, optimizer, artifacts, native serving | Core runtime |
| `riftco_transformer::compiler` | Cajal types, AST, evaluator, encoding, compiler | Standard library only |
| `riftco_transformer::analysis` | Matrices, representation traces, PCA, interventions, ablations | Standard library only |
| `riftco_transformer::lowering` | Cajal/multilinear-map to neural modules | Compiler + runtime |
| `riftco_transformer::programmed` | Programmed sequence cores, placement, and task-neutral learned/programmed model composition | Analysis + lowering |
| `riftco_transformer::c_api` | Stable C ABI 2.8 shared library | Runtime and programmed composition behind opaque handles |

The exported target definitions live in
[`CMakeLists.txt`](https://github.com/quangng2000/riftco-transformer/blob/main/CMakeLists.txt)
and
[`RiftcoTransformerInstall.cmake`](https://github.com/quangng2000/riftco-transformer/blob/main/cmake/RiftcoTransformerInstall.cmake).
The installed CMake package uses same-minor version compatibility. Only the C
surface carries an explicit binary ABI version; C++ consumers should rebuild
against the selected package release.

## C++ core runtime

All names below are in `riftco_transformer` unless another namespace is shown.

| Header | Principal API | Contract and ownership |
| --- | --- | --- |
| [`core/backend.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/core/backend.hpp) | `ExecutionBackend`, `execution_backend_available`, `set_execution_backend`, `ScopedExecutionBackend` | The construction default is thread-local. Existing tensors retain their intrinsic backend. A scope guard is non-copyable, non-movable, and must die on its creating thread. |
| [`core/tensor.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/core/tensor.hpp) | `Tensor(shape, values, backend)`, `zeros`, `full`, `reshape`, `to`, `shape`, `strides`, `data` | Owns contiguous row-major FP32 storage. Copying performs a deep backend allocation; moving transfers it. `to()` returns an independent copy. |
| [`core/tensor_ops.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/core/tensor_ops.hpp) | Elementwise arithmetic, `matmul`, layout, reductions, math, softmax | Pure tensor operations validate shapes and preserve input backend identity. Mixed-backend numerical inputs are rejected. |
| [`core/autograd.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/core/autograd.hpp) | `Variable(Tensor, requires_gradient)`, `backward`, differentiable operators, `custom_gradient`, `checkpoint` | A `Variable` copy shares its graph node. Scalar outputs may use implicit seed `1`; non-scalars require a same-shape seed. |
| [`core/quantized_weight.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/core/quantized_weight.hpp) | `QuantizedWeight::quantize_nf4`, `quantize_nf4_double_quantized`, `from_packed_nf4`, `dequantize`, `to`, `memory_usage` | Copies share immutable packed storage. Readback copies packed payload; `dequantize()` is the explicit FP32 materialization boundary. |

Representative tensor and autograd signatures:

```cpp
Tensor tensor({2, 3}, values, ExecutionBackend::Cpu);
Tensor product = tensor_ops::matmul(left, right);

Variable x(Tensor({2}, {2.0F, 3.0F}));
Variable loss = sum(x * x);
loss.backward();
```

See [Tensor](TENSOR.md), [Tensor operations](TENSOR_OPS.md), and
[Autograd](AUTOGRAD.md) for behavioral detail.

## Modules, models, and optimization

| Header | Principal API | Contract and ownership |
| --- | --- | --- |
| [`nn/parameter.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/nn/parameter.hpp) | `Parameter`, `ParameterHandle`, `NamedParameter`, `ParameterList`, `move_parameters_to`, `parameter_count` | Handles retain canonical parameter state; copied lists remain valid independently of the originating wrapper. |
| [`nn/module.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/nn/module.hpp) | `Module::parameters`, `Module::to`, `ModuleList` | Modules are non-copyable and non-movable. Registered child links are non-owning; `ModuleList` owns repeated children. |
| [`nn/linear.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/nn/linear.hpp) | `Linear::forward`, LoRA attachment/merge, NF4 conversion | Owns dense parameters or an immutable packed base weight, never both as trainable base state. |
| [`nn/embedding.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/nn/embedding.hpp) | `Embedding::forward` | Gathers rows from a registered embedding table. |
| [`nn/layer_norm.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/nn/layer_norm.hpp) | `LayerNorm::forward` | Registered scale and bias; differentiable normalization. |
| [`nn/rms_norm.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/nn/rms_norm.hpp) | `RMSNorm::forward`, `rms_norm` | Scale-only root-mean-square normalization with a fully differentiable reference composition. |
| [`nn/low_rank_adapter.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/nn/low_rank_adapter.hpp) | `LowRankAdapter::forward`, `weight_delta` | Owns floating-point A/B adapter parameters. |
| [`nn/activations.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/nn/activations.hpp) and [`nn/loss.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/nn/loss.hpp) | `gelu`, `relu`, `softmax`, `cross_entropy`, `cross_entropy_time_range` | Differentiable operations over `Variable`; loss returns a scalar mean over all positions or one contiguous per-batch time range. |
| [`model/feed_forward.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/model/feed_forward.hpp) | `FeedForward`, `FeedForwardActivation` | Position-wise expand/activate/project module with GELU or ReLU. |
| [`model/causal_self_attention.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/model/causal_self_attention.hpp) | `CausalSelfAttention`, `FullSequenceAttentionKind`, head split/merge, diagnostic materialized attention | Full-sequence materialized/Flash policy is independent of incremental decode. |
| [`model/transformer_block.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/model/transformer_block.hpp) | `TransformerBlock::forward` | Pre-normalized attention and feed-forward residual composition. |
| [`model/decoder_kv_cache.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/model/decoder_kv_cache.hpp) | `DecoderKeyValueCache` | Abstract, caller-owned per-request cache mutated transactionally by token decode. |
| [`model/decoder_only_transformer.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/model/decoder_only_transformer.hpp) | `DecoderOnlyTransformer::forward`, `decode_token`, `to`, NF4/LoRA lifecycle, `parameters` | Model is non-copyable/non-movable. Full forward builds an autograd graph; token decode returns detached logits and mutates a caller-owned cache. |
| [`model/llama_mistral_transformer.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/model/llama_mistral_transformer.hpp) | `LlamaMistralConfig`, `LlamaMistralTransformer::forward` | Experimental native C++ dense full-context RMSNorm/RoPE/GQA/SwiGLU runtime. Narrow sliding windows and external checkpoint/tokenizer reinterpretation are rejected. |
| [`optim/adam.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/optim/adam.hpp) | `Adam(ParameterList, AdamOptions)`, `step`, `zero_gradients`, `state`, `load_state` | Retains parameter handles and owns first/second moments. Updates and logical-state restoration are transactional across the registered list. |

Model construction and forward:

```cpp
std::mt19937 random(42);
DecoderOnlyTransformer model(
    TransformerDimensions{256, 16, 32, 4, 2, 64}, random
);
Variable logits = model.forward(token_ids, {batch, time});
```

`forward()` returns `[batch, time, vocabulary]`. Training parameters come from
`model.parameters()`; after LoRA attachment, adapter parameters come from
`model.lora_parameters()`. Quantized frozen weights deliberately do not appear
in either optimizer list. See [Modules](MODULES.md), [Transformer](TRANSFORMER.md),
[LoRA](LORA.md), [QLoRA](QLORA.md), and [Adam](ADAM.md).
The distinct dense family topology and its current non-goals are listed in
[Dense Llama and Mistral runtime boundary](LLAMA_MISTRAL.md).

## Data, native serving, and artifacts

| Header | Principal API | Lifetime notes |
| --- | --- | --- |
| [`data/tokenizer.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/data/tokenizer.hpp) | `TokenizerStrategy`, `ByteTokenizer`, `BytePairTokenizer`, `make_tokenizer` | Tokenizers own immutable vocabulary state and return owned token vectors/strings. |
| [`data/token_batch.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/data/token_batch.hpp) | `TokenBatch`, `make_next_token_batch`, `sample_next_token_batch` | Batch owns rectangular input and target token arrays. Caller owns the seeded RNG. |
| [`stages/serving/stack.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/stages/serving/stack.hpp) | `ServingStack(snapshot, config)`, `generate` | Restores model/tokenizer inference state and owns its cache factory. |
| [`artifacts/state.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/artifacts/state.hpp) | `ModelSnapshot`, `capture_snapshot`, `load_model_state`, `restore_tokenizer` | In-memory, backend-neutral value handoff; no optimizer state, lineage, or persistence. |

The stage umbrella is
[`stages/stages.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/stages/stages.hpp).
Configuration defaults are listed in [Configuration reference](CONFIGURATION_REFERENCE.md).

## Compiler, lowering, and interpretation

| Namespace and header | Principal API |
| --- | --- |
| [`compiler::cajal`](https://github.com/quangng2000/riftco-transformer/tree/main/include/riftco_transformer/compiler/cajal) | Immutable `Type`, `Expression`, and `Value`; `type_check`, `evaluate`, `encode`, `decode`, `compile`; `CompiledProgram` and `MultilinearMap` |
| [`lowering`](https://github.com/quangng2000/riftco-transformer/tree/main/include/riftco_transformer/lowering) | `NeuralLoweringConfig`, `LoweringRegistry`, `analyze_neural_lowering`, `lower_to_neural`, `LoweredMultilinearModule` |
| [`programmed/sequence_placement.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/programmed/sequence_placement.hpp) | `ProgrammedSequenceCore`, `ProgrammedSequenceAdapter`, projection sharing, placement, steering, and batch-roll ablation options |
| [`programmed/program_augmented_model.hpp`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/programmed/program_augmented_model.hpp) | `ProgramAugmentedModelConfig`, move-only `ProgramBranch`, `ProgramAugmentedForwardOptions`, `ProgramAugmentedModel` |
| [`analysis`](https://github.com/quangng2000/riftco-transformer/tree/main/include/riftco_transformer/analysis) | `Matrix`, `RepresentationTrace`, `fit_pca`, `transform_pca`, `apply_intervention`, `summarize_ablation` |

The compiler and analysis libraries do not depend on the tensor runtime.
Lowering is the explicit one-way bridge into differentiable modules. Cajal is
a finite, first-order language constructed through C++ APIs; it is not a text
parser or a general lambda-calculus implementation. See
[Compiling to transformers](COMPILING_TO_TRANSFORMERS.md).

## Program-augmented composition

`ProgramAugmentedModel` is a fixed-context, task-neutral composition. For
token-plus-position state $x$, residual width $D$, and $N\geq1$ independently
parameterized causal-attention branches, its learned path is

```math
\begin{aligned}
r_1 &= x + \mathrm{FFN}_{\mathrm{ReLU}}(x),\\
h &= W_A[\mathrm{Attn}_1(r_1);\ldots;
             \mathrm{Attn}_N(r_1)] + b_A.
\end{aligned}
```

Without a program, $r_2=r_1+h$. With a `ProgramBranch`, the model selects the
configured source span, runs `ProgrammedSequenceCore`, places the raw program
output at the arbitrary configured target offset with zeros elsewhere, and
computes

```math
r_2=r_1+W_M[h;\mathrm{place}(p)]+b_M.
```

A final learned projection produces vocabulary logits. The branch config owns
source/target offsets, core input layouts and shared projection groups, a
lowered module, and optional merge bias. Forward options support affine
program-input steering and one shared positive batch-roll shift for learned
attention, selected program inputs, and/or raw program output. These are graph
interventions: gradients still flow through placement and the selected model
paths.

When capture is enabled, the owning host trace uses these stable names:

- `embedding.sum`
- `residual.pre_attention`
- `learned_attention.merged`
- `program.source` when a branch exists
- `program.input.N` and `program.input.N.projected` for each logical input
- `program.output.raw` and `program.output.placed` when a branch exists
- `residual.post_merge`
- `logits`

`cross_entropy_time_range(logits, targets, time_offset, time_count)` selects
the same contiguous time interval independently in every `[batch,time,vocab]`
row before taking mean cross entropy. Ordinary `cross_entropy` remains the
all-position objective.

## Stable C ABI

[`c_api.h`](https://github.com/quangng2000/riftco-transformer/blob/main/include/riftco_transformer/c_api.h)
defines C ABI 2.8. It uses fixed-width constants, status returns, versioned
value structures, and opaque handles:

```c
rt_context* context = NULL;
rt_status status = rt_context_create(RT_BACKEND_CPU, &context);
/* use context */
rt_context_release(context);
```

| Family | Functions |
| --- | --- |
| ABI and errors | `rt_abi_version`, `rt_status_string`, `rt_last_error` |
| Tokenization | `rt_tokenizer_options_init`, create/restore, vocabulary/merge queries, encode/decode, `rt_tokenizer_release` |
| Backend and tensors | `rt_backend_is_available`, context create/query/release, FP32 tensor create/query/copy/matmul/release |
| Model | Decoder config initialization, create/transfer/query, attention/checkpointing selection, forward, NF4 conversion, packed-state size/copy/transactional load, LoRA attach/query/merge, memory statistics, release |
| Dense Llama/Mistral | `rt_llama_mistral_config_init`, create/transfer/backend, full-sequence forward, base-parameter list, release |
| Serving | Decode-session options, create, step, reset, cache queries, release |
| Multilinear maps | Dense or sparse output-major import into `rt_multilinear_map`, copied ownership, release |
| Programmed model | Versioned model/branch/lowering/forward configs; create, transfer, query, parameters, forward, release |
| Representation traces | Owning trace count/name/shape/value queries and release |
| Parameters and autograd | Base/LoRA/programmed parameter lists, shape/value transfer, checkpoint-safe frozen-base restore under the sole adapter Adam, model forward variables, cross-entropy, backward, release |
| Optimization | Adam options, create, step, zero gradients, state diagnostics, logical state size/copy/load, release |

Initialize every versioned structure with its matching `rt_*_init` function
and its actual `sizeof(...)`. Every successful create returns one handle that
must be released exactly once. Derived handles retain model state, but a live
decode session pins model backend and parameter values. Raw C calls are not
internally lifetime-synchronized; callers must serialize operations involving
the same state. Variable-size outputs support a null-output/zero-capacity size
query. Copy `rt_last_error()` before the next status-returning call on that
thread.

## Python API

Python 3.10+ has no runtime package dependencies. The top-level
[`riftco_transformer`](https://github.com/quangng2000/riftco-transformer/blob/main/python/riftco_transformer/__init__.py)
module re-exports the native layer:

```python
from riftco_transformer import (
    Adam, Context, DecoderOnlyTransformer, LoraConfig, Tensor,
    LlamaMistralConfig, LlamaMistralTransformer, Tokenizer,
    TransformerConfig, Variable, backend_available, cross_entropy,
    cross_entropy_time_range,
)
```

Native wrappers are closeable context managers and cannot be copied. They
retain related state and synchronize operations through shared locks. Important
high-level packages are:

| Package | Public entry points |
| --- | --- |
| [`artifacts`](https://github.com/quangng2000/riftco-transformer/tree/main/python/riftco_transformer/artifacts) | `ModelBundle`, `ModelRuntime`, `ParameterSpec`, `TokenizerSpec` |
| [`checkpoints`](https://github.com/quangng2000/riftco-transformer/tree/main/python/riftco_transformer/checkpoints) | `TrainingCheckpoint.capture`, `save`, `load`, `restore`; v2 packed QLoRA plus v1 dense loading; `TrainingCheckpointRestore` |
| [`data`](https://github.com/quangng2000/riftco-transformer/tree/main/python/riftco_transformer/data) | Hugging Face HTTP client, dataset adapters, stable splitting, serializers, preparation and verification |
| [`interchange`](https://github.com/quangng2000/riftco-transformer/tree/main/python/riftco_transformer/interchange) | `load_model`, `export_model`, `convert_model`; F32 SafeTensors, Riftco Hugging Face directory, GGUF v3, and strict canonical ONNX interchange |
| [`training`](https://github.com/quangng2000/riftco-transformer/tree/main/python/riftco_transformer/training) | Batch sources, `TrainingLoopConfig`, `CausalLanguageModelTrainer`, backend selection |
| [`pretraining`](https://github.com/quangng2000/riftco-transformer/tree/main/python/riftco_transformer/pretraining) | `PretrainingConfig`, `pretrain_text`, `pretrain_splits`, `pretrain_file`, `pretrain_files` |
| [`post_training`](https://github.com/quangng2000/riftco-transformer/tree/main/python/riftco_transformer/post_training) | Instruction loading/splits, `PostTrainingConfig`, `post_train`, held-out evaluation |
| [`programmed`](https://github.com/quangng2000/riftco-transformer/tree/main/python/riftco_transformer/programmed) | `MultilinearMap.from_dense/from_sparse`, lowering and branch configs, `ProgramAugmentedModel`, interventions, owning traces |
| [`serving`](https://github.com/quangng2000/riftco-transformer/tree/main/python/riftco_transformer/serving) | Samplers, `TextGenerator`, `ModelService`, dependency-free HTTP server |

Repository research protocols are intentionally outside this installed API.
Top-level [`labs`](https://github.com/quangng2000/riftco-transformer/tree/main/labs)
contains Python-owned fine-tuning, LoRA-rank, and conditional-reversal labs;
they compose the public packages above and write ignored `runs/` output.
In particular, the framework exposes no native F/P/T/I experiment type:
conditional-reversal program construction, training, evaluation, PCA policy,
and reporting remain in the Python lab.

Python loads the native library from the wheel, recognized source-build
directories, the system loader, or the explicit `RIFTCO_TRANSFORMER_LIBRARY`
path. See [Troubleshooting](TROUBLESHOOTING.md) for loader and ABI errors.
