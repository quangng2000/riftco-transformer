# Architecture

## Implemented end-to-end dataflow

```text
UTF-8 corpus
    │
    ▼
byte tokenizer
    │ token ids [batch, time]
    ▼
token embedding + positional embedding
    │ hidden state [batch, time, d_model]
    ▼
┌──────────────── transformer block × N ────────────────┐
│ layer norm → causal multi-head attention → residual   │
│ layer norm → GELU feed-forward network → residual     │
└────────────────────────────────────────────────────────┘
    │
    ▼
final layer norm → language-model head
    │ logits [batch, time, vocabulary]
    ▼
cross-entropy loss → backward pass → parameter gradients
    ↓
global-norm clipping → Adam → updated parameter values
```

## First model

The initial configuration deliberately stays small:

| Property | Value |
| --- | ---: |
| Context length | 16 |
| Model width | 32 |
| Attention heads | 4 |
| Head width | 8 |
| Transformer blocks | 2 |
| Feed-forward width | 64 |

These dimensions are large enough to exercise the real transformer equations
and small enough to inspect in a debugger.

The mathematical sections use the following shape symbols:

| Symbol | Meaning |
| --- | --- |
| $B$ | batch size |
| $T$ | sequence or context length |
| $D$ | model feature width |
| $H$ | number of attention heads |
| $d_h = D/H$ | feature width of one attention head |
| $F$ | feed-forward hidden width |
| $V$ | vocabulary size |

Each pre-normalized block applies:

```math
\begin{aligned}
a &= x + \mathrm{CausalSelfAttention}
    \left(\mathrm{LayerNorm}(x)\right), \\
y &= a + \mathrm{FeedForward}
    \left(\mathrm{LayerNorm}(a)\right).
\end{aligned}
```

Here, $x$ is the block input, $a$ is the state after attention, and $y$ is the
block output. All three have shape $[B,T,D]$. The residual paths preserve this
shape and give gradients a direct route around each learned sublayer.

## Tensor conventions

- Storage is contiguous and row-major.
- Activations use `float`.
- Activation shape order is $[B,T,D]$.
- Linear weights use $[D_{\mathrm{out}},D_{\mathrm{in}}]$.
- Attention scores use $[B,H,T_q,T_k]$.
- The causal mask prevents a key position $t_k$ from being used when
  $t_k > t_q$.
- Tests fail immediately on incompatible shapes.

## Module, parameter, and autograd boundaries

The framework has separate composition, differentiation, and update
boundaries:

```text
registered Module tree ──→ ordered ParameterList ──→ Adam / decoder state / to()
          │
          └─ typed forward methods ──→ Variable operation graph ──→ autograd
                                           ▲
                                           └─ custom_gradient VJP seam
```

`Linear`, `Embedding`, `LayerNorm`, `LowRankAdapter`, `FeedForward`,
`CausalSelfAttention`, `TransformerBlock`, and `DecoderOnlyTransformer`
inherit `Module`. Direct parameters and child modules are registered during
construction. Traversal derives recursively qualified names in stable
registration order; `ModuleList` gives repeated children numeric segments such
as `blocks.0` and `blocks.1`.

Direct member-child links are non-owning, so modules cannot copy or move after
registration; `ModuleList` owns repeated children through shared pointers.
Attached static subtrees are sealed against later registration mutation.
Parameter entries instead own `ParameterHandle` values that retain canonical
parameter state. The raw `NamedParameter::parameter` field remains a
compatibility view, while copied lists and optimizers keep the state alive.
Recursive backend transfer prepares every changed value and gradient before
committing the tree. A polymorphic extra-parameter hook includes dynamically
attached LoRA storage without changing the stable base-parameter schema.

There is no generic module `forward()` or module-level `backward()`. Concrete
components keep type-specific forward APIs, and central autograd owns reverse
topological traversal, accumulation, and chain-rule composition. The public
`custom_gradient` operation lets a fused or backend-specific tensor result
provide a validated VJP without exposing graph internals. LoRA factors remain
an explicit parameter group outside the model's stable base-parameter tree.
See [MODULES.md](MODULES.md) and [AUTOGRAD.md](AUTOGRAD.md) for these contracts.

## Training boundary

Adam is downstream of the transformer and consumes the generic parameter-list
contract:

```text
parameters → forward pass → loss → gradients → Adam → updated parameters
```

It is not a transformer layer. Keeping the optimizer and global gradient norm
generic over `ParameterList` lets their equations remain independently
testable with known values and usable with a full model, LoRA-only list, or
custom module. The training executable now repeats this transaction across
deterministically sampled batches and records one CSV metrics row after every
successful update. Each iteration performs a new forward pass and therefore
builds a fresh autograd graph from the parameters updated by the previous
iteration. See [TRAINING.md](TRAINING.md) for the loop boundary and metrics
schema.

## Serving boundary

Training keeps the full-sequence `forward` path above. Serving adds a detached
one-token path over the same model parameters:

```text
prompt token IDs
    │ one token at a time
    ▼
decode_token(token, request cache)
    │ append per-layer K/V and attend through a logical page table
    ▼
latest logits [1, 1, vocabulary]
    │ sample one token
    └───────────────────────────────────────────────┐
                                                    ▼
                                             next decode_token
```

Prompt prefill and autoregressive decode currently use the same one-token
operation. The request cache owns a logical page table; the default
`PagedKvCachePool` leases physical pages from backend-resident key/value pools
with one pool pair per transformer layer. `ContiguousKvCacheFactory` supplies a
fixed-layout reference strategy behind the same `KeyValueCacheFactory`
interface. CPU and Metal both read the paged layout directly.

The model uses learned absolute positions. At maximum context, generation
cannot preserve the existing cropped-window result by evicting only the oldest
page: the remaining tokens must be renumbered. It resets the cache and replays
the retained suffix from position zero. See [SERVING.md](SERVING.md) for the
layout, lifecycle, stable C/Python session surface, and current scheduling
limits.

## Execution-backend boundary

Every tensor owns backend-specific storage. Tensor operations validate shapes
and backend compatibility, allocate outputs on the input backend, and then call
a focused adapter capability:

```text
tensor_ops / NN / model operation
    │ validate shapes and create output
    ▼
fixed backend registry
    │ choose by the tensor's ExecutionBackend
    ▼
BackendAdapter capabilities
    ├── StorageCapability
    ├── MatmulCapability
    ├── Elementwise / Reduction / Layout
    ├── Softmax / Indexing / Normalization / Loss
    ├── Materialized-causal, Flash-causal, and paged-decode attention
    └── AdamCapability
         │
         ├── CpuBackendAdapter
         └── MetalBackendAdapter / unavailable platform stub
```

The adapter normalizes different implementation technologies behind one
synchronous storage contract. CPU storage owns a vector; Metal storage owns a
persistent shared `MTLBuffer`. Selecting CPU or Metal is the Strategy choice;
adapting each implementation to the common capabilities is the Adapter role.
Capability-specific base interfaces keep storage, generic tensor math, neural
operations, attention, and optimizer growth independent. CPU delegates the
capabilities to readable reference functions. Metal implements the same
requests with compute pipelines over persistent shared buffers. The registry
is immutable after compilation, so lookup is deterministic and thread-safe
without runtime registration races.

Autograd graph traversal remains host control flow, but its values, saved
tensors, seeds, and accumulated gradients retain one backend. Local backward
rules route layout, elementwise, reduction, GELU, LayerNorm, softmax,
gather/scatter, loss, matmul, and full-sequence attention vector-Jacobian
products through that backend. Paged decode attention is serving-only and
returns a detached context. Adam computes the overflow-safe global norm on
host-visible
shared storage, then batch-encodes one out-of-place fused update dispatch per
parameter tensor in a single command buffer for safe, well-conditioned
arithmetic. The precise-math kernel requests a whole-batch retry when its
actual operation sequence becomes subnormal, non-finite, or ill-conditioned;
that retry uses the shared `double` reference without changing tensor storage.
Live parameter and moment state is committed only after the whole batch
succeeds.

Metal matmul, neural, attention, and Adam shader sources build focused pipeline
states lazily. Failure to build one operation therefore does not invalidate
persistent storage or another operation's already-usable pipeline.

The adapter vtable stays below `src/` and is not part of the installed C++ ABI.
Framework users consume public C++ headers, the versioned C ABI, or the Python
wrapper. A future third-party binary-plugin system would require a separately
versioned C function table rather than exposing compiler-specific C++ class
layout.

## Intentional limits

Metal storage is persistent but shared and synchronously host-addressable.
Adapter calls wait for command completion before returning; graph operations
are not scheduled into asynchronous streams. Full-sequence attention defaults
to the materialized probability path, with an opt-in exact tile-8 Flash path
on CPU and Metal. Flash saves `[B,H,T]` row maxima and exponential sums and
reconstructs probabilities during backward instead of retaining a
`[B,H,T,T]` tensor. This memory contract is not itself a claim of measured
speedup.

Transformer-block activation checkpointing is another independent opt-in
policy. It drops internal block graph nodes after forward and rebuilds them in
an isolated nested VJP during backward. The numerical work automatically uses
the model's CPU or Metal backend. Checkpointing reduces retained activations,
not parameters or Adam state, and adds one block replay per backward pass.

Incremental serving has paged KV caching but no batched prefill,
continuous-batching scheduler, or shared-prefix cache. Its prefill remains
token-at-a-time and does not use the full-sequence Flash path. Private GPU
memory, asynchronous command batching, flattened multi-tensor arenas, broader
fusion, SIMD, mixed precision, dropout, distributed training, and memory
mapping remain future work.
