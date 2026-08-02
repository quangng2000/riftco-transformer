# Neural-Network Primitives and Feed-Forward Layer

The project now has the bridge from integer token IDs to trainable feature
vectors and a scalar language-model loss:

```text
token IDs [batch, time]
    ↓ token Embedding + learned positional Embedding
features [batch, time, d_model]
    ↓ pre-normalized residual TransformerBlock × N
contextual features [batch, time, d_model]
    ↓ final LayerNorm
normalized features
    ↓ Linear language-model head
logits [batch, time, vocabulary]
    ↓ CrossEntropy
scalar loss
```

The smoke executable runs this complete path and calls `backward()`, proving
that gradients traverse both embedding tables, every normalization, attention
and feed-forward sublayer, residual block, and output head.

## Parameters and registration

A `Parameter` owns a trainable leaf `Variable`. It exposes its value and
gradient and permits a same-shaped value replacement. Its shared state also
has one canonical identity retained by `ParameterHandle`:

```text
Parameter
├── Tensor value
├── Tensor gradient
├── shared autograd leaf identity
└── canonical ParameterHandle identity
```

`Linear`, `Embedding`, `LayerNorm`, `LowRankAdapter`, `FeedForward`,
`CausalSelfAttention`, `TransformerBlock`, and `DecoderOnlyTransformer`
inherit `Module`. Constructors register direct parameters and child modules.
`parameters()` traverses that tree in stable registration order and returns
recursively qualified names:

```text
Module registration tree
    ↓ depth-first traversal
name + owning ParameterHandle → value, gradient, optimizer state
```

Each `NamedParameter` retains the canonical parameter state and keeps a public
`.parameter` raw-pointer compatibility view. A copied parameter list therefore
remains valid if the originating wrapper or module leaves scope. Callers that
extract the raw view must keep a corresponding list, handle, or optimizer
alive.

Direct member-child registrations are non-owning. Modules cannot copy or move,
so a parent's registered child addresses remain stable for its usable
lifetime. Attached subtrees are sealed against later static registration.
`ModuleList` shared-owns repeated children and registers them as `0`, `1`, and
so on; the decoder uses it for names such as
`blocks.0.attention.query.weight`.

The module lifecycle does not prescribe one virtual `forward()` and does not
add a per-module `backward()`. Concrete layers retain typed forward methods,
while central autograd applies the chain rule to their composed `Variable`
operations. Fused operations can use the validated public
`custom_gradient(output, inputs, vjp)` seam described in
[AUTOGRAD.md](AUTOGRAD.md).

Parameter values must only be changed after the current forward/backward pass
is finished. Training rebuilds the computation graph after every Adam update.
See [MODULES.md](MODULES.md) for registration validation, ownership, recursive
naming, transfer, and extension examples.

## Embedding lookup

Let $V$ be the vocabulary size and $D = d_{\text{model}}$ be the feature
width. An embedding table has shape:

```text
[vocabulary, d_model]
```

Equivalently, the table is a matrix $E \in \mathbb{R}^{V \times D}$.
Token ID $k$ selects row $E_{k,:}$. Therefore:

```text
token IDs [batch, time]
       ↓
features [batch, time, d_model]
```

The backward rule scatter-adds gradients into table rows. Scatter-add matters
because a token can occur multiple times in one batch; every occurrence must
contribute to that token's embedding gradient.

## Linear layer

The project uses:

```text
weight [output_width, input_width]
bias   [output_width]
```

For an input `X[..., input_width]`, `Linear` temporarily flattens all leading
dimensions:

```text
X [..., input]
    ↓ reshape
[rows, input]
    ↓ matrix multiply by weight transpose
[rows, output]
    ↓ reshape and broadcast bias
[..., output]
```

This lets one matrix plane of the shared batched-matmul kernel process rank-3
transformer activations without hiding the reshape.

For one flattened input row $\mathbf{x}$, the calculation is:

```math
\mathbf{y} = W\mathbf{x} + \mathbf{b},
```

where $W$ is the weight matrix and $\mathbf{b}$ is the bias vector.

Weights use Xavier-uniform initialization and biases start at zero. A shared
standard-library random generator is passed through layer construction.

`Linear(weight)` and `Linear(input, output, random, false)` construct a
biasless projection, $\mathbf{y}=W\mathbf{x}$. `has_bias()` distinguishes the
two contracts. For source compatibility, `bias()` still returns an inert zero
parameter for a biasless layer, but that object is excluded from forward,
`parameters()`, and Adam state. Code must check `has_bias()` before treating a
bias as trainable. This form is used for the paper's shared program projection
and residual merge.

### Packed quantized linear

For QLoRA, `Linear` replaces its dense weight `Parameter` with an immutable
`QuantizedWeight`; the bias and any attached LoRA factors remain FP32. The
packed operation produces the same output shape and registers a VJP only for
its input. There is deliberately no base-weight gradient or Adam state.

Legacy NF4 stores one FP32 scale per weight block. The default QLoRA conversion
double-quantizes those first-level scales into U8 codes, second-level FP32
scales, and one FP32 offset. Forward and input backward decode values while
accumulating rather than retaining a full FP32 matrix. See
[QLORA.md](QLORA.md) for the exact encoding and payload formulas.

## Layer normalization

Layer normalization operates independently on every final-axis feature
vector $\mathbf{x} \in \mathbb{R}^{D}$. First, it calculates the feature
mean $\mu$ and population variance $\sigma^2$:

```math
\mu = \frac{1}{D}\sum_{i=1}^{D} x_i,
\qquad
\sigma^2 = \frac{1}{D}\sum_{i=1}^{D}(x_i-\mu)^2.
```

It then normalizes and applies a learned scale and bias:

```math
\hat{x}_i = \frac{x_i-\mu}{\sqrt{\sigma^2+\varepsilon}},
\qquad
y_i = \gamma_i\hat{x}_i+\beta_i.
```

Here, $\varepsilon > 0$ prevents division by zero, while
$\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ are the learned `scale`
and `bias`. Both have shape `[d_model]` and are explicitly broadcast across
batch and time. Differentiable broadcasting reduces their backward gradients
back to the original parameter shape.

The implementation uses population variance and keeps epsilon inside the square
root. A constant feature slice therefore remains finite and returns the bias.

## GELU, ReLU, and feed-forward

The exact Gaussian Error Linear Unit is:

```math
\mathrm{GELU}(x)
= \frac{x}{2}\left(1+\mathrm{erf}\left(\frac{x}{\sqrt{2}}\right)\right),
```

where $\mathrm{erf}$ is the Gaussian error function. The public equation
is exact rather than the common tanh approximation. Its implementation routes
one GELU forward request and one local derivative request through the backend;
CPU uses the readable reference formula, Metal and CUDA use focused kernels,
and TPU uses the host reference path over its mirror.

The learned program experiment also exposes rectified linear activation:

```math
\mathrm{ReLU}(x)=\max(0,x),
\qquad
\frac{\partial\mathrm{ReLU}}{\partial x}=
\begin{cases}1,&x>0,\\0,&x\le0.\end{cases}
```

The derivative at zero is deliberately chosen as zero. NaNs remain NaN in
both forward and backward rather than being silently converted to zero.

`FeedForward` accepts `FeedForwardActivation::Gelu` or
`FeedForwardActivation::Relu`; GELU remains the default used by ordinary
decoder blocks. Its shape is:

```text
input [..., d_model]
    ↓ Linear(d_model, d_ff)
hidden [..., d_ff]
    ↓ configured GELU or ReLU
activated [..., d_ff]
    ↓ Linear(d_ff, d_model)
output [..., d_model]
```

It does not contain layer normalization or a residual connection. Those belong
to the transformer block that composes layers:

```text
x → LayerNorm → FeedForward → add x
```

In equation form, if $F$ is the feed-forward transformation, the surrounding
block computes $\mathbf{x} + F(\mathrm{LayerNorm}(\mathbf{x}))$.

Keeping the residual outside makes `FeedForward` reusable and keeps its job
precise.

## Backend execution

The layers contain no device-specific branches. They construct validated
requests and let the input tensor backend choose the Adapter:

- embedding uses row-gather forward and repeated-row scatter-add backward;
- `Linear` uses matmul plus routed broadcast/add operations;
- packed `Linear` uses a quantized-linear forward/input-backward capability and
  never exposes its frozen weight as a gradient parent;
- LayerNorm has fused row normalization and dedicated input/scale/bias
  backward requests;
- GELU and softmax have dedicated forward and vector-Jacobian-product requests;
- cross-entropy computes a stable mean loss and saved base gradient in one
  backend request.

CPU implements these contracts in `backend/nn/reference/`; attention has its
own reference implementations under `backend/attention/reference/`. Packed
linear has a focused `backend/nn/quantized_linear/` subtree. Metal and CUDA
retain packed NF4 payloads in accelerator-visible buffers and decode inside
their quantized-linear kernels. TPU retains packed host payloads, uploads U8
codes and scale metadata, and reconstructs/dequantizes inside StableHLO. TPU
sends materialized attention and paged decode through PJRT but retains the
reference Flash and generic-NN paths over host-mirrored storage.

The CUDA and TPU packed source paths are implemented, but actual NVIDIA GPU and
Cloud TPU execution was not available on this macOS host. Their outstanding
acceptance work is hardware validation, not a missing quantized-linear backend.
Different reduction order and device math mean the parity contract uses
tolerances rather than bitwise equality.

## Stable softmax and cross-entropy

For a vector of logits $\mathbf{z}$, softmax subtracts the maximum
$m = \max_j z_j$ before exponentiation:

```math
\mathrm{softmax}(\mathbf{z})_i
= \frac{\exp(z_i-m)}
       {\sum_j \exp(z_j-m)}.
```

This stays finite for logits such as `[10000, 10001, 10002]`. It supports any
axis and permits $-\infty$ for future causal attention masks. A fully masked
slice is rejected.

Cross-entropy is fused directly from logits rather than computing
`-log(softmax)` as separate float operations. For $N$ token positions, let
$p_{n,k}$ be the softmax probability for vocabulary item $k$ at position
$n$, and let $y_n$ be the target token ID. The mean loss is:

```math
L = -\frac{1}{N}\sum_{n=1}^{N}\log p_{n,y_n}.
```

Its gradient with respect to logit $z_{n,k}$ is:

```math
\frac{\partial L}{\partial z_{n,k}}
= \frac{p_{n,k}-\mathbf{1}[k=y_n]}{N},
```

where $\mathbf{1}[k=y_n]$ is $1$ when $k$ is the target and $0$
otherwise. Targets remain integer `TokenId` values and correspond to the
flattened leading dimensions of `[batch, time, vocabulary]` logits.

## Where attention is

`CausalSelfAttention` is a separate model component, not hidden inside
`FeedForward`. It reuses `Linear`, parameter registration, stable softmax,
general axis permutations, and batched matrix multiplication for:

```text
input
  ├─ query projection ─┐
  ├─ key projection ───┼─ split heads → scaled causal attention
  └─ value projection ─┘                    ↓
                              merge heads → output projection
```

See [ATTENTION.md](ATTENTION.md) for the equations, tensor shapes, mask, and
gradient checks.

`TransformerBlock` composes attention and feed-forward into a complete
pre-normalized residual block:

```text
x ──→ LayerNorm ──→ CausalSelfAttention ──→ add x ──→ y
│                                             ▲
└─────────────────────────────────────────────┘

y ──→ LayerNorm ──→ FeedForward ───────────→ add y ──→ output
│                                             ▲
└─────────────────────────────────────────────┘
```

## Module backend transfer

`Linear`, `Embedding`, `LayerNorm`, `LowRankAdapter`, `FeedForward`,
`CausalSelfAttention`, `TransformerBlock`, and `DecoderOnlyTransformer`
expose `to(ExecutionBackend)`. Their registered base-parameter tree delegates
to one common transactional transfer utility, keeping device policy out of
layer equations. Every changed value and fresh zero gradient is prepared
before any parameter is committed. Transfer a module before constructing a
forward graph or an optimizer; values move to independent storage and moved
leaf gradients restart at zero on the destination backend.

LoRA factors remain a deliberate separate parameter group rather than dynamic
members of the base registration tree. The common `Module::to()` traversal
uses a polymorphic extra-parameter hook at each child, so attached or retained
adapter storage participates in the same transaction even through a
`Module&`. `parameters()` continues to mean base parameters only.

Immutable NF4 weights intentionally sit outside that Parameter tree. Because
they own backend storage of their own, `Module::to()` is virtual: quantized
`Linear` and decoder overrides prepare packed-buffer transfers before the
ordinary parameter transaction. This keeps calls through `Module&` correct and
prevents a model from ending up with parameters and packed weights on different
backends.

## Source map

```text
nn/parameter.hpp/.cpp       trainable state, handles, and named lists
nn/module.hpp/.cpp          registered module lifecycle and ModuleList
nn/embedding.hpp/.cpp       token-row lookup and scatter-add backward
nn/linear.hpp/.cpp          arbitrary-rank final-axis projection
nn/quantized_linear.hpp/.cpp packed NF4 projection and input VJP
nn/layer_norm.hpp/.cpp      final-axis normalization
nn/activations.hpp/.cpp     erf-form GELU and stable softmax
nn/loss.hpp/.cpp            fused cross-entropy
nn/initialization.hpp/.cpp  uniform and Xavier initialization
model/feed_forward.hpp/.cpp Linear → GELU → Linear
model/causal_self_attention.hpp/.cpp
                             scaled causal multi-head self-attention
model/transformer_block.hpp/.cpp
                             pre-normalized residual composition
model/decoder_only_transformer.hpp/.cpp
                             embeddings → block stack → logits
```

Tests compare gradients with centered finite differences and cover rank-3
shapes, repeated embedding IDs, constant normalization slices, large logits,
masked softmax values, parameter registration, feed-forward composition, exact
causal probabilities, attention projection parameters, residual paths,
whole-model causality, end-to-end gradients, CPU reference requests, and
conditional real-Metal forward/backward parity. Focused quantized-linear tests
also compare legacy and double-quantized forward/input gradients with a decoded
CPU oracle on every available accelerator and verify that packed residency is
preserved.
