# Causal Multi-Head Self-Attention

Self-attention lets every token build a new feature vector from tokens at its
own position and earlier positions. “Causal” means a token cannot read future
tokens.

## Dataflow and shapes

Let $B$ be the batch size, $T$ the sequence length, $D$ the model
width, and $H$ the number of attention heads. Each head has width:

```math
d_h = \frac{D}{H}.
```

The layer preserves the input shape:

```text
input [B, T, D]
  ├─ Linear → queries [B, T, D] ─┐
  ├─ Linear → keys    [B, T, D] ─┼─ split + permute → [B, H, T, d_h]
  └─ Linear → values  [B, T, D] ─┘

scores        = queries @ transpose(keys) / sqrt(d_h)  [B, H, T, T]
masked scores = scores + causal bias                   [B, H, T, T]
probabilities = softmax(masked scores, key axis)       [B, H, T, T]
context       = probabilities @ values                 [B, H, T, d_h]

context ─ merge heads → [B, T, D] ─ Linear → output [B, T, D]
```

Here, `@` means batched matrix multiplication. Splitting heads first reshapes
`[B, T, D]` to `[B, T, H, d_h]`, then permutes the axes to
`[B, H, T, d_h]`. Merging applies the inverse permutation and
reshape. The data is copied in this first implementation so the layout remains
simple and contiguous.

## Queries, keys, and values

Each input token is projected three different ways:

- a query describes what the token is looking for;
- a key describes what each candidate token offers;
- a value is the information copied when that candidate receives weight.

For one head, each query-key dot product produces a compatibility score.
Let $\mathbf{q}_{b,h,i}$ be the query for batch item $b$, head $h$, and
query position $i$, and let $\mathbf{k}_{b,h,j}$ be the key at candidate
position $j$. Their scaled score is:

```math
S_{b,h,i,j}
= \frac{\mathbf{q}_{b,h,i}\cdot\mathbf{k}_{b,h,j}}{\sqrt{d_h}}.
```

Dividing by $\sqrt{d_h}$ keeps scores from growing with the head width and
driving softmax into an overly saturated range.

## The causal mask

For query position $i$ and key position $j$, the additive causal mask is:

```math
M_{i,j} =
\begin{cases}
0,       & j \le i, \\
-\infty, & j > i.
\end{cases}
```

The attention probability assigned by query $i$ to key $j$ is:

```math
P_{b,h,i,j}
= \frac{\exp(S_{b,h,i,j}+M_{i,j})}
       {\sum_{\ell=0}^{T-1}\exp(S_{b,h,i,\ell}+M_{i,\ell})}.
```

Softmax maps the $-\infty$ entries to exactly zero. A row at position $i$
therefore distributes probability only across positions $0$ through $i$,
and every row still sums to one.

If $\mathbf{v}_{b,h,j}$ is the value vector at position $j$, the output
context for one head and query position is the weighted sum:

```math
\mathbf{c}_{b,h,i}
= \sum_{j=0}^{T-1}P_{b,h,i,j}\mathbf{v}_{b,h,j}.
```

The mask is an implicit, non-trainable rule inside the causal-attention
request. Gradients flow through allowed scores, future-score gradients are
exactly zero, and there is no mask value to differentiate.

## Layer boundary

`CausalSelfAttention` owns only:

```text
query projection
key projection
value projection
output projection
```

It deliberately does not own layer normalization or a residual connection.
Those operations belong to `TransformerBlock`, which composes:

```text
x → LayerNorm → CausalSelfAttention → add x
```

Its registered parameters are named `query.weight`, `query.bias`,
`key.weight`, `key.bias`, `value.weight`, `value.bias`, `output.weight`, and
`output.bias`.

## Two exact full-sequence algorithms

`causal_scaled_dot_product_attention` is one Adapter-level operation over
rank-four `Q`, `K`, and `V` tensors. It returns two differentiable values:

```text
probabilities [B, H, T, T]
context       [B, H, T, d_h]
```

This probability-returning function is a diagnostic and teaching surface. It
always uses the materialized algorithm, regardless of the model's selector,
because returning probabilities necessarily creates the
`[B, H, T, T]` result.

`CausalSelfAttention` consumes only context and can select either exact
full-sequence implementation:

- **materialized** is the default. It saves the complete
  `[B, H, T, T]` probability tensor for the context backward pass;
- **Flash** processes eight queries and eight keys per tile on both CPU and
  Metal. It computes context with online softmax and saves only row maxima and
  exponential sums, each `[B, H, T]`. Backward reconstructs each needed
  probability from `Q`, `K`, and the saved row statistics, so no global
  `[B, H, T, T]` probability buffer is allocated in either direction.

For row $i$, the Flash path saves:

```math
m_i = \max_{j \le i} S_{i,j},
\qquad
\ell_i = \sum_{j \le i}\exp(S_{i,j}-m_i).
```

It can then reconstruct an allowed probability while traversing tiles:

```math
P_{i,j} = \frac{\exp(S_{i,j}-m_i)}{\ell_i},
\qquad j \le i.
```

The forward pass updates the maximum, normalizer, and context numerator as
each key tile arrives. The backward pass recomputes scores and probabilities
and applies the same exact softmax vector-Jacobian product as the materialized
path. “Exact” means this is the standard causal-attention equation rather than
an approximate attention model; ordinary float32 ordering differences still
require numerical tolerances.

Both algorithms dispatch vector-Jacobian products rather than rebuilding the
forward equation from scalar graph operations:

- a context upstream produces query, key, and value gradients;
- a probability upstream produces query and key gradients;
- autograd accumulates both contributions if a caller uses both outputs.

The last two cases apply to the materialized probability-returning diagnostic;
the Flash model path has only a context output. CPU uses readable tile-8
reference loops. Metal uses 64-thread groups, tile-8 threadgroup-memory
working sets, online-softmax forward, and recomputing backward kernels. The
implementation is dependency-free. Metal calls remain synchronous over shared
buffers, and no speedup is claimed without measuring a concrete workload.

Metal's dynamic threadgroup working storage is device-limited. In float
elements, the forward kernel requests `32*d_h + 96`; the three backward
phases request `32*d_h + 136`, `40*d_h + 64`, and `48*d_h + 128`,
respectively, in addition to any static pipeline storage. Before launching a
Flash forward, the runtime validates all four pipelines against the device's
threadgroup-memory limit. An unsupported head width therefore fails before a
training graph is returned rather than later during `backward()`. CPU has no
corresponding GPU scratch limit. On Metal, reduce `d_h` by using more heads or
select materialized attention when this validation fails.

## Selecting the full-sequence path

Materialized attention remains the default for compatibility and transparent
probability inspection. C++ callers can select Flash at construction or for
future forwards:

```cpp
std::mt19937 random(7);
transformer_lab::DecoderOnlyTransformer model(
    dimensions,
    random,
    1.0e-5F,
    transformer_lab::FullSequenceAttentionKind::Flash
);

model.set_full_sequence_attention_kind(
    transformer_lab::FullSequenceAttentionKind::Materialized
);
```

C ABI 1.7 adds stable
`TL_FULL_SEQUENCE_ATTENTION_MATERIALIZED` and
`TL_FULL_SEQUENCE_ATTENTION_FLASH` values plus
`tl_model_set_full_sequence_attention` and
`tl_model_full_sequence_attention`.

Python exposes the same runtime policy:

```python
model = DecoderOnlyTransformer(config, attention="flash")
assert model.full_sequence_attention == "flash"
model.set_full_sequence_attention("materialized")
```

`PretrainingConfig` and `PostTrainingConfig` also accept
`attention="materialized"` or `attention="flash"`. This is execution policy,
not model state: changing the algorithm does not change parameter names,
shapes, or persisted artifacts.

Activation checkpointing composes with either choice. A checkpointed block
replays the same forward-time attention kind during backward; a Flash replay
still uses FlashAttention's own probability rematerialization. See
[ACTIVATION_CHECKPOINTING.md](ACTIVATION_CHECKPOINTING.md).

## Algorithm and cache boundaries

Backend attention code is grouped under `src/core/backend/attention/`:

```text
attention/
├── contracts.hpp
├── capability.hpp
├── dispatch.hpp
├── dispatch.cpp
├── reference/
│   ├── materialized_causal.*
│   ├── flash_causal.*
│   └── paged_decode.*
└── metal/
    ├── materialized_causal_kernels.hpp
    ├── flash_causal_kernels.hpp
    └── paged_decode_kernels.hpp
```

The names describe computation:

- **materialized causal attention** is the default full-sequence
  training/autograd algorithm;
- **Flash causal attention** is the memory-linear, tile-8 alternative for the
  context-only full-sequence forward/backward path;
- **paged decode attention** is the one-query inference algorithm that reads a
  logical-to-physical K/V page table.

KV allocation remains separately grouped under
`src/stages/serving/cache/`. Paged storage and FlashAttention are therefore
not two choices in one enum: paging controls where K/V state lives, while
FlashAttention controls how a full sequence is calculated.

A future serving split can use batched Flash-style prompt prefill followed by
paged decode attention. The current prefill is token-at-a-time and already
uses the paged decode operation, so selecting full-sequence Flash does not
alter or accelerate the current serving path.

## Verification

`tests/model/test_causal_self_attention.cpp` checks:

- exact head split/merge ordering and inverse gradients;
- hand-calculated attention probabilities and contexts;
- multi-batch equality with an independent loop-based reference and batch
  isolation;
- exactly zero probability and gradient influence from future positions;
- distinct query, key, value, and output projections checked against an
  independent module calculation;
- centered finite-difference gradients for inputs, $Q$, $K$, $V$, and
  every projection parameter;
- CPU-reference and conditional Metal forward/VJP parity for the materialized
  and Flash backend requests;
- Flash/materialized forward and backward parity, tile-boundary shapes,
  reconstructed probabilities, finite differences, causal zeros, invalid
  rows, large logits, and input immutability;
- full-module and complete-model parity with the runtime selector on CPU;
- conditional end-to-end Metal Flash model forward, loss, and parameter
  gradient parity;
- wide Metal tiles and the invariant that a device resource limit is reported
  before forward returns, never first during backward;
- invalid ranks, incompatible shapes, and invalid head counts.

The implementation remains dependency-free. CPU stays the readable oracle,
while Metal comparisons use numerical tolerances rather than bitwise equality.

The algorithmic reference is the original
[FlashAttention paper](https://arxiv.org/abs/2205.14135); the
[official implementation](https://github.com/Dao-AILab/flash-attention) is a
useful production comparison. Riftco Transformer does not link either project or
use an external attention library.
