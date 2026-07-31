# Decoder-Only Transformer

The completed model turns token IDs into next-token logits using only the
tokens at the current position and to its left. It is **decoder-only** because
there is no encoder input or cross-attention: every block contains causal
self-attention followed by a feed-forward layer.

The model returns logits. Choosing next-token targets, computing loss, and
updating parameters remain separate responsibilities.

## Dimensions and shape flow

Let $B$ be the batch size, $T$ the sequence length, $V$ the vocabulary
size, $C$ the maximum context length, $D$ the model width, $H$ the
number of attention heads, $F$ the feed-forward width, and $N$ the number
of transformer blocks.

The complete forward pass is:

```text
token IDs                                      [B, T]
  ├─ token embedding table                     [V, D]
  └─ learned position embedding table          [C, D]
                 ↓ lookup and add
hidden state                                   [B, T, D]
                 ↓ TransformerBlock × N
contextual hidden state                        [B, T, D]
                 ↓ final LayerNorm
normalized hidden state                        [B, T, D]
                 ↓ Linear(D, V)
logits                                         [B, T, V]
```

The language-model head uses weight `[V, D]` and bias `[V]`. In mathematical
terms, it maps each hidden vector $\mathbf{h}_{b,t} \in \mathbb{R}^{D}$ to
a logits vector $\mathbf{z}_{b,t} \in \mathbb{R}^{V}$. Token embedding and
head weights are separate parameters; this first implementation does not tie
them.

## Learned token and position embeddings

Each token ID selects one row from the token table. Position IDs are generated
inside the model as $0, 1, \ldots, T-1$.

They restart at zero for every batch row and select rows from the learned
position table. The two lookup results have the same `[B, T, D]` shape and are
added before the first block. If $E_{\text{token}}$ and
$E_{\text{position}}$ are the two embedding tables, the initial hidden
vector at batch item $b$ and time $t$ is:

```math
\mathbf{h}^{(0)}_{b,t}
= E_{\text{token}}[\operatorname{id}_{b,t}]
+ E_{\text{position}}[t].
```

Because the position table contains $C$ rows, a forward pass requires
$T \le C$.

## Pre-normalized residual blocks

For block input $\mathbf{x}$, the implementation applies exactly two
residual equations:

```math
\mathbf{a}
= \mathbf{x}
+ \operatorname{CausalSelfAttention}
  \left(\operatorname{AttentionNorm}(\mathbf{x})\right),
```

```math
\mathbf{y}
= \mathbf{a}
+ \operatorname{FeedForward}
  \left(\operatorname{FeedForwardNorm}(\mathbf{a})\right).
```

This is a **pre-normalized** block because each sublayer receives normalized
features while the residual path carries the unnormalized state around it.
Both equations preserve `[B, T, D]`, so blocks can be stacked without changing
shape.

Attention owns its query, key, value, and output projections. Feed-forward owns
its $D \to F \to D$ projections. The block owns the two normalizations and the two
residual additions that compose those reusable components.

After all $N$ blocks, one final layer normalization is applied before the
language-model head.

## Why the whole model remains causal

Causal self-attention assigns negative infinity to every score whose key
position is later than its query position. Softmax therefore gives future
positions exactly zero probability.

The remaining operations—embedding lookup, layer normalization, feed-forward
projections, residual addition, and the language-model head—operate
independently at each time position. Consequently, changing tokens after
position $t$ cannot change logits at or before $t$.

See [ATTENTION.md](ATTENTION.md) for the attention equations and mask.

## Parameters

Named parameters follow the model hierarchy:

```text
token_embedding.weight
position_embedding.weight

blocks.0.attention_norm.scale
blocks.0.attention_norm.bias
blocks.0.attention.query.weight
blocks.0.attention.query.bias
blocks.0.attention.key.weight
blocks.0.attention.key.bias
blocks.0.attention.value.weight
blocks.0.attention.value.bias
blocks.0.attention.output.weight
blocks.0.attention.output.bias
blocks.0.feed_forward_norm.scale
blocks.0.feed_forward_norm.bias
blocks.0.feed_forward.expand.weight
blocks.0.feed_forward.expand.bias
blocks.0.feed_forward.project.weight
blocks.0.feed_forward.project.bias

blocks.1...
final_norm.scale
final_norm.bias
language_model_head.weight
language_model_head.bias
```

One block contains the following number of scalar parameters:

```math
4D^2 + 2DF + 9D + F.
```

The complete untied model therefore contains:

```math
2VD + CD + N\left(4D^2 + 2DF + 9D + F\right) + 2D + V.
```

For the tiny corpus and configuration
$V=27$, $C=16$, $D=32$, $F=64$, and $N=2$, that is $19{,}419$
trainable scalars.

`parameters()` returns stable names paired with owning `ParameterHandle`
entries for Adam. Each entry keeps `.parameter` as a raw compatibility view of
the same canonical state. The complete model is intentionally non-copyable and
non-movable because its registered tree contains stable direct-child
addresses; its repeated block list additionally shared-owns each block.
Parameter-list copies independently retain parameter state.

`Module::to(ExecutionBackend)` transfers every registered base parameter plus
any attached adapter storage discovered through each child's dynamic transfer
hook, while preserving canonical identities. The transfer prepares all
changed values and zero gradients before committing any of them. Call it
before constructing Adam:

```cpp
model.to(ExecutionBackend::Metal);
Adam optimizer(model.parameters(), options);
```

Forward values, autograd gradients, causal masks, and loss scalars then preserve
the model backend. CPU and conditional Metal tests compare full-model forward
and backward results after a transfer while leaving the thread's construction
default on CPU. On Metal, embedding gather/scatter, layout transforms,
elementwise/residual math, reductions, GELU, LayerNorm, causal softmax,
materialized or Flash attention/VJPs, cross-entropy, and matmul all route to
backend kernels. Autograd traversal itself remains host control flow.

`FullSequenceAttentionKind::Materialized` is the constructor default.
`FullSequenceAttentionKind::Flash` selects the exact tile-8 context-only
forward/backward path, which saves `[B,H,T]` row maxima and exponential sums
instead of `[B,H,T,T]` probabilities. The explicit probability-returning
diagnostic remains materialized. Storage is shared and host-visible, and every
backend operation completes before returning; selecting Flash is not a claim
of measured speedup. See [ATTENTION.md](ATTENTION.md) for selection examples
and the separation from paged serving decode.

`ActivationCheckpointingKind::Disabled` is the compatibility default.
`ActivationCheckpointingKind::TransformerBlock` retains each block boundary
and reconstructs the block internals during backward. The checkpoint node
registers all base and active-LoRA block parameters as explicit leaf
dependencies, while replay preserves the forward-time attention choice.
Changing checkpointing affects only future full-sequence forwards and never
`decode_token()`. See
[ACTIVATION_CHECKPOINTING.md](ACTIVATION_CHECKPOINTING.md).

## Validation

Construction rejects:

- zero vocabulary, context, model width, head count, block count, or
  feed-forward width;
- a model width that is not divisible by the head count;
- vocabulary or context indices that cannot fit in `TokenId`;
- a non-finite or non-positive layer-normalization $\varepsilon$.

A forward pass rejects:

- a token shape other than `[batch, time]`;
- zero batch or time dimensions;
- a token count different from $\text{batch}\times\text{time}$;
- a sequence longer than the maximum context;
- a token ID outside the vocabulary.

Each block also checks that its input is exactly `[batch, time, model_width]`.

## Loss and optimizer boundary

`DecoderOnlyTransformer::forward` stops at logits:

```text
token IDs → model → logits
```

The caller supplies shifted next-token targets and computes:

```text
logits → cross-entropy → backward → parameter gradients
```

Cross-entropy is not a model layer, and Adam is not part of the transformer.
The caller-owned optimizer consumes the model's named parameter list after
`backward()` and updates those leaf values. See [ADAM.md](ADAM.md).

## Source and tests

```text
include/transformer_lab/model/transformer_block.hpp
include/transformer_lab/model/activation_checkpointing.hpp
src/model/transformer_block.cpp
tests/model/test_transformer_block.cpp

include/transformer_lab/model/decoder_only_transformer.hpp
src/model/decoder_only_transformer.cpp
tests/model/test_decoder_only_transformer.cpp
```

The block tests cover residual composition, shapes, parameter registration, and
gradients. The full-model tests cover learned positions, deterministic seeded
initialization, parameter counts, context validation, end-to-end gradients,
and the guarantee that future tokens cannot influence earlier logits.
