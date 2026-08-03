# Glossary

Terms use their meaning inside Riftco Transformer. For the system view, see
[Architecture](ARCHITECTURE.md); for symbols and shapes, see
[Tensor](TENSOR.md) and [Attention](ATTENTION.md).

## A

**Ablation**

An intervention that removes, replaces, or resamples a component and measures
paired metric change. Batch-roll ablation replaces one row's representation
with another row's representation.

**Activation**

An intermediate tensor produced by a layer. It is distinct from an activation
function such as ReLU or GELU.

**Activation checkpointing**

A memory/computation trade: discard a transformer's internal block graph
during forward and replay the block during backward. `block` and `disabled`
are the implemented policies.

**Adam**

A gradient-based optimizer with first and second moment estimates, bias
correction, and framework-level global-norm clipping. It updates parameters;
it is not a transformer layer.

**Adapter**

Context-dependent. A backend adapter implements storage and operation
contracts. A LoRA adapter is a learned low-rank update. A dataset adapter maps
external rows into a framework record schema.

**Autograd**

Reverse-mode automatic differentiation over `Variable` graph nodes. It
computes gradients by applying vector-Jacobian products in reverse topological
order.

## B

**Backend**

The intrinsic storage and dispatched implementation associated with a tensor:
CPU, Metal, CUDA, or TPU. Backend identity is explicit and is never silently
changed to CPU.

**Batch**

A rectangular group of examples processed together. Transformer activations
normally place batch on their first axis.

**BPE (byte-pair encoding)**

A tokenizer that begins with all 256 byte tokens and learns ordered merges for
frequent adjacent token pairs.

## C

**C ABI**

The versioned C-compatible binary interface in `c_api.h`. It uses opaque
handles, fixed-width constants, status codes, and size-prefixed structures.

**Cajal**

The project's finite, deterministic, first-order language for constructing,
evaluating, and compiling inspectable programs into multilinear maps. It is
not a general lambda calculus or a text-based language parser.

**Causal attention**

Self-attention whose mask prevents position `t` from reading future positions
greater than `t`.

**Checkpoint**

Potentially ambiguous. Activation checkpointing replays computation during
backward. A `.riftckpt` `TrainingCheckpoint` preserves model/optimizer state,
RNG state, and built-in batch-source progress for exact continuation. A
`ModelSnapshot` or `.rift` `ModelBundle` is a model handoff, not an
exact-resume training checkpoint.

**Context length**

The number of token positions in one training window or model request. A
next-token window also needs one following target token.

**Cross-entropy**

The scalar classification loss between logits and target token IDs. It creates
the objective whose gradient autograd computes.

## D

**Decode session**

Request-local incremental inference state. Each `step` consumes one token,
appends per-layer K/V state, and returns next-token logits.

**Decoder-only transformer**

A stack of causal self-attention and feed-forward blocks trained to predict
the next token. “Decoder-only” contrasts with encoder-only and encoder-decoder
architectures; it does not mean the model only performs serving decode.

**Double quantization**

QLoRA's additional compression of first-level NF4 block scales into 8-bit
codes plus second-level FP32 scales and an offset.

## E

**Embedding**

A learned row table mapping a discrete token or position ID to a dense feature
vector. A `[batch,time]` token matrix becomes `[batch,time,model_width]`.

**Epoch**

One pass through a finite training dataset. The Python corpus-window training
workflow is step-based sampling with replacement rather than epoch traversal.

## F

**Feature width**

The size of a tensor's last axis used to represent one token or observation.
Model width `D`, head width `D/H`, and feed-forward width are different feature
widths.

**Feed-forward network (FFN)**

The position-wise expand → GELU → project sublayer inside each transformer
block.

**Fine-tuning**

Updating a pretrained model on task or instruction data. Implemented methods
are full fine-tuning, LoRA, and QLoRA.

**Flash attention**

The exact memory-linear full-sequence attention implementation. It avoids
persisting the complete `[B,H,T,T]` probability tensor; it is separate from
paged KV-cache layout.

**F/P/T/I**

Conditional-reversal lab controls: F is a frozen compiled conditional program,
P a frozen unconditional reverse program, T a randomized trainable program
with F's shape, and I omits the program branch. The task-specific C++ prototype
is retired; the current Python lab constructs and executes these variants over
the generic installed `riftco_transformer.programmed` surface.

## G

**Generalization**

Performance on examples not used for optimizer updates. Python evaluators and
labs report train/validation/test losses and gaps; one split or one experiment does not
prove broad generalization.

**Gradient**

The derivative of an objective with respect to a tensor or parameter. It is a
rate of change, not the cross-entropy value itself.

**Gradient clipping**

Scaling all parameter gradients by one shared factor when their global L2 norm
exceeds the configured maximum.

## H

**Head**

One parallel attention subspace. With model width `D` and `H` heads, each head
has width `D/H`; heads run conceptually in parallel and are concatenated.

**Held-out split**

Validation, test, or probe data excluded from optimizer updates. Validation
may guide selection; test should be read only after selection is complete.

## K

**Kernel**

A focused backend implementation of an operation such as matmul, attention,
or Adam update. The public backend folder is application-space runtime code,
not an operating-system kernel.

**KV cache**

Per-layer keys and values retained during incremental generation so earlier
tokens do not need full transformer recomputation on every decode step.

## L

**Layer normalization**

Per-token normalization across the feature axis followed by learned scale and
bias.

**Logit**

An unnormalized score for one vocabulary item. The model returns logits shaped
`[batch,time,vocabulary]`.

**LoRA**

Low-rank adaptation. A frozen linear weight receives a trainable low-rank
delta, reducing the number of optimized parameters.

**Lowering**

The explicit bridge from a compiled multilinear program to a differentiable
neural `Module`. Strategy selection, coefficient precision, initialization,
and trainability are configurable.

## M

**Materialized attention**

The readable attention algorithm that explicitly creates the full score and
probability matrices. It is the default full-sequence implementation.

**Matmul**

Matrix multiplication over the final two axes, with leading batch dimensions
supported by the framework contract. It is the primitive behind linear
layers, but a linear layer may also add bias, LoRA, or packed weights.

**Model width (`D`)**

The feature width of the residual stream. Every transformer block preserves
`[batch,time,D]` around its residual connections.

**Module**

A composition object that registers parameters and child modules. Modules do
not expose a generic `forward()`; concrete types define typed forward methods.

**Multilinear map**

A function linear in each input separately. Cajal compilation represents a
finite program as inspectable coefficients that can be lowered to a neural
module.

## N

**NF4**

NormalFloat 4, the 16-value codebook used for blockwise four-bit base-weight
quantization. Two values share each packed byte.

## P

**Paged Adam**

Adam state split into bounded tensor allocations and update requests. It still
stores two FP32 moments per trainable scalar and is not disk spill or a general
virtual-memory manager.

**Paged KV cache**

A serving layout mapping logical sequence blocks to physical K/V pages. It is
orthogonal to Flash attention.

**Parameter**

A trainable leaf `Variable` with canonical shared state. A `ParameterHandle`
keeps that state alive; a `ParameterList` carries stable qualified names.

**PCA (principal component analysis)**

A deterministic linear analysis that finds directions of greatest centered
variance. PCA is observational; ablation or steering is needed for causal
evidence.

**Perplexity**

`exp(mean cross-entropy)`. Lower is better, but comparisons require the same
tokenization and evaluation protocol.

**Post-training**

Supervised training of an existing model artifact. Architecture and tokenizer
come from the base artifact rather than being redefined by the stage config.

**Pretraining**

Self-supervised next-token training from raw text before task-specific
post-training.

**Probe split**

Held-out observations used to fit an analysis model such as PCA, separate from
validation and final test scoring.

**Program-augmented model**

A fixed-context generic model combining residual ReLU feed-forward and one or
more learned causal-attention branches with an optional lowered multilinear
program. It contains no task-specific F/P/T/I policy.

## Q

**QLoRA**

LoRA training over immutable blockwise-NF4 base linear weights. Only floating-
point adapter parameters receive gradients and Adam state; packed weights must
remain packed during computation to satisfy this contract.

**Quantized weight**

Immutable packed base-weight storage with block scales and optional double-
quantized scales. It is intentionally separate from the ordinary FP32
`Tensor` type.

## R

**Reference backend**

Readable CPU implementations used as correctness oracles for accelerated
backends. “Reference” describes algorithm ownership, not a separate public
backend value.

**Representation trace**

An owning, named collection of captured activations flattened to observation
rows while retaining their original leading shape.

**Residual connection**

An addition from a sublayer's input to its output. It preserves model width and
provides a direct signal/gradient path around the sublayer.

## S

**Serving**

Inference over an immutable model artifact. The current stack offers
token-at-a-time prefill/decode and request-local caches, not continuous
batching or a vLLM-equivalent scheduler.

**Snapshot**

`artifacts::ModelSnapshot`, an in-memory backend-neutral copy of model and
tokenizer state. It has no checksum, lineage, optimizer, RNG, or progress.

**Steering**

An explicit representation intervention, such as scaling or offsetting a
compiled selector coordinate, followed by measurement of changed behavior.

**Stride**

The number of contiguous scalar elements skipped when one index along a tensor
axis advances by one. Contiguous row-major strides are derived from shape.

## T

**Tensor**

An owning multidimensional FP32 array with shape, contiguous row-major strides,
storage backend, and scalar data. Tensor rank is its number of axes.

**Token**

A discrete tokenizer ID. It may represent one byte or a learned BPE byte
sequence; a token is not necessarily a word.

**Tokenizer**

The reversible mapping between byte strings and token IDs. Tokenizer state is
part of every reproducible model artifact.

**Transformer block**

A pre-normalized causal-attention residual sublayer followed by a
pre-normalized feed-forward residual sublayer.

## V

**Variable**

A tensor value paired with an optional reverse-mode autograd graph node.
Copying a Variable shares the node; it does not deep-copy the graph.

**Vector-Jacobian product (VJP)**

The product of an arriving output gradient vector and an operation's Jacobian.
Backward rules compute VJPs without materializing full Jacobian matrices.

## W

**Weight tying**

Sharing one parameter across multiple uses. The current decoder uses separate
token-embedding and language-model-head parameters unless a component's API
explicitly documents sharing.

## Symbols

| Symbol | Meaning |
| --- | --- |
| `B` | Batch size |
| `T` | Sequence/context length |
| `D` | Model/residual feature width |
| `H` | Attention head count |
| `d_h = D/H` | One attention head's feature width |
| `F` | Feed-forward hidden width |
| `V` | Vocabulary size |
| `L` | Conditional-reversal source length in experiment documentation |

Continue with [API reference](API_REFERENCE.md),
[Configuration reference](CONFIGURATION_REFERENCE.md), or
[Troubleshooting](TROUBLESHOOTING.md).
