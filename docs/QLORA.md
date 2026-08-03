# QLoRA Fine-Tuning

Riftco Transformer's QLoRA path keeps eligible frozen `Linear` weights in
blockwise 4-bit NormalFloat (`NF4`) storage for the complete training loop.
Only the ordinary FP32 LoRA factors are passed to Adam. The design follows the
central memory invariant from the [QLoRA paper](https://arxiv.org/abs/2305.14314):
gradients flow through a frozen quantized base model into trainable adapters,
without first expanding the base model into a persistent FP32 copy.

## NF4 representation

For a contiguous block of $B$ weights, the encoder first calculates one
absolute maximum

\[
s_b = \max_{i \in b} |w_i|
\]

and replaces each normalized value $w_i / s_b$ with the nearest entry in the
fixed 16-value NF4 codebook. The constants match the canonical mapping in the
[official bitsandbytes implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py).
Two four-bit code indices share one byte. The
first logical value occupies the low nibble and the next occupies the high
nibble. An all-zero block has scale zero and uses the exact-zero NF4 code.

Legacy scale storage keeps each $s_b$ as FP32. With $M=\lceil N/B\rceil$
weight blocks, that representation occupies

\[
\left\lceil\frac{N}{2}\right\rceil
+ 4\left\lceil\frac{N}{B}\right\rceil
\]

payload bytes. With the default $B=64$, this is about 4.5 bits per weight
before metadata. The public memory diagnostic reports packed-code bytes, scale
bytes, persistent payload bytes, and the equivalent FP32 byte count separately.

QLoRA post-training enables double-quantized scales by default. For a
scale-group width $C$, it stores the shared FP32 mean

\[
\mu = \frac{1}{M}\sum_{b=0}^{M-1}s_b,
\]

one FP32 second-level absolute maximum $t_g$ per group, and one centered
unsigned-byte code per first-level scale:

\[
t_g = \max_{b\in g}|s_b-\mu|,
\qquad
q_b = 128 + \mathrm{clamp}\!\left(
\mathrm{round}\!\left(127\frac{s_b-\mu}{t_g}\right),
-127, 127
\right).
\]

The zero-range case uses code 128. Decoding reconstructs

\[
\widehat{s}_b = \mathrm{clamp}\!\left(
\mu + t_g\frac{q_b-128}{127},
0,
\mathrm{FLT\_MAX}
\right).
\]

With $S=\lceil M/C\rceil$, the double-quantized payload occupies

\[
\left\lceil\frac{N}{2}\right\rceil + M + 4S + 4
\]

bytes: packed NF4 codes, first-level scale codes, FP32 second-level scales,
and one FP32 offset. With $B=64$ and $C=256$, this approaches 4.13 bits per
weight before object metadata. No resolved FP32 first-level scale vector is
retained.

`QuantizedWeight` is intentionally separate from `Tensor`. It is immutable,
shared cheaply by autograd closures, and can never become a `Parameter` or an
Adam input. Calling `dequantize()` is an explicit diagnostic/export operation;
the training kernels do not call it.

## Forward and backward

For input $X\in\mathbb{R}^{R\times I}$ and packed base weight
$W_q\in\mathbb{R}^{O\times I}$, the fused operation computes

\[
Y_{r,o}=\sum_{i=0}^{I-1} X_{r,i}\,\mathrm{NF4}(W_{q,o,i}).
\]

The autograd node has exactly one parent: `input`. Its vector-Jacobian product
computes

\[
\frac{\partial L}{\partial X_{r,i}}
=\sum_{o=0}^{O-1}
\frac{\partial L}{\partial Y_{r,o}}
\mathrm{NF4}(W_{q,o,i}).
\]

There is no weight `Variable`, weight gradient, or first/second Adam moment.
CPU, Metal, and CUDA decode each nibble inside these sums; they never allocate
a full-size floating-point weight matrix. TPU keeps packed host storage,
uploads packed U8 codes and scale metadata, and performs scale reconstruction,
dequantization, and the dot product inside one compiled StableHLO computation.
That computation may transiently realize decoded values during execution, but
the model never retains a persistent FP32 base matrix.

## Training lifecycle

Direct C++ model setup is explicit:

```cpp
model.quantize_linear_weights_nf4_double_quantized(64, 256);
const auto packed_memory = model.quantized_memory_usage();
model.attach_lora(lora_config);

AdamOptions adam_options;
adam_options.state_storage = AdamStateStorageKind::Paged;
adam_options.page_size = 4096;
Adam optimizer(model.lora_parameters(), adam_options);
// Build a fresh forward/loss graph, call backward(), then optimizer.step().

model.merge_lora();  // Explicitly materializes the serving/export weights.
```

The direct C++ sequence above documents reusable model and optimizer
primitives, not a native high-level training stage. Python owns the configured
post-training lifecycle and accepts `fine_tuning_method="qlora"`:

```python
from riftco_transformer import LoraConfig
from riftco_transformer.post_training import PostTrainingConfig

config = PostTrainingConfig(
    fine_tuning_method="qlora",
    nf4_block_size=64,
    double_quantization=True,
    nf4_scale_block_size=256,
    optimizer_state="auto",
    optimizer_page_size=4096,
    lora=LoraConfig(rank=8, alpha=16.0),
)
```

The runtime performs these steps:

1. Restore the ordinary FP32 base artifact on the selected backend.
2. Pack every eligible transformer `Linear` weight as NF4 and release its FP32
   parameter and gradient storage.
3. Attach unchanged FP32 `LowRankAdapter` factors to the selected projections.
4. Train only `model.adapter_parameters()` with Adam. Python's automatic
   optimizer policy selects bounded-page state for QLoRA.
5. At export, explicitly dequantize all packed weights, add trained LoRA deltas
   where present, and restore the ordinary FP32 parameter schema.
6. Capture the existing serving-ready `.rift` artifact.

Step 5 intentionally has an export-time FP32 memory spike. It happens after
the optimizer and training graph are released and does not weaken the packed
training-memory guarantee. Exact-resume `.riftckpt` v2 archives can instead
retain packed state during training; packed serving/distribution `.rift`
artifacts remain a separate milestone.

## Exact-resume checkpoints

`TrainingCheckpoint.capture()` preserves every NF4 nibble and all legacy or
double-quantized scale metadata alongside FP32 adapters, Adam moments, random
state, and batch-source position. Restore requires a freshly constructed,
fully quantized model with the same LoRA configuration and adapter-only Adam.
It transactionally replaces the target's immutable bases from canonical bytes
without allocating a full FP32 base matrix. Tests compare packed payload bytes,
resident memory accounting, and the resumed optimization trajectory against an
uninterrupted QLoRA run.

The quantization targets and LoRA targets are independent: all
$6\times\text{block_count}+1$ eligible linear weights are packed, while the
adapter target mask may stay at its query/value default.

## Backend status

| Backend | QLoRA linear path |
| --- | --- |
| CPU | Readable packed reference forward and input backward |
| Apple Metal | Persistent packed buffers with inline-decode Metal kernels |
| NVIDIA CUDA | Persistent managed packed allocations with inline-decode CUDA kernels |
| Cloud TPU | Persistent packed host payload with U8 PJRT inputs and in-program StableHLO dequantization |

Python `backend="auto"` uses the normal TPU → CUDA → Metal → CPU priority for
QLoRA as well as the other training methods. Explicit CUDA and TPU requests
still require their opt-in source builds and available runtimes/devices.

A backend is not advertised as supporting QLoRA merely because it can load an
NF4 checkpoint. The backend must retain packed weights and consume them without
a persistent FP32 expansion.

CUDA and TPU packed paths are source-integrated, and their conditional parity
tests run when those backends are available. The TPU sources passed the local
strict C++ syntax check, including StableHLO generation. This macOS host has
neither a CUDA compiler/NVIDIA GPU nor a Cloud TPU runtime/device, so CUDA
compilation and both backends' hardware acceptance runs remain pending. Source
completion is not a production-performance claim.

## Current scope

This milestone implements base NF4, optional legacy FP32 scales, default
double-quantized scales, adapter-only Adam, bounded-page Adam state, packed
linear execution across all four backends, and explicit FP32 export.
Embeddings, normalization parameters, and biases stay FP32; the large eligible
linear matrices are the packed base-weight scope. Packed serving artifacts
remain a separate format milestone; packed training continuation is covered by
`.riftckpt` v2.

Paged Adam divides each trainable parameter's two FP32 moment vectors into
pages of at most `page_size` elements and submits one page update at a time.
It does not reduce the total moment payload. CUDA pages are managed-memory
allocations, but the implementation is not a general OS spill/page-fault
manager: it has no explicit eviction, host/disk budget, prefetch policy, or
out-of-core scheduler.

See [ADAM.md](ADAM.md) for page diagnostics, complete candidate allocation,
and whole-step transactionality.

Tests assert both scale encodings' exact packed byte counts, forward and
input-gradient parity, immutable packed payloads across Adam steps, paged-state
accounting, exclusion from parameter lists, and restoration of the ordinary
FP32 schema at export.
