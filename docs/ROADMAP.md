# Learning Roadmap

Every milestone must compile, pass tests, and leave the main branch runnable.

## 0. Scaffold — complete

- CMake presets and strict compiler warnings
- validated model/training configuration
- tiny corpus and smoke test

Acceptance check: configure, build, and run CTest from a clean build directory.

## 1. Tensor storage — complete

- contiguous `float` storage
- dimensions, strides, checked indexing, reshape
- reusable raw tensor operations separated from storage
- elementwise arithmetic, explicit broadcasting, reductions, and matrix
  multiplication

Acceptance check: compare small hand-calculated matrices and reject invalid
shapes.

## 2. Reverse-mode automatic differentiation — complete

- operation graph
- gradient accumulation
- backward functions for arithmetic, matrix multiplication, reductions,
  exponentials, logarithms, and square roots

Acceptance check: compare every analytical gradient with centered finite
differences.

## 3. Extensible tokenizers and batches — complete

- deterministic corpus-derived byte vocabulary
- Strategy/factory selection between byte and byte-pair encoding
- BPE with all 256 base bytes, deterministic learned merges, configurable
  vocabulary limit and minimum pair frequency
- encode/decode round trips
- checked integer next-token training windows

Acceptance check: both methods round-trip their inputs, BPE compresses repeated
pairs and encodes unseen bytes, and targets are inputs shifted by one token.

## 4. Neural-network primitives — complete

- parameter registration
- embedding lookup
- linear layer
- layer normalization
- GELU
- softmax and cross-entropy
- feed-forward composition

Acceptance check: shape tests, finite-difference gradient tests, and stable
softmax for large logits.

## 5. Causal self-attention — complete

- query, key, and value projections
- split and merge attention heads
- general axis permutation and strict batched matrix multiplication
- scaled dot products
- causal mask
- output projection

Acceptance check: future tokens receive exactly zero attention probability,
rows sum to one, and gradients pass finite-difference checks.

## 6. Decoder-only transformer — complete

- token and positional embeddings
- pre-normalization transformer blocks
- attention and feed-forward residual paths
- final normalization and language-model head

Acceptance check: logits have shape `[batch, time, vocabulary]`, parameter
counts are reported, a forward pass is deterministic for a fixed seed, future
tokens cannot alter earlier logits, and loss gradients reach every component.

## 7. Adam — complete

Implement, without a library:

```math
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1-\beta_1)g_t, \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2)g_t^2, \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t}, \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t}, \\
\theta_t &= \theta_{t-1}
  - \alpha\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}.
\end{aligned}
```

Here, $g_t$ is the current gradient, $\theta_t$ is the parameter value, and
$\alpha$ is the learning rate. Add global gradient-norm clipping before the
update:

```math
\begin{aligned}
G &= \sqrt{\sum_i \lVert g_i\rVert_2^2}, \\
s &=
\begin{cases}
\dfrac{C}{G}, & G>C, \\
1, & G\le C,
\end{cases} \\
g_i^{\mathrm{clipped}} &= s g_i,
\end{aligned}
```

where $C$ is the configured maximum global norm. The second branch includes
$G=0$, so clipping never divides by zero.

Acceptance check: match hand-calculated Adam updates for several steps and
verify that zero gradients with zero optimizer state leave parameters
unchanged. After momentum exists, later zero gradients continue to follow the
standard Adam equations.

## 8. Training loop — complete

- deterministic seeded mini-batches sampled with replacement
- a sampling random engine independent of model initialization
- forward, loss, backward, clip, update
- a fresh computation graph for every step
- CSV step, loss, gradient-norm, and clipping metrics
- tiny-batch overfitting test

Acceptance check: loss falls substantially on one repeated tiny batch, and a
seeded run reproduces its batch sequence.

## 9. Staged pipeline, generation, and persistence — active

- native in-memory model/tokenizer state capture and restoration through
  `ModelSnapshot` — complete slice
- native stage-neutral batch-source, optimizer-strategy, Adam-adapter, and
  causal-language-model trainer contracts — complete slice
- native `PretrainingStack`, with the existing CLI migrated to that
  composition root — complete slice
- native full-sequence supervised `PostTrainingStack` with full-parameter or
  adapter-only Adam selection and a merged, detached result snapshot —
  complete slice
- reusable low-rank linear adapters, transformer target selection, and
  one-way merge into serving-ready base weights — complete slice
- artifact guards that reject active, unmerged adapters — complete slice
- inference-only native `ServingStack` with greedy and seeded
  temperature/top-k generation — complete slice
- detached one-token model decode with transactional per-request KV-cache
  updates; full-sequence training forward remains unchanged — complete slice
- logical page tables over shared per-layer K/V pools, direct CPU/Metal paged
  attention, and a swappable contiguous reference factory — complete slice
- native artifact, shared-training, stage-contract, stage-stack, and
  serving-generation tests — complete slice
- shared stage-neutral Python causal training engine — complete slice
- self-supervised pretraining that emits an immutable base artifact —
  complete slice
- full-sequence supervised full/LoRA post-training that emits a
  lineage-linked child artifact with merged base weights — complete slice
- greedy and seeded temperature/top-k autoregressive generation — complete
  slice
- stable C/Python decode sessions and `TextGenerator` integration, using paged
  caching by default — complete slice
- versioned `ModelBundle` save/load for configuration, named float32 weights,
  and exact byte/BPE tokenizer state — complete slice
- dependency-free in-process and local HTTP serving adapters — complete slice
- a persistent native artifact contract with integrity and identity
- response-only post-training loss masks
- resumable `TrainingCheckpoint` state, including Adam moments/step and
  data/random-generator progress
- reproducible saved sample artifacts

Native handoff acceptance check: a captured `ModelSnapshot` restores exact
model parameter and tokenizer state; post-training leaves its input snapshot
unchanged; serving restores a compatible inference runtime without importing
training or optimizer policy.

Python artifact acceptance check: a reloaded `ModelBundle` produces identical
tokenizer state, parameter values, logits, and seeded token IDs. Persistent
native-artifact acceptance criteria remain future work.

Checkpoint acceptance check: a resumed `TrainingCheckpoint` produces the same
next batch, loss, gradients, and Adam update as an uninterrupted run.

## 10. Optional performance work

Only after correctness:

- backend-neutral matmul dispatch and first Metal kernel — complete slice
- stable tensor C ABI and low-level Python client — complete slice
- persistent backend-owned CPU/Metal tensor storage — complete slice
- backend-preserving autograd and module transfer — complete slice
- transactional fused Metal Adam in one batch submission, with unsafe-value
  and cancellation-sensitive reference retry — complete slice
- Adapter-routed CPU reference operations — complete slice
- Metal layout, elementwise, reduction, GELU, LayerNorm, softmax/causal-mask,
  gather/scatter, and fused cross-entropy forward/backward kernels — complete
  slice
- materialized causal-attention forward and vector-Jacobian-product Metal
  kernels — complete slice
- C ABI 1.2 tokenizer/model/variable/Adam ownership and end-to-end Python text
  training — complete slice
- C ABI 1.3 selectable byte/BPE tokenizers and end-to-end Python BPE training
  — complete slice
- C ABI 1.4 exact tokenizer reconstruction and deterministic named-parameter
  state transfer for `ModelBundle` — complete slice
- C ABI 1.5 LoRA configuration, attachment, adapter-parameter access, and
  merge lifecycle — complete slice
- C ABI 1.6 contiguous/paged incremental decode-session lifecycle — complete
  slice
- dependency-free exact tile-8 Flash causal-attention forward/backward on CPU
  and Metal, with `[B,H,T]` row statistics instead of saved `[B,H,T,T]`
  probabilities — complete slice
- C ABI 1.7 and Python/native-stage full-sequence attention selection —
  complete slice
- exception-atomic nested VJPs plus transformer-block activation
  checkpointing on CPU/Metal, including LoRA dependencies, C ABI 1.8, Python,
  stage, and CLI selection — complete slice
- optimized CPU matrix multiplication
- parallel CPU loops
- SIMD
- Apple Accelerate comparison
- private-memory and asynchronous Metal scheduling
- attention algorithm contracts separated from serving KV-cache layout —
  complete
- one-token paged KV caching on CPU and Metal — complete slice
- batched Flash-style serving prefill (current prefill remains
  token-at-a-time)
- continuous batching and request scheduling
- immutable prefix detection, page sharing, and copy-on-write
