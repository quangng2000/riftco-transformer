# Learning Roadmap

Every milestone must compile, pass tests, and leave the main branch runnable.

## Working rules

1. Add one concept at a time.
2. Write its test before connecting it to the Transformer.
3. Check every tensor shape at runtime while the project is small.
4. Compare analytical gradients with finite differences.
5. Prefer obvious loops over clever optimizations.
6. Optimize only after the tiny model can overfit a tiny batch.

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
- immutable Python step, loss, gradient-norm, clipping, and validation metrics
- tiny-batch overfitting test

Acceptance check: loss falls substantially on one repeated tiny batch, and a
seeded run reproduces its batch sequence.

## 9. Staged pipeline, generation, and persistence — active

- native in-memory model/tokenizer state capture and restoration through
  `ModelSnapshot` — complete slice
- reusable low-rank linear adapters, transformer target selection, and
  one-way merge into serving-ready base weights — complete slice
- artifact guards that reject active, unmerged adapters — complete slice
- inference-only native `ServingStack` with greedy and seeded
  temperature/top-k generation — complete slice
- detached one-token model decode with transactional per-request KV-cache
  updates; full-sequence training forward remains unchanged — complete slice
- logical page tables over shared per-layer K/V pools, backend-owned
  CPU/Metal/CUDA/TPU paged attention, and a swappable contiguous reference
  factory — complete slice
- native artifact and serving-generation tests — complete slice
- shared stage-neutral Python causal training engine, batch sources, metrics,
  and evaluation policy — complete slice
- self-supervised pretraining that emits an immutable base artifact —
  complete slice
- full-sequence supervised full/LoRA/QLoRA post-training that emits a
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
model parameter and tokenizer state, and serving restores a compatible
inference runtime without importing training or optimizer policy.

Python artifact acceptance check: a reloaded `ModelBundle` produces identical
tokenizer state, parameter values, logits, and seeded token IDs. Persistent
native-artifact acceptance criteria remain future work.

Checkpoint acceptance check: a resumed `TrainingCheckpoint` produces the same
next batch, loss, gradients, and Adam update as an uninterrupted run.

## 10. Cajal-lite program compiler — active

- immutable finite `Unit`, `Sum`, `Product`, and `Dictionary` types — complete
  slice
- programmatically constructed expression AST and finite runtime values —
  complete slice
- Cajal-style linear context checker with additive products and multiplicative
  context splitting — complete slice
- deterministic reference interpreter with unique-key dictionary lookup —
  complete slice
- dense encodings for finite values, with discrete decoding and bounded
  enumeration for `Unit`, `Sum`, and `Product` — complete slice
- backend-neutral dense `MultilinearMap` for arbitrary arity, including
  constant, linear, and higher-arity contractions — complete slice
- recursive `compile(expression, ordered_context)` lowering from checked
  expressions to explicitly ordered multilinear maps — complete slice
- exhaustive compiler-versus-interpreter tests over every finite input to the
  current representative program suite, plus non-discrete multilinearity
  checks — complete slice
- optional, separately linked neural-lowering target with configurable strategy
  registry, analysis, explicit fallback, resource limits, coefficient precision,
  seeded initialization, trainability, and backend placement — complete slice
- differentiable constant, unary-linear, bilinear identity-kernel
  linear-attention, and arbitrary-arity dense modules with inspectable
  coefficient metadata — complete slice
- three-way interpreter/map/module equivalence plus input/coefficient gradient,
  leading-shape, configuration, and package-consumer tests — complete slice
- exact MLP or other factored lowering strategy (the current GELU feed-forward
  is deliberately not presented as an exact arbitrary-map lowering)
- sequence-placement adapter that composes a programmed module with learned
  projections, source/target spans, capture sites, batch-roll ablation, and
  privileged-basis steering — complete slice
- raw programmed-sequence core with explicit shared projection groups,
  biasless projections, logical-input interventions, and pre-merge output —
  complete slice
- separately linked, standard-library-only interpretation analysis with named
  representation matrices, deterministic covariance/Jacobi PCA,
  fit/transform/reconstruct, row-wise interventions, and paired ablation
  statistics — complete slice
- Python-owned conditional-reversal task, deterministic source-disjoint
  10k/5k/1k/1k train/probe/validation/test protocol, branch-stratified metrics,
  exact oracle, and copy/reverse controls — complete slice
- one curated historical F-variant record from the retired task-specific C++
  prototype, clearly separated from current executable evidence — complete
  slice
- public task-neutral program-augmented model composition that Python labs can
  use without importing experiment-specific C++ — complete slice
- additive C ABI 2.5 multilinear-map import, generic programmed-model,
  representation-trace, and contiguous time-range cross-entropy surfaces,
  plus installed `riftco_transformer.programmed` wrappers — complete slice
- Python-owned F/P/T/I construction, training/evaluation, PCA, batch-roll
  ablation, steering, and quick/paper profile orchestration over the generic
  public model — complete executable slice; fresh reviewed run evidence is a
  separate milestone
- full paper-scale multi-seed F/P/T/I runs with archived configurations,
  validation-based selection, test metrics, checkpoints, and comparison to the
  reported analysis
- full Cajal conformance work: nondeterminism and relation-based lookup

Frontend acceptance check: malformed programs are rejected with useful
diagnostics; every accepted test program evaluates to a value whose runtime
type equals its checked type; variables cannot be dropped or duplicated by the
same multiplicative path, while additive product components share one context.

Compiler acceptance check — complete for the current finite compiler suite:
for every exhaustively enumerated input to each accepted fixture, the compiled
multilinear-map coordinates match the encoded reference-interpreter result.
Non-dictionary outputs also decode to the same source value. Dictionary
coefficients remain explicit key/value outer products.

Program-integration acceptance check: strategy analysis is inspectable before
allocation; exact mode rejects lossy FP32 materialization; automatic lowering
selects bilinear linear attention; and frozen coefficients stay out of Adam.
Python can now compose the generic programmed model, train surrounding
parameters with target-time-range loss, fit PCA on probe/train captures, and
apply held-out ablation/steering without a task-specific installed C++ target.
The remaining evidence acceptance check is to archive fresh quick smoke output
and then full multi-seed paper-profile configurations/results without using
test outcomes for selection.

## 11. Optional performance work

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
- C ABI tokenizer/model/variable/Adam ownership and end-to-end Python text
  training — complete slice
- selectable byte/BPE tokenizers and end-to-end Python BPE training
  — complete slice
- exact tokenizer reconstruction and deterministic named-parameter
  state transfer for `ModelBundle` — complete slice
- LoRA configuration, attachment, adapter-parameter access, and
  merge lifecycle — complete slice
- immutable blockwise NF4 weights, optional FP32 or default double-quantized
  scales, backend-neutral quantized-linear forward and input backward, CPU
  reference, Metal/CUDA kernels, TPU StableHLO execution, adapter-only QLoRA,
  packed-memory diagnostics, and explicit FP32 export — complete slice;
  packed artifacts remain future work, and real CUDA/TPU hardware acceptance
  remains pending
- contiguous or bounded-page Adam moment storage — complete slice; paged
  storage caps each moment allocation and update request, but retains two FP32
  moment values per trainable scalar and is not a general spill/page-fault
  manager
- contiguous/paged incremental decode-session lifecycle — complete
  slice
- dependency-free exact memory-linear Flash causal-attention forward/backward,
  tile-8 on CPU/Metal and block-parallel on CUDA, with `[B,H,T]` row statistics
  instead of saved `[B,H,T,T]` probabilities — complete slice
- C ABI and Python full-sequence attention selection —
  complete slice
- exception-atomic nested VJPs plus transformer-block activation
  checkpointing across the registered backends, including LoRA dependencies,
  C ABI and Python selection — complete slice
- additive ABI 2.1 CUDA identity, unavailable stub, optional CUDA Toolkit 12+
  source build, managed tensor/packed-weight storage, synchronous GPU matmul,
  quantized linear, and all attention contracts — complete slice; real-GPU
  parity remains required
- additive ABI 2.2 TPU identity, unavailable stub, optional Linux x86-64 PJRT
  source adapter, host-mirrored tensor and packed-weight storage, and
  one-device StableHLO quantized linear, matmul, materialized attention/VJPs,
  and paged decode — implemented;
  real-Cloud-TPU acceptance remains pending
- additive ABI 2.5 dense/sparse multilinear-map import, generic
  `ProgramAugmentedModel` lifecycle/forward/parameters, owning representation
  traces, and contiguous time-range cross entropy — complete slice
- TPU device-resident storage, genuinely tiled StableHLO Flash attention,
  additional native operations, asynchronous execution, SPMD partitioning,
  and multi-host support
- CUDA kernels for layout, elementwise, reductions, indexing, normalization,
  loss, and Adam's transactional candidate-state update — complete slice;
  a device-native global gradient-norm reduction remains future work
- optimized CPU matrix multiplication
- parallel CPU loops
- SIMD
- Apple Accelerate comparison
- private-memory and asynchronous Metal scheduling
- attention algorithm contracts separated from serving KV-cache layout —
  complete
- one-token paged KV caching on CPU, Metal, CUDA, and TPU — complete slice
- batched Flash-style serving prefill (current prefill remains
  token-at-a-time)
- continuous batching and request scheduling
- immutable prefix detection, page sharing, and copy-on-write
