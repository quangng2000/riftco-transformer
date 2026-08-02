# Training Loop

The Python training loop connects the tokenizer, native model, autograd engine,
and Adam optimizer. Python owns batch policy, iteration, evaluation, and metric
delivery; C++ executes tensor operations, forward, loss, backward, and Adam.
One iteration is:

```text
sample window starts
        ↓
build input and next-token target batch
        ↓
fresh model forward pass
        ↓
cross-entropy loss
        ↓
backward pass
        ↓
global gradient clipping and Adam update
        ↓
publish one immutable metric record
```

The loop coordinates these components; it does not move their responsibilities.
The model calculates logits, cross-entropy creates the scalar objective,
autograd calculates parameter gradients, and Adam consumes those gradients to
replace the parameter values.

## Seeded sampling with replacement

For a corpus with $N$ tokens and context length $T$, a training window needs
$T$ input tokens plus the following target token. Its valid starting offsets
are therefore:

```math
0, 1, \ldots, N-T-1
```

There are $N-T$ valid starts. For every row of every batch, the training loop
draws one valid offset uniformly and independently. Sampling is **with
replacement**, so:

- two rows in one batch may use the same window;
- a window may reappear in later steps;
- the batch size may exceed the number of distinct windows.

Duplicates are expected samples, not an error. Sampling with replacement keeps
each draw simple and gives every valid window the same chance on every row.

The Python sampler owns a dedicated `random.Random` seeded by the workflow
config, while native model initialization receives its own seed. Their state is independent, so
consuming more random numbers while constructing the model does not silently
change the training batches. With the same corpus, configuration, command-line
overrides, and implementation, a seeded run reproduces its batch sequence.

The corpus must contain more than `context_size` tokens. A corpus containing
exactly `context_size + 1` tokens has one valid window, which can be sampled
repeatedly for every batch row.

## A fresh graph on every step

In simplified Python, the loop is:

```python
for _ in range(config.steps):
    batch = source.next_batch()
    with model(batch.inputs) as logits:
        with cross_entropy(logits, batch.targets) as loss:
            loss_value = loss.item()
            loss.backward()
            stats = optimizer.step()
    publish_metric(stats.step, loss_value, stats)
```

Every call to `model.forward` creates new operation nodes that remember the
values needed by their backward rules. `optimizer.step()` then replaces the
trainable leaf values. The old graph describes the old forward pass and must
not be reused after that update. The next iteration therefore performs another
forward pass and builds a fresh graph from the updated parameters.

The loss is copied to `loss_value` before the optimizer step because it
describes the prediction made with the pre-update parameters. Adam reports the
gradient norm and clip scale used for that same update.

## Metric records

`CausalLanguageModelTrainer.run()` returns immutable `TrainingMetric` values
and can send each one to a caller-supplied metric sink. Examples print them;
labs may serialize them into run reports. The framework does not prescribe a
CSV file or overwrite policy.

| Column | Meaning |
| --- | --- |
| `step` | one-based successful Adam step |
| `loss` | scalar cross-entropy from the forward pass before that update |
| `gradient_norm` | global parameter-gradient norm before clipping |
| `clip_scale` | shared multiplier applied to all gradients |

`clip_scale` is `1` when the norm is already within the configured limit. When
clipping is required it is `gradient_clip / gradient_norm`, a value below `1`.
Together, these fields show whether learning is progressing and whether
updates are regularly being clipped.

## Running a short example

From the project directory:

```bash
PYTHONPATH=python python3 examples/python/train_tiny.py \
  --steps 5 --backend cpu
```

The example exposes the Python workflow controls directly:

```bash
PYTHONPATH=python python3 examples/python/train_tiny.py \
  --steps 20 \
  --backend cpu \
  --attention flash \
  --activation-checkpointing block
```

- `--steps N` selects a positive optimizer-step count.
- `--backend cpu|metal|cuda|tpu` selects model, activation, gradient, and optimizer
  storage. Metal routes the training graph's layout, elementwise, reduction,
  GELU, LayerNorm, softmax/causal-mask, embedding gather/scatter,
  cross-entropy, materialized or Flash attention/VJP, matmul, and fused Adam
  work to compute kernels on supported Apple systems. Its QLoRA path also keeps
  packed NF4 codes/scales in Metal buffers and decodes them inside forward/input-
  backward kernels. The overflow-safe global gradient norm and an unsafe Adam
  retry use the CPU reference over the same host-visible shared buffers.
- CUDA requires a source build configured with
  `-DRIFTCO_TRANSFORMER_ENABLE_CUDA=ON`, CUDA Toolkit 12+, a compatible NVIDIA
  driver, and an NVIDIA GPU. CUDA storage uses managed allocations; matmul,
  packed NF4 linear forward/input backward, materialized/Flash attention and
  their gradients, and paged decode run as CUDA kernels. Layout, elementwise,
  reduction, indexing, normalization, loss, and Adam's out-of-place candidate-
  state update also use native CUDA kernels.
  Adam's overflow-safe global gradient norm and autograd graph traversal remain
  host control flow over managed storage. The backend supports complete
  pretraining, Full fine-tuning, LoRA, QLoRA, and evaluation, but does not imply
  a fully device-resident execution graph or an end-to-end speedup. The packed
  source path is implemented; actual NVIDIA-hardware validation was not
  available on this macOS host. Standard wheels contain the recognized
  unavailable CUDA stub.
- TPU requires a Linux x86-64 source build configured with
  `-DRIFTCO_TRANSFORMER_ENABLE_TPU=ON`, a compatible `libtpu.so`, and an
  addressable Google Cloud TPU device. It compiles and executes packed NF4
  linear forward/input backward, batched matmul, materialized attention and its
  gradients, and paged decode through PJRT/StableHLO; Adam, Flash attention,
  loss, and other operations use synchronous reference paths over host-mirrored
  storage. Complete pretraining, Full fine-tuning, LoRA, QLoRA, and evaluation
  are wired, but real hardware validation is pending and this phase is not an
  end-to-end TPU speedup claim. Standard wheels contain the recognized
  unavailable TPU stub.
- `--attention materialized|flash` selects the full-sequence attention
  implementation. Materialized remains the default. Flash uses an exact
  memory-linear algorithm and saves `[B,H,T]` row statistics rather than
  `[B,H,T,T]` probabilities for backward. CPU and Metal use tile-8 paths;
  CUDA uses cooperating thread blocks. TPU uses StableHLO for materialized
  attention and the CPU reference implementation for Flash.
- `--activation-checkpointing disabled|block` retains the ordinary graph or
  discards and replays each transformer's internal block graph during
  backward. Disabled remains the default.
- Options may appear in any order.

These are Python command options; they do not change persisted model
configuration. Metal,
CUDA, and TPU execution are synchronous and do not imply
asynchronous streams, bitwise equality with CPU, or an unmeasured speedup from
selecting an accelerator or Flash.

Block checkpointing lowers retained activation state but executes every
selected block forward a second time during backward. It composes with both
attention algorithms and with full, LoRA, or QLoRA post-training; it changes
neither the optimizer equations nor persisted model state. See
[ACTIVATION_CHECKPOINTING.md](ACTIVATION_CHECKPOINTING.md).

## Adam state storage

The general Adam API defaults to contiguous moment tensors. Callers can instead
split each parameter's first and second moment vectors into bounded pages:

```cpp
AdamOptions options;
options.state_storage = AdamStateStorageKind::Paged;
options.page_size = 4096;
Adam optimizer(parameters, options);
```

Paged and contiguous modes use identical Adam equations and transactional
whole-step commit semantics. Paged mode bounds each moment-state allocation and
backend update request to at most `page_size` scalar elements, but it still
stores two FP32 moment values per trainable scalar and prepares full next-value
parameter candidates for transactionality. CUDA page tensors use managed
allocations; there is no explicit eviction, host/disk spill, prefetch policy,
memory budget, or general OS page-fault manager.

Python post-training chooses paged state automatically for QLoRA and contiguous
state for Full or LoRA. Override that policy with
`optimizer_state="contiguous"|"paged"`. See [ADAM.md](ADAM.md) for
diagnostics and failure atomicity.

## Tiny-batch overfitting acceptance

The acceptance test intentionally trains repeatedly on one fixed tiny batch.
That is not a realistic evaluation setup. It is a wiring test with a very
useful expectation:

```text
same examples repeatedly → final loss substantially below initial loss
```

Passing it demonstrates that the model forward pass, loss, autograd graph,
gradient accumulation, clipping, Adam state, and parameter replacement work
together across multiple steps. It does not demonstrate generalization, good
language generation, or useful model quality. Those require held-out data and
the later generation/checkpoint milestone.
