# Resume training from an exact checkpoint

Use a `TrainingCheckpoint` when an interrupted Adam run must continue from the
same model values, optimizer moments, random state, and sampled-data position.
The `.riftckpt` contract is separate from both `.rift` distribution artifacts
and activation checkpointing.

## Before you begin

Checkpoint v2 supports ordinary FP32 full-parameter training, active LoRA, and
packed QLoRA. It restores into a freshly constructed compatible model,
optimizer, and batch source. The loader remains compatible with dense v1
archives.

The built-in `RandomWindowBatchSource`, `SequenceWindowBatchSource`, and
`ExampleWindowBatchSource` implement the checkpoint protocol. Their state
contains a SHA-256 dataset/configuration fingerprint, emitted-batch count, and
private `random.Random` state.

## Capture at a clean boundary

Capture only after `Adam.step()` has committed and cleared gradients. The
runtime tracks whether backward produced a pending gradient separately from
its numeric values, so an exactly-zero gradient still requires a step or an
explicit `optimizer.zero_gradients()` before capture:

```python
from riftco_transformer import Adam, DecoderOnlyTransformer, TransformerConfig
from riftco_transformer.checkpoints import TrainingCheckpoint
from riftco_transformer.training import (
    CausalLanguageModelTrainer,
    RandomWindowBatchSource,
    TrainingLoopConfig,
)

config = TransformerConfig(
    vocabulary_size=8,
    maximum_context=4,
    model_width=8,
    head_count=2,
    block_count=1,
    feed_forward_width=16,
)
tokens = tuple(range(8)) * 16
source = RandomWindowBatchSource(
    tokens,
    batch_size=2,
    context_size=4,
    random_seed=17,
)

with DecoderOnlyTransformer(config) as model:
    with model.parameters() as parameters:
        with Adam(parameters, learning_rate=1.0e-3) as optimizer:
            CausalLanguageModelTrainer(model, optimizer).run(
                source,
                TrainingLoopConfig(steps=10),
            )
            checkpoint = TrainingCheckpoint.capture(
                model,
                optimizer,
                source=source,
                global_step=10,
                metadata={"run": "example"},
            )
            checkpoint.save("runs/example-step-10.riftckpt")
```

Passing a non-`None` source that does not implement checkpoint capture and
restore is an error. Pass `source=None` only when intentionally omitting data
position; such an archive cannot by itself reproduce the next batch.

## Restore into fresh objects

Recreate the same executable topology, full-sequence attention algorithm,
activation-checkpointing policy, and optimizer hyperparameters, then restore
before constructing a new forward graph. The fresh model's initializer seed
and initial values may differ because the archive replaces all persistent base
state. Adam must be constructed from the exact model instance passed to
`restore()`; an optimizer belonging to another model is deliberately rejected:

```python
checkpoint = TrainingCheckpoint.load("runs/example-step-10.riftckpt")
source = RandomWindowBatchSource(
    tokens,
    batch_size=2,
    context_size=4,
    random_seed=17,
)

with DecoderOnlyTransformer(config) as model:
    with model.parameters() as parameters:
        with Adam(
            parameters,
            learning_rate=1.0e-3,
            state_storage="paged",
            page_size=4096,
        ) as optimizer:
            restored = checkpoint.restore(model, optimizer, source)
            assert restored.optimizer_step == 10
            CausalLanguageModelTrainer(model, optimizer).run(
                source,
                TrainingLoopConfig(steps=10),
            )
```

Contiguous versus paged Adam storage is a physical restoration choice. It is
not serialized into the logical moment arrays, so a checkpoint can restore
between the two layouts and across available backends. Cross-backend results
promise numerical equivalence, not bitwise-identical future arithmetic.

For LoRA, attach a compatible `LoraConfig` (same rank, alpha, and targets) and
construct Adam over the complete `model.adapter_parameters()` list before
restoring. Initializer seeds may differ. The checkpoint restores the archived
frozen base, adapter values, and adapter moments; the target does not need to
start from the same pretrained or adapter values.

For QLoRA, quantize every eligible Linear first, attach the same `LoraConfig`,
and construct adapter-only Adam. The target may initially use a different NF4
block size, scale encoding, or initializer values: restore transactionally
replaces the packed Linear bases and all remaining frozen parameters with the
archived representation. The native import never constructs a full FP32
Linear base tensor.

## What the archive contains

An independently versioned `.riftckpt` ZIP stores:

- model configuration, exact parameter names, shapes, and FP32 values;
- for QLoRA, exact packed NF4 nibbles, weight/block shapes, block sizes, and
  either FP32 scales or scale codes, second-level scales, block size, and
  offset for double quantization;
- full-sequence attention and activation-checkpointing policies;
- active LoRA configuration and adapter values when applicable;
- Adam options, step, beta powers, and first/second moments;
- the global training step and Python random state;
- the built-in batch-source fingerprint, position, and RNG state; and
- checksums, a deterministic checkpoint ID, and user metadata.

Saving writes beside the destination, flushes the completed archive, and
atomically replaces the destination. Loading validates structure, member
sizes, checksums, parameter signatures, optimizer semantics, RNG state, and
the data fingerprint before restoration mutates live state.

Restore snapshots the prior optimizer, packed NF4 base when applicable,
source, and Python RNG states. A source callback runs before the native Adam
commit; if any component then fails, the operation restores prior state in
reverse order. Built-in batch sources
validate before mutation and provide the strong all-or-nothing path. A custom
source must make `checkpoint_state()` side-effect-free and its restore method
rollback-safe. If custom rollback itself fails, restoration raises an explicit
indeterminate-state error instead of claiming success.

## Three different checkpoint meanings

| Contract | Purpose | Persistent training state |
| --- | --- | --- |
| Activation checkpoint | Recompute forward activations during backward to save memory | No |
| `.rift` `ModelBundle` | Immutable inference, distribution, and stage handoff | Weights and tokenizer only |
| `.riftckpt` `TrainingCheckpoint` | Continue the same Adam run | Weights, moments, counters, RNG, and data position |

## Current limits

- A checkpoint restores execution state but does not embed a byte/BPE
  tokenizer; sampled token IDs and their fingerprint define the data stream.
- Only the complete base-parameter list or complete active LoRA adapter list
  can own Adam state.
- There is no learning-rate scheduler or gradient-scaler state because those
  components do not yet exist in the training engine.
- Process-wide Python random state is restored; applications sharing that
  global generator should coordinate checkpoint restoration.

## Verify exact continuation

The acceptance test compares an uninterrupted $N$-step CPU run with a fresh
process-shaped $K$-step capture/load/restore followed by $N-K$ steps. Final
weights, both moment arrays, beta powers, optimizer step, and the next batch
source state must match exactly. The QLoRA variant additionally requires byte-
identical packed state before and after restoration and resident packed-memory
accounting throughout continuation. Corruption and fingerprint mismatch must
fail without changing model, optimizer, or random state.

## Related guides

- [Training](TRAINING.md)
- [Adam](ADAM.md)
- [Activation checkpointing](ACTIVATION_CHECKPOINTING.md)
- [Model interchange](MODEL_INTERCHANGE.md)
- [Pipeline](PIPELINE.md)
