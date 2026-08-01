# LoRA Fine-Tuning

Riftco Transformer supports low-rank adaptation (LoRA) as an explicit
post-training alternative to full-parameter fine-tuning. The implementation
uses only the framework's native tensor operations, autograd graph, Adam
optimizer, and CPU, Metal, or optional source-built CUDA and TPU backends.

## What LoRA changes

For a linear weight with shape `[output, input]`, LoRA composes a small
trainable branch with the existing `Linear` module:

```text
A: [rank, input]
B: [output, rank]
scale = alpha / rank

linear(x) = x Wᵀ + bias + scale · (x Aᵀ) Bᵀ
```

`A` uses deterministic Xavier initialization and `B` starts at exact zero.
Attaching an adapter therefore preserves the model's output initially. During
LoRA post-training, Adam receives only the adapter parameter list, so the base
parameter values remain unchanged. After training, a transactional merge
computes:

```text
W_merged = W + scale · B A
```

The resulting model has the ordinary base parameter schema and needs no LoRA
branch during serving.

The current default targets are the query and value projections in every
Transformer block. The selectable targets are:

- attention query, key, value, and output projections;
- feed-forward expand and project projections; and
- the language-model head.

Embeddings, normalization parameters, and biases are not LoRA targets.

## Python post-training

```python
from riftco_transformer import LoraConfig
from riftco_transformer.artifacts import ModelBundle
from riftco_transformer.post_training import (
    PostTrainingConfig,
    post_train_jsonl,
)

base = ModelBundle.load("results/stages/tiny_pretrained.rift")
result = post_train_jsonl(
    base,
    "data/post_training/tiny_instructions.jsonl",
    PostTrainingConfig(
        steps=10,
        backend="metal",
        fine_tuning_method="lora",
        lora=LoraConfig(
            rank=4,
            alpha=8.0,
            targets=("attention.query", "attention.value"),
            random_seed=41,
        ),
    ),
)
result.bundle.save("results/stages/tiny_lora_merged.rift")
```

The returned `ModelBundle` is merged and serving-ready. Its metadata records
that LoRA was used, which parameter scope Adam optimized, and that adapter and
optimizer state are not included.

Full-parameter fine-tuning remains the default:

```python
PostTrainingConfig(fine_tuning_method="full")
```

## Comparing ranks without test leakage

`python/riftco_transformer/experiments/lora_rank.py` builds a controlled
experiment around the post-training API. It verifies prepared Dolly
train/validation/test files, starts each candidate from the same immutable
base artifact, and fixes:

- dataset fingerprints and split membership;
- training and adapter-initialization seeds;
- optimizer steps, learning rate, batch size, context, sampler, attention, and
  activation checkpointing;
- target projections and resolved CPU/Metal/CUDA/TPU backend; and
- the scale ratio `alpha / rank`.

Only rank and the derived `alpha = rank × alpha_over_rank` change. This keeps
the branch scale comparable while varying adapter capacity and trainable
parameter count.

```bash
PYTHONPATH="$PWD/python" \
python3 examples/python/compare_lora_ranks.py \
  --base results/stages/tinystories_pretrained.rift \
  --data data/external/huggingface/dolly-lora-v1 \
  --output results/experiments/dolly-lora-ranks \
  --ranks 1,2,4,8 \
  --alpha-over-rank 2 \
  --steps 20 \
  --context 16 \
  --batch-size 2 \
  --backend cpu \
  --prompt "Explain attention."
```

Every candidate and the base model are scored on validation. The lowest
validation loss wins, with lower rank as a deterministic tie-breaker. Only
after selection does the evaluator read test examples, once for the base and
once for the winner. Nonwinning trials therefore have no test measurement.
Prepared records are checked for exact overlap, and formatted
whitespace-normalized model inputs are checked again so formatting cannot hide
cross-split leakage.

Evaluation is deterministic, read-only, and target-token weighted. It scores
each formatted sequence in non-overlapping causal chunks, with model context
and learned positions reset at each chunk boundary. Its metric matches the
current `full_sequence_causal_sft` objective: user prompt, delimiters, and
assistant response all contribute to loss. It is not response-only SFT.

The CLI stages each merged candidate and `comparison.json`, then atomically
publishes the whole output directory without replacing an existing run.
Failures clean staging. The summary contains controls, data fingerprints,
baselines, validation measurements, the selected test measurement, parameter
counts, optional greedy generations, and the full verified prepared-data
manifest plus its SHA-256. Because all candidates are merged into ordinary
model weights, they have identical serving topology. Do not interpret the
included generation timings as evidence that a lower or higher LoRA rank
serves faster.

To compare the validation-selected LoRA recipe with full-parameter
fine-tuning, use `examples/python/compare_fine_tuning.py`. It evaluates both
methods with the same exhaustive train/validation/test metric, reports actual
generalization gaps, and keeps separate learning-rate controls. See
[Post-training generalization](GENERALIZATION.md). Reporting both final test
results consumes that test split, so it must be retired afterward.

The data preparation and end-to-end commands are documented in
[DATASETS_AND_LORA_EXPERIMENTS.md](DATASETS_AND_LORA_EXPERIMENTS.md).
HH-RLHF's chosen/rejected records are not inputs to this experiment; the lab
does not yet implement a preference-training objective.

## Direct Python lifecycle

The low-level API exposes the adapter parameter collection when a custom
training loop is useful:

```python
from riftco_transformer import (
    Adam,
    DecoderOnlyTransformer,
    LoraConfig,
    TransformerConfig,
    cross_entropy,
)

model = DecoderOnlyTransformer(TransformerConfig(
    vocabulary_size=256,
    maximum_context=16,
    model_width=64,
    head_count=4,
    block_count=2,
    feed_forward_width=256,
)).to("metal")

tokens = [[0, 1, 2]]
targets = [[1, 2, 3]]
model.attach_lora(LoraConfig(rank=4, alpha=8.0))
with model.adapter_parameters() as adapters:
    with Adam(adapters, learning_rate=1.0e-3) as optimizer:
        with cross_entropy(model(tokens), targets) as loss:
            loss.backward()
            optimizer.step()

# Adapter parameter views, optimizers, and graphs must be closed first.
model.merge_lora()
```

`model.parameters()` remains the stable base-model collection before, during,
and after LoRA. This preserves the existing artifact format and full-training
API. `model.adapter_parameters()` returns only active LoRA factors.

## Native C++ lifecycle

```cpp
#include "riftco_transformer/model/decoder_only_transformer.hpp"
#include "riftco_transformer/model/lora.hpp"
#include "riftco_transformer/optim/adam.hpp"

riftco_transformer::LoraConfig lora;
lora.rank = 4;
lora.alpha = 8.0F;
lora.targets =
    riftco_transformer::kLoraAttentionQuery |
    riftco_transformer::kLoraAttentionValue;

model.attach_lora(lora);
{
    riftco_transformer::Adam optimizer(
        model.lora_parameters(),
        adam_options
    );
    // Build a fresh loss graph, call backward(), and step Adam.
}
model.merge_lora();
```

The model-level merge prepares and validates every selected weight before
committing any of them. It is one-way for that model instance. Create a fresh
model from a base artifact to run another independent adapter experiment.

C ABI 2.0 exposes the LoRA lifecycle through `rt_model_attach_lora`,
`rt_model_lora_parameters`, and `rt_model_merge_lora`, alongside
incremental decode sessions, full-sequence attention selection, and
activation-checkpointing selection. Its size-versioned `rt_lora_config` uses
fixed-width target-mask bits.

## Artifacts and current limits

An active adapter changes model output but is not part of the ordinary
`ModelState` or `ModelBundle` schema. Capturing either artifact while LoRA is
active is rejected instead of silently dropping the adapter. Merge first,
then capture the ordinary serving artifact.

Serving therefore sees an ordinary model with the LoRA update already folded
into its base linear weights. The one-token decode and paged KV-cache paths do
not need adapter-specific branches or artifact fields. A live decode session
pins parameter state, so LoRA attachment or merge is rejected until that
session is released. See [SERVING.md](SERVING.md) for the session lifecycle.

This milestone deliberately does not define a distributable adapter-only
artifact or resumable optimizer checkpoint. The raw adapter values can be
inspected through the adapter parameter list, but a portable adapter file also
needs a base-artifact identity and a versioned topology contract.

The base parameters are excluded from Adam and remain value-frozen. The
current autograd engine can still calculate gradients for reachable base
leaves; eliminating that extra gradient work is a separate graph-freezing
optimization. LoRA is therefore parameter-efficient now, while further
training-memory optimization remains future work.
