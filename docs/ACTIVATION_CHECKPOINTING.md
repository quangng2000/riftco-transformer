# Activation Checkpointing

Activation checkpointing, also called gradient checkpointing, trades extra
forward computation for lower training activation memory. It does not change
the model equation, parameter count, loss, or optimizer.

## The ordinary training graph

For transformer blocks

```math
h_{i+1} = f_i(h_i; \theta_i),
```

an ordinary forward pass retains each operation inside every $f_i$. Backward
needs those values to apply the chain rule. In this lab, every retained
`Variable` node owns both its forward `Tensor` and a same-shaped gradient
`Tensor`, so deep graphs retain many node-owned buffers.

## The checkpointed graph

The block policy keeps only block boundaries and parameter leaves:

```text
h_i ── checkpoint(f_i, θ_i) ── h_(i+1)
          │
          └── direct graph parents: every base and active-LoRA parameter
```

During the original forward:

1. evaluate the block normally;
2. copy out its output value;
3. discard the internal block graph;
4. retain one checkpoint node connected to the block input and parameters.

During backward:

1. create a fresh leaf holding the saved block input;
2. evaluate the same uncheckpointed block again;
3. compute a local vector–Jacobian product;
4. add the input VJP to the preceding graph boundary;
5. add each parameter VJP to its parameter leaf;
6. release the temporary replay graph.

For an upstream gradient $g_{i+1}$, replay calculates

```math
g_i =
\frac{\partial f_i}{\partial h_i}^{\mathsf T}g_{i+1},
\qquad
\nabla_{\theta_i}L =
\frac{\partial f_i}{\partial \theta_i}^{\mathsf T}g_{i+1}.
```

The equations are identical to ordinary backpropagation. Only the lifetime of
the intermediate forward values changes.

## Isolated nested VJPs

A replay must not call ordinary `backward()` against live parameters. Ordinary
backward starts a fresh pass and would erase gradients already contributed by
later blocks, shared branches, or other checkpoints.

`Variable::backward()` now evaluates every pass in an isolated gradient
context. Backward closures accumulate into that context, and public node
gradients are committed only after the complete pass succeeds. A checkpoint
replay creates a nested context, extracts gradients for its fresh input and
declared parameter leaves, then adds them to the outer context.

This gives three important properties:

- shared parameters receive every contribution exactly once;
- a replay exception cannot partially overwrite public gradients;
- repeated `backward()` calls remain fresh passes while values are unchanged.

The existing parent-version validation still runs before differentiation.
Changing a parameter value, transferring its backend, or applying Adam between
forward and backward rejects the stale graph before gradients are committed.

## Replay contract

The reusable core primitive is:

```cpp
Variable checkpoint(
    const Variable& input,
    std::span<const Variable> dependencies,
    std::function<Variable(const Variable&)> recompute
);
```

Every dependency must be a unique differentiable leaf. The implementation
rejects:

- missing or duplicate dependencies;
- captured external graph nodes outside the declared boundary;
- newly created undeclared differentiable leaves;
- a callable that does not depend on its input or every dependency;
- output shape/backend changes between forward and replay.

The callable must be deterministic and replay-safe. Current transformer blocks
meet that contract: they have no dropout, random sampling, or mutable cache.
If dropout is added later, checkpoint replay must capture and restore the
forward RNG state so it reconstructs the same mask.

Incremental `decode_token()` is deliberately never checkpointed. It mutates
the K/V cache and therefore is not replay-safe.

## Public model policy

C++:

```cpp
using riftco_transformer::ActivationCheckpointingKind;

DecoderOnlyTransformer model(
    dimensions,
    random,
    1.0e-5F,
    FullSequenceAttentionKind::Flash,
    ActivationCheckpointingKind::TransformerBlock
);

model.set_activation_checkpointing_kind(
    ActivationCheckpointingKind::Disabled
);
```

Python:

```python
model = DecoderOnlyTransformer(
    config,
    attention="flash",
    activation_checkpointing="block",
).to("metal")

assert model.activation_checkpointing == "block"
model.set_activation_checkpointing("disabled")
```

Stage configuration:

```python
PretrainingConfig(
    attention="flash",
    activation_checkpointing="block",
)

PostTrainingConfig(
    fine_tuning_method="lora",
    activation_checkpointing="block",
)
```

Native and Python training CLIs accept:

```text
--activation-checkpointing disabled|block
```

`disabled` is the compatibility default. `block` checkpoints every transformer
block. The enum and execution-policy boundary allow later policies, such as
multi-block segments, without coupling checkpoint mechanics to the trainer.

## C ABI

ABI 2.0 exposes:

```c
RT_ACTIVATION_CHECKPOINTING_DISABLED
RT_ACTIVATION_CHECKPOINTING_TRANSFORMER_BLOCK

rt_model_set_activation_checkpointing(...)
rt_model_activation_checkpointing(...)
```

The policy is deliberately absent from `rt_transformer_config` and the
persisted model-state schema. Stage pipelines do record the selected policy as
descriptive training metadata. It changes how a future training graph retains
activations, not model architecture or weights. It is safe to change while a
decode session exists because decode does not use it.

## Relationship to other memory techniques

These features save different memory:

| Technique | What it reduces |
| --- | --- |
| LoRA | Trainable gradients and Adam state by optimizing small adapters |
| FlashAttention | Saved quadratic full-sequence attention probabilities |
| Activation checkpointing | Intermediate activations across transformer blocks |

They compose. A checkpointed block may replay FlashAttention; Flash backward
then performs its own score/probability rematerialization. LoRA replay
dependencies include both all base parameters used by the block and every
active adapter parameter, even when Adam updates only adapters.

Checkpointing does not reduce parameter storage, Adam state for fully
trainable parameters, or the final logits/loss. It also adds computation:
each checkpointed block executes its forward calculation once during the
original forward and once again during backward.

## Diagnostics and tests

`Variable::graph_statistics()` reports reachable graph-node count and the
number of `Tensor` elements owned directly by those nodes. It excludes backend
workspaces and tensors captured by backward closures, so it is a structural
diagnostic rather than a process-memory profiler.

Tests verify:

- forward, loss, parameter-gradient, and Adam-step parity;
- materialized and Flash attention on CPU and Metal;
- base and active-LoRA dependency gradients;
- multiple checkpoints sharing parameters;
- recomputation count and reduced retained graph statistics;
- stale-value and structural-mutation rejection;
- exception-atomic backward behavior;
- graph lifetime after the original C++/C/Python model handle is released;
- incremental decode independence;
- C ABI, Python, native-stage, CLI, and installed-package contracts.

The original systematic treatment is
[Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174).
The user-facing intuition is also summarized by
[Intuitive Papers](https://intuitivepapers.ai/gradient-checkpointing/).
