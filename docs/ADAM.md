# Adam Optimizer

Adam is the project's parameter-update algorithm. It is separate from the
decoder-only transformer: the model produces logits, cross-entropy and autograd
produce gradients, and Adam consumes those gradients to replace the trainable
leaf values.

One optimizer step has this boundary:

```text
existing parameter values
        ↓ forward
logits → cross-entropy → backward
        ↓ parameter gradients
global gradient-norm clipping
        ↓ clipped gradients
bias-corrected Adam update
        ↓
new parameter values
```

Adam does not choose batches, run the forward or backward passes, repeat steps,
write metrics, or decide when training is finished. Those responsibilities
belong to the training loop described in [TRAINING.md](TRAINING.md).

## Public interface

The optimizer is declared in:

```text
include/riftco_transformer/optim/adam.hpp
```

`AdamOptions` contains:

| Option | Default | Requirement |
| --- | ---: | --- |
| `learning_rate` | `0.001` | finite and greater than zero |
| `beta1` | `0.9` | finite and strictly between zero and one |
| `beta2` | `0.999` | finite and strictly between zero and one |
| `epsilon` | `0.00000001` | finite and greater than zero |
| `maximum_gradient_norm` | `1.0` | finite and greater than zero |

A typical caller constructs the optimizer from the model's named parameters:

```cpp
riftco_transformer::AdamOptions options;
options.learning_rate = config.learning_rate;
options.beta1 = config.adam_beta1;
options.beta2 = config.adam_beta2;
options.epsilon = config.adam_epsilon;
options.maximum_gradient_norm = config.gradient_clip;

riftco_transformer::Adam optimizer(model.parameters(), options);
```

The optimizer copies the `ParameterList`. Each `NamedParameter` owns a
`ParameterHandle` to canonical shared parameter state and exposes
`.parameter` as a raw-pointer compatibility view. Adam therefore updates the
same state as the model rather than copies of its tensors, and its copied list
keeps that state alive independently of the caller's list or wrapper lifetime.

The remaining public operations are:

```text
options()                 configured hyperparameters
backend()                 intrinsic backend of parameters and moment state
step_count()              number of successful updates
parameter_tensor_count()  number of registered parameter tensors
step()                    clip gradients and apply one update
zero_gradients()          explicitly clear every registered gradient
global_gradient_norm()    measure any ParameterList without updating it
```

`parameter_tensor_count()` counts parameter tensors, not the total number of
scalar values. `global_gradient_norm()` is generic over `ParameterList`; it
does not require a decoder model and can measure a complete registered module,
an explicit LoRA-only list, or a custom parameter collection.

## Per-parameter state

For every registered parameter tensor, Adam owns two same-shaped tensors:

- the first moment, $m$, is an exponential average of gradients;
- the second moment, $v$, is an exponential average of squared gradients.

Both start at zero on the same backend as the registered parameters. The
optimizer also stores:

- the number of successful steps, $t$;
- the current power $\beta_1^t$;
- the current power $\beta_2^t$.

The parameter list is fixed when the optimizer is constructed. Moment tensors
remain aligned with that list for the optimizer's lifetime.

## Global gradient clipping

Before updating any moment, Adam measures one norm across every scalar gradient
in every registered parameter tensor. If $g_i$ is the raw gradient of scalar
element $i$, the global norm is:

```math
\lVert g \rVert_2
=
\sqrt{\sum_i g_i^2}
```

The implementation accumulates this norm with a scaled sum-of-squares algorithm
in `double`. This avoids overflow or underflow that a direct sum of `float`
squares could cause.

Let $G_{\max}$ be `maximum_gradient_norm`. One shared scale $s$ is chosen:

```math
s
=
\begin{cases}
\dfrac{G_{\max}}{\lVert g \rVert_2},
& \text{if } \lVert g \rVert_2 > G_{\max}, \\[6pt]
1,
& \text{otherwise.}
\end{cases}
```

Each scalar gradient is then clipped with the same multiplier:

```math
\widetilde{g}_i = s g_i
```

A zero norm falls into the “otherwise” case and uses $s=1$. Clipping is global
rather than per tensor, so it preserves the direction of the complete model
gradient. The clipped gradient $\widetilde{g}$—not the original gradient
$g$—is used to update both Adam moments.

## Adam equations

For successful step $t$, Adam applies the following equations independently to
each parameter scalar:

```math
m_t
=
\beta_1 m_{t-1}
+
(1-\beta_1)\widetilde{g}_t
```

```math
v_t
=
\beta_2 v_{t-1}
+
(1-\beta_2)\widetilde{g}_t^2
```

Because both moment tensors start at zero, their early values are biased toward
zero. Adam corrects that bias:

```math
\widehat{m}_t
=
\frac{m_t}{1-\beta_1^t}
\qquad
\widehat{v}_t
=
\frac{v_t}{1-\beta_2^t}
```

Finally, it updates the parameter:

```math
\theta_t
=
\theta_{t-1}
-
\alpha
\frac{\widehat{m}_t}
{\sqrt{\widehat{v}_t}+\epsilon}
```

The symbols mean:

| Symbol | Meaning |
| --- | --- |
| $t$ | successful optimizer step, beginning at $1$ |
| $\theta_t$ | one parameter scalar after step $t$ |
| $\widetilde{g}_t$ | that scalar's globally clipped gradient at step $t$ |
| $m_t$ | first-moment value for that scalar |
| $v_t$ | second-moment value for that scalar |
| $\widehat{m}_t$ | bias-corrected first moment |
| $\widehat{v}_t$ | bias-corrected second moment |
| $\alpha$ | `learning_rate` |
| $\beta_1$ | `beta1`, controlling first-moment memory |
| $\beta_2$ | `beta2`, controlling second-moment memory |
| $\epsilon$ | `epsilon`, preventing division by zero |

The square root covers only $\widehat{v}_t$; $\epsilon$ is added afterward. The
stored beta powers start at $1$. They are multiplied before each update, so the
first successful call uses $\beta_1^1$ and $\beta_2^1$ for bias correction.

The portable reference performs intermediate update arithmetic in `double` and
checks each stored `float`. CUDA performs the same candidate calculation with
device `double` intermediates and reports a non-finite candidate through one
shared device flag. Metal uses native `float` arithmetic for its normal fast
path and similarly reports unsafe arithmetic. Accelerator parity tests use
numerical tolerances rather than requiring bit equality.

Apple GPU float arithmetic may flush subnormal intermediates. The precise-math
kernel checks its actual operation sequence for subnormal, non-finite, and
ill-conditioned cancellation results. One shared device flag requests a retry
when native `float` is unsafe. After the command completes, that rare batch is
evaluated by the portable `double` reference directly over the host-visible
shared candidate buffers. This preserves the optimizer contract, backend
identity, and transactional commit boundary; it does not move tensors to CPU
storage or add a duplicate host pass to ordinary fused updates.

All parameter values and moments are written to out-of-place candidate tensors.
Dispatch rejects a candidate that aliases any live buffer or another candidate,
including aliases across parameter tensors in the same batch. No live optimizer
state changes while a kernel is running. Only after the complete batch succeeds
are candidates moved into the parameters and moment slots.

## Backend execution

Adam captures one intrinsic backend when it is constructed and requires every
parameter value, gradient, and moment to remain on it. Move a model before
constructing its optimizer:

```cpp
model.to(ExecutionBackend::Metal);
Adam optimizer(model.parameters(), options);
```

The global norm remains an overflow-safe `double` host reduction. Metal tensors
use shared storage, so this does not require copying every gradient into a
temporary buffer. On the fused fast path, a clipping scale below the scalar
`float` range is decomposed into a mantissa and exponent and reconstructed with
`ldexp`; scale `1` is passed through directly. If the kernel observes that any
resulting Adam intermediate leaves the normal `float` range or suffers severe
cancellation, the reference retry described above handles the entire batch.

The safe, well-conditioned Metal path uses one fused kernel per parameter
tensor. Each thread computes the clipped gradient, both new moments, both bias
corrections, and the new parameter value. All parameter dispatches are encoded
into one command buffer and synchronized once:

```text
validate and measure all gradients
        ↓
allocate every candidate value and moment tensor
        ↓
encode fused update for parameter 0 ... parameter N
        ↓ one commit and wait
check command status + shared reference-request flag
        ├── safe: keep fused candidates
        └── unsafe: rewrite all candidates with wide reference
        ↓
move every candidate into live state
```

CUDA tensors use managed allocations. The CUDA Adam module preflights every
parameter, gradient, moment, and candidate native handle, launches one
grid-stride update kernel per parameter tensor, then synchronizes once for the
batch. Each thread uses `double` intermediates before checked conversion to the
stored `float`. A shared device status flag turns any non-finite candidate into
an exception before the public optimizer commits live values or moments.

TPU currently calls the same portable reference implementation over its host
mirror. Neither the native Metal nor CUDA candidate update changes the separate
global-norm boundary: gradient-norm clipping remains host control flow on all
backends.

This is per-tensor fusion in one submission, not a flattened multi-tensor
kernel. Flattened parameter arenas can be considered later if profiling shows
dispatch overhead matters.

## Step statistics

`step()` returns an `AdamStepStats` value:

| Field | Meaning |
| --- | --- |
| `step` | successful step count after this update |
| `gradient_norm` | global norm before clipping |
| `clip_scale` | shared multiplier applied to every gradient |

The norm and scale are reported as `double`. A successful all-zero first step
still increments the step count and reports a norm of `0` and a clip scale of
`1`.

## Gradient consumption and reset

`step()` consumes the gradients currently stored in the registered
`Parameter` objects. During commit it calls `Parameter::set_value()` for every
new value. Replacing a leaf value preserves its identity and shape but resets
its gradient tensor to zeros. Consequently, gradients inspected after a
successful optimizer step are zero rather than the gradients used for that
step.

`zero_gradients()` is also available when the caller needs to discard gradients
without changing parameter values or optimizer moments.

There are two different zero-gradient cases:

1. With a fresh optimizer, zero gradients and zero moments produce a zero
   update, so parameters remain unchanged.
2. After an earlier nonzero gradient, a later zero gradient decays the stored
   moments. The remaining first moment can still produce a parameter update.

The second behavior is standard Adam momentum. “Zero gradients leave parameters
unchanged” is therefore guaranteed only while the relevant optimizer state is
also zero.

## Validation and failure behavior

Construction rejects:

- an empty parameter list;
- an empty or duplicate parameter name;
- a null or duplicate `Parameter*`;
- a non-finite initial parameter value;
- a gradient whose shape differs from its parameter value;
- parameter values or gradients split across different backends;
- a non-finite initial gradient;
- an invalid or non-finite optimizer option.

Each step rechecks gradient shapes, backend identity, finite gradients, and
finite parameter values. Moving a parameter after optimizer construction is
therefore rejected before an update. A step also rejects counter overflow and
any moment or parameter result that cannot be represented as a finite `float`.

The update is transactional. Norm validation, candidate allocation, and alias
validation happen before device work; backend command failures or Metal
wide-reference retry failures leave parameter values, gradients, moments, beta
powers, and the successful-step counter unchanged.

## Lifetime and computation-graph rules

Adam retains the owning handles from its copied `ParameterList`. The canonical
parameter state therefore outlives an originating `Parameter` wrapper or
module that leaves scope. Moving a wrapper also resolves to the same canonical
identity, so aliases are still detected as duplicates.

The public `.parameter` field is a compatibility view, not independent
ownership. Code that extracts that raw pointer must keep a corresponding
`NamedParameter`, `ParameterHandle`, or optimizer alive while using it.
Parameter shape and backend must not change while Adam exists. Moving a
parameter to another backend after optimizer construction is rejected before
an update because its moment state remains on the original backend.

Parameter values must be updated only after the current backward pass has
finished. Once `step()` replaces the leaf values, the old computation graph is
consumed; the next forward pass must build a new graph from the updated values.

Milestone 7 isolated and proved this single transaction:

```text
prepared gradients → clip → update moments → replace parameter values
```

Milestone 8 now orchestrates it repeatedly:

```text
choose batch → forward → loss → backward → Adam step → record metrics → repeat
```

The next iteration builds a fresh graph from the newly replaced leaf values.
Keeping the milestones separate let the optimizer equations and state be
verified independently from batch scheduling and training-control behavior.
