# Autograd Milestone

Autograd answers the question training depends on:

```text
If this parameter changes slightly, how does the final loss change?
```

The answer is the parameter's gradient. Our implementation applies the chain
rule to an operation graph built during the forward pass.

The public `Variable` contract stays in
`include/riftco_transformer/core/autograd.hpp`. Its implementation is split by
responsibility so each algorithm can be studied independently:

```text
src/core/autograd/
├── graph.cpp              # nodes, traversal, accumulation, and VJPs
├── operations.cpp         # differentiable arithmetic and tensor transforms
├── custom_gradient.cpp    # validated user-supplied VJP boundary
├── checkpoint.cpp         # discard-and-recompute activation checkpointing
└── detail/node.hpp        # private graph-node representation
```

`detail/node.hpp` is private implementation glue, not a second public API.

## Tensor versus Variable

`Tensor` remains a plain numeric container:

```text
Tensor = backend-owned values + shape + strides
```

`Variable` wraps a tensor in an autograd graph node:

```text
Variable node
├── forward Tensor value
├── same-shaped Tensor gradient
├── requires-gradient flag
├── value version
├── parent nodes
├── parent versions captured during forward
└── local backward rule
```

Keeping these responsibilities separate means raw tensor operations can be used
inside backward rules without accidentally constructing another graph.
Those calculations go through the shared `tensor_ops` layer:

```text
Variable/autograd → tensor_ops → Tensor
```

## Forward values must stay unchanged

Some local backward rules read a parent's forward value. For example, the
gradient of `x * y` needs the values of both `x` and `y`. Every operation
therefore records the value version of each parent when its forward result is
created.

Before changing any gradient, `backward()` validates all recorded versions in
the reachable graph. Replacing a parameter directly, applying an Adam step, or
moving a parameter to another backend increments that parameter's version. An
older graph then throws `std::logic_error` instead of silently differentiating
with the new value. Repeated backward passes remain valid while the forward
values are unchanged.

## A tiny graph

For:

```math
z = xy + x
```

the forward graph is:

```text
x ──────┬── multiply ──┐
        │       ▲       │
        │       │       ▼
        │       y      add ── z
        └───────────────▲
```

When `z.backward()` runs, the output gradient starts at one:

```math
\frac{\partial z}{\partial z} = 1
```

The graph is processed from `z` toward its leaves:

```math
\begin{aligned}
\frac{\partial z}{\partial x} &= y + 1, \\
\frac{\partial z}{\partial y} &= x.
\end{aligned}
```

If $x=2$ and $y=3$, the gradients are `x.grad = 4` and `y.grad = 2`.

## Local backward rules

Each operation only needs to know its own derivative and the gradient arriving
from later operations, called the upstream gradient $G$.

| Forward operation | Gradient sent left | Gradient sent right |
| --- | --- | --- |
| $A + B$ | $G$ | $G$ |
| $A - B$ | $G$ | $-G$ |
| $A \odot B$ | $G \odot B$ | $G \odot A$ |
| $A \oslash B$ | $G \oslash B$ | $-G \odot A \oslash B^2$ |
| $AB$ | $GB^{\mathsf T}$ | $A^{\mathsf T}G$ |

Here, $\odot$ and $\oslash$ mean elementwise multiplication and division.
Writing $AB$ without either symbol means matrix multiplication.

Unary rules include:

```math
\begin{aligned}
\exp(A):\quad
    &G \odot \exp(A), \\
\log(A):\quad
    &G \oslash A, \\
\sqrt{A}:\quad
    &\frac{G}{2\sqrt{A}}, \\
\mathrm{erf}(A):\quad
    &G \odot \frac{2}{\sqrt{\pi}} \exp(-A^2).
\end{aligned}
```

These are raw `Tensor` calculations performed by `tensor_ops`. They do not
create new `Variable` nodes.

## Why gradients accumulate

A value can influence the loss through multiple paths:

```math
\begin{aligned}
a &= x^2, \\
\mathrm{loss} &= a^2 + a.
\end{aligned}
```

Both uses of $a$ contribute to `a.grad`, and both uses of $x$ in $x \times x$
contribute to `x.grad`. The engine therefore adds gradient contributions rather
than replacing an existing gradient.

The topological traversal visits each node once, but every graph edge still
contributes its derivative.

## Backward ordering

The engine performs a depth-first traversal through parent links and stores
nodes in postorder. Reversing that order gives:

```text
loss → later operations → earlier operations → leaf parameters
```

This ensures a shared node receives all downstream gradient contributions
before its own backward rule runs.

Every backward call creates a fresh isolated gradient context. Local rules
accumulate there, and the engine commits all reachable node gradients only
after the pass succeeds. Repeating `backward()` therefore recomputes the same
gradients instead of silently doubling them. A thrown local rule leaves the
previous public gradients unchanged.

## Seeds and vector outputs

A scalar loss has an implicit seed of one:

```cpp
loss.backward();
```

A non-scalar output needs an explicit same-shaped seed:

```cpp
output.backward(seed);
```

This computes a vector-Jacobian product without building a full Jacobian
matrix, which would be much larger.

The seed must also use the output tensor's backend. Every graph node allocates
its gradient beside its value, scalar constants are created “like” their
operand, and accumulation rejects cross-backend contributions. Graph traversal
is CPU control flow, while numerical rules operate on backend-preserving
tensors. Layout, elementwise, reduction, GELU, softmax, embedding
gather/scatter, LayerNorm, loss, matmul, and causal-attention local rules all
dispatch through that backend. Attention uses dedicated
vector-Jacobian-product requests for its materialized context/probability
outputs and its Flash context-only path; Flash reconstructs probabilities from
saved `[B,H,T]` row statistics during backward. Thus a Metal
graph keeps its numerical backward work on Metal even though graph scheduling
and node accumulation remain host-controlled.

`concatenate_last_axis` is also a graph operation. If
$Y=[X_1;\ldots;X_n]$ along the feature axis, its VJP partitions the upstream
gradient into the same feature ranges:

```math
\nabla_{X_i}L
= \nabla_YL[...,\,o_i:o_i+d_i],
\qquad o_i=\sum_{j<i}d_j.
```

This lets attention and program branches merge activations without introducing
a special layer or duplicating gradient logic.

## Public custom gradients

`custom_gradient(output, inputs, vjp)` connects an externally computed
`Tensor` result to the graph without exposing graph nodes or gradient
accumulation. The callback receives the upstream output gradient and returns
one positional contribution for every input:

```cpp
const Variable x(Tensor({2}, {2.0F, 3.0F}));
const Variable y(Tensor({2}, {4.0F, 5.0F}));
const Tensor x_value = x.value();
const Tensor y_value = y.value();
const std::vector<Variable> inputs{x, y};

const Variable product = custom_gradient(
    tensor_ops::multiply(x_value, y_value),
    inputs,
    [x_value, y_value](const Tensor& upstream) {
        return std::vector<Tensor>{
            tensor_ops::multiply(upstream, y_value),
            tensor_ops::multiply(upstream, x_value),
        };
    }
);
```

Every input and the output must use the same backend. During backward, the VJP
must return exactly `inputs.size()` tensors, including positions for inputs
that do not require gradients. Each result must match its input's shape and
backend. Core validates the complete result before accumulating anything, and
the normal isolated gradient context commits only after the full backward pass
succeeds. A callback exception or malformed result therefore leaves previously
visible gradients unchanged.

The callback should capture only the forward tensors or metadata needed for
its derivative. It must not mutate graph inputs or perform a nested parameter
update. Higher-order differentiation through the callback is not recorded.

## Activation checkpoint replay

`checkpoint(input, dependencies, recompute)` evaluates a deterministic region
once, keeps only its result plus boundary nodes, and evaluates it again during
backward. The replay runs a nested vector-Jacobian product in its own gradient
context. It returns gradients for the fresh replay input and declared
differentiable leaves, which the checkpoint node adds to the outer pass.

This isolation is essential when multiple checkpointed blocks share
parameters: an ordinary nested `backward()` would clear gradients already
contributed by later blocks. The primitive also validates unique leaf
dependencies, rejects captured external graph state, and checks that
shape/backend/dependencies remain stable across replay.

The decoder can apply this primitive at every transformer-block boundary.
See
[ACTIVATION_CHECKPOINTING.md](ACTIVATION_CHECKPOINTING.md)
for its algorithm, execution-policy APIs, tests, and memory/compute tradeoff.

## Ownership

`Variable` is a small handle containing a `std::shared_ptr` to its graph node.
A child node owns its parent nodes. This lets intermediate C++ handles go out
of scope while the final loss still keeps the complete graph alive.

Parents do not own their children, and backward closures never capture the
output node. Consequently, the graph has no ownership cycle and is released
when its final handles disappear.

## Gradient checking

An analytical gradient is compared with a centered finite difference:

```math
\frac{\partial f}{\partial x}
\approx
\frac{f(x + \varepsilon) - f(x - \varepsilon)}
     {2\varepsilon}
```

The tests perturb every element of non-square and batched matmul inputs and
compare the numerical result with autograd. Composite tests also cover
arithmetic, `exp`, `log`, `sqrt`, reductions, reshape, transpose, and arbitrary
axis permutation.

Finite differences are slow and approximate, but they are an independent way
to catch an incorrect backward formula.

## Current limits

- Differentiable elementwise operations require equal shapes; broadcasting is
  always an explicit `broadcast_to` graph operation.
- Broadcast backward uses `sum_to_shape` so bias and normalization gradients
  return to their original parameter shapes.
- Embedding lookup gathers table rows and scatter-adds repeated-row gradients.
- Batched matmul requires identical leading batch dimensions; it does not
  perform implicit batch broadcasting.
- Permutation backward applies the inverse axis order.
- `log` and differentiable `sqrt` require strictly positive inputs.
- The graph traversal is recursive and intended for this tiny learning model.
- Higher-order gradients are not recorded.

These boundaries keep the first gradient engine small enough to understand.
