# Tensor Operations

`Tensor` stores values and describes their layout. `tensor_ops` performs
calculations on those values. Keeping those jobs separate gives the project one
reusable numerical layer without turning `Tensor` into a large do-everything
class.

```text
autograd and neural-network code
                  │
                  ▼
             tensor_ops
                  │
                  ▼
       Tensor storage and layout
```

The public numerical interface is:

```text
include/riftco_transformer/core/tensor_ops.hpp
```

The numerical implementation is split by operation family, then routed through
the backend dispatch layer:

```text
src/core/tensor/
├── elementwise.cpp
├── matmul.cpp
├── layout_ops.cpp
├── indexing.cpp
├── reductions.cpp
├── softmax.cpp
└── detail/validation.{hpp,cpp}
src/core/backend/adapter.hpp
src/core/backend/adapters/
src/core/backend/nn/reference/
src/core/backend/attention/
src/core/backend/registry.cpp
```

## Current operations

The namespace `riftco_transformer::tensor_ops` contains:

- elementwise `add`, `subtract`, `multiply`, and `divide`;
- `negate` and scalar `scale`;
- strict batched `matmul`, general `permute`, and `transpose_2d`;
- checked final-axis concatenation for two tensors or a span of tensors;
- explicit `broadcast_to` and its inverse reduction `sum_to_shape`;
- checked row gathering for embedding lookup;
- full-tensor and axis-based `sum` and `mean`;
- elementwise `exp`, `log`, `sqrt`, and `erf`;
- erf-form GELU and its local backward kernel;
- stable arbitrary-axis softmax and its local backward kernel;
- fused scale + causal mask + softmax and its backward kernel.

Each function consumes plain `Tensor` values and returns a plain `Tensor` on
the same backend. Multi-input functions reject mixed storage backends. The
layer does not know about computation graphs or gradients.

For example:

```cpp
const Tensor scores = tensor_ops::matmul(queries, keys_transposed);
const Tensor scaled = tensor_ops::scale(scores, attention_scale);
```

## Why not a generic helper file?

A name such as `math_helpers` does not describe what belongs in it and tends to
become a collection of unrelated utilities. `tensor_ops` has a precise rule:
an operation belongs here when it numerically transforms one or more tensors
without recording an autograd graph.

This also avoids giving raw `Tensor` and differentiable `Variable` the same
operators. A call site makes the layer explicit:

```cpp
tensor_ops::multiply(raw_left, raw_right);  // numerical calculation only
left_variable * right_variable;             // calculation plus graph node
```

## Shapes and broadcasting

Elementwise binary operations require identical shapes and a common backend.
Broadcasting is an explicit step:

```text
[2, 1] --broadcast_to [2, 3]--> [2, 3]
```

Dimensions are matched from the right. Each input dimension must either equal
the output dimension or be `1`. A scalar with shape `[]` can be broadcast to
any valid shape.

Keeping broadcasting explicit makes shape changes visible while the project is
still focused on learning correctness.

Final-axis concatenation preserves every leading dimension:

```text
[B, T, D1] + [B, T, D2] --concatenate_last_axis--> [B, T, D1 + D2]
```

Inputs must have the same rank, leading shape, and backend. Scalars, an empty
input list, mixed backends, and final-axis overflow are rejected. The
differentiable overload slices the upstream gradient back into the original
width of every input.

## Relationship to autograd

Autograd owns graph construction, traversal, gradient accumulation, and local
derivative rules. It delegates the actual tensor calculations to `tensor_ops`.
For matrix multiplication:

```text
A [..., M, K] @ B [..., K, N] → C [..., M, N]
```

```math
\begin{aligned}
\text{forward:}\quad C &= AB, \\
\text{upstream gradient:}\quad G &= \nabla_C L, \\
\text{backward to }A:\quad \nabla_A L &= G B^{\mathsf T}, \\
\text{backward to }B:\quad \nabla_B L &= A^{\mathsf T} G.
\end{aligned}
```

Here, $L$ is the final loss and $G$ is the gradient arriving at $C$ from the
later operations.

The leading batch dimensions must match exactly; this reference kernel does
not implicitly broadcast them. All three matrix multiplications use the same
raw `tensor_ops::matmul` kernel. The backward pass therefore reuses the
numerical implementation without creating more graph nodes.

`permute` materializes an arbitrary axis ordering. Its backward operation
applies the inverse permutation, which is what attention uses to split
`[batch, time, model_width]` into
`[batch, head, time, head_width]` and merge it again.

An intentional domain distinction remains: raw `tensor_ops::sqrt(0)` is valid,
but differentiable `sqrt(0)` is rejected because its derivative contains
$\frac{1}{2\sqrt{0}}$.

## Testing and future acceleration

`tests/core/test_tensor_ops.cpp` checks every public operation with
hand-calculated values and invalid-shape/domain cases. Autograd tests then
check derivative rules independently with centered finite differences.

The functions in `backend/nn/reference/` and
`backend/attention/reference/`, plus the direct CPU matmul loop, remain the
readable reference implementation.
The operation layer dispatches
layout, elementwise, reduction, GELU, softmax, indexing, and matmul requests
through focused Adapter capabilities. The ordinary overloads run on the input
tensor's intrinsic backend. The explicit matmul overload can execute through
another available implementation for comparison while preserving the input
storage backend.

Metal tensors retain shared buffers, so routed kernels bind input and output
storage directly without changing backend identity. Calls remain synchronous:
each operation waits for its command buffer and checks device status before
returning. Optional source-built CUDA tensors retain managed allocations and
run the tensor/NN contracts, matmul, attention, and Adam's candidate-state
update through native GPU kernels. The experimental TPU adapter keeps a host
mirror and stages batched matmul,
materialized attention, and paged decode through PJRT to one Cloud TPU device;
Flash attention and other capabilities retain host reference paths.
CPU/accelerator tests compare with numerical tolerances because device math
and reduction order are not bitwise identical. See
[BACKENDS_AND_PYTHON.md](BACKENDS_AND_PYTHON.md) for the dispatch, neural
kernel, and ABI boundaries.
