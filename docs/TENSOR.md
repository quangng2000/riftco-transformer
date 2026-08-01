# Tensor Milestone

A tensor is a multidimensional owner of a contiguous sequence of numbers. It
performs neither numerical operations nor automatic differentiation itself.
Its private storage is supplied by its intrinsic backend:

- CPU storage owns a `std::vector<float>`;
- Metal storage owns a persistent shared `MTLBuffer`.

Both expose the same synchronous contiguous `float` contract. Shared Metal
storage keeps the existing checked indexing API useful while allowing GPU
kernels to consume the allocation directly.

## Contract and implementation

The public contract lives in:

```text
include/riftco_transformer/core/tensor.hpp
```

Code using a tensor includes that header and only needs to know the storage and
layout interface. The implementation is grouped by responsibility:

```text
src/core/tensor/
├── storage.cpp       # allocation, ownership, transfer, and raw access
└── layout.cpp        # shape, strides, indexing, and reshape
```

Numerical calculations have their own reusable public interface and focused
implementation files:

```text
include/riftco_transformer/core/tensor_ops.hpp
src/core/tensor/
├── elementwise.cpp
├── matmul.cpp
├── layout_ops.cpp
├── indexing.cpp
├── reductions.cpp
├── softmax.cpp
└── detail/validation.{hpp,cpp}
```

This gives the project three deliberate layers:

```text
Tensor      = storage + shape + strides
tensor_ops  = raw numerical calculations
Variable    = autograd graph + gradients
```

## Backend identity and transfer

Every tensor has an intrinsic `ExecutionBackend`, available through
`backend()`. Ordinary constructors use the calling thread's current backend as
their default. Derived tensor operations preserve the backend of their inputs,
and multi-input operations reject mixed backends.

Call `to(backend)` for an explicit deep transfer:

```cpp
Tensor cpu({2, 2}, {1.0F, 2.0F, 3.0F, 4.0F});
Tensor metal = cpu.to(ExecutionBackend::Metal);
```

Copy construction, copy assignment, `reshape()`, and `to()` all preserve value
semantics: the result owns independent storage. Moving a tensor transfers
ownership without copying.

## Shape, stride, and offset

Consider a tensor with shape `[2, 3, 4]`. It contains:

```math
2 \times 3 \times 4 = 24
```

In row-major order its strides are `[12, 4, 1]`. Moving one position along:

- dimension 0 skips 12 values;
- dimension 1 skips 4 values;
- dimension 2 skips 1 value.

The index `[1, 2, 3]` becomes:

```math
\text{flat offset}
= 1 \times 12 + 2 \times 4 + 3 \times 1
= 23
```

Every checked multidimensional access performs this calculation after verifying
the index rank and each dimension boundary.

## Scalars

An empty shape `[]` represents a scalar. Its rank is zero, but it contains one
value. This convention will be useful when the entire computation eventually
produces one loss value.

## Reshape

`reshape()` preserves the flat value order and requires the same number of
elements:

```text
[2, 3] → [3, 2]  valid: both contain 6 values
[2, 3] → [4, 2]  invalid: 6 and 8 values differ
```

The current implementation returns a copy. A later performance pass could add
non-owning views, but ownership is deliberately obvious for now.

## Numerical operations

For tensor $A$ with shape `[..., m, k]` and tensor $B$ with shape
`[..., k, n]`, with the same leading batch dimensions, the result $C$ has
shape `[..., m, n]`:

```math
C_{i,j} = \sum_{r=0}^{k-1} A_{i,r} B_{r,j}
```

`tensor_ops::matmul` dispatches to the input tensor's backend: CPU uses three
direct loops, Metal and CUDA use batched device kernels, and the experimental
TPU adapter compiles a StableHLO `dot_general` through PJRT.
The same layer also owns elementwise arithmetic, reductions, broadcasting,
transposition, and elementary functions. CPU uses readable reference
implementations; Metal routes those operations to compute kernels over the
same persistent shared storage. CUDA owns native tensor/NN, attention, and
Adam-update kernels. TPU owns StableHLO materialized-attention and paged-decode
programs while its remaining operations use portable reference paths. See
[TENSOR_OPS.md](TENSOR_OPS.md) for the complete contract and
[BACKENDS_AND_PYTHON.md](BACKENDS_AND_PYTHON.md) for the synchronous execution
boundary.

## Safety choices

- zero-sized dimensions are rejected;
- shape multiplication is checked for integer overflow;
- flat and multidimensional indexing are bounds checked;
- raw elementwise operations require exactly equal shapes;
- multi-input operations require one common backend;
- broadcasting is available only through an explicit operation;
- batched matrix multiplication requires matching leading dimensions;
- axis permutations materialize a new contiguous tensor.

The tests in `tests/core/test_tensor.cpp` cover storage and layout.
`tests/core/test_tensor_ops.cpp` checks numerical operations with
hand-calculated values, so neither suite depends on another tensor library as
an oracle.
