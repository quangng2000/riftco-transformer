# Execution Backends and the Python ABI

The backend layer owns tensor storage and focused accelerated operations
without changing the transformer or autograd equations:

```text
Python stages/tokenizer/tensors/model/Adam ──ctypes──→ stable C ABI 2.4
                                         │
                                         ▼
                                public C++ operations
                                         │
                                         ▼
                                  fixed registry
                                         │
                                         ▼
                       private capability-specific Adapter
                     storage · layout · elementwise · reduction
                matmul · packed NF4 linear · softmax · indexing · LayerNorm
       cross-entropy · materialized/Flash/paged-decode attention · Adam
             │                 │                 │                 │
             ▼                 ▼                 ▼                 ▼
      CPU reference      Metal kernels      CUDA kernels      TPU host mirror
                           over shared       over managed      + PJRT/StableHLO
                           MTLBuffers          storage          selected graphs
                                │                 │                  │
                                └──────── synchronous ───────────────┘
                                             explicit host fallbacks
```

CPU remains the default and the readable correctness reference. On Apple
platforms with an available Metal device, tensors own persistent shared
`MTLBuffer` storage. The operations used by the training graph dispatch to real
Metal kernels: layout transforms, elementwise math, reductions, GELU,
LayerNorm, softmax, causal masking, embedding gather/scatter, cross-entropy,
materialized and tile-8 Flash causal attention and their vector-Jacobian
products, matmul, packed NF4 linear forward/input backward, and Adam. Packed
weights own persistent Metal buffers and are decoded inside the quantized-linear
kernels. `Variable::matmul` captures the input tensor backend during the
forward pass, so both matrix-gradient products use that backend even if the
calling thread's construction default changes before `backward()`.

CUDA is an optional source-build backend for an NVIDIA GPU. It requires CUDA
Toolkit 12 or newer and a compatible NVIDIA driver. CUDA tensors own persistent
managed allocations, which keeps the framework's host-visible tensor contract
intact. Packed NF4 codes and scale metadata likewise own managed allocations;
quantized-linear forward and input backward decode them inside CUDA kernels.
Batched matmul, materialized causal attention and its VJPs, memory-linear
Flash attention and its VJP, and paged decode also execute as CUDA kernels.
Layout, elementwise, reduction, indexing, normalization, loss, and Adam's
out-of-place candidate-state update use CUDA kernels. Adam's overflow-safe
global gradient norm remains a host reduction over managed storage. All calls
are synchronous. This makes the whole framework—including Full, LoRA, and
QLoRA training, generalization evaluation, and serving—functionally available
on `cuda`, but it is not a claim that every piece of control flow is
device-resident or that an end-to-end workload is faster. The CUDA source and
conditional parity coverage are implemented; actual NVIDIA hardware was not
validated on the macOS host used for this milestone.

TPU is an experimental, opt-in Linux x86-64 source backend for Google Cloud
TPUs. It dynamically loads Google's `libtpu.so`, obtains the versioned PJRT C
API, compiles shape-specialized StableHLO programs, and executes packed NF4
linear forward/input backward, batched matmul, materialized causal attention
and its VJPs, and paged decode on one addressable TPU device. TPU tensors retain
a host mirror, while quantized weights retain packed U8 host payloads that are
dequantized inside the StableHLO computation. Flash attention and the remaining
capabilities use the same synchronous reference implementations as CPU. Full,
LoRA, and QLoRA training, Adam, evaluation, and serving are therefore
functionally wired, but transfers occur around each PJRT program and this is
not an end-to-end acceleration claim. The source/ABI boundary and no-device
behavior are tested. A tests-only fake PJRT plugin also exercises the existing
loader, client, compile, transfer, execute, and download paths. It recognizes
the generated quantized-linear program contract and checks legacy and
double-quantized forward/input-backward results against CPU oracles. Real
`libtpu` and Cloud TPU hardware validation remain pending.

## Selecting a backend

Low-level C++ and C APIs use explicit CPU, Metal, CUDA, or TPU selectors.
High-level Python workflows and labs additionally accept `auto`: it selects
TPU when available, then CUDA, Metal, and CPU. Requesting an unavailable
explicit backend fails instead of silently changing a run.

For example, the Python-owned training example accepts the backend alongside
independent attention and activation-retention policies:

```bash
PYTHONPATH=python python3 examples/python/train_tiny.py \
  --steps 20 \
  --backend metal \
  --attention flash \
  --activation-checkpointing block
```

Use `cpu` in every build, `metal` on a compatible Mac, `cuda` in a
CUDA-enabled source build, or `tpu` in a TPU-enabled Cloud TPU build.

The public C++ default-selection interface is:

```cpp
#include "riftco_transformer/core/backend.hpp"

using riftco_transformer::ExecutionBackend;

if (riftco_transformer::execution_backend_available(
        ExecutionBackend::Cuda
    )) {
    const riftco_transformer::ScopedExecutionBackend use_cuda(
        ExecutionBackend::Cuda
    );
    // Tensors and modules constructed in this scope default to CUDA.
}
```

Existing tensors keep their intrinsic backend. Transfer them explicitly:

```cpp
Tensor on_cuda = on_cpu.to(ExecutionBackend::Cuda);
model.to(ExecutionBackend::Cuda);
Adam optimizer(model.parameters(), options);
```

Derived operations return tensors on the input backend, and multi-input
operations reject mixed storage backends. The ordinary
`tensor_ops::matmul(left, right)` runs on that intrinsic backend. An explicit
`matmul(left, right, execution_backend)` overload can stage through another
available implementation for comparison while returning storage on the input
backend; it never changes thread-local state.

## Adapter and selection contracts

`BackendAdapter` is an internal facade over segregated capabilities:

- report its stable name;
- report whether its base runtime and storage can run;
- allocate and clone backend-owned storage;
- execute validated synchronous layout, elementwise, reduction, matmul,
  packed quantized-linear, softmax, indexing, normalization, loss, and
  attention requests;
- execute a validated transactional Adam update batch.

The Adapter normalizes CPU loops, Metal APIs, CUDA runtime calls, and the
TPU/PJRT boundary. The selected `ExecutionBackend` is the Strategy used by
ordinary tensor operations.

| Backend value | Name | Availability | Selection behavior |
| --- | --- | --- | --- |
| `Cpu` | `cpu` | Always | Succeeds |
| `Metal` | `metal` | Runtime/build dependent | Succeeds or throws `runtime_error` |
| `Cuda` | `cuda` | CUDA build, driver, and device dependent | Succeeds or throws `runtime_error` |
| `Tpu` | `tpu` | TPU build, `libtpu`, and Cloud TPU device dependent | Succeeds or throws `runtime_error` |
| Unknown value | None | False | Throws `invalid_argument` |

The framework guarantees:

- no silent backend-identity change from Metal, CUDA, or TPU to CPU;
- the documented Adam numerical-safety retry executes on the host over the
  original Metal candidate storage;
- failed selection leaves the previous selection unchanged;
- construction default is thread-local and each new thread starts on CPU;
- existing tensors and derived results preserve intrinsic backend identity;
- mixed-backend numerical inputs are rejected;
- explicit `matmul(..., backend)` never changes thread-local selection;
- shape validation happens before an adapter is invoked;
- an adapter finishes writing its output before returning;
- autograd values, seeds, and gradients remain on their graph tensor backend;
- Adam requires parameters, gradients, and moments on one backend.

The registry is fixed at build time. This avoids mutable global registration,
duplicate-name rules, adapter-lifetime races, and an unstable public C++
vtable. It is an extension point for built-in backends, not yet a binary plugin
system.

For Metal, `execution_backend_available(...)` confirms the device, command
queue, and persistent-storage runtime. Matmul, Adam, and neural kernels compile
pipeline states lazily on first use. Neural functions share one lazily compiled
source library but cache their pipeline construction independently. A neural
source-compilation failure affects the neural capabilities, while storage,
matmul, and Adam remain independent; deterministic compilation and pipeline
failures are cached and rethrown.

For CUDA, availability requires both a CUDA-enabled build and at least one
device visible through the CUDA runtime. The default build and all standard
wheels instead compile a recognized unavailable stub: querying
`execution_backend_available(ExecutionBackend::Cuda)` or
`backend_available("cuda")` returns false, while explicitly selecting CUDA
raises the ordinary backend-unavailable error. The stable name and numeric ABI
value therefore do not depend on how a particular binary was built.

For TPU, availability requires a TPU-enabled Linux x86-64 build, a compatible
PJRT table exported by `libtpu.so`, and at least one addressable device. The
loader checks `RIFTCO_TRANSFORMER_TPU_LIBRARY`, then `TPU_LIBRARY_PATH`, then
the system loader path for `libtpu.so`. Initialization is attempted once and a
failure is cached. Default builds and standard wheels compile the same stable
`tpu` identity to an unavailable stub.

Native callers can pair `execution_backend_available(...)` with
`execution_backend_unavailability_reason(...)` to distinguish a backend that
was not compiled from a TPU loader, PJRT compatibility, or device-discovery
failure. The returned view is process-lifetime state and is empty for available
or unknown backend values. Selection errors and C API error messages include
the same diagnostic automatically.

`ScopedExecutionBackend` objects must be properly nested and destroyed on the
same thread that constructed them. The type is non-copyable and non-movable,
but code that transfers a pointer to the object must still honor that lifetime
rule.

### Adding another built-in backend

1. Add a stable `ExecutionBackend` identity without renumbering existing
   values.
2. Implement `BackendAdapter` in a platform-specific source file.
3. Provide an unavailable stub where the technology cannot be built.
4. Add exactly one entry to `find_backend_adapter()` in
   `backend/registry.cpp`.
5. Wire the implementation/stub choice in CMake.
6. Add CPU-reference parity, transfer, unavailable, autograd, and optimizer
   tests.
7. If the backend is public through C or Python, append stable mappings there;
   never derive their numeric ABI values from a C++ registry position.

For a true third-party binary plugin, add a separately versioned C
function-table ABI. Do not expose `BackendAdapter` itself: C++ vtables, standard
library types, and compiler exception conventions are not a stable plugin ABI.

On Apple, the real adapter is enabled by default. Its deterministic stub path
can be tested explicitly:

```bash
cmake -S . -B build/stub \
  -DRIFTCO_TRANSFORMER_ENABLE_METAL=OFF
cmake --build build/stub
ctest --test-dir build/stub --output-on-failure
```

CUDA is disabled by default on every platform so ordinary builds and release
wheels do not acquire a CUDA runtime dependency. Build it explicitly with:

```bash
cmake -S . -B build/cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DRIFTCO_TRANSFORMER_ENABLE_CUDA=ON
cmake --build build/cuda
ctest --test-dir build/cuda --output-on-failure
```

This path requires CUDA Toolkit 12 or newer at configure time and a compatible
NVIDIA driver and GPU at runtime. CUDA and the project's sanitizer option
cannot currently be enabled in the same build. An installed CUDA-enabled CMake
package records its CUDA Toolkit dependency for downstream CMake consumers.
To build the Python package from source with the same option, pass it through
the build backend:

```bash
CMAKE_ARGS="-DRIFTCO_TRANSFORMER_ENABLE_CUDA=ON" \
  python3 -m pip install .
```

TPU is also disabled by default. Its source adapter compiles without linking
`libtpu`, then loads the runtime dynamically so a TPU-enabled binary still has
clean unavailable behavior off-device. Configure it only on Linux x86-64:

```bash
export RIFTCO_TRANSFORMER_TPU_LIBRARY=/absolute/path/to/libtpu.so
cmake --preset tpu-release
cmake --build --preset tpu-release
ctest --preset tpu-release
```

`RIFTCO_TRANSFORMER_TPU_LIBRARY` is the clearest way to select the runtime; the
loader then falls back to `TPU_LIBRARY_PATH` and `libtpu.so`. Google distributes
`libtpu` separately for Cloud TPU environments. It is an unavoidable external
runtime for real TPU execution and is never bundled in this repository or its
standard wheels. On a real device, configure the preset with
`-DRIFTCO_TRANSFORMER_TEST_REQUIRE_TPU=ON` so missing hardware fails the test
run. The ordinary TPU CI job intentionally verifies only compilation and the
no-runtime/no-device contract plus fake-PJRT happy paths. The fake checks the
project's PJRT calls, matmul/attention parity, and the generated
quantized-linear contract for both scale encodings. It emulates the resulting
math; it is not a StableHLO compiler or a substitute for `libtpu` on real
hardware.

## What the CUDA slice does

Each CUDA tensor owns a `cudaMallocManaged` allocation for its lifetime. The
same pointer is host-visible and device-addressable, so existing tensor,
artifact, autograd, optimizer, and cache contracts do not need a CUDA-specific
public representation. Quantized weights use separate managed allocations for
packed codes and either legacy FP32 or double-quantized scale metadata:

```text
persistent managed tensor or packed NF4 inputs
          ↓
CUDA NN, matmul, quantized-linear, attention, or Adam-update kernel family
          ↓
device synchronization and error check
          ↓
persistent managed outputs
```

The matmul kernel uses a grid-stride loop and produces one output element per
thread visit. Attention is separated into materialized causal, memory-linear
Flash causal, and paged-decode modules. Their backward requests also dispatch
through the captured tensor backend.

Quantized-linear kernels reconstruct either scale encoding and decode each NF4
nibble while accumulating forward or input-gradient products. They do not
allocate or retain a full FP32 base-weight matrix. This makes QLoRA a packed
execution path, rather than merely loading a quantized checkpoint and expanding
it before training.

The CUDA NN module covers layout, elementwise/GELU, axis reductions,
softmax/causal softmax, embedding gather/scatter, LayerNorm, and fused
cross-entropy. Its optimizer module writes Adam's next parameter and moment
buffers with double intermediates and rejects a whole update batch before live
state is committed if any candidate is non-finite. Complete model forward,
backward, Adam, Full fine-tuning, LoRA, QLoRA, exhaustive
train/validation/test evaluation, artifact capture, and incremental serving
retain CUDA tensor identity. Host and device access can migrate managed pages,
and the framework synchronizes each CUDA kernel sequence before returning.
Autograd graph traversal and Adam's global gradient norm remain host control
flow. There are no CUDA streams, broad graph fusion, cuBLAS/cuDNN integration,
device-resident optimizer reductions, or multi-GPU selection. Benchmark the
exact workload; do not infer a broad speedup merely from `backend="cuda"`.
Actual NVIDIA-device validation remains an acceptance run outside the macOS
development host.

## What the TPU slice does

TPU tensor storage keeps the public host-readable representation. TPU
quantized-weight storage separately retains packed U8 codes and either FP32 or
double-quantized scale metadata. The runtime specializes StableHLO programs
for quantized-linear forward/input backward, batched matmul, materialized
attention forward/backward, and paged decode, compiles them through the PJRT C
API, and caches shape- and operation-specific loaded executables:

```text
host-mirrored tensor or packed NF4 inputs
        ↓ upload
one-device PJRT execution of StableHLO
        ↓ download every result and validate
atomic commit to host-mirrored outputs
```

Output is copied into the tensor only after execution and download succeed.
The runtime validates the PJRT major/minor and every function-table entry it
uses, initializes once, serializes execution, and supports one process with one
addressable execution device. The PJRT client, executable cache, and loaded
`libtpu` mapping intentionally have process lifetime: this avoids calling PJRT
after `libtpu` has torn down its own process-global state during exit. The OS
reclaims them. It does not implement multi-host coordination,
SPMD partitioning, persistent device-resident model state, asynchronous graph
scheduling, or mixing with another framework's `libtpu` ownership.

Layout, elementwise, normalization, loss, Flash attention, and Adam requests
execute through audited host reference paths. Quantized linear reconstructs
the selected scale encoding and dequantizes packed codes inside its StableHLO
program; the model retains no persistent FP32 base matrix. Matmul—including
the two matmuls in its autograd rule—plus materialized attention and its VJPs
and paged decode also use PJRT. This makes complete Full, LoRA, and QLoRA
workflows testable with `backend="tpu"`, but host-mirrored tensors and frequent
transfer make it an educational integration rather than a performance-ready
TPU stack. TPU Flash stays on the reference path because a naive StableHLO
score matrix would break the API's linear-storage Flash contract. The source
and generated StableHLO are checked, but actual Cloud TPU execution remains a
pending hardware acceptance run.

## What the Metal slice does

Each Metal tensor owns one `MTLResourceStorageModeShared` buffer for its
lifetime. A normal Metal matmul therefore has no staging copies:

```text
persistent input MTLBuffers
          ↓ bind
batched Metal matmul kernel
          ↓ write
persistent output MTLBuffer
          ↓
wait for command completion
```

The kernel grid is `[columns, rows, batch]`; one GPU thread calculates one
output element. Forward matrix multiplication and the two matrix
multiplications used by its backward rule all follow the tensor backend.
The explicit execution-override overload stages only when its requested
implementation differs from storage—for example, asking Metal to execute a
comparison on CPU tensors.

Adam uses its own Metal kernel. It keeps parameters, gradients, first moments,
second moments, and out-of-place candidates in persistent buffers. One thread
per scalar fuses clipping, both moment updates, bias correction, and the
parameter update. Every parameter tensor is encoded into one command buffer
and synchronized once. A shared flag requests a wide retry when the precise
float operation sequence encounters subnormal, non-finite, or
cancellation-sensitive arithmetic; candidates are committed only after the
entire batch succeeds.

The matmul and Adam shader sources compile into separate pipeline states on
first use, so an Adam-pipeline failure cannot disable storage or matmul. Neural
pipelines are likewise cached per operation. When the Adam kernel requests a
retry, the portable `double` reference rewrites the same shared, out-of-place
Metal candidate buffers. Ordinary batches incur no duplicate host Adam pass.
Tensor backend identity and the atomic commit boundary remain unchanged.

The neural runtime routes these operations through Metal:

| Capability | Forward work | Backward work |
| --- | --- | --- |
| Layout | copy, permutation, broadcast | inverse permutation and `sum_to_shape` |
| Elementwise | unary/binary math, scalar scale, erf-form GELU | routed arithmetic and fused GELU derivative |
| Reduction | axis sum and mean | broadcast and shape-reduction rules |
| Softmax | arbitrary-axis stable softmax; scale + causal mask + softmax | softmax and causal-softmax vector-Jacobian products |
| Indexing | embedding row gather | repeated-row scatter-add |
| LayerNorm | normalized output plus saved mean/inverse deviation | input, scale, and bias gradients |
| Loss | stable mean cross-entropy plus saved base gradient | backend scale and accumulation |
| Attention | materialized probabilities/context; Flash context; paged decode context | materialized probability/context VJPs; Flash recomputing VJP |

“Fused” is operation-specific here. GELU evaluates its whole formula in one
kernel, causal softmax combines scaling, masking, and normalization, and
cross-entropy produces loss plus its saved base gradient together. The runtime
does not fuse arbitrary adjacent elementwise graph nodes into a generated
kernel.

The CPU GELU reference calls `std::erf`. Metal Shading Language does not expose
`erf` on every supported SDK target, so the Metal kernel uses an internal
float approximation with about `1.5e-7` maximum absolute error. This keeps the
erf-form equation and float32 parity without claiming bitwise equality.

Full-sequence attention has two exact backend requests. The default
materialized path returns probabilities and context and retains
`[batch, heads, time, time]` probabilities for its VJPs. The opt-in Flash path
returns context through a memory-linear traversal and saves only
`[batch, heads, time]` row maxima and exponential sums. Its backward kernels
recompute scores and probabilities from `Q`, `K`, and those row statistics
instead of allocating a global quadratic probability buffer. CPU supplies the
readable tile-8 reference implementation; Metal uses tile-8 GPU kernels and
CUDA uses cooperating thread blocks. TPU runs materialized attention through
StableHLO but keeps Flash on the CPU reference implementation until a
genuinely memory-linear StableHLO program can preserve the same storage
contract. The explicit
probability-returning diagnostic always uses the materialized path.

Metal Flash scratch usage grows with head width. In float elements, forward
uses `32*d_h + 96`, while the largest backward phase uses
`48*d_h + 128`, plus any static pipeline storage. The runtime preflights the
forward and every backward pipeline against
`maxThreadgroupMemoryLength` before dispatching the forward kernel. A head
width unsupported by that device therefore produces an immediate,
descriptive error; callers can use more heads or select materialized
attention. The CPU Flash reference has no GPU threadgroup-memory ceiling.

Autograd graph construction, reverse-topological traversal, and gradient-node
bookkeeping remain host control flow. Local numerical backward rules dispatch
through the same backend capabilities. Metal and CUDA attention graphs launch
their backend VJP kernels. TPU matmul and materialized-attention gradients
dispatch through PJRT, while TPU Flash and the remaining VJPs use host
reference paths over the TPU tensor's host mirror.
Adam's global clipping norm remains an overflow-safe host reduction over
host-visible storage. Metal then uses its fused update with a whole-batch wide
reference retry when necessary; CUDA uses its double-intermediate native
candidate-state kernel, and TPU uses the portable Adam reference path.

Adam can store each parameter's first and second moments contiguously or as
fixed-size tensor pages. The paged form submits one bounded page at a time to
the same backend update contract. It still owns two FP32 moment values per
trainable scalar; its benefit is bounded allocation and update granularity,
not a smaller total moment payload. CUDA pages use managed allocations, but
there is no explicit eviction, spill budget, disk paging, prefetch policy, or
general OS page-fault manager.

The runtime remains deliberately synchronous. Metal finishes its command
buffer, CUDA synchronizes each kernel sequence, and TPU awaits PJRT execution
and download before the Adapter call returns. There are no
asynchronous graph streams or scheduling of several operations into one
submission. Small workloads may therefore
be slower on Metal because launch and synchronization costs dominate. CUDA
kernels have the same per-operation synchronization boundary, and managed-page
migration can dominate small or host-control-heavy CUDA workloads. TPU transfer
and shape-specialized compilation can dominate its small workloads. CPU,
Metal, CUDA, and TPU matmul use different floating-point implementations and
reduction orders, so the contract is numerical parity within
operation-appropriate tolerances—not bitwise equality.

## Incremental attention

The inference-only decoder path issues one paged-decode-attention operation per
transformer layer and token. Query and output use `[1,H,1,d_h]`; the backend
request views them as `[H,d_h]`. Key and value pools use
`[physical_page,H,page_offset,d_h]`, and a request-local table maps each
logical page to a fixed-width physical page ID.

The CPU reference loops over the logical sequence and resolves each K/V
address through that table. Metal and CUDA run backend-owned probability and
context kernels while reading the page layout. TPU stages the validated paged
request through a shape-specialized StableHLO program. No path gathers the
complete K/V sequence into a public
contiguous tensor. This operation is detached and has no backward rule;
full-sequence materialized or Flash causal attention remains the
training/autograd operation. Current serving prefill calls this same paged
operation one token at a time, so the full-sequence Flash selector does not
affect serving.

The serving-level `KeyValueCacheFactory` separates allocation policy from the
model. `PagedKvCachePool` leases pages from shared per-layer pools;
`ContiguousKvCacheFactory` gives a request one fixed maximum-context page as a
reference strategy. Both reach the same validated backend request. See
[SERVING.md](SERVING.md) for page ownership, decode sessions, and context
rollover.

## Stable C ABI

The language-neutral header is:

```text
include/riftco_transformer/c_api.h
```

It builds as:

```text
libriftco_transformer_c.dylib   macOS
libriftco_transformer_c.so      Linux
riftco_transformer_c.dll        Windows
```

The ABI uses:

- opaque tokenizer, context, tensor, model, decode-session, parameter-list,
  variable, and Adam handles;
- fixed-width integers for status codes, backends, shapes, and counts;
- fixed-layout tokenizer-options, model-config, decode-session-options,
  Adam-options, and step-stat structures beginning with their caller-supplied
  byte size;
- an explicit `cdecl` calling convention on Windows;
- explicit create/release ownership;
- host-copy functions instead of exposing storage pointers;
- thread-local error details;
- status returns that catch every C++ exception.

No `std::vector`, `std::string`, C++ class layout, allocator, or exception
crosses the binary boundary. Tensor handles are immutable. A context selects
storage at construction; the tensor then owns that storage and reports its
intrinsic backend. The public context handle may be released without
invalidating already-created tensors.

Model-derived handles share native model ownership. An already-created decode
session, parameter list, variable, or Adam optimizer therefore remains valid
if the caller releases the public model handle. Adam's copied parameter list
also owns `ParameterHandle` entries; its internal raw `Parameter*` views refer
to that retained canonical state. `rt_model_to` rejects a transfer while a
decode session, variable graph, or optimizer is alive, preventing backend
drift under caches, saved graphs, and optimizer moments.

The C training graph is single-use. A successful `rt_variable_backward`
consumes the shared graph, and a successful Adam step advances the model's
parameter epoch so any older, unconsumed graph is rejected. This makes stale
graph mistakes explicit while preserving the normal
forward → loss → backward → step transaction.

The raw C handles deliberately contain no hidden operation locks. Callers must
externally synchronize calls involving the same handle or handles derived from
the same model, and must never release a handle while a call using it is in
flight. Python uses one shared reentrant lock for a model and its derived
objects, which is why its concurrent `close()` behavior is stronger than the
raw C contract.

ABI version `0x00020004` represents the current version 2.4: the upper 16 bits
are the major and the lower 16 bits are the minor. Version 2.0 was the
intentional breaking namespace reset. It exports only the `rt_` function/type
prefix and `RT_` constants; no legacy symbol-prefix aliases are provided.
Version 2.1 additively appends the stable `RT_BACKEND_CUDA = 2` value without
renumbering CPU (`0`) or Metal (`1`). Version 2.2 additively appends
`RT_BACKEND_TPU = 3` without changing those values. Version 2.3 additively
exposes NF4 model conversion and exact packed-memory
statistics without changing existing structures or numeric values. Version
2.4 extends `rt_adam_options` with bounded-page moment-state selection while
continuing to accept the original 2.3 structure prefix. Future major changes may
break callers, while a minor change may only add compatible API. Clients accept
the same major and an equal or newer minor. Published status and backend
integer values must never be renumbered. CMake checks the public header's
major/minor against the shared-library version during configuration so their
metadata cannot drift.

The current 2.x surface includes immutable byte/BPE tokenizer handles and binary-safe
size-query APIs; exact tokenizer reconstruction; deterministic named-parameter
inspection and transactional float32 copy/load; LoRA attachment, adapter-only
optimization, NF4/double-scale conversion and memory accounting, one-way QLoRA
export, and contiguous or bounded-page Adam moment storage;
contiguous and paged decode sessions;
materialized or Flash full-sequence attention; and disabled or block-level
activation checkpointing. A live decode session pins its model backend and
parameter epoch. Parameter loading, transfer, LoRA lifecycle changes, and Adam
updates are rejected while derived state would make the operation unsafe.
Attention and checkpointing are runtime policies: they do not change weights,
the persisted artifact schema, or the separate decode-session cache policy.

Tokenizer selection is independent from execution-backend selection. A native
factory maps the stable method value to a strategy behind one tokenizer
interface; the C and Python layers only hold the opaque facade. Another
built-in tokenizer method can therefore extend that factory and ABI mapping
without changing model, autograd, backend Adapter, or optimizer contracts.
This is a source-level built-in extension point, not a third-party tokenizer
plugin ABI.

`rt_tokenizer_options_init`, `rt_transformer_config_init`,
`rt_lora_config_init`, `rt_decode_session_options_init`, and
`rt_adam_options_init` receive the caller's actual structure size. Their
fields use explicit fixed-width layouts, including reserved words rather than
ambiguous tail padding. Future additive minors can inspect the supplied size
without overwriting an older caller's smaller allocation.

The shared library also carries ABI major version `2` in its platform library
metadata. CMake install exports `riftco_transformer::c_api` and
`riftco_transformer::library`; the private adapter interface is not exported.

## Python client

The default runtime-dependency-free client in `riftco_transformer.native` uses
`ctypes`; the package root re-exports this public low-level API without
installing a legacy package-name alias. A released platform wheel bundles this
same C ABI implementation rather than substituting a separate Python numerical
path. Only an explicitly TPU-enabled source build adds the external `libtpu`
runtime:

```python
from riftco_transformer import Context, Tensor, backend_available

backend = (
    "tpu"
    if backend_available("tpu")
    else "cuda"
    if backend_available("cuda")
    else "metal"
    if backend_available("metal")
    else "cpu"
)

with Context(backend) as context:
    with Tensor.from_data(
        context,
        (2, 3),
        [1, 2, 3, 4, 5, 6],
    ) as left:
        with Tensor.from_data(
            context,
            (3, 2),
            [7, 8, 9, 10, 11, 12],
        ) as right:
            with left @ right as product:
                print(product.shape)
                print(product.tolist())
```

Output:

```text
(2, 2)
[58.0, 64.0, 139.0, 154.0]
```

The same package exposes low-level Python wrappers around native model,
autograd, loss, and Adam objects:

```python
from riftco_transformer import (
    Adam,
    DecoderOnlyTransformer,
    Tokenizer,
    TransformerConfig,
    cross_entropy,
)

corpus = "hello hello hello"
tokenizer = Tokenizer(
    corpus,
    method="bpe",
    vocabulary_size=272,
    minimum_pair_frequency=2,
)
encoded = tokenizer.encode(corpus)
tokens = [encoded[:-1]]
targets = [encoded[1:]]

config = TransformerConfig(
    vocabulary_size=tokenizer.vocab_size,
    maximum_context=len(tokens[0]),
    model_width=64,
    head_count=4,
    block_count=2,
    feed_forward_width=256,
    random_seed=7,
)

model = DecoderOnlyTransformer(
    config,
    attention="flash",
    activation_checkpointing="block",
).to("cuda")
optimizer = Adam(model.parameters())

loss = cross_entropy(model(tokens), targets)
loss.backward()
stats = optimizer.step()

print(loss.item(), stats.gradient_norm)
```

`DecoderOnlyTransformer` defaults to `attention="materialized"`. The
`full_sequence_attention` property reports the current policy, and
`set_full_sequence_attention("flash")` changes future full-sequence forwards.
The same choice is available as `attention` on `PretrainingConfig` and
`PostTrainingConfig`. It is independent of
`to("cpu" | "metal" | "cuda" | "tpu")` and of the
serving session's paged/contiguous cache choice.

`activation_checkpointing` defaults to `"disabled"`. Select `"block"` to
retain block boundaries and recompute each block during backward. The
`activation_checkpointing` property and
`set_activation_checkpointing(...)` method expose the same policy, and both
training-stage configs accept it.

`Tokenizer` accepts a UTF-8 `str` or an arbitrary bytes-like corpus. Its
Strategy-style facade selects `"byte"` or `"bpe"` while the model continues to
consume the same `uint32` IDs. `Tokenizer(corpus)` preserves the original
corpus-derived byte behavior. BPE starts with all 256 bytes and learns
deterministically ordered pair pieces, so it can encode bytes absent from its
training corpus. The common `vocabulary` property exposes every token as
bytes; `vocabulary_bytes` remains available in byte mode.

`encode()`/`decode()` are strict UTF-8 conveniences, while
`encode_bytes()`/`decode_bytes()` preserve arbitrary bytes. Use `"cpu"` when
no optional accelerator backend is available. Tokens and targets may be one flat row
or a rectangular Python batch and are checked before crossing as `uint32`.
Tokenizer
`Tokenizer.from_state(...)` restores an ordered byte vocabulary or ordered BPE
merge table without retraining. The higher-level `ModelBundle` stores that
state beside model configuration, named parameter shapes, float32 weights,
checksums, stage metadata, and parent lineage.
`Variable` exposes backend, shape, flat values, scalar `item()`, and
`backward()`. `ParameterList` exposes stable names and retains model state;
`Adam(model.parameters())` is therefore safe even though the temporary
parameter-list expression is not assigned separately.

### Installation and library discovery

After a release has been published to PyPI, a normal installation is:

```bash
python3 -m pip install riftco-transformer
python3 -c "from riftco_transformer import Context; print(Context().backend)"
```

Each wheel installs its native library under `riftco_transformer/.libs`:

```text
riftco_transformer/.libs/libriftco_transformer_c.so       Linux
riftco_transformer/.libs/libriftco_transformer_c.dylib    macOS
riftco_transformer/.libs/riftco_transformer_c.dll          Windows
```

The library contains the statically linked framework implementation behind the
stable C ABI. A standard installed wheel has no third-party runtime
dependencies; users of a matching wheel do not need CMake, a C++ compiler, a
system-wide native installation, or `RIFTCO_TRANSFORMER_LIBRARY`. Standard
wheels recognize `cuda` and `tpu` through ABI 2.4 but build their unavailable
stubs, so they do not require or silently load CUDA or `libtpu` runtimes.

The initial binary matrix provides Linux `x86_64` and `aarch64` wheels for
both glibc (`manylinux`) and musl (`musllinux`), macOS `x86_64` and `arm64`,
and Windows `AMD64` wheels. Every wheel includes CPU. The macOS builds also
include Metal, whose availability is still checked at runtime. Because
`ctypes` calls the language-neutral C ABI instead of CPython's extension ABI,
a build produces one `py3-none-<platform>` wheel for its platform and
architecture. Package metadata requires Python 3.10 or newer.

Install from a source checkout at the repository root with:

```bash
python3 -m pip install .
```

A default source install compiles the native C++20 target and its CUDA and TPU
stubs. It therefore needs a supported compiler and platform SDK, but no CUDA
Toolkit or `libtpu`.
Set `CMAKE_ARGS="-DRIFTCO_TRANSFORMER_ENABLE_CUDA=ON"` only for an explicitly
CUDA-enabled source build; that build additionally requires CUDA Toolkit 12+
and a compatible NVIDIA driver/GPU. A TPU-enabled library is built with
`RIFTCO_TRANSFORMER_ENABLE_TPU=ON` on Linux x86-64 and discovers `libtpu.so` at
runtime as described above. The build backend and wheel builder are build-time
tools, not installed runtime dependencies.

For an in-tree native development cycle, the explicit build-and-test route
remains available:

```bash
cmake --preset debug
cmake --build --preset debug

PYTHONPATH=python \
RIFTCO_TRANSFORMER_LIBRARY=build/debug/libriftco_transformer_c.dylib \
python3 tests/python/test_python_binding.py
```

`RIFTCO_TRANSFORMER_LIBRARY` is an advanced override and takes precedence when it
is set. Without it, the loader searches the package-local `.libs` directory,
standard project build directories (release before debug), and then the system
library path. It recognizes the configuration postfixes emitted by
multi-config generators and deliberately skips local sanitizer builds, which
cannot be safely loaded into an arbitrary Python interpreter. The loader checks
C ABI compatibility before exposing the selected library.

### Release workflow

The repository's `.github/workflows/release.yml` builds, repairs where needed,
and tests the source distribution and all supported wheels in isolated Python
environments. A manual `workflow_dispatch` performs build and verification
without publishing. Pushing `v<version>` creates a GitHub Release only after
the artifacts pass. When the repository variable `PUBLISH_TO_PYPI` is `true`,
the tag path publishes to PyPI first and creates the GitHub Release after that
publication succeeds.

PyPI uses OIDC Trusted Publishing, not a long-lived API token. Configure its
publisher as project `riftco-transformer`, GitHub owner `quangng2000`, repository
`riftco-transformer`, workflow `release.yml`, and environment `pypi`. This
repository and its Python package are licensed under Apache-2.0. Keep
`PUBLISH_TO_PYPI` disabled until the Trusted Publisher is in place.

Python `Tokenizer`, `Context`, `Tensor`, `DecoderOnlyTransformer`,
`DecodeSession`, `ParameterList`, `Variable`, and `Adam` objects support
context managers and idempotent `close()`. The owning wrappers are
intentionally non-copyable: use ordinary Python references to share them.
Native operations and `close()` synchronize handle lifetimes, so a concurrent
close cannot free a handle during an in-flight call. Model-derived objects
share the model lock and native owner. Native tensors remain valid after their
context closes, and a decode session, loss, or optimizer remains valid after
its public model handle closes. Native failures raise `RiftcoTransformerError` with
both a stable status code and the copied thread-local diagnostic.

Construct a fresh logits/loss graph for every update. Calling `backward()`
twice, using an old graph after `step()`, or moving a model while variables or
an optimizer are alive produces an explicit native error.

## Verification

The backend tests compare:

- non-square CPU matmul with hand-calculated values;
- batched CPU/accelerator outputs;
- CPU/accelerator autograd forward values;
- CPU/accelerator left/right gradients;
- deep-copy and CPU/Metal transfer value semantics;
- backend-preserving tensor operations, autograd values, and gradients;
- CPU-reference and conditional-accelerator layout, elementwise, reduction, GELU,
  LayerNorm, softmax/causal-softmax, gather/scatter, cross-entropy, and
  materialized-causal-, Flash-causal-, and paged-decode-attention results;
- CPU-oracle packed quantized-linear forward/input backward for legacy FP32
  scales and double-quantized scales, plus packed-residency invariants and
  unavailable-backend rejection;
- routed forward results and vector-Jacobian products, including repeated-row
  scatter-add, materialized outputs, and Flash recomputing backward;
- backward behavior after changing the thread-local construction default;
- multi-step CPU/accelerator Adam parity, clipping, momentum tails, and
  contiguous-versus-paged state parity/accounting;
- extreme clipping below scalar `float` range, minimum-normal epsilon, and the
  unclipped minimum-normal boundary;
- fused-path and reference-retry counters, rounded cancellation, float-square
  overflow, and whole-batch retry behavior;
- mixed-device and post-construction optimizer-backend rejection;
- failed fused-update atomicity and retry behavior;
- typed unknown/unavailable errors and transactional selection;
- explicit-dispatch side-effect freedom;
- the forced Metal-stub path on an Apple build.

CUDA verification additionally covers its stable name and unavailable-stub
contract in ordinary builds. A CUDA-enabled NVIDIA system must exercise
CPU/CUDA transfer, NN-operation parity, matmul/autograd parity, both packed
quantized-linear scale encodings and their input VJPs, every attention
forward/VJP, paged decode, contiguous/paged Adam update parity, and full-model,
Full fine-tuning, LoRA, QLoRA, evaluation, and serving smoke paths. The
toolkit-only CI job proves compilation and no-device behavior; it cannot
establish kernel numerical parity without a visible NVIDIA GPU. That real-GPU
acceptance run was not available on the macOS development host.

TPU verification covers its additive ABI identity, unavailable-stub contract,
C ABI/Python recognition, TPU-runtime source compilation on Linux, and fake-PJRT
matmul, packed quantized-linear, and materialized/paged-attention execution. On
a Cloud TPU host it
additionally requires CPU/TPU transfer, batched matmul/autograd parity, both
packed quantized-linear scale encodings and their input VJPs, materialized
attention/VJP and paged-decode parity, and complete pretraining,
Full/LoRA/QLoRA, evaluation, and serving smoke paths. Fake PJRT is an API
emulator, not a StableHLO compiler or real-device acceptance test; actual
Cloud TPU execution was not available on the macOS development host.

On a real NVIDIA test host, make device absence a hard failure instead of a
skip:

```bash
cmake -S . -B build/cuda-gpu -G Ninja \
  -DRIFTCO_TRANSFORMER_ENABLE_CUDA=ON \
  -DRIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA=ON
cmake --build build/cuda-gpu
ctest --test-dir build/cuda-gpu --output-on-failure
```

The repository's hosted CUDA workflow is intentionally a compile and
no-device contract job. It does not replace this real-GPU run.

The equivalent hardware-required Cloud TPU gate is:

```bash
export RIFTCO_TRANSFORMER_TPU_LIBRARY=/absolute/path/to/libtpu.so
cmake --preset tpu-release \
  -DRIFTCO_TRANSFORMER_TEST_REQUIRE_TPU=ON
cmake --build --preset tpu-release
ctest --preset tpu-release
```

The hosted TPU workflow covers the compile/no-device boundary and a tests-only
fake-PJRT execution path. It does not replace this real-device run.

The pure C11 test verifies the ABI independently of C++, including tokenizer
option validation, method selection, token-piece and merge queries,
full-sequence attention selection, decode-session lifecycle, state
restoration, parameter metadata/transactional loading, canaries, binary round
trips, and error contracts. The Python tests cover byte compatibility,
deterministic BPE compression and unseen bytes, tokenizer and decode-session
lifecycle, full-sequence attention selection and backward, scalar/shape
access, zeros, error translation, CPU matmul, mixed-backend rejection,
conditional real-Metal parity, model-derived lifetime retention, graph
invalidation, incremental generation, and complete
text → BPE → model → loss → backward → Adam execution.

The intended validation matrix is:

| Configuration | Purpose |
| --- | --- |
| Debug with real Metal when available | Assertions, CPU reference tests, Metal kernel routing, and full-model training |
| Release | Optimized-build behavior and ABI/Python execution |
| ASan + UBSan | Host memory, lifetime, and undefined-behavior checks around the same interfaces |
| `RIFTCO_TRANSFORMER_ENABLE_METAL=OFF` | Deterministic recognized-but-unavailable stub behavior |
| Default CUDA-disabled build | Stable `cuda` identity with deterministic unavailable-stub behavior |
| CUDA Toolkit 12+ on an NVIDIA runner | Managed tensor/packed-weight transfer, GPU matmul and quantized-linear parity, paged Adam, and full-framework smoke coverage |
| Default TPU-disabled build | Stable `tpu` identity with deterministic unavailable-stub behavior |
| TPU-enabled Linux runner | Dynamic PJRT boundary, clean runtime-unavailable behavior, fake-PJRT matmul/quantized-linear parity, and quantized-linear source compilation |
| Google Cloud TPU hardware runner | PJRT matmul/quantized-linear parity plus full-framework functional smoke coverage |
| Installed-package consumer | Public C++ and C targets without private Adapter headers |

Metal, CUDA, and TPU comparisons use documented absolute/relative tolerances.
Different device math and reduction order make bitwise accelerator/CPU parity
an invalid acceptance criterion.
