# Execution Backends and Python ABI

The backend layer owns tensor storage and focused accelerated operations
without changing the transformer or autograd equations:

```text
Python stages/tokenizer/tensors/model/Adam ──ctypes──→ stable C ABI 1.8
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
                     matmul · softmax · indexing · LayerNorm
       cross-entropy · materialized/Flash/paged-decode attention · Adam
                                 │                 │
                                 ▼                 ▼
                         CPU reference math   Metal kernels over
                                              shared MTLBuffers
```

CPU remains the default and the readable correctness reference. On Apple
platforms with an available Metal device, tensors own persistent shared
`MTLBuffer` storage. The operations used by the training graph dispatch to real
Metal kernels: layout transforms, elementwise math, reductions, GELU,
LayerNorm, softmax, causal masking, embedding gather/scatter, cross-entropy,
materialized and tile-8 Flash causal attention and their vector-Jacobian
products, matmul, and Adam. `Variable::matmul` captures the input tensor
backend during the forward pass, so both matrix-gradient products use that
backend even if the calling thread's construction default changes before
`backward()`.

## Selecting a C++ backend

The training executable accepts:

```bash
./build/release/transformer_lab \
  --config configs/tiny.conf \
  --steps 20 \
  --backend cpu \
  --attention flash \
  --activation-checkpointing block
```

On a Metal-capable Mac:

```bash
./build/release/transformer_lab \
  --config configs/tiny.conf \
  --steps 20 \
  --backend metal \
  --attention flash \
  --activation-checkpointing block
```

`--attention materialized|flash` selects the full-sequence algorithm;
materialized is the default. It is independent from
`--backend cpu|metal`.
`--activation-checkpointing disabled|block` controls full-sequence autograd
retention independently of both selectors.

The public C++ default-selection interface is:

```cpp
#include "transformer_lab/core/backend.hpp"

using transformer_lab::ExecutionBackend;

if (transformer_lab::execution_backend_available(
        ExecutionBackend::Metal
    )) {
    const transformer_lab::ScopedExecutionBackend use_metal(
        ExecutionBackend::Metal
    );
    // Tensors and modules constructed in this scope default to Metal.
}
```

Existing tensors keep their intrinsic backend. Transfer them explicitly:

```cpp
Tensor on_metal = on_cpu.to(ExecutionBackend::Metal);
model.to(ExecutionBackend::Metal);
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
  softmax, indexing, normalization, loss, and attention requests;
- execute a validated transactional Adam update batch.

The Adapter normalizes CPU loops and Metal APIs. The selected
`ExecutionBackend` is the Strategy used by ordinary tensor operations.

| Backend value | Name | Availability | Selection behavior |
| --- | --- | --- | --- |
| `Cpu` | `cpu` | Always | Succeeds |
| `Metal` | `metal` | Runtime/build dependent | Succeeds or throws `runtime_error` |
| Unknown value | None | False | Throws `invalid_argument` |

The framework guarantees:

- no silent backend-identity or storage migration from Metal to CPU;
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
  -DTRANSFORMER_LAB_ENABLE_METAL=OFF
cmake --build build/stub
ctest --test-dir build/stub --output-on-failure
```

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
returns context, processes tile-8 query/key blocks, and saves only
`[batch, heads, time]` row maxima and exponential sums. Its backward kernels
recompute scores and probabilities from `Q`, `K`, and those row statistics
instead of allocating a global quadratic probability buffer. CPU supplies the
readable reference implementation; Metal uses tile-8 threadgroup-memory
forward and backward kernels. The explicit probability-returning diagnostic
always uses the materialized path.

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
through the same backend capabilities, so Metal graphs launch Metal VJP
kernels rather than copying their arithmetic to CPU. Adam's global clipping
norm remains an overflow-safe host reduction over host-visible shared storage
before the fused update.

The runtime remains deliberately synchronous and shared-memory based. Every
Adapter call completes its command buffer and checks any device status before
returning. There are no asynchronous streams, private GPU heaps, or scheduling
of several graph operations into one submission. Small workloads may therefore
be slower on Metal because launch and synchronization costs dominate. CPU and
Metal use different floating-point implementations and reduction orders, so
the contract is numerical parity within operation-appropriate tolerances—not
bitwise equality.

## Incremental attention

The inference-only decoder path issues one paged-decode-attention operation per
transformer layer and token. Query and output use `[1,H,1,d_h]`; the backend
request views them as `[H,d_h]`. Key and value pools use
`[physical_page,H,page_offset,d_h]`, and a request-local table maps each
logical page to a fixed-width physical page ID.

The CPU reference loops over the logical sequence and resolves each K/V
address through that table. Metal runs the corresponding probability and
context kernels while reading the same backend-resident page layout. Neither
backend gathers the complete K/V sequence into a public contiguous tensor.
This operation is detached and has no backward rule; full-sequence
materialized or Flash causal attention remains the training/autograd
operation. Current serving prefill calls this same paged operation one token
at a time, so the full-sequence Flash selector does not affect serving.

The serving-level `KeyValueCacheFactory` separates allocation policy from the
model. `PagedKvCachePool` leases pages from shared per-layer pools;
`ContiguousKvCacheFactory` gives a request one fixed maximum-context page as a
reference strategy. Both reach the same validated backend request. See
[SERVING.md](SERVING.md) for page ownership, decode sessions, and context
rollover.

## Stable C ABI

The language-neutral header is:

```text
include/transformer_lab/c_api.h
```

It builds as:

```text
libtransformer_lab_c.dylib   macOS
libtransformer_lab_c.so      Linux
transformer_lab_c.dll        Windows
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
to that retained canonical state. `tl_model_to` rejects a transfer while a
decode session, variable graph, or optimizer is alive, preventing backend
drift under caches, saved graphs, and optimizer moments.

The C training graph is single-use. A successful `tl_variable_backward`
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

ABI version `0x00010008` represents version 1.8: the upper 16 bits are the
major and the lower 16 bits are the minor. A major change may break callers;
a minor change may only add compatible API. Clients accept the same major and
an equal or newer minor. Published status and backend integer values must
never be renumbered. CMake checks the public header's major/minor against the
shared-library version during configuration so their metadata cannot drift.

ABI 1.2 added the immutable `tl_tokenizer` handle and binary-safe
encode/decode size-query APIs. ABI 1.3 adds a size-versioned
`tl_tokenizer_options`, stable byte/BPE method values, selectable construction,
method inspection, and a per-token byte-piece query. The original
`tl_tokenizer_create` remains byte-compatible.

ABI 1.4 adds exact byte-vocabulary and ordered-BPE-merge reconstruction,
ordered BPE-merge inspection, parameter rank/shape/element-count inspection,
and deterministic flattened float32 parameter copy/load. Parameter loading is
transactional, clears gradients, and is rejected while a graph or optimizer
derived from the model is alive. These additions are the native persistence
boundary used by Python `ModelBundle`.

ABI 1.5 adds the size-versioned LoRA configuration, stable projection-target
mask values, adapter attachment and inspection, adapter-only parameter-list
access for Adam, and one-way merge into base weights. Base parameter names and
ordering remain unchanged while an adapter is attached. Merge is rejected
while a variable graph, optimizer, or adapter parameter-list handle is alive.

ABI 1.6 adds a size-versioned decode-session configuration, stable contiguous
and paged cache-kind values, and the opaque `tl_decode_session` lifecycle. A
session appends one token per `tl_decode_session_step` and returns the logits
for the following token. Size, capacity, cache kind, block size, reset, and
release are explicit. A live session pins its model backend and parameter
epoch; transfer, parameter loading, LoRA lifecycle changes, and Adam updates
are rejected until all sessions derived from that model are released.

ABI 1.7 adds stable materialized/Flash full-sequence-attention values plus
`tl_model_set_full_sequence_attention` and
`tl_model_full_sequence_attention`. This runtime policy changes future
full-sequence model forwards without changing weights, artifact schemas, or
the separate decode-session cache policy.

ABI 1.8 adds stable disabled/transformer-block activation-checkpointing values
plus `tl_model_set_activation_checkpointing` and
`tl_model_activation_checkpointing`. This is another future-forward runtime
policy, so `tl_transformer_config` and the persisted model-state schema remain
unchanged. Stage pipelines record the selection in descriptive training
metadata. Decode sessions ignore it.

Tokenizer selection is independent from execution-backend selection. A native
factory maps the stable method value to a strategy behind one tokenizer
interface; the C and Python layers only hold the opaque facade. Another
built-in tokenizer method can therefore extend that factory and ABI mapping
without changing model, autograd, backend Adapter, or optimizer contracts.
This is a source-level built-in extension point, not a third-party tokenizer
plugin ABI.

`tl_tokenizer_options_init`, `tl_transformer_config_init`,
`tl_lora_config_init`, `tl_decode_session_options_init`, and
`tl_adam_options_init` receive the caller's actual structure size. Their
fields use explicit fixed-width layouts, including reserved words rather than
ambiguous tail padding. Future additive minors can inspect the supplied size
without overwriting an older caller's smaller allocation.

The shared library also carries ABI major version `1` in its platform library
metadata. CMake install exports `transformer_lab::c_api` and
`transformer_lab::library`; the private adapter interface is not exported.

## Python client

The runtime-dependency-free client in `transformer_lab.native` uses `ctypes`;
the package root re-exports this low-level API for compatibility. A released
platform wheel bundles this same C ABI implementation rather than substituting
a separate Python numerical path:

```python
from transformer_lab import Context, Tensor, backend_available

backend = "metal" if backend_available("metal") else "cpu"

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

The same package exposes the high-level native training objects:

```python
from transformer_lab import (
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
).to("metal")
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
`PostTrainingConfig`. It is independent of `to("cpu" | "metal")` and of the
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
`encode_bytes()`/`decode_bytes()` preserve arbitrary bytes. Use `"cpu"` on a
system without Metal. Tokens and targets may be one flat row or a rectangular
Python batch and are checked before crossing as `uint32`. Tokenizer
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
python3 -m pip install transformer-lab
python3 -c "from transformer_lab import Context; print(Context().backend)"
```

Each wheel installs its native library under `transformer_lab/.libs`:

```text
transformer_lab/.libs/libtransformer_lab_c.so       Linux
transformer_lab/.libs/libtransformer_lab_c.dylib    macOS
transformer_lab/.libs/transformer_lab_c.dll          Windows
```

The library contains the statically linked framework implementation behind the
stable C ABI. The installed package has no third-party runtime dependencies;
users of a matching wheel do not need CMake, a C++ compiler, a system-wide
native installation, or `TRANSFORMER_LAB_LIBRARY`.

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

A source install compiles the native C++20 target and therefore needs a
supported compiler and platform SDK. The build backend and wheel builder are
build-time tools, not installed runtime dependencies.

For an in-tree native development cycle, the explicit build-and-test route
remains available:

```bash
cmake --preset debug
cmake --build --preset debug

PYTHONPATH=python \
TRANSFORMER_LAB_LIBRARY=build/debug/libtransformer_lab_c.dylib \
python3 tests/python/test_python_binding.py
```

`TRANSFORMER_LAB_LIBRARY` is an advanced override and takes precedence when it
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
publisher as project `transformer-lab`, GitHub owner `quangng2000`, repository
`transformer-lab`, workflow `release.yml`, and environment `pypi`. This
repository currently has no declared software license; one must be chosen and
recorded in both the repository and package metadata before publishing for
third-party reuse. Keep `PUBLISH_TO_PYPI` disabled until both that license and
the Trusted Publisher are in place.

Python `Tokenizer`, `Context`, `Tensor`, `DecoderOnlyTransformer`,
`DecodeSession`, `ParameterList`, `Variable`, and `Adam` objects support
context managers and idempotent `close()`. The owning wrappers are
intentionally non-copyable: use ordinary Python references to share them.
Native operations and `close()` synchronize handle lifetimes, so a concurrent
close cannot free a handle during an in-flight call. Model-derived objects
share the model lock and native owner. Native tensors remain valid after their
context closes, and a decode session, loss, or optimizer remains valid after
its public model handle closes. Native failures raise `TensorLabError` with
both a stable status code and the copied thread-local diagnostic.

Construct a fresh logits/loss graph for every update. Calling `backward()`
twice, using an old graph after `step()`, or moving a model while variables or
an optimizer are alive produces an explicit native error.

## Verification

The backend tests compare:

- non-square CPU matmul with hand-calculated values;
- batched CPU and Metal outputs;
- CPU and Metal autograd forward values;
- CPU and Metal left/right gradients;
- deep-copy and CPU/Metal transfer value semantics;
- backend-preserving tensor operations, autograd values, and gradients;
- CPU-reference and real-Metal layout, elementwise, reduction, GELU,
  LayerNorm, softmax/causal-softmax, gather/scatter, cross-entropy, and
  materialized-causal-, Flash-causal-, and paged-decode-attention results;
- routed forward results and vector-Jacobian products, including repeated-row
  scatter-add, materialized outputs, and Flash recomputing backward;
- backward behavior after changing the thread-local construction default;
- multi-step CPU/Metal fused Adam parity, clipping, and momentum tails;
- extreme clipping below scalar `float` range, minimum-normal epsilon, and the
  unclipped minimum-normal boundary;
- fused-path and reference-retry counters, rounded cancellation, float-square
  overflow, and whole-batch retry behavior;
- mixed-device and post-construction optimizer-backend rejection;
- failed fused-update atomicity and retry behavior;
- typed unknown/unavailable errors and transactional selection;
- explicit-dispatch side-effect freedom;
- the forced Metal-stub path on an Apple build.

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
| `TRANSFORMER_LAB_ENABLE_METAL=OFF` | Deterministic recognized-but-unavailable stub behavior |
| Installed-package consumer | Public C++ and C targets without private Adapter headers |

Metal comparisons use documented absolute/relative tolerances. Different
device math and reduction order make bitwise CPU/Metal parity an invalid
acceptance criterion.
