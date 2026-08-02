# Troubleshooting

Start with a Release build and its complete tests. This distinguishes a source
or platform problem from a workload configuration problem:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

Each section below gives a symptom, likely cause, corrective action, and a
small verification command.

## Configure and build

### CMake cannot find Ninja or a compiler

**Symptom:** preset configuration fails before generating the build.

**Cause:** the provided presets use Ninja and require a C++20 compiler.

**Fix:** install Ninja and a current AppleClang, Clang, or GCC toolchain; or
configure manually with another generator.

```bash
cmake --version
c++ --version
ninja --version
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release
```

**Verify:** `cmake --build build/local` completes.

### Metal was enabled on a non-Apple platform

**Symptom:** configure reports that `RIFTCO_TRANSFORMER_ENABLE_METAL` requires
an Apple platform.

**Cause:** Metal was explicitly enabled where Apple frameworks are unavailable.

**Fix:** remove the override or configure with
`-DRIFTCO_TRANSFORMER_ENABLE_METAL=OFF`.

**Verify:** inspect `RIFTCO_TRANSFORMER_ENABLE_METAL` in the generated
`CMakeCache.txt` and rebuild.

### CUDA compiler or toolkit is missing

**Symptom:** configure reports that CUDA needs an NVIDIA compiler or cannot
find CUDA Toolkit 12.

**Cause:** CUDA was enabled without Toolkit 12+, or the toolkit is not visible
to CMake.

**Fix:** install/select the toolkit and configure with the CUDA preset. Do not
combine the repository sanitizer option with CUDA.

```bash
nvcc --version
cmake --preset cuda-release
cmake --build --preset cuda-release
```

**Verify:** `ctest --preset cuda-release` runs on the CUDA host. A successful
source build still requires a compatible driver and visible NVIDIA GPU at
runtime.

### TPU configure is rejected

**Symptom:** configure says TPU requires Linux x86-64.

**Cause:** the experimental PJRT adapter has that explicit platform boundary.

**Fix:** build the default TPU stub locally, or configure `tpu-release` on a
Linux x86-64 Cloud TPU host.

```bash
export RIFTCO_TRANSFORMER_TPU_LIBRARY=/absolute/path/to/libtpu.so
cmake --preset tpu-release
cmake --build --preset tpu-release
```

**Verify:** run `ctest --preset tpu-release`. Fake-PJRT tests validate the
loader contract off-device; they are not real TPU hardware validation.

## Native backend selection

### `requested backend is unavailable`

**Symptom:** the CLI or Python raises a backend-unavailable error.

**Cause:** the binary contains a recognized stub, the runtime/device is absent,
or backend initialization failed. Explicit selections do not fall back.

**Fix:** query availability before constructing tensors or models, and use the
reported reason in C++:

```cpp
using riftco_transformer::ExecutionBackend;
if (!riftco_transformer::execution_backend_available(
        ExecutionBackend::Cuda)) {
    std::cerr << riftco_transformer::
        execution_backend_unavailability_reason(ExecutionBackend::Cuda);
}
```

Python can use:

```python
from riftco_transformer import backend_available
print({name: backend_available(name) for name in ("cpu", "metal", "cuda", "tpu")})
```

**Verify:** select `cpu`, or rebuild with the desired backend and repeat the
availability query.

### Mixed-backend operation is rejected

**Symptom:** matmul, autograd, Adam, or a neural operation reports mismatched
storage backends.

**Cause:** one input, parameter, gradient, optimizer, or cache was constructed
before a backend switch. Changing the thread-local default does not migrate
existing values.

**Fix:** transfer explicitly before building the graph or optimizer:

```cpp
model.to(ExecutionBackend::Metal);
Tensor right_on_metal = right.to(ExecutionBackend::Metal);
Adam optimizer(model.parameters(), options);
```

**Verify:** compare every participating object's `backend()` and then rerun
the operation.

### Metal fails only on the first neural operation

**Symptom:** storage creation succeeds, but the first matmul, attention, Adam,
or neural operation throws a pipeline compilation error.

**Cause:** Metal pipeline states are compiled lazily. Availability proves the
device/queue/storage runtime, not every kernel compilation.

**Fix:** retain the complete first error, check macOS/Xcode toolchain support,
and run the backend tests. The deterministic failure is cached for the process,
so restart after changing the environment.

**Verify:** `ctest --test-dir build/release --output-on-failure` passes on the
same machine.

## Configuration and training

### A Python workflow cannot open an input

**Symptom:** a pretraining, post-training, or lab command reports that its
corpus, instruction data, prepared dataset, or base artifact cannot be read.

**Cause:** script paths are resolved from the current working directory.
Source-only lab modules also require both `python/` and the repository root on
the import path.

**Fix:** run from the repository root, use explicit paths, and invoke labs as
modules:

```bash
PYTHONPATH=python:. python3 -m labs.lora_rank.run --help
```

**Verify:** the command reaches argument validation or starts the requested
workflow without an import/path error.

### `d_model must be divisible by n_heads`

**Symptom:** model configuration validation fails.

**Cause:** every attention head must receive the same integer head width.

**Fix:** choose `d_model` and `n_heads` so `d_model % n_heads == 0`.

**Verify:** calculate `head_width = d_model / n_heads`, then rerun the Python
smoke command.

### Corpus is too short

**Symptom:** batch construction reports that the corpus/token sequence must
contain more than the context size.

**Cause:** each next-token example requires `context_size` inputs plus one
following target.

**Fix:** provide at least `context_size + 1` encodable tokens or reduce the
context size.

**Verify:** a one-step Python run reports a finite training metric.

### Loss or gradient becomes non-finite

**Symptom:** backward or Adam rejects NaN/Inf values and no update is committed.

**Cause:** an unstable learning rate, invalid input values, or numerical
overflow. Adam validates complete candidate state transactionally.

**Fix:** inspect the first failing step, lower the learning rate, retain finite
gradient clipping, and reproduce on CPU with the same seed and batch.

**Verify:** `loss`, `gradient_norm`, and `clip_scale` remain finite in the
reported metrics; the successful step counter advances by one.

## Tensor and autograd errors

### Shape or axis validation fails

**Symptom:** an operation reports incompatible shapes, invalid permutation,
out-of-range axis/index, or an incorrect element count.

**Cause:** tensor shape contracts are checked before backend dispatch.

**Fix:** print `shape()`, `strides()`, and `numel()` for each input. For model
activations use `[batch, time, feature]`; linear weights use
`[output_feature, input_feature]`.

**Verify:** reproduce the operation with a tiny CPU tensor and run
`ctest --test-dir build/release -R 'tensor|autograd' --output-on-failure`.

### `backward()` requires a seed

**Symptom:** backward on a non-scalar output fails.

**Cause:** implicit seed 1 is defined only for a scalar output.

**Fix:** reduce to a scalar loss or pass a seed tensor with the same shape and
backend:

```cpp
output.backward(Tensor::full(output.value().shape(), 1.0F,
                             output.value().backend()));
```

**Verify:** the intended leaf gradient has the expected shape and backend.

### An old graph is reused after Adam

**Symptom:** training uses stale forward values or fails after parameter
replacement.

**Cause:** one graph describes one forward pass. Adam replaces leaf values.

**Fix:** perform a new model forward and loss construction for every optimizer
step.

**Verify:** the loop order is forward → loss → backward → Adam, repeated from
forward on the next step. See [Training](TRAINING.md).

## LoRA and QLoRA

### LoRA attachment or merge is rejected

**Symptom:** attachment is attempted twice, merge is repeated, or mutation is
rejected while a decode session is live.

**Cause:** LoRA attachment is one-time, merge is one-way, and live decode
sessions pin model parameters.

**Fix:** close every session, attach before optimizer construction, and treat
merge as the final export transition.

**Verify:** C++ `has_lora()` or Python `lora_attached` reports the expected
state and a new decode session can be created after mutation completes.

### QLoRA uses more memory than expected

**Symptom:** peak memory does not resemble four-bit storage.

**Cause:** embeddings, biases, normalization, LoRA factors, activations, and
Adam state remain FP32; export also intentionally materializes an FP32 model.
Paged Adam bounds allocation granularity but does not reduce two FP32 moments
per trainable adapter scalar.

**Fix:** inspect C++ `quantized_memory_usage()` or Python `quantized_memory`,
confirm all eligible linear weights are packed, and
measure during training rather than FP32 export.

**Verify:** packed storage remains present and immutable across adapter Adam
steps; `resident_payload_bytes` does not hide a persistent full FP32 mirror.
See [QLoRA](QLORA.md).

## Python loading and ABI

### `could not load libriftco_transformer_c`

**Symptom:** the first native Python operation raises `OSError`.

**Cause:** no wheel-local, recognized source-build, or system shared library
could be loaded. Sanitizer builds are deliberately skipped by automatic source
discovery.

**Fix:** install the wheel, build a normal Release library, or set the exact
path:

```bash
export RIFTCO_TRANSFORMER_LIBRARY="$PWD/build/release/libriftco_transformer_c.dylib"
python3 -c 'import riftco_transformer as rt; print(rt.backend_available("cpu"))'
```

Use `.so` on Linux or `.dll` on Windows.

**Verify:** the command prints `True`.

### C ABI mismatch

**Symptom:** Python reports that it requires ABI 2.5 or a newer compatible
minor but loaded another library.

**Cause:** `RIFTCO_TRANSFORMER_LIBRARY` or system search found a stale or
incompatible shared library.

**Fix:** remove the override or point it at the library produced from the same
release. Rebuild after pulling version changes.

**Verify:** in C, `rt_abi_version() == RT_ABI_VERSION`; in Python, importing and
calling `backend_available("cpu")` succeeds.

## Serving

### Decode session reaches capacity

**Symptom:** a raw native/C/Python session cannot append another token.

**Cause:** a decode session has fixed `maximum_context` capacity. The raw
session does not choose a rollover policy.

**Fix:** use `TextGenerator`, which crops and replays the newest suffix, or
reset/recreate the session and explicitly replay the desired context.

**Verify:** `session.size` stays at or below `session.capacity` and generation
continues from a replayed cache.

### Model mutation is rejected during serving

**Symptom:** transfer, parameter loading, NF4 conversion, LoRA mutation, or
Adam is rejected.

**Cause:** a live decode session pins backend and parameters.

**Fix:** close or release all sessions before mutating the model.

**Verify:** the mutation succeeds, then create a fresh session on the model's
current backend. See [Serving](SERVING.md).

## Collecting a useful report

Include the exact command, commit, platform/compiler, CMake cache backend
values, first complete error, and the narrow failing test:

```bash
git rev-parse HEAD
cmake --build build/release
ctest --test-dir build/release --output-on-failure
```

Do not report a backend speed claim without the workload, dimensions, build
type, warm-up policy, backend, and hardware.
