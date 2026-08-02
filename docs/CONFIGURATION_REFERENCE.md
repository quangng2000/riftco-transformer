# Configuration reference

Riftco Transformer has four independent configuration layers: CMake chooses
what is built, backend selectors choose storage and kernels, stage configs
choose runtime behavior, and experiment configs choose evaluation protocols.
Changing one layer does not silently rewrite another.

## Build requirements

| Requirement | Value |
| --- | --- |
| CMake | 3.24 or newer |
| C++ | C++20, extensions disabled |
| C ABI consumers | C11, extensions disabled |
| Required dependency | Platform threads |
| Python | 3.10 or newer; zero runtime package dependencies |
| CUDA build | CUDA Toolkit 12 or newer plus a compatible driver/GPU |
| TPU build | Linux x86-64 plus a compatible `libtpu.so` and Cloud TPU |

## CMake options

| Cache variable | Top-level default | Effect and constraints |
| --- | --- | --- |
| `RIFTCO_TRANSFORMER_BUILD_CLI` | `ON` | Builds `riftco-transformer`. |
| `RIFTCO_TRANSFORMER_BUILD_EXPERIMENTS` | `ON` | Builds both conditional-reversal lab executables. |
| `RIFTCO_TRANSFORMER_ENABLE_INSTALL` | `ON` | Generates install rules and the exported CMake package. |
| `RIFTCO_TRANSFORMER_BUILD_PYTHON_WHEEL` | `OFF` | Installs only the C shared library into the wheel layout; driven by `pyproject.toml`. |
| `RIFTCO_TRANSFORMER_BUILD_TESTS` | `BUILD_TESTING` at top level; `OFF` as a subproject | Builds the repository test suite. |
| `RIFTCO_TRANSFORMER_ENABLE_METAL` | `ON` on Apple; otherwise `OFF` | Requires an Apple platform. Disable explicitly to exercise the stub. |
| `RIFTCO_TRANSFORMER_ENABLE_CUDA` | `OFF` | Requires CUDA Toolkit 12+. Cannot be combined with sanitizers. |
| `RIFTCO_TRANSFORMER_ENABLE_TPU` | `OFF` | Requires Linux x86-64. The runtime is loaded dynamically. |
| `RIFTCO_TRANSFORMER_ENABLE_SANITIZERS` | `OFF` | Enables AddressSanitizer and UndefinedBehaviorSanitizer with AppleClang, Clang, or GCC. |

Source: [`CMakeLists.txt`](https://github.com/quangng2000/riftco-transformer/blob/main/CMakeLists.txt),
[`RiftcoTransformerBackends.cmake`](https://github.com/quangng2000/riftco-transformer/blob/main/cmake/RiftcoTransformerBackends.cmake),
and [`RiftcoTransformerSanitizers.cmake`](https://github.com/quangng2000/riftco-transformer/blob/main/cmake/RiftcoTransformerSanitizers.cmake).

When tests are enabled, `RIFTCO_TRANSFORMER_BUILD_PYTHON_TESTS` defaults to
`ON`. The test-only switches `RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL`,
`RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA`, and
`RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU` default to `OFF`; turn one on to make
absence of that accelerator fail the relevant conditional tests instead of
skip them.

### Presets

| Preset | Build directory | Important settings |
| --- | --- | --- |
| `debug` | `build/debug` | Debug, tests on |
| `release` | `build/release` | Release, tests on |
| `sanitize` | `build/sanitize` | Debug, tests, ASan + UBSan; unavailable on Windows |
| `cuda-release` | `build/cuda-release` | Release, tests, CUDA on; hidden on macOS |
| `tpu-release` | `build/tpu-release` | Release, tests, TPU on, Metal off; Linux only |

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
```

## Backend selection

| Surface | Accepted values | Default |
| --- | --- | --- |
| C++ | `ExecutionBackend::{Cpu, Metal, Cuda, Tpu}` | Thread-local `Cpu` |
| Native CLI | `cpu`, `metal`, `cuda`, `tpu` | `cpu` |
| Python stage configs | `auto`, `cpu`, `metal`, `cuda`, `tpu` | `auto` |

Python `auto` selects the first available backend in this order: TPU, CUDA,
Metal, CPU. Explicit unavailable selections fail; they never fall back. C++
operations use the intrinsic backend of their tensor inputs, and newly created
objects capture the calling thread's construction default. Use `Tensor::to`,
`Module::to`, or Python `.to()` for explicit transfer.

Backend limitations and availability probes are documented in
[Backends and Python ABI](BACKENDS_AND_PYTHON.md).

## Training configuration file

`riftco-transformer` reads strict `key=value` text through `Config::load`.
Blank lines and text following `#` are ignored. Unknown keys, duplicate keys,
missing values, and trailing characters in numbers are errors. Relative
`corpus` and `results` paths follow the repository convention: they are
resolved against the directory above the configuration file's parent. A file
at `configs/run.conf` therefore resolves them against the project root.

| Key | Struct default | Requirement or use |
| --- | ---: | --- |
| `corpus` | none | Required input path. Corpus must produce more than `context_size` tokens. |
| `results` | none | Required directory; default metrics file is `results/metrics.csv`. |
| `seed` | `42` | Unsigned 32-bit model and batch seed. |
| `context_size` | `16` | Positive. |
| `batch_size` | `4` | Positive. |
| `d_model` | `32` | Positive and divisible by `n_heads`. |
| `n_heads` | `4` | Positive. |
| `n_layers` | `2` | Positive. |
| `d_ff` | `64` | Positive. |
| `training_steps` | `500` | Positive; overridden by CLI `--steps`. |
| `sample_every` | `100` | Positive; controls console metric reporting interval. |
| `sample_length` | `120` | Positive and retained in `Config`; the current CLI does not generate text samples. |
| `learning_rate` | `0.001` | Finite and positive. |
| `adam_beta1` | `0.9` | Finite, strictly between zero and one. |
| `adam_beta2` | `0.999` | Finite, strictly between zero and one. |
| `adam_epsilon` | `1e-8` | Finite and positive. |
| `gradient_clip` | `1.0` | Finite and positive global-norm limit. |

Only `corpus` and `results` have no usable default; both must appear in the
file. See [`configs/tiny.conf`](https://github.com/quangng2000/riftco-transformer/blob/main/configs/tiny.conf)
and [Command-line reference](CLI_REFERENCE.md).

## Native C++ stage settings

### `stages::pretraining::PretrainingConfig`

Defaults: 100 steps, context 32, batch 4, model width 16, 4 heads, 1 block,
feed-forward width 32, epsilon `1e-5`, BPE vocabulary 272 with minimum pair
frequency 2, Adam learning rate `1e-2`, seeds 7, CPU, materialized attention,
and disabled activation checkpointing. Vocabulary size is derived from the
fitted tokenizer.

### `stages::post_training::PostTrainingConfig`

Defaults: 20 steps, context 16, batch 2, Adam learning rate `1e-3`, batch seed
29, CPU, full fine-tuning, materialized attention, and disabled checkpointing.
LoRA defaults to rank 4, alpha 8, seed 5489, query/value targets. QLoRA defaults
to NF4 block size 64, double-quantized scales with block size 256, and paged
Adam with page size 4096.

### `stages::serving::ServingConfig`

Defaults: CPU, at most 256 new tokens, paged KV cache, 16-token blocks, and
`kv_cache_block_count=0`. Zero block count allocates enough blocks for one
maximum-length context. Serving has no optimizer or gradient settings.

The authoritative declarations are under
[`include/riftco_transformer/stages`](https://github.com/quangng2000/riftco-transformer/tree/main/include/riftco_transformer/stages).

## Python stage settings

Python configs are immutable dataclasses and validate at construction.

### `PretrainingConfig`

In addition to native model/training choices, Python controls a validation
fraction (`0.1`), validation batch count (`4`), evaluation interval (`10`),
loss-average window (`10`), validation seed (`17`), and Adam state layout.
Defaults are BPE, `backend="auto"`, materialized attention, disabled block
checkpointing, contiguous optimizer state, and 4096-element state pages.

### `PostTrainingConfig`

| Setting | Default | Accepted values or rule |
| --- | --- | --- |
| `fine_tuning_method` | `"full"` | `full`, `lora`, `qlora` |
| `sampling_strategy` | `"example_uniform"` | `example_uniform`, `window_uniform` |
| `backend` | `"auto"` | `auto`, `cpu`, `metal`, `cuda`, `tpu` |
| `attention` | `"materialized"` | `materialized`, `flash` |
| `activation_checkpointing` | `"disabled"` | `disabled`, `block` |
| `nf4_block_size` | `64` | Power-of-two choice from 32 through 4096 |
| `nf4_scale_block_size` | `256` | Power-of-two choice from 32 through 4096 |
| `double_quantization` | `True` | Boolean |
| `optimizer_state` | `"auto"` | `auto`, `contiguous`, `paged`; auto selects paged for QLoRA |
| `optimizer_page_size` | `4096` | Positive scalar count |

### `serving.ServingConfig`

Defaults: `backend="auto"`, 256 maximum new tokens, 1 MiB maximum request,
`kv_cache="paged"`, and block size 16. The HTTP server is dependency-free and
is configured through `create_http_server` or `serve_model`; there is no
installed server CLI.

Source: [`python/riftco_transformer`](https://github.com/quangng2000/riftco-transformer/tree/main/python/riftco_transformer).

## Neural-lowering settings

`lowering::NeuralLoweringConfig` belongs only to the optional compiler-to-neural
bridge.

| Field | Default | Meaning |
| --- | --- | --- |
| `strategy` | `"auto"` | `auto`, `linear`, `linear_attention`, `dense`, `mlp`, or a registered strategy ID |
| `automatic_strategy_order` | linear, linear-attention, dense | First supported exact strategy wins. |
| `unsupported_strategy` | `Reject` | Reject or permit dense fallback. |
| `precision` | exact FP32 required | Reject rounded coefficients or allow them. |
| `initialization` | compiled | Preserve compiled coefficients or seeded random-uniform control. |
| `trainable` | `false` | Expose coefficients through `Module::parameters()`. |
| `backend` | CPU | Storage/backend for the lowered module. |
| `seed` / `random_scale` | `42` / `0.02` | Random-control initialization. |
| `max_coefficient_elements` | `2^24` | Pre-allocation safety limit. |
| `attention_query_axis` | unset | Query input for bilinear linear-attention lowering. |

`mlp` is a registered diagnostic strategy, but the current GELU MLP cannot
exactly preserve a general multilinear map. It reports unsupported unless the
configured policy permits a dense fallback.

See [Compiling to transformers](COMPILING_TO_TRANSFORMERS.md).

## Conditional-reversal experiment settings

The exact `riftco-conditional-reverse` lab is intentionally fixed: sequence
length 5, alphabet `abcde`, reverse trigger `a`, seed 2026, three disjoint
balanced splits of eight examples, and a 4096-element coefficient limit.

The learned lab exposes only the flags in [Command-line reference](CLI_REFERENCE.md).
Smoke mode uses F, CPU, seed 42, sequence length 3, width 8, two base heads,
feed-forward width 24, 128/64/64/64 source-disjoint splits, eight epochs,
batch 16, evaluation batch 64, and at most 64 optimizer steps. `--paper`
switches to sequence length 15, width 20, two base heads, feed-forward width
80, 10,000/5,000/1,000/1,000 independently sampled splits, ten epochs, batch
128, and evaluation batch 256. Both modes set Adam learning rate to `0.01`.

## Environment variables

| Variable | Use |
| --- | --- |
| `RIFTCO_TRANSFORMER_LIBRARY` | Exact native shared-library path used by the Python loader. |
| `RIFTCO_TRANSFORMER_TPU_LIBRARY` | Preferred exact `libtpu.so` path. |
| `TPU_LIBRARY_PATH` | TPU loader fallback before the system `libtpu.so` lookup. |
| `CMAKE_ARGS` | Standard build-backend mechanism for passing options while installing the Python package from source. |

Example CUDA wheel build:

```bash
CMAKE_ARGS="-DRIFTCO_TRANSFORMER_ENABLE_CUDA=ON" python3 -m pip install .
```

For failure diagnosis, use [Troubleshooting](TROUBLESHOOTING.md).
