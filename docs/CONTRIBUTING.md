# Contributing

Riftco Transformer is a learning framework with a strict engineering goal:
keep the complete transformer stack inspectable while preserving clear seams
for optimized backends. A useful contribution improves correctness,
explanation, or extensibility without hiding behavior behind a new dependency.

This guide describes the repository's current contribution contract. It does
not promise that an issue or proposal will be accepted; it makes the technical
expectations explicit so a change can be evaluated on evidence.

## Before you begin

You need:

- Git;
- CMake 3.24 or newer;
- Ninja;
- a C++20 compiler; and
- Python 3.10 or newer when the Python tests are enabled.

Clone the repository and establish a clean baseline:

```bash
git clone https://github.com/quangng2000/riftco-transformer.git
cd riftco-transformer
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

If the baseline fails, record the platform, compiler, configuration command,
and first failing test before changing code. Do not treat an unavailable CUDA
or TPU device as a CPU failure; optional accelerator builds have separate
preconditions.

## Choose the smallest ownership boundary

Place a change in the lowest reusable layer that owns its semantics:

| Change | Primary location | Must not depend on |
| --- | --- | --- |
| Tensor storage or numerical operation | `include/riftco_transformer/core/`, `src/core/` | `nn`, `model`, stages |
| Reusable layer, activation, loss, or parameter behavior | `include/riftco_transformer/nn/`, `src/nn/` | `model`, stages |
| Transformer-specific composition | `include/riftco_transformer/model/`, `src/model/` | training or serving policy |
| Optimizer rule | `include/riftco_transformer/optim/`, `src/optim/` | a concrete model |
| Tokenization or batch representation | `include/riftco_transformer/data/`, `src/data/` | model policy |
| High-level training, evaluation, or dataset policy | `python/riftco_transformer/` | private C++ layouts |
| Native serving composition | `include/riftco_transformer/stages/serving/`, `src/stages/serving/` | training or optimizer policy |
| Symbolic compiler or interpretation analysis | its separately linked component | the tensor runtime unless it is the explicit lowering bridge |
| Small public-API demonstration | `examples/python/` | experiment-specific policy |
| Research protocol, comparison, or reporting | `labs/` | non-public framework internals |

Read [Project structure](PROJECT_STRUCTURE.md) before moving a responsibility
across layers. The public declaration normally lives below
`include/riftco_transformer/`; its implementation mirrors that path below
`src/`.

## Make one coherent change

A reviewable contribution has one primary purpose. It normally contains:

1. a failing or missing acceptance test;
2. the smallest implementation that satisfies the contract;
3. validation for invalid shapes, values, lifetimes, and backend combinations;
4. documentation for public behavior and limitations; and
5. no unrelated formatting or renaming.

Large milestones can be split vertically. For example, a new operation can land
with its public contract, readable CPU implementation, autograd rule, and
tests before an optional accelerator kernel. The intermediate state must still
build and tell the truth about dispatch.

## Preserve the public boundaries

### Tensor and autograd

- Validate ranks, shapes, element counts, and backend identity before
  performing numerical work.
- Keep forward values immutable after graph construction.
- Accumulate gradient contributions when a node has multiple downstream uses.
- Compare new analytical gradients with centered finite differences.
- Do not make `core` depend on a neural-network layer to implement an
  operation.

### Modules and parameters

- Register every trainable parameter exactly once under a stable hierarchical
  name.
- Preserve the nonmoving module lifetime assumptions described in
  [Module and parameter lifecycle](MODULES.md).
- Transfer the complete parameter set transactionally before constructing a
  graph or optimizer.
- Keep frozen quantized base weights out of autograd and optimizer state.

### Training and serving

- Build a fresh computation graph for every training step.
- Keep loss and optimizer policy outside the model.
- Keep high-level training loops, dataset policy, and evaluation orchestration
  in Python; C++ owns the numerical primitives beneath the C ABI.
- Keep research hypotheses, fixed comparisons, and generated reports in
  source-only `labs/`, not installed framework packages.
- Keep serving independent of training and optimizer dependencies.
- Treat `ModelSnapshot` as an in-memory native handoff and `ModelBundle` as the
  persisted Python artifact; neither is an exact resumable training
  checkpoint.

### Stable C and Python interfaces

The Python client binds the C ABI through `ctypes`; it does not bind C++ class
layouts. Additive ABI work must preserve the version and compatibility rules
documented in [Execution backends and the Python ABI](BACKENDS_AND_PYTHON.md).
Do not expose an internal C++ layout through an opaque C handle.

## Add backend work in layers

The framework uses a readable reference implementation as the correctness
oracle. For a new backend-aware operation:

1. define a backend-neutral contract and validate it before dispatch;
2. implement or extend the CPU reference path;
3. add exact small examples and gradient tests where applicable;
4. add dispatch without changing the public tensor or layer API;
5. implement each accelerator adapter in its own backend directory; and
6. compare accelerator results with the CPU oracle on hardware that is
   actually available.

Do not advertise source integration or no-device CI as hardware validation.
State separately whether a path compiles, whether unavailable-device behavior
is tested, whether real hardware was exercised, and whether performance was
measured.

The current backend layout is documented in
[Execution backends and the Python ABI](BACKENDS_AND_PYTHON.md).

## Write tests at the contract boundary

Tests are plain executables registered through CTest. Put a test beside the
subsystem it protects:

```text
tests/core/          tensor, autograd, quantization
tests/nn/            reusable layers and modules
tests/model/         attention, blocks, Transformer, LoRA/QLoRA
tests/optim/         Adam and state layouts
tests/stages/        stage handoffs and serving generation
tests/compiler/      Cajal checking, interpretation, compilation
tests/analysis/      PCA and causal interventions
tests/python/        C ABI client and Python workflows
tests/package/       installed exported-target consumers
```

A numerical test should cover more than a happy-path output. Depending on the
contract, include:

- hand-calculated small values;
- exact output shapes and backend identity;
- invalid rank, extent, index, and option rejection;
- deterministic replay from a fixed seed;
- finite-difference gradients;
- causality or batch-isolation invariants;
- transactional behavior when validation fails; and
- CPU-oracle comparison for an available accelerator.

Avoid calling a tiny-batch overfit test “generalization.” It validates training
wiring. Generalization requires disjoint held-out data and a selection protocol
that does not inspect the test set.

## Run proportionate validation

During development, build and run the narrowest relevant test:

```bash
cmake --build --preset debug --target tensor_tests
ctest --preset debug -R '^tensor$'
```

Before handing off a C++ change, run the full default suite:

```bash
cmake --build --preset debug
ctest --preset debug
```

On a supported non-Windows host, exercise address and undefined-behavior
sanitizers for changes involving storage, indexing, lifetimes, or ownership:

```bash
cmake --preset sanitize
cmake --build --preset sanitize
ctest --preset sanitize
```

Use the optional `cuda-release` or `tpu-release` preset only when its toolchain
requirements are satisfied. A backend change is not fully validated until the
relevant real hardware path has been exercised; keep that limitation visible
when it has not.

For documentation and release metadata, run:

```bash
python3 .github/scripts/check_docs_math.py
python3 .github/scripts/check_release_version.py
python3 .github/scripts/build_pages.py --output build/pages --check
```

These checks validate renderer-safe math, aligned release versions, generated
site links, and internal anchors. They do not replace technical review of the
claims in prose.

## Keep warnings clean

The project enables strict warnings:

- MSVC uses `/W4` and `/permissive-`;
- other supported compilers use `-Wall`, `-Wextra`, `-Wpedantic`,
  `-Wconversion`, and `-Wshadow`.

Resolve new warnings in the implementation. Avoid broad warning suppression;
when an external ABI forces a narrowly scoped exception, document the reason
next to the build rule.

There is no repository-wide autoformatter contract. Match the style of the
surrounding file, keep headers self-contained, prefer explicit types when they
clarify ownership or numeric conversion, and let strict compilation expose
portability mistakes.

## Update documentation as part of the contract

Public behavior is incomplete until a reader can discover it. Update the page
that owns the feature, then add or revise a manifest record in
`site/docs_manifest.json` when the documentation set changes.

Follow [Documentation style](DOCUMENTATION_STYLE.md). In particular:

- identify whether a page is a tutorial, how-to, concept, reference, overview,
  or explanation;
- separate implemented behavior from future work;
- qualify backend and hardware evidence precisely;
- give commands from a stated working directory; and
- use relative links between documentation pages.

## Keep dependencies intentional

The default C++ runtime and released Python runtime are designed to remain free
of third-party runtime dependencies. Build tools, platform APIs, CUDA, and
`libtpu` are separate concerns. A proposal that adds a runtime dependency must
explain which component owns it, whether it remains optional, how it affects
packaging, and why a standard-library or platform implementation is
insufficient.

Do not copy code, data, or documentation under a license incompatible with the
project's Apache-2.0 license. Cite papers and external specifications without
presenting their reported results as results reproduced by this repository.

## Prepare the handoff

Before requesting review, summarize:

- the contract that changed;
- the source and documentation surfaces affected;
- the tests run and their outcomes;
- the platforms or hardware actually exercised; and
- known limitations or deliberately deferred work.

A useful handoff makes a claim no stronger than its evidence. “CUDA sources
compile in no-device CI” and “validated on an NVIDIA GPU” are different claims;
both can be valuable when labeled accurately.

## Contribution checklist

- [ ] The change has one primary purpose.
- [ ] It respects the dependency direction in
      [Project structure](PROJECT_STRUCTURE.md).
- [ ] Public inputs are validated before mutation or dispatch.
- [ ] New numerical behavior has small oracle tests.
- [ ] New differentiable behavior has gradient tests.
- [ ] Backend claims name the hardware actually tested.
- [ ] The default build remains dependency-free at runtime.
- [ ] Relevant narrow tests and the full default suite pass.
- [ ] Sanitizers were run when the change affects memory or ownership.
- [ ] User-facing behavior and limitations are documented.
- [ ] The generated documentation site passes its link and math checks.
