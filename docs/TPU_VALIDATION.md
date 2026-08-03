# Validate the Cloud TPU Backend

Riftco Transformer has two deliberately different TPU gates:

- `tpu-release` validates the Linux x86-64 source boundary and the complete
  fake-PJRT execution contract without requiring a Cloud TPU; and
- `tpu-hardware` requires a loadable `libtpu.so` and an addressable device, so
  device absence is a test failure rather than a skip.

Passing the first gate does not establish real-device compatibility. The fake
plugin implements the PJRT calls used by this repository, but it is not a
StableHLO compiler and it does not emulate TPU numerics, memory, or scheduling.

## Source and fake-PJRT gate

Run this gate on Linux x86-64:

```bash
cmake --preset tpu-release
cmake --build --preset tpu-release
ctest --preset tpu-release -L tpu
```

The TPU-labelled suite covers:

| Contract | Evidence |
| --- | --- |
| PJRT lifecycle | dynamic load, version/table validation, client/device discovery, compile, host upload, execute, event wait, and host download |
| Tensor/autograd | rank-2 and batched matmul plus both matrix gradients |
| Neural operations | materialized causal attention forward/VJPs, paged decode, stability checks, and host-reference operations |
| Quantization | packed NF4 and double-quantized NF4 forward/input backward against the CPU oracle |
| Training | native nonzero full/LoRA/packed-QLoRA optimizer steps plus Python pretraining and full/LoRA/QLoRA post-training smokes |
| Packed-storage invariant | Every QLoRA code, scale, offset, shape, and block-size value remains bit-identical across the optimizer step; allocation accounting stays fixed and no persistent FP32 base appears |
| Public surfaces | C ABI TPU tensors/model training plus Python training, artifact handoff, and serving |

The repository CI runs this gate. It is suitable for catching source, emitted
StableHLO contract-marker, PJRT-call-sequence, workflow-routing, and
packed-storage regressions. It does not parse or compile StableHLO and is not a
hardware acceptance result.

## Real Cloud TPU gate

Use a Linux x86-64 Cloud TPU VM with one addressable device. Point the runtime
at the Cloud TPU `libtpu.so` supplied for that environment:

```bash
export RIFTCO_TRANSFORMER_TPU_LIBRARY=/absolute/path/to/libtpu.so
cmake --preset tpu-hardware
cmake --build --preset tpu-hardware
ctest --preset tpu-hardware
```

The `tpu-hardware` preset sets
`RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU=ON` and rejects the exported sentinel in
Riftco's tests-only fake PJRT plugin. The run therefore fails during ABI,
backend, NN, quantized-linear, Adam, or dedicated workflow acceptance if the
runtime cannot load, the PJRT table is incompatible, or no device is
addressable. It also registers `tpu_hardware_acceptance`, which executes full,
LoRA, and packed-QLoRA training on the selected device.

Do not set `RIFTCO_TRANSFORMER_TPU_LIBRARY` to a different emulator when
collecting a hardware result. The preset mechanically rejects this
repository's fake plugin; the recorded external runtime hash and Cloud TPU host
identity remain part of the evidence boundary.

## Record reproducible evidence

Keep the host and runtime identity beside the unedited test log:

```bash
uname -a | tee tpu-host.txt
sha256sum "$RIFTCO_TRANSFORMER_TPU_LIBRARY" | tee tpu-runtime.sha256
ctest --preset tpu-hardware --output-on-failure 2>&1 | tee tpu-ctest.log
```

A real-device claim requires all of the following:

1. the host reports Linux x86-64;
2. the runtime hash refers to the external Cloud TPU `libtpu.so`, not the
   repository fake;
3. `tpu_hardware_acceptance` is present and passes;
4. no TPU-required test is skipped; and
5. the complete CTest command exits successfully.

## Current validation status

The Linux x86-64 build and all eight TPU-labelled fake-PJRT tests pass in an
isolated x86-64 container. The development host is Apple silicon and has no
Cloud TPU or `libtpu.so`, so a real-device run has not been performed here.
Until a Cloud TPU evidence log satisfies the protocol above, the TPU backend
remains experimental rather than production-supported.

## Scope limits

The backend targets one addressable device in one process. Tensors retain a
host mirror, and transfers surround shape-specialized PJRT programs. Packed
linear, matmul, materialized attention and its VJPs, and paged decode use
StableHLO. Flash attention and the remaining operations use audited host
reference paths. The gate establishes functional correctness within that
scope; it does not establish multi-host behavior, performance, fault
tolerance, or full device residency.
