# Command-line reference

Riftco Transformer does not install a training console script. Python owns
high-level training and experiment orchestration, while C++ supplies the
native runtime through the stable C ABI. The commands below are source-tree
examples and labs, so run them from the repository root.

Install the package and bundled native library first:

```bash
python3 -m pip install .
```

## Framework examples

The small scripts under `examples/python/` demonstrate supported public APIs:

```bash
python3 examples/python/train_tiny.py --help
python3 examples/python/pretrain_stage.py --help
python3 examples/python/post_train_stage.py --help
python3 examples/python/serve_stage.py --help
python3 examples/python/prepare_huggingface_data.py --help
```

- `train_tiny.py` exposes tokenization, next-token batching, validation,
  forward/loss/backward, and Adam in one readable Python loop.
- `pretrain_stage.py` creates an immutable base `.rift` bundle.
- `post_train_stage.py` applies Full, LoRA, or QLoRA training and creates a
  child bundle.
- `serve_stage.py` starts the dependency-free local chat and JSON service.
- `prepare_huggingface_data.py` prepares verified, content-hash-split data.

Use `--help` on a script for its exact defaults and accepted values. Backend
selectors accept `auto`, `cpu`, `metal`, `cuda`, and `tpu`; an explicitly
unavailable accelerator fails rather than silently falling back.

## Research labs

Controlled protocols live under top-level `labs/`, are Python-owned, and are
not installed in wheels or exported by CMake. Include both `python/` and the
repository root on the import path:

```bash
PYTHONPATH=python:. python3 -m labs.lora_rank.run --help
PYTHONPATH=python:. python3 -m labs.fine_tuning.run --help
PYTHONPATH=python:. python3 -m labs.conditional_reverse.run --help
```

Typical comparison runs are:

```bash
PYTHONPATH=python:. python3 -m labs.lora_rank.run \
  --base results/stages/tinystories_pretrained.rift \
  --data data/external/huggingface/dolly-lora-v1 \
  --output runs/lora-rank

PYTHONPATH=python:. python3 -m labs.fine_tuning.run \
  --base results/stages/tinystories_pretrained.rift \
  --data data/external/huggingface/dolly-lora-v1 \
  --output runs/fine-tuning

PYTHONPATH=python:. python3 -m labs.conditional_reverse.run \
  --profile quick --variants F --backend cpu \
  --output runs/conditional-reverse/quick.json
```

Lab output defaults to ignored `runs/` paths, and commands refuse to overwrite
existing evidence. The conditional-reversal lab owns the task, source-disjoint
splits, F/P/T/I construction, training/evaluation policy, PCA, ablations,
steering, and reports. It composes the generic installed
`riftco_transformer.programmed` model rather than importing task-specific C++.
`quick` is the smoke profile and `paper` is the long configuration. Because
the source lab CLI can evolve independently of the installed package, verify
the exact accepted syntax with `--help` before starting either profile. See
[Compiling programs to transformers](COMPILING_TO_TRANSFORMERS.md).

## Tests

The C++ suite remains driven through CTest:

```bash
ctest --preset release
# or
ctest --test-dir build/release --output-on-failure
```

Python and lab tests are registered by the configured CMake test suite. None
of these test runners is installed as an end-user command.
