# Command-line reference

Riftco Transformer builds three user-facing executables. They are enabled by
the top-level CMake defaults; `RIFTCO_TRANSFORMER_BUILD_CLI` controls the
training executable and `RIFTCO_TRANSFORMER_BUILD_EXPERIMENTS` controls both
labs. Python exposes library entry points, not installed console scripts.

```bash
cmake --preset release
cmake --build --preset release
```

The examples below therefore use `build/release/`. Installed builds place the
same executable names in the configured binary directory.

## `riftco-transformer`

Runs one self-supervised next-token pretraining stage and writes CSV metrics.

```text
riftco-transformer [--config PATH] [--steps COUNT] [--metrics PATH]
                   [--backend cpu|metal|cuda|tpu]
                   [--attention materialized|flash]
                   [--activation-checkpointing disabled|block]
```

| Option | Default | Behavior |
| --- | --- | --- |
| `--config PATH` | `configs/tiny.conf` | Loads the strict `key=value` training configuration. |
| `--steps COUNT` | `training_steps` from config | Positive integer override for this run. |
| `--metrics PATH` | `<results>/metrics.csv` | Truncates or creates the CSV and creates missing parent directories. |
| `--backend NAME` | `cpu` | Selects tensor, model, gradient, and optimizer backend. Unavailable explicit backends fail. |
| `--attention NAME` | `materialized` | Selects materialized or memory-linear Flash full-sequence attention. |
| `--activation-checkpointing NAME` | `disabled` | `block` replays each transformer block during backward. |

Each option may appear at most once, in any order, and requires a separate
value token. There is currently no `--help`/`-h` branch; an invalid or
incomplete invocation prints the usage line as an error and exits with status
1. With no options, the command uses the defaults above.

Example:

```bash
./build/release/riftco-transformer \
  --config configs/tiny.conf \
  --steps 20 \
  --metrics results/quick/metrics.csv \
  --backend metal \
  --attention flash \
  --activation-checkpointing block
```

The metrics schema is exactly:

```csv
step,loss,gradient_norm,clip_scale
```

One row is flushed after every successful Adam update. Console summaries are
reported at step 1, the final step, and multiples of `sample_every`. The
configuration's `sample_length` is validated but the current executable does
not generate samples. See [Training](TRAINING.md) and
[Configuration reference](CONFIGURATION_REFERENCE.md).

## `riftco-conditional-reverse`

Runs the fixed exact compiled-circuit lab:

```bash
./build/release/riftco-conditional-reverse
```

This executable has no command-line options. It builds deterministic,
source-disjoint train/validation/test batches; checks exact compiled behavior
and a seeded randomized-head control; captures representations; runs PCA,
batch-roll ablation, and condition steering; then prints the report. Any failed
acceptance check exits with status 1.

Its settings are fixed in
[`apps/experiments/conditional_reverse.cpp`](https://github.com/quangng2000/riftco-transformer/blob/main/apps/experiments/conditional_reverse.cpp).

## `riftco-conditional-reverse-learned`

Runs the learned F/P/T/I replication and interpretation pipeline.

```text
riftco-conditional-reverse-learned
    [--variant F|P|T|I]
    [--backend cpu|metal|cuda|tpu]
    [--epochs N]
    [--steps N]
    [--paper]
```

| Option | Default | Behavior |
| --- | --- | --- |
| `--variant F|P|T|I` | `F` | Selects the program control described below; letters are case-insensitive. |
| `--backend NAME` | `cpu` | Explicit execution backend; no `auto` value. |
| `--epochs N` | 8 smoke / 10 paper | Positive epoch-count override. |
| `--steps N` | 64 smoke / unlimited within configured epochs in paper mode | Positive maximum Adam-step override. |
| `--paper` | off | Uses the larger L=15, width-20, 10k/5k/1k/1k protocol. |
| `--help`, `-h` | — | Prints usage and exits successfully. |

Variants:

| Variant | Program branch |
| --- | --- |
| `F` | Frozen compiled conditional program |
| `P` | Frozen unconditional reverse program |
| `T` | Randomized trainable program with F's shape |
| `I` | Learned transformer components only; no program |

Quick deterministic smoke run:

```bash
./build/release/riftco-conditional-reverse-learned \
  --variant F --backend cpu
```

Paper-scale Metal run with an explicit step cap:

```bash
./build/release/riftco-conditional-reverse-learned \
  --variant F --backend metal --paper --steps 790
```

The executable trains, evaluates validation/test generalization, fits PCA on
the probe split only, reports paired ablations, and runs selector steering for
F. `--paper` reproduces configured dimensions and split sizes; a single run is
not a multi-seed paper reproduction. See
[Compiling to transformers](COMPILING_TO_TRANSFORMERS.md) and
[Generalization](GENERALIZATION.md).

## Exit and output behavior

All three executables return 0 only after the requested run completes. Runtime,
validation, unavailable-backend, allocation, and I/O exceptions are written to
standard error and return 1. Lab reports go to standard output. The training
CLI additionally writes its metrics file; neither lab writes an artifact by
default.

Tests are driven separately through CTest:

```bash
ctest --preset release
# or
ctest --test-dir build/release --output-on-failure
```
