# Conditional string reversal

This lab trains and analyzes the F/P/T/I conditional-reversal study through
the installed, task-neutral `riftco_transformer` API. The task, datasets,
variant definitions, training loop, evaluation, interventions, and reports
remain ordinary Python under `labs/`; none is shipped as framework behavior.

A source string is reversed when its first character is a configured trigger
and copied otherwise. Inputs are teacher-forced `source|target` sequences, and
training supervises only logits positions `[L, 2L)`.

## Run the learned study

From the repository root, run the bounded quick smoke profile on all four
variants:

```bash
python3 -m labs.conditional_reverse.run \
  --profile quick \
  --variants all \
  --backend auto \
  --output runs/conditional-reverse/quick-all-seed-42.json
```

The quick profile uses `L=3`, alphabet `abcdefghij`, reverse triggers `ae`,
model width 8, two heads in each of two parallel attention branches (four
learned heads total), feed-forward width 24, 128/64/64/64 examples, batch size
16, learning rate 0.01, and 8 epochs (64 Adam steps per variant).

Select individual controls with a comma-separated list and optionally cap the
run at a different number of steps:

```bash
python3 -m labs.conditional_reverse.run \
  --profile quick --variants F,T --steps 16 \
  --backend cpu --seed 7 \
  --output runs/conditional-reverse/quick-ft-seed-7.json
```

The paper-scale profile uses `L=15`, the full lowercase alphabet and vowel
condition, width 20, two heads, feed-forward width 80, 10,000/5,000/1,000/1,000
examples, batch size 128, learning rate 0.01, and 10 epochs:

```bash
python3 -m labs.conditional_reverse.run \
  --profile paper --variants all --backend auto \
  --output runs/conditional-reverse/paper-all-seed-42.json
```

`paper` is a paper-scale, source-disjoint run of the current protocol. It is
not a claim of bit-for-bit reproduction of the historical dataset RNG or an
archived external artifact.

Every learned output is strict finite JSON with command, Git, Python, package,
ABI, exact native-library hash, backend, seed, split-fingerprint, parameter,
training, validation, test, hypothesis, PCA, ablation, and steering evidence.
Source and runtime identity are checked before and after training. An existing
output path is rejected before model construction and is never overwritten.

## Protocol-only audit

The protocol path imports no `riftco_transformer` module and needs no native
library:

```bash
python3 -m labs.conditional_reverse.run \
  --protocol-only --profile quick \
  --output runs/conditional-reverse/quick-protocol.json
```

It validates source-disjoint splits and reports the conditional oracle,
copy-only, and reverse-only controls.

## Variant contract

- **F** uses the exact frozen bilinear conditional map. Its two whole-source
  inputs share projection group `(0, 0)`.
- **P** uses the exact frozen unary reversal map with projection group `(0,)`.
- **T** has F's logical map shape and shared projections, but its coefficient
  tensor is random-uniform and trainable.
- **I** has no program branch.

Python retains only F/P's sparse unit-coefficient indices. It never allocates
a dense `D³` coefficient list. The generic native lowering owns the eventual
tensor. Probe PCA streams only `program.output.raw` into online covariance and
is an unsupervised variance diagnostic, not causal confirmation; normal
evaluation, ablation, and steering runs retain no representation trace. F
steering masks and amplifies input 0 at source position 0 with an
only-reverse-basis scale `(100, 0, ...)` or only-copy-basis scale
`(0, 100, ...)`. Because this is multiplicative, the reported behavior is
empirical rather than a guaranteed coordinate assignment.

## Layout

- `protocol.py`: task semantics, deterministic splits, and control metrics.
- `config.py`: F/P/T/I and quick/paper experiment configuration.
- `data.py`: token IDs and deterministic rectangular batches.
- `programs.py`: sparse output-major F/P maps and T/I policy.
- `model.py`: translation to the public program-augmented model API.
- `training.py`: Adam lifecycle and target-half loss.
- `evaluation.py`: metrics, hypotheses, streamed PCA, ablations, and steering.
- `analysis.py`: dependency-free Jacobi PCA and paired statistics.
- `reporting.py`: provenance and strict no-overwrite JSON.
- `run.py`: CLI orchestration only.
- `tests/`: pure unit tests plus a native-gated QUICK F integration test.

Full generated reports belong under ignored `runs/conditional-reverse/`.
`reports/` contains compact reviewed records for the current ABI 2.5 QUICK and
paper-F runs alongside the clearly separated retired-prototype record.
