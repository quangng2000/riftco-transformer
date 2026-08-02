# Conditional string reversal

This Python-owned lab specifies the conditional reversal task and its data and
evaluation protocol. A source string is reversed when its first character is a
configured trigger (vowels by default); otherwise it is copied. The lab emits
source-disjoint train, probe, validation, and test splits and reports metrics
separately for reverse and copy examples.

Run the protocol audit from the repository root:

```bash
python3 -m labs.conditional_reverse.run \
  --output runs/conditional-reverse/protocol.json
```

This command compares the exact conditional oracle against copy-only and
reverse-only controls. It verifies the task, split, teacher-forcing, and metric
contracts; it does not train a model.

## Framework boundary

The former F/P/T/I learned hybrid was an experiment-specific C++ target. It was
removed from the installed/exported framework when the project adopted this
boundary:

- C++ owns reusable tensor, autograd, model, loss, Adam, backend, compiler, and
  analysis primitives.
- Python owns datasets, training loops, experimental variants, evaluation,
  ablations, steering, and reports.

The generic Cajal compiler, lowering, programmed-sequence, PCA, ablation, and
intervention components remain in the C++ framework. Rebuilding the learned
F/P/T/I study now requires a public, task-neutral program-augmented model API
that Python can compose. Until that exists, this lab deliberately does not
claim to reproduce or execute those variants.

The [`reports/`](reports/) directory retains one historical F-variant record
from the retired C++ prototype with explicit provenance. It is evidence from a
past source state, not a current executable benchmark or paper reproduction.

## Files

- `protocol.py` owns deterministic data generation, teacher-forced examples,
  split fingerprints, and branch-stratified evaluation.
- `run.py` creates a machine-readable protocol/control report.
- `tests/` protects task semantics, disjointness, determinism, and metrics.
- `reports/` stores small reviewed historical evidence.
