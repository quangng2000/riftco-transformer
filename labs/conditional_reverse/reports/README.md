# Curated conditional-reversal records

This directory stores small, reviewed records for completed experiment runs. A
record is evidence about one configuration on one source state and one machine;
it is not automatically a benchmark or a paper reproduction.

## Current ABI 2.5 Python-lab records

[`m4-max-metal-f-seed-42-abi-2.5.json`](m4-max-metal-f-seed-42-abi-2.5.json)
records a clean paper-profile `F` run through the public generic model, C ABI,
and Python wrapper. On its source-disjoint 1,000-example test split it reached
100% target-token and exact-sequence accuracy after 790 Adam steps. Rolling the
learned attention output changed token accuracy by 0 points; rolling the frozen
compiled-program output reduced it by 95.8 percentage points. The raw
program-output PCA is explicitly observational. Multiplicative selector-basis
masking shifted behavior in the intended directions on a 386-example balanced
held-out subset.

[`m4-max-metal-quick-fpti-seed-42-abi-2.5.json`](m4-max-metal-quick-fpti-seed-42-abi-2.5.json)
is the clean all-variant integration smoke with 64 Adam steps per variant. It
establishes that `F`, `P`, `T`, and `I` all train and evaluate over the current
public API; the compact record also retains branch-resampling effects where
applicable. Its small synthetic profile is not a paper result.

Both records identify commit `09cf334`, ABI 2.5, and the exact native dylib
SHA-256. They remain single-seed local evidence, not a hardware benchmark or a
multi-seed reproduction.

## Historical pre-boundary record

[`m4-max-metal-f-seed-42.json`](m4-max-metal-f-seed-42.json) records the first
complete paper-scale local run of the learned conditional-reversal `F` variant.
It used 790 Adam steps and achieved 100% target-token and exact-sequence
accuracy on its 1,000-example held-out test split. Batch-rolling the compiled
program output reduced paired token accuracy by 96.15 percentage points, while
batch-rolling learned attention had no measured effect in this run.

That record came from the retired experiment-specific C++ prototype before the
framework/lab boundary was enforced. It is retained as historical evidence,
not as evidence for the current path. A credible paper comparison still
requires multiple predeclared seeds for paper-profile `F`, `P`, `T`, and `I`,
validation-based selection, checkpoints, and an aggregate statistical report.
