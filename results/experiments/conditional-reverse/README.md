# Conditional-reversal experiment records

This directory stores machine-readable records for completed local experiment
runs. A record is evidence about one configuration on one source state and one
machine; it is not automatically a benchmark or a paper reproduction.

## Apple M4 Max, Metal, F variant, seed 42

[`m4-max-metal-f-seed-42.json`](m4-max-metal-f-seed-42.json) records the first
complete paper-scale local run of the learned conditional-reversal `F` variant.
It used 790 Adam steps and achieved 100% target-token and exact-sequence
accuracy on its 1,000-example held-out test split. Batch-rolling the compiled
program output reduced paired token accuracy by 96.15 percentage points, while
batch-rolling learned attention had no measured effect in this run.

The record is deliberately labeled as a dirty-worktree, single-seed run. A
credible comparison with the source paper still requires multiple archived
seeds for `F`, `P`, `T`, and `I`, validation-based selection, checkpoints, and
an aggregate statistical report.
