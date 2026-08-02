# LoRA rank comparison

This lab trains several LoRA ranks from the same immutable base artifact with
shared data, seeds, optimizer settings, and `alpha / rank` scaling. Validation
selects the best rank before the held-out test split is consumed.

From the repository root:

```bash
PYTHONPATH=python python -m labs.lora_rank.run \
  --base path/to/base.rift \
  --data path/to/prepared-data \
  --output runs/lora-rank
```

Use `python -m labs.lora_rank.run --help` for all controls. Python owns the
sweep and reporting; native C++ still executes tensor math, autograd, and Adam
through the stable Python bindings.

Files:

- `protocol.py` defines the controlled rank comparison.
- `run.py` is the repository-owned CLI and report writer.
- `tests/` protects protocol and CLI behavior.

This package is a lab, not part of the installed `riftco_transformer` API.
