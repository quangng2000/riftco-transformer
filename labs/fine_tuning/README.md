# Full fine-tuning versus LoRA

This lab compares full fine-tuning and LoRA candidates from the same immutable
base artifact. Python owns candidate construction, training orchestration,
validation selection, held-out evaluation, and report publication. Numerical
forward, loss, backward, and Adam operations execute in the C++ runtime through
the public Python API.

From the repository root:

```bash
PYTHONPATH=python python -m labs.fine_tuning.run \
  --base path/to/base.rift \
  --data path/to/prepared-data \
  --output runs/fine-tuning
```

Use `python -m labs.fine_tuning.run --help` for all controls. The output path
must not already exist, which prevents an old report from being overwritten.

Files:

- `protocol.py` defines candidates, selection, and generalization metrics.
- `run.py` is the repository-owned CLI and report writer.
- `tests/` protects protocol and CLI behavior.

This package is a lab, not part of the installed `riftco_transformer` API.
