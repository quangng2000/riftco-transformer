# Research labs

Labs contain experimental policy: hypotheses, controlled comparisons, fixed
seeds, validation rules, ablations, and reports. They are Python-owned and are
not installed with the `riftco-transformer` wheel or exported by CMake.

The framework remains underneath them:

- `python/riftco_transformer/` provides stable Python workflows and bindings.
- `include/` and `src/` provide native tensors, autograd, models, losses,
  optimizers, compiler primitives, serving primitives, and hardware backends.
- `labs/` composes those public APIs into research protocols.

Run a lab from the repository root with module syntax, for example:

```bash
PYTHONPATH=python python -m labs.lora_rank.run --help
```

Generated checkpoints and reports default to `runs/`, which is intentionally
ignored. Small, reviewed evidence records belong in a lab's `reports/` folder.

## Available labs

- [`fine_tuning/`](fine_tuning/) compares full fine-tuning and LoRA with
  validation-only selection and a final held-out test.
- [`lora_rank/`](lora_rank/) compares LoRA ranks from one immutable base model.
- [`conditional_reverse/`](conditional_reverse/) owns the conditional string
  reversal research protocol and its curated historical evidence.
