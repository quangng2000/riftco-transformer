# Python example

This folder trains directly from UTF-8 text through the stable C ABI. It uses
the native BPE tokenizer by default, samples shifted next-token windows, and
runs forward, cross-entropy, backward, and Adam updates. A deterministic
held-out tail and rolling training-loss average make progress easier to
interpret. It needs no NumPy, PyTorch, or other third-party Python package.

```text
examples/python/
├── README.md
├── train_tiny.py
├── prepare_huggingface_data.py
├── pretrain_stage.py
├── post_train_stage.py
├── compare_lora_ranks.py
└── serve_stage.py
```

From the repository root, install the package and its native library together:

```bash
python3 -m pip install .
```

After a release is available on PyPI, `python3 -m pip install transformer-lab`
provides the same self-contained package without a source checkout or compiler.
The wheel has no third-party runtime dependencies. Run the local example with:

```bash
python3 examples/python/train_tiny.py
```

For the artifact-based workflow, run the three explicit stage commands:

```bash
python3 examples/python/pretrain_stage.py \
  --backend cpu \
  --activation-checkpointing block

python3 examples/python/post_train_stage.py \
  --backend cpu \
  --activation-checkpointing block

python3 examples/python/serve_stage.py --backend cpu
```

The serving command prints a local chat URL. Open it in a browser to send
single-turn prompts through the same dependency-free runtime and paged
KV-cache used by the JSON API. The browser page itself has no external
assets or packages.

They hand off immutable `.tlab` bundles under `results/stages/`. See
[the staged pipeline guide](../../docs/PIPELINE.md) for the artifact contract,
HTTP request examples, and the distinction between a model bundle and a
future resumable training checkpoint.

## Prepare real learning data

The preparation example uses Python's standard-library `urllib` and `json`
with the official Hugging Face Dataset Viewer API. It does not install
`datasets`, `huggingface_hub`, NumPy, or PyTorch. The available presets are
`tinystories`, `dolly`, and `hh-rlhf`.

For pretraining, preserve TinyStories' published validation boundary:

```bash
python3 examples/python/prepare_huggingface_data.py \
  --preset tinystories \
  --source-split train \
  --output data/external/huggingface/tinystories-train \
  --limit 1000 \
  --seed tinystories-v1 \
  --train-fraction 1 \
  --validation-fraction 0 \
  --test-fraction 0

python3 examples/python/prepare_huggingface_data.py \
  --preset tinystories \
  --source-split validation \
  --output data/external/huggingface/tinystories-validation \
  --limit 200 \
  --seed tinystories-v1 \
  --train-fraction 0 \
  --validation-fraction 1 \
  --test-fraction 0

python3 examples/python/pretrain_stage.py \
  --file data/external/huggingface/tinystories-train/train.txt \
  --validation-file \
    data/external/huggingface/tinystories-validation/validation.txt \
  --output results/stages/tinystories_pretrained.tlab \
  --backend cpu \
  --steps 100 \
  --context 32
```

The tokenizer is fitted only on the explicit training file. Prepared
directories include source revision/ranges, license metadata, split settings,
counts, and SHA-256 file digests in `manifest.json`; their destinations must
not already exist.

Prepare Dolly and run a fair rank sweep:

```bash
python3 examples/python/prepare_huggingface_data.py \
  --preset dolly \
  --output data/external/huggingface/dolly-lora-v1 \
  --limit 2000 \
  --seed lora-v1

python3 examples/python/compare_lora_ranks.py \
  --base results/stages/tinystories_pretrained.tlab \
  --data data/external/huggingface/dolly-lora-v1 \
  --output results/experiments/dolly-lora-ranks \
  --ranks 1,2,4,8 \
  --alpha-over-rank 2 \
  --steps 20 \
  --context 16 \
  --batch-size 2 \
  --backend cpu \
  --prompt "Explain attention to a beginner."
```

All ranks start from the same artifact and share the same split fingerprints,
seeds, optimizer settings, and `alpha / rank`. Validation chooses the rank;
only then is the winner evaluated on held-out test data. The objective scores
the full prompt/response sequence, not the response alone. Merged adapters
have identical serving topology, so per-rank generation timings are smoke
checks rather than rank-speed evidence. The output directory must not exist;
the CLI stages and atomically publishes the complete run. Its
`comparison.json` embeds the verified prepared manifest and manifest SHA-256.

HH-RLHF is prepared only as `chosen`/`rejected` preference data. The current
SFT and rank scripts do not consume it because the lab has no pairwise
preference objective yet. See
[the full data and experiment guide](../../docs/DATASETS_AND_LORA_EXPERIMENTS.md)
for dataset-card and license links, CPU/Metal sample-size guidance, and every
reproducibility caveat.

Select adapter-only LoRA post-training while keeping the output artifact
serving-ready:

```bash
python3 examples/python/post_train_stage.py \
    --backend cpu \
    --fine-tuning-method lora \
    --lora-rank 4 \
    --lora-alpha 8
```

The stage optimizes only query/value LoRA factors, merges them into the base
weights, and then saves the ordinary child `.tlab` bundle. See
[the LoRA guide](../../docs/LORA.md) for target selection, direct APIs, and
the current adapter-checkpoint limitation.

The script selects Metal when available and otherwise uses CPU. A backend can
also be selected explicitly:

```bash
python3 examples/python/train_tiny.py --backend cpu --steps 3
```

Add `--activation-checkpointing block` to recompute transformer blocks during
backward and reduce retained activation graph state. The default is
`disabled`; this option is independent of `--attention materialized|flash`.

Train literal text with BPE:

```bash
python3 examples/python/train_tiny.py \
    --text "hello transformer! tiny models learn from text. hello tokenizer! validation checks unseen model-training text." \
    --tokenizer bpe \
    --vocab-size 272 \
    --min-pair-frequency 2 \
    --context 4 \
    --steps 10
```

Or train a UTF-8 file:

```bash
python3 examples/python/train_tiny.py \
    --file data/pretraining/tiny_corpus.txt \
    --tokenizer bpe \
    --context 16 \
    --batch-size 4 \
    --validation-fraction 0.1 \
    --validation-batches 4 \
    --eval-every 10 \
    --loss-average-window 10 \
    --steps 10
```

Compare the byte strategy while leaving the model and optimizer unchanged:

```bash
python3 examples/python/train_tiny.py \
    --file data/pretraining/tiny_corpus.txt \
    --tokenizer byte \
    --context 16 \
    --steps 10
```

`--vocab-size` and `--min-pair-frequency` configure BPE only. BPE begins with
all 256 single-byte tokens and learns up to `vocab-size - 256` merge tokens.
The requested size is a maximum; learning stops if no pair occurs often
enough. Byte mode instead keeps its backward-compatible corpus-derived
vocabulary.

The reported `corpus_tokens` value shows the encoded sequence length.
`--context` counts those tokens—not Unicode characters or UTF-8 bytes—so two
strategies can create different numbers of windows from the same text.

After tokenization, the script reserves a contiguous tail using
`--validation-fraction` as a target; at least `context + 1` tokens are held out.
Training windows never use those held-out model targets, and both regions need
at least `context + 1` encoded tokens. The tokenizer itself is fitted on the
complete corpus before the split; this preserves the corpus-derived byte
tokenizer's ability to represent every validation byte. For a
publication-grade benchmark, split the raw documents first, fit the tokenizer
only on the training partition, freeze it, and then encode validation data.

Each `train_loss` is measured on the current random mini-batch before its Adam
update. `train_loss_average` is the rolling mean of the latest
`--loss-average-window` mini-batches. Validation batches are sampled once with
an independent deterministic random generator, reused for every evaluation,
and averaged into `validation_loss`. Validation is reported before training,
every `--eval-every` updates, and after the final update. It never calls
`backward()` or `optimizer.step()`.

When automatic native-library discovery is unsuitable, provide its path:

```bash
TRANSFORMER_LAB_LIBRARY="$PWD/build/release/libtransformer_lab_c.dylib" \
python3 examples/python/train_tiny.py
```

This override is for native development and is unnecessary with a released
wheel. Use `libtransformer_lab_c.so` instead on Linux or
`transformer_lab_c.dll` on Windows. The model must move to its backend before
creating the parameter view or optimizer. Each optimizer update invalidates the
previous computation graph, so the loop creates a fresh forward and loss every
step.

The data flow is:

```text
UTF-8 text
  → selected byte/BPE strategy
  → integer token IDs
  → shifted input/target windows
  → Transformer
  → cross-entropy
  → backward
  → Adam
```
