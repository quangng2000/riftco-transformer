# Python example

This folder trains directly from UTF-8 text through the stable C ABI. It uses
the native BPE tokenizer by default, samples shifted next-token windows, and
runs forward, cross-entropy, backward, and Adam updates. A deterministic
held-out tail and rolling training-loss average make progress easier to
interpret. It needs no NumPy, PyTorch, or other third-party Python package.
These files are small API examples; controlled comparisons, fixed protocols,
and report generation belong in the repository's top-level `labs/` directory.

```text
examples/python/
├── README.md
├── train_tiny.py
├── prepare_huggingface_data.py
├── pretrain_stage.py
├── post_train_stage.py
├── convert_model.py
└── serve_stage.py
```

From the repository root, install the package and its native library together:

```bash
python3 -m pip install .
```

After a release is available on PyPI, `python3 -m pip install riftco-transformer`
provides the same self-contained package without a source checkout or compiler.
The standard wheel has no third-party runtime dependencies. It recognizes the
`cuda` and `tpu` backends but contains their unavailable stubs; both require
explicit source builds. Run the local example with:

```bash
python3 examples/python/train_tiny.py
```

To install the CUDA backend, use an NVIDIA GPU and compatible driver
with CUDA Toolkit 12 or newer:

```bash
CMAKE_ARGS="-DRIFTCO_TRANSFORMER_ENABLE_CUDA=ON" \
  python3 -m pip install .
```

CUDA tensors and packed NF4 weights use managed memory. NN operations, matmul,
packed quantized-linear forward/input backward, materialized/Flash attention
and its gradients, paged decode, and Adam's candidate update run as native GPU
kernels. Full, LoRA, and QLoRA examples are functionally supported, but
selecting CUDA alone is not evidence that the complete run is device-resident
or faster. The source path is implemented; actual NVIDIA-hardware validation
was not available on this macOS host.

The experimental TPU path requires Linux x86-64, a Google Cloud TPU, and
Google's separately installed `libtpu.so`:

```bash
export RIFTCO_TRANSFORMER_TPU_LIBRARY=/absolute/path/to/libtpu.so
CMAKE_ARGS="-DRIFTCO_TRANSFORMER_ENABLE_TPU=ON" \
  python3 -m pip install .
```

TPU packed quantized-linear forward/input backward, matmul, materialized
attention and its gradients, and paged decode run through PJRT/StableHLO; Flash
attention, Adam, and other operations use host reference paths. Full, LoRA, and
QLoRA examples are functionally wired, but real-hardware validation is pending
and this phase does not claim end-to-end TPU acceleration.

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

They hand off immutable `.rift` bundles under `results/stages/`. See
[the staged pipeline guide](../../docs/PIPELINE.md) for the artifact contract,
HTTP request examples, and the distinction between a model bundle and an
exact-resume `.riftckpt` training checkpoint.

Convert a complete model by naming both formats explicitly:

```bash
python3 examples/python/convert_model.py \
  results/stages/tiny_pretrained.rift \
  results/stages/tiny_pretrained.onnx \
  --from rift --to onnx
```

The same command imports or exports the current Riftco Hugging Face-style,
GGUF, and canonical ONNX representations. SafeTensors is available as a
lower-level named-tensor container. ONNX import requires the generated
adjacent `.onnx.riftco.json` tokenizer/artifact sidecar and rejects rewritten
or foreign graphs. See the
[model interchange guide](../../docs/MODEL_INTERCHANGE.md).

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
  --output results/stages/tinystories_pretrained.rift \
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

PYTHONPATH=python:. python3 -m labs.lora_rank.run \
  --base results/stages/tinystories_pretrained.rift \
  --data data/external/huggingface/dolly-lora-v1 \
  --output runs/lora-rank \
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

Compare a fixed Full recipe with validation-selected LoRA ranks under the same
held-out protocol:

```bash
PYTHONPATH=python:. python3 -m labs.fine_tuning.run \
  --base results/stages/tinystories_pretrained.rift \
  --data data/external/huggingface/dolly-lora-v1 \
  --output runs/fine-tuning \
  --methods full,lora \
  --lora-ranks 1,2,4,8 \
  --backend cpu
```

Use `--backend cuda` with the CUDA-enabled source build or `--backend tpu` with
the experimental TPU-enabled Cloud TPU build. Training,
validation-based selection, and final held-out evaluation use one resolved
backend for all candidates. The final test comparison consumes that test
split; retire it rather than tuning against the result.

HH-RLHF is prepared only as `chosen`/`rejected` preference data. The current
SFT workflows and rank lab do not consume it because the lab has no pairwise
preference objective yet. See
[the full data and experiment guide](../../docs/DATASETS_AND_LORA_EXPERIMENTS.md)
for dataset-card and license links, sample-size guidance, and every
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
weights, and then saves the ordinary child `.rift` bundle. See
[the LoRA guide](../../docs/LORA.md) for target selection, direct APIs, and
the adapter lifecycle. Active FP32 LoRA runs can be resumed with
`TrainingCheckpoint`; still-packed QLoRA state is not supported by checkpoint
v1.

For a packed frozen base, select QLoRA:

```bash
python3 examples/python/post_train_stage.py \
    --backend auto \
    --fine-tuning-method qlora \
    --nf4-block-size 64 \
    --lora-rank 4 \
    --lora-alpha 8
```

QLoRA defaults to double-quantized scales and bounded-page Adam state. Eligible
base matrices remain packed during training on CPU, Metal, CUDA, or TPU; only
the floating-point LoRA factors receive gradients and moments. Paged state
still contains two FP32 moments per trainable adapter scalar. CUDA pages are
managed allocations, not an OS spill or page-fault manager. The final `.rift`
bundle is deliberately materialized as ordinary merged FP32 weights. See
[the QLoRA guide](../../docs/QLORA.md) for storage accounting and hardware
validation status.

The scripts' `auto` setting selects TPU when available, then CUDA, Metal, and
CPU. A backend can also be selected explicitly; an unavailable explicit
backend is an error:

```bash
python3 examples/python/train_tiny.py --backend cpu --steps 3
```

The accepted explicit names are `cpu`, `metal`, `cuda`, and `tpu`.

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
RIFTCO_TRANSFORMER_LIBRARY="$PWD/build/release/libriftco_transformer_c.dylib" \
python3 examples/python/train_tiny.py
```

This override is for native development and is unnecessary with a released
wheel. Use `libriftco_transformer_c.so` instead on Linux or
`riftco_transformer_c.dll` on Windows. The model must move to its backend before
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
