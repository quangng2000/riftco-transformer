# Hugging Face Data and LoRA-Rank Experiments

This workflow prepares bounded samples from Hugging Face, pretrains a small
base model, and compares LoRA ranks under the same experimental controls. It
uses only Python's standard-library `urllib`, `json`, hashing, and filesystem
modules. It does not require `datasets`, `huggingface_hub`, NumPy, PyTorch, or
another machine-learning framework.

The examples are for learning and pipeline verification. Small samples and
the lab's small model do not produce a production-quality assistant.

## Audited dataset presets

| Preset | Source and license | Prepared representation | Intended use |
| --- | --- | --- | --- |
| `tinystories` | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), CDLA Sharing 1.0 | UTF-8 `train.txt`, `validation.txt`, and `test.txt` corpora | Causal-language-model pretraining |
| `dolly` | [Databricks Dolly 15K](https://huggingface.co/datasets/databricks/databricks-dolly-15k), CC BY-SA 3.0 | JSONL `prompt`, `response`, and `category` records | Supervised instruction post-training and LoRA-rank comparison |
| `hh-rlhf` | [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf), MIT | JSONL `chosen` and `rejected` pairs | Data preparation only; a future preference objective |

Dataset licenses and usage conditions are separate from this repository.
Review the linked cards before redistributing data or trained artifacts.
HH-RLHF can also contain disturbing dialogue. Its own dataset card says the
preference pairs are not intended for supervised dialogue-agent training, and
the lab does not silently reinterpret them as SFT records.

The adapters make the source-to-lab mapping explicit:

- TinyStories' `text` becomes one plain-text document.
- Dolly's `instruction` becomes `prompt`. When `context` is nonempty, the
  prompt becomes `instruction + "\n\nContext:\n" + context`. `response` and
  `category` are preserved.
- HH-RLHF's `chosen` and `rejected` strings remain a preference pair.

For Dolly, record identity uses only `prompt` and `response`; changing or
duplicating category metadata cannot put the same example in different
partitions.

## Dependency-free preparation contract

`examples/python/prepare_huggingface_data.py` calls the official Hugging Face
Dataset Viewer [`/splits` and `/rows`
API](https://huggingface.co/docs/dataset-viewer/rows). The API limits each
`/rows` request to 100 records, so the client paginates and validates every
response. `HF_TOKEN` is an optional environment variable for access that
requires authentication; it is deliberately not a command-line argument and
is never written to the manifest. The standard-library transport rejects all
HTTP redirects before a second request is made, so an `Authorization` header
cannot be forwarded to a different origin or an HTTPS-to-HTTP destination.

The default `--selection seeded_pages` chooses deterministic API pages spread
across the eligible source range. This avoids always taking the first `N`
rows, but it is page-clustered sampling rather than a claim of statistically
uniform row sampling. Use `--selection sequential` when contiguous source
rows are intentional.

Each adapted record is assigned by a seeded SHA-256 content hash. Assignment
does not depend on download order, and exact duplicates share an identity, so
they cannot leak across train, validation, and test. Duplicates are removed
before serialization. Split fractions must be nonnegative and sum to one.
For a very small sample, hash assignment can legitimately leave a partition
empty; inspect the printed/manifest counts and increase `--limit` before a
three-way experiment.

Preparation writes and synchronizes a staging directory, then uses an atomic
platform no-replace rename. A destination that appears concurrently—even an
empty directory—is never replaced. The implementation fails closed when the
platform or filesystem cannot guarantee that contract. Every prepared
directory contains
`manifest.json` with:

- dataset, configuration, source split, dataset-card URL, and declared
  license;
- the revision reported by the server's `X-Revision` header, when available,
  plus a revision URL;
- the exact selected source-row ranges and selection settings;
- adapter, serializer, identity fields, split fractions, and split seed;
- source, duplicate, adapted, and partition counts;
- Dolly category counts for inspection; and
- file sizes, record counts, and SHA-256 digests.

Prepared downloads belong under `data/external/`, which is ignored by Git.
Experiment artifacts belong under `results/`, which is also ignored. Commit
the code and small hand-authored fixtures, not downloaded corpora or generated
model artifacts.

## Build and environment

From the repository root, install the Python package and bundled native
library, then run the examples directly:

```bash
python3 -m pip install .
```

After a release is published, `python3 -m pip install riftco-transformer` installs
the matching self-contained platform wheel instead. Neither installation adds
third-party runtime dependencies. `RIFTCO_TRANSFORMER_LIBRARY` remains available
only as an explicit override for testing a particular local native build.

Inspect the exact preparation controls at any time:

```bash
python3 examples/python/prepare_huggingface_data.py --help
```

## Stage 1: TinyStories pretraining

Keep TinyStories' published training and validation sources distinct. The
first command assigns every sampled training story to `train.txt`; the second
assigns every sampled official validation story to `validation.txt`:

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
```

Pretrain with the two files explicitly:

```bash
python3 examples/python/pretrain_stage.py \
  --file data/external/huggingface/tinystories-train/train.txt \
  --validation-file \
    data/external/huggingface/tinystories-validation/validation.txt \
  --output results/stages/tinystories_pretrained.rift \
  --backend cpu \
  --steps 100 \
  --context 32 \
  --batch-size 4 \
  --tokenizer bpe \
  --vocab-size 512 \
  --model-width 32 \
  --heads 4 \
  --blocks 2 \
  --feed-forward-width 64
```

With `--validation-file`, the tokenizer is fitted only on training text and
the supplied validation corpus remains held out. The `--validation-fraction`
option is ignored in this explicit-split path. The stage rejects aliased
files and identical effective UTF-8 corpora. Its model artifact records the
SHA-256 of the exact training and validation text consumed, so those values
can be checked directly against the prepared dataset manifest.

The example above is a CPU-friendly pipeline run, not a quality recipe. A
useful progression on the current small architecture is:

| Purpose | TinyStories train/validation rows | Dolly rows | Rank sweep |
| --- | ---: | ---: | --- |
| Fast CPU smoke test | 300 / 100 | 300 | `1,2,4`, 5–20 steps |
| Longer CPU experiment | 1,000 / 200 | 1,000 | `1,2,4,8`, 20–50 steps |
| Metal experiment | 10,000 / 1,000 | 2,000–5,000 | `1,2,4,8`, 50–200 steps |

Actual time depends on model dimensions, context length, and machine. Increase
one axis at a time and record it; row counts alone do not determine token
counts or model quality.

## Stage 2: Dolly and LoRA-rank comparison

Prepare deterministic, disjoint Dolly splits:

```bash
python3 examples/python/prepare_huggingface_data.py \
  --preset dolly \
  --output data/external/huggingface/dolly-lora-v1 \
  --limit 2000 \
  --seed lora-v1 \
  --train-fraction 0.8 \
  --validation-fraction 0.1 \
  --test-fraction 0.1
```

The default sampler chooses each instruction example uniformly and then
chooses one causal window within that example. This prevents long instructions
from receiving more training probability merely because they contain more
token windows. `window_uniform` remains available as an explicit comparison.

Run the controlled rank sweep from the same immutable pretrained artifact:

```bash
python3 examples/python/compare_lora_ranks.py \
  --base results/stages/tinystories_pretrained.rift \
  --data data/external/huggingface/dolly-lora-v1 \
  --output results/experiments/dolly-lora-ranks \
  --ranks 1,2,4,8 \
  --alpha-over-rank 2 \
  --steps 50 \
  --context 16 \
  --batch-size 2 \
  --learning-rate 0.001 \
  --seed 29 \
  --adapter-seed 5489 \
  --sampling-strategy example_uniform \
  --backend cpu \
  --prompt "Explain a tensor to a beginner." \
  --max-new-tokens 16
```

Use `--backend metal` on a Metal-capable Mac, `--backend cuda` with the
optional CUDA Toolkit 12+ source build, or `--backend tpu` with the
experimental Linux x86-64 Cloud TPU build. CUDA runs its NN, matmul, attention,
loss, and Adam candidate-state capabilities as native kernels, while its graph
traversal and gradient norm remain host control flow. TPU accelerates matmul,
materialized attention, and paged decode while retaining reference paths for
Flash and the other operations. These remain functional experiment backends
rather than broad performance claims. The TPU
path still requires real-hardware
validation. The base artifact's maximum context must be at least the experiment's
`--context` value. The `--output`
directory must not exist: the CLI stages every rank artifact and
`comparison.json` together, then atomically publishes the complete directory
without replacing an earlier run. A failed run removes its staging directory.

### What makes the comparison fair

Every candidate:

1. starts from a fresh runtime instantiated from the same immutable base
   artifact;
2. uses the same train, validation, and test fingerprints;
3. uses the same training seed, adapter-initialization seed, sampling
   strategy, optimizer steps, batch size, context, learning rate, LoRA
   targets, attention/checkpointing policy, and resolved backend; and
4. keeps `alpha / rank` constant by setting `alpha = rank ×
   alpha_over_rank`.

The query and value projections are the default LoRA targets. Rank changes
adapter capacity and trainable parameter count; the shared controls reduce
confounding but do not turn a small sweep into a statistically conclusive
benchmark.

The loader first rejects exact prepared-record overlap. The experiment also
formats examples with the configured instruction template and rejects overlap
again on the normalized model input. This prevents whitespace differences in
raw prompt/response records from hiding leakage after formatting.

All ranks are evaluated on validation data. The lowest validation loss wins,
with lower rank as the deterministic tie-breaker. Only after that selection
is fixed does the experiment evaluate held-out test data for the base model
and the selected candidate. The test split therefore does not choose the
rank. `comparison.json` records configuration, dataset fingerprints,
baseline measurements, per-rank validation results, the selected test result,
artifact paths, and optional greedy inference samples. Its
`dataset_provenance` contains the full verified preparation manifest and the
manifest's own SHA-256, including source revision/ranges/license and split
file hashes.

Evaluation is read-only and weights loss by target-token count. It scores each
formatted sequence in deterministic non-overlapping causal chunks. Context
does not carry across chunk boundaries, so learned positions restart at zero
for each chunk.

### Objective and inference limitations

Current post-training uses `full_sequence_causal_sft`. Loss covers the entire
formatted sequence—delimiters, user prompt, and assistant response—not only
the response. A response-only mask is not implemented, so interpret
validation and test loss as full-sequence causal loss.

Each trial merges its LoRA factors into ordinary base weights before saving.
Consequently all saved ranks have the same serving topology and use the same
paged-decode-attention path. Generation timings are smoke measurements only:
do not
claim that one merged rank serves faster based on this experiment. Rank
changes training-time adapter capacity and parameter count, not the topology
of the merged serving artifact.

## Stage 3: full fine-tuning versus LoRA

Use the same verified prepared dataset to measure both post-training methods
with one held-out metric:

```bash
python3 examples/python/compare_fine_tuning.py \
  --base results/stages/tinystories_pretrained.rift \
  --data data/external/huggingface/dolly-lora-v1 \
  --output results/experiments/dolly-full-vs-lora \
  --methods full,lora \
  --lora-ranks 1,2,4,8 \
  --alpha-over-rank 2 \
  --steps 50 \
  --context 16 \
  --batch-size 2 \
  --full-learning-rate 0.001 \
  --lora-learning-rate 0.005 \
  --seed 29 \
  --adapter-seed 5489 \
  --sampling-strategy example_uniform \
  --backend cpu
```

The full recipe and every LoRA rank start independently from the same base.
All candidates receive exhaustive train and validation measurements. The
lowest-validation-loss LoRA rank is fixed before any test forward pass; then
the base, full recipe, and selected LoRA recipe receive final test scores.
Nonwinning ranks keep `null` test fields.

`comparison.json` reports held-out loss/perplexity and the generalization gaps
`validation_loss - train_loss` and `test_loss - train_loss`. It also records
trainable parameter counts and fractions, making the memory/capacity tradeoff
visible beside quality. Full and LoRA have separate learning-rate controls;
the run compares the supplied recipes, not the best possible version of each
method.

Because the command reports test results for both methods, the test split is
consumed for that final comparison. Do not adjust rank, learning rate, steps,
or another choice from those results and rerun the same test. See
[Post-training generalization](GENERALIZATION.md) for the metric formulas,
native C++ API, and interpretation limits.

## Preference data is a future stage

The following command can prepare HH-RLHF for inspecting a future
chosen-versus-rejected objective:

```bash
python3 examples/python/prepare_huggingface_data.py \
  --preset hh-rlhf \
  --output data/external/huggingface/hh-rlhf-preference \
  --limit 1000 \
  --seed preference-v1
```

The result contains `chosen` and `rejected`, not `prompt` and `response`.
`post_train_stage.py` and `compare_lora_ranks.py` intentionally reject that
schema. The lab does not yet implement pairwise reward-model training, DPO,
PPO, or another preference objective.

## Reproducibility checklist

- Preserve each prepared `manifest.json` with experiment records.
- Preserve the published `comparison.json`; its embedded manifest and
  `manifest_sha256` bind the run to its prepared source.
- Record the base artifact ID and the native/Python revision used to run.
- Keep dataset, source split, offset, limit, selection strategy, page size,
  split seed, and split fractions fixed.
- Keep all rank-sweep controls fixed except rank and the derived alpha.
- Select on validation, then inspect each final method's test result once and
  retire that test split.
- Treat generated text and latency as qualitative smoke checks.
- Do not compare runs that used different data fingerprints as if only rank
  changed.
