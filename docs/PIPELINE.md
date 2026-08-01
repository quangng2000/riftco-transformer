# Staged Model Pipeline

The lab exposes the same three-stage lifecycle through two surfaces:

- native C++ composition roots for explicit in-process wiring; and
- a higher-level Python workflow with persisted, immutable artifacts and a
  local HTTP adapter.

Both surfaces use the same native model implementation; Python reaches it
through the C ABI. Their handoff contracts are intentionally different.

## Native C++ composition roots

```text
UTF-8 corpus
  → PretrainingStack
  → in-memory ModelSnapshot
  → PostTrainingStack
  → in-memory ModelSnapshot
  → ServingStack
```

The composition roots live below
`include/riftco_transformer/stages/{pretraining,post_training,serving}` with
matching implementations below `src/stages/`.

### Pretraining

`PretrainingStack` owns the tokenizer, model, native Adam optimizer, Adam
strategy adapter, causal-language-model trainer, and random-window batch
source for one run. It returns metrics and a `ModelSnapshot` after performing
the repeated forward → cross-entropy → backward → global clipping → Adam
transaction.

The `riftco-transformer` command-line executable, whose source is
`apps/pretraining/train.cpp`, now constructs this stack rather than assembling
the training loop itself. It can write CSV metrics, but its snapshot remains
in memory and disappears when the process exits.

### Supervised post-training

`PostTrainingStack` accepts a const base snapshot, restores a separate model
and tokenizer, formats instruction examples, and trains with a fresh Adam
optimizer through the same stage-neutral trainer. Its sequence-window batch
source never crosses from one formatted example into another. The returned
snapshot contains the updated copy; the input snapshot is not mutated.

`PostTrainingConfig::fine_tuning_method` selects the parameter-update policy.
`Full` is the compatibility default and gives Adam every base-model parameter.
`Lora` attaches low-rank factors and gives Adam only those adapter parameters.
`Qlora` first packs every eligible Linear base weight as blockwise NF4, then
attaches the same floating-point LoRA factors and gives Adam only those
adapters. Quantized weights are immutable buffers rather than Parameters, so
they have neither gradients nor Adam moments.
QLoRA defaults to double-quantized first-level scales with scale-group width
256 and bounded-page Adam with 4096-element pages. The config exposes legacy
FP32 scales and contiguous Adam as explicit alternatives. CPU uses the readable
packed reference path, Metal and CUDA decode inside kernels, and TPU uploads
packed U8 data and dequantizes inside StableHLO. CUDA and TPU source paths are
implemented, although actual NVIDIA/Cloud-TPU hardware validation was not
available on this macOS host.

`LoraConfig` controls rank, alpha, initialization seed, and target
projections; its default targets are the query and value projections in every
transformer block.

At the end of a LoRA or QLoRA run, the stack releases the trainer, optimizer
adapter, and Adam before performing the one-way merge. QLoRA also explicitly
dequantizes every packed projection during this export step. It then captures
the same ordinary FP32 base-parameter schema used by full training. The returned
`ModelSnapshot` is therefore directly consumable by `ServingStack`; it
contains merged weights, not separate adapter factors. Capturing or restoring
model state while an unmerged adapter is active is rejected, preventing an
accidentally incomplete handoff.

The QLoRA export has a documented one-time FP32 memory spike after training.
During forward, backward, and Adam steps, the base linear weights remain packed.
Paged Adam bounds individual moment allocations and update requests but still
stores two FP32 moment values per trainable adapter scalar. CUDA pages use
managed memory; this is not an OS spill, eviction, disk-paging, or page-fault
manager.

See [QLORA.md](QLORA.md) for the storage and kernel contracts.

The native objective is currently `full_sequence_causal_sft`.
Cross-entropy applies to every shifted token in the formatted sequence,
including the prompt and response. A response-only loss mask is not
implemented.

### Serving

`ServingStack` restores only the tokenizer and model needed for inference. Its
dependency boundary includes `artifacts`, `model`, and `data`, and deliberately
excludes `training` and `optim`.

The native generation engine supports greedy decoding and seeded
temperature/top-k sampling. It creates a request-local KV cache, prefills the
prompt one token at a time, then decodes one token at a time. Paged caching is
the default; a contiguous reference factory is available through the same
cache interface. CPU, Metal, and CUDA execute attention through backend-owned
paged implementations; TPU stages paged decode through PJRT from its host
mirror. The full-sequence,
autograd-producing training forward is unchanged.

At context rollover, learned absolute positions require the retained suffix to
be reset and replayed from position zero. Evicting only the oldest page would
leave different position IDs and therefore different K/V state. The engine
returns exact decoded bytes. This native layer is an in-process generation API;
it does not provide an HTTP server. See [SERVING.md](SERVING.md) for the cache
layout and session contracts.

### Native snapshot contract

`ModelSnapshot` is a value-like, in-memory handoff. It contains:

- transformer dimensions and layer-normalization epsilon;
- ordered named parameter shapes and float values; and
- exact corpus-byte vocabulary or ordered BPE merge state.

It deliberately contains no Adam moments or step, random-engine state, batch
position, persistence format, checksum, artifact ID, metadata, or lineage.
It is sufficient to initialize another native stage in the same process, but
it is neither a native model file nor a resumable training checkpoint.

CTest registers the native artifact-state, shared-training, stage-contract,
stage-stack, and serving-generation tests. These verify the in-memory handoff
and the three native composition paths.

### Minimal native handoff

This example keeps both snapshots in one process:

```cpp
#include "riftco_transformer/stages/stages.hpp"

#include <iostream>
#include <string>
#include <vector>

namespace pre = riftco_transformer::stages::pretraining;
namespace post = riftco_transformer::stages::post_training;
namespace serve = riftco_transformer::stages::serving;

int main() {
    pre::PretrainingConfig pre_config;
    pre_config.steps = 2;
    pre_config.context_size = 3;
    pre_config.batch_size = 1;
    pre_config.model_width = 4;
    pre_config.head_count = 2;
    pre_config.block_count = 1;
    pre_config.feed_forward_width = 8;
    pre_config.attention =
        riftco_transformer::FullSequenceAttentionKind::Flash;
    pre_config.tokenizer = {
        riftco_transformer::TokenizerMethod::BytePair,
        260,
        2,
    };

    const std::string corpus =
        "tensor vectors learn patterns. "
        "tensor vectors learn patterns.";
    pre::PretrainingStack pretraining(corpus, pre_config);
    const auto pretrained = pretraining.run();

    post::PostTrainingConfig post_config;
    post_config.steps = 1;
    post_config.context_size = 3;
    post_config.batch_size = 1;
    post_config.attention =
        riftco_transformer::FullSequenceAttentionKind::Flash;
    post_config.fine_tuning_method = post::FineTuningMethod::Lora;
    post_config.lora.rank = 2;
    post_config.lora.alpha = 4.0F;
    post::PlainChatFormatter formatter;
    post::PostTrainingStack post_training(
        pretrained.snapshot,
        std::vector<post::InstructionExample>{
            {"What is a tensor?", "A multidimensional array."},
        },
        formatter,
        post_config
    );
    const auto tuned = post_training.run();

    serve::ServingConfig serve_config;
    serve_config.maximum_new_tokens = 4;
    serve_config.kv_cache_kind = serve::KvCacheKind::Paged;
    serve_config.kv_cache_block_size = 16;
    serve::ServingStack serving(tuned.snapshot, serve_config);
    const auto generated =
        serving.generate("Tensor:", serve::GenerationConfig{4});
    std::cout << generated.text << '\n';
}
```

`pretrained.snapshot` and `tuned.snapshot` are ordinary in-memory values here.
The latter contains the merged LoRA update and no live adapter. This example
does not create a native model file.

## Native snapshot versus Python bundle

| Property | Native `ModelSnapshot` | Python `ModelBundle` |
| --- | --- | --- |
| Primary purpose | In-process stage handoff | Persisted stage handoff and distribution |
| Storage | In memory only | Immutable versioned ZIP |
| Model/tokenizer state | Yes | Yes |
| Checksum and artifact identity | No | Yes |
| Metadata and parent lineage | No | Yes |
| Optimizer, RNG, and batch progress | No | No |
| LoRA stage result | Merged base weights | Merged base weights |

The Python bundle is not a serialized form of the native snapshot contract,
even though both carry enough model and tokenizer state to reconstruct an
inference runtime. Neither contract can resume the exact next training step.

## Python persisted workflow

```text
prepared UTF-8 corpus
  → self-supervised pretraining
  → immutable base ModelBundle
  → prepared prompt/response splits
  → supervised Full/LoRA/QLoRA post-training or controlled rank selection
  → immutable child ModelBundle
  → local generation service
```

Each arrow is a narrow handoff. Training code owns mutable model and optimizer
state only while a stage is running. A `ModelBundle` is the portable,
read-only result passed to the next Python stage or to serving.

### External dataset preparation

`python/riftco_transformer/data` is an input-boundary package, not part of the
training engine. It uses the standard-library `urllib` and `json` modules to
read bounded slices from Hugging Face's official Dataset Viewer API. Dataset
adapters and serializers convert three audited presets:

- TinyStories to blank-line-separated UTF-8 text for pretraining;
- Dolly 15K to `prompt`/`response`/`category` JSONL for SFT; and
- HH-RLHF to `chosen`/`rejected` JSONL for future preference training only.

A seeded SHA-256 splitter assigns canonical record identities independently
of source order. Exact duplicates are removed before serialization and cannot
cross partitions. The output directory is published atomically and includes
a manifest with source coordinates, server-reported revision, license,
selected row ranges, split controls, counts, and per-file SHA-256. Generated
datasets live below ignored `data/external/`.

TinyStories pretraining should pass separately prepared published train and
validation files to `pretrain_stage.py --file ... --validation-file ...`.
That keeps the validation source held out and fits the tokenizer only on
training text. Full commands, license links, and sample-size guidance are in
[DATASETS_AND_LORA_EXPERIMENTS.md](DATASETS_AND_LORA_EXPERIMENTS.md).

### Python pretraining

`pretrain_text()` and `pretrain_file()` implement causal next-token
pretraining. They split one raw text into training and validation regions
before fitting the tokenizer. `pretrain_splits()` and `pretrain_files()` keep
already separated training and validation corpora explicit instead. All paths
create shifted token windows and run the shared forward → cross-entropy →
backward → Adam transaction.

`PretrainingConfig.attention` selects `"materialized"` (the default) or
`"flash"` for the full-sequence forward/backward path. The same runtime policy
exists on `PostTrainingConfig`; it changes neither parameter state nor the
persisted artifact schema.

Both configs also expose
`activation_checkpointing="disabled"|"block"`. Block mode reconstructs each
transformer block during backward and is recorded in training metadata, but
it is not part of model configuration or the persisted artifact schema.

The result contains metrics and a `ModelBundle` with stage `"pretraining"`.
The bundle captures:

- the complete transformer configuration;
- the byte vocabulary or ordered BPE merge rules;
- ordered parameter names and shapes;
- little-endian float32 parameter values;
- SHA-256 fingerprints of the exact effective training and validation text;
- stage metadata and a content-derived artifact ID.

Explicit training and validation inputs must be distinct. The pipeline rejects
both paths that identify the same file and separate files whose consumed text
is identical.

The Adam optimizer is temporary. Its moments and step count are not included
in the bundle.

### Python immutable artifact boundary

`ModelBundle` is the persisted contract between Python pretraining,
post-training, and serving. Saving it creates a versioned ZIP artifact
containing exactly `manifest.json` and `weights.f32le`. Loading validates the
format version, tokenizer/model compatibility, parameter identities and
shapes, byte count, weight checksum, and artifact ID before a live model is
created. The manifest's canonical format identifier is
`riftco-transformer-model-bundle`, and persisted bundles use the `.rift`
extension.

Post-training never mutates its input bundle. It instantiates a live copy,
trains that copy, and captures a new child bundle whose
`parent_artifact_id` points to the base artifact. Serving likewise instantiates
its own closeable runtime from a bundle. `ModelBundle.capture()` rejects a
model with an active adapter; the adapter must first be merged into the base
weights.

#### Model bundle versus training checkpoint

These are intentionally different persistence contracts:

- `ModelBundle` is the current inference, distribution, and stage-handoff
  contract. It stores model configuration, tokenizer state, named weights,
  lineage, and metadata.
- `TrainingCheckpoint` is a future exact-resumption contract. It must add Adam
  moments and step, the training step, data position, random-generator states,
  and any schedule state.

Loading a `ModelBundle` and creating a new Adam optimizer starts a new training
stage. It does **not** reproduce the next update of the run that created the
bundle. Until `TrainingCheckpoint` exists, do not describe model-bundle
save/load as resumable training.

### Python supervised post-training

`post_train()` accepts `InstructionExample` values. `post_train_jsonl()` loads
UTF-8 JSON Lines records with this shape:

```json
{"prompt":"What is a tensor?","response":"A multidimensional array."}
```

`PlainChatFormatter` renders each record as readable user/assistant text.
The default `ExampleWindowBatchSource` selects an instruction uniformly and
then one causal window inside it, so longer examples do not receive more
sampling probability just because they contain more windows.
`sampling_strategy="window_uniform"` selects `SequenceWindowBatchSource` as
an explicit alternative. Neither source crosses from one formatted example
into another. Post-training creates a fresh Adam optimizer and returns a child
bundle with stage `"post_training"`.

`PostTrainingConfig(fine_tuning_method="full")` preserves full-parameter
training. Selecting `"lora"` and supplying a native `LoraConfig` restricts
Adam to adapter parameters:

```python
from riftco_transformer import LoraConfig
from riftco_transformer.post_training import PostTrainingConfig

config = PostTrainingConfig(
    attention="flash",
    activation_checkpointing="block",
    fine_tuning_method="lora",
    lora=LoraConfig(
        rank=4,
        alpha=8.0,
        targets=("attention.query", "attention.value"),
    ),
)
```

Selecting `"qlora"` uses the same LoRA factors over an NF4-packed frozen base:

```python
config = PostTrainingConfig(
    backend="auto",
    fine_tuning_method="qlora",
    nf4_block_size=64,
    double_quantization=True,
    nf4_scale_block_size=256,
    optimizer_state="auto",       # resolves to paged for QLoRA
    optimizer_page_size=4096,
    lora=LoraConfig(rank=4, alpha=8.0),
)
```

The ordinary `auto` backend order—TPU, CUDA, Metal, then CPU—applies to QLoRA.
`optimizer_state="contiguous"` opts out of its default paged state, while
`double_quantization=False` selects legacy FP32 block scales.

After training, Python closes the adapter parameter list and Adam handle,
merges the factors, explicitly dequantizes a QLoRA base when needed, and
captures a normal child `ModelBundle`. Metadata records the method, parameter
scope, quantization, and optimizer-state diagnostics, but the persisted weights
use the standard base-model schema. This release does not define an adapter-only
or packed-weight artifact or an unmerge operation.

The current objective is deliberately named
`full_sequence_causal_sft`. Cross-entropy applies to every shifted token in
the formatted sequence: delimiters, user prompt, and assistant response.
There is no response-only loss mask yet, so this is not the masked instruction
loss commonly used by production post-training systems.

### Full-versus-LoRA generalization

`compare_fine_tuning()` applies one exhaustive metric to full fine-tuning and
LoRA. Every candidate starts from the same immutable base, trains only on the
training split, and is scored on training and validation. The training score
uses the same exhaustive target-token weighting as validation, so the reported
validation generalization gap is comparable rather than a difference from one
sampled minibatch.

Candidates are selected within method groups using validation loss. A fixed
full-fine-tuning recipe and the validation-selected LoRA rank receive final
test measurements only after all selections are frozen. The accompanying
`compare_fine_tuning.py` command records separate learning rates, trainable
parameter fractions, split fingerprints, base deltas, and validation/test
generalization gaps. Using the test result for both methods consumes that test
split; it must not guide another tuning run.

The native `PostTrainingStack` exposes the same exhaustive split metric through
its `InstructionSplits` overload and optional `PostTrainingEvaluationMetrics`.
That overload is for one pre-registered candidate and evaluates test on each
run; grouped validation selection remains in `compare_fine_tuning()`. The
legacy single-example-vector constructor remains training-only. See
[Post-training generalization](GENERALIZATION.md) for the command, formulas,
and interpretation limits.

### Controlled LoRA-rank selection

`python/riftco_transformer/experiments` adds a reproducible rank sweep above the
post-training API. `load_prepared_instruction_splits()` verifies the prepared
manifest and file hashes before loading disjoint train, validation, and test
JSONL. It rejects both exact record overlap and overlap in the formatted,
whitespace-normalized model inputs. `compare_lora_ranks()` then:

1. instantiates every rank from the same immutable base bundle;
2. fixes dataset fingerprints, training and adapter seeds, sampler, optimizer
   controls, target projections, backend, and `alpha / rank`;
3. trains on only the training split and scores every rank on validation;
4. selects the lowest validation loss, breaking ties by lower rank; and
5. evaluates test data only after selection, for the base model and winner.

The evaluator performs no backward pass. It scores every usable target once
in deterministic causal chunks and weights the mean by target-token count.
Like training, this remains full-sequence causal loss; context and learned
positions restart at each evaluation chunk.

The CLI stages merged serving-ready artifacts and `comparison.json`, then
atomically publishes the whole output directory without replacing an existing
run. Failures remove staging. The summary contains configuration,
fingerprints, baseline results, per-rank validation results, the selected test
result, optional greedy generations, and the full verified prepared-data
manifest plus its SHA-256. Because merge folds every adapter into the ordinary
weight topology, different ranks do not have different serving architectures.
Generation latency is reported only as a smoke measurement, not as evidence
that one rank serves faster.

HH-RLHF is deliberately outside this path. The current lab has no DPO,
pairwise reward-model, PPO, or other preference objective, and does not treat
chosen/rejected pairs as SFT examples.

### Python generation and local serving

`TextGenerator` performs single-request autoregressive generation. For a
native `DecoderOnlyTransformer`, it uses the current stable ABI 2.4
`DecodeSession` surface, prefills one token at a time,
and then appends one generated token per step.
Paged caching with 16-token pages is the default; callers can select the
contiguous reference strategy. Protocol-style alternate models retain the
full-forward compatibility path.

When a native session reaches `model.config.maximum_context`,
`TextGenerator` resets it and replays the retained suffix so learned absolute
positions are rebased exactly as in the original cropped full-forward path.
There is no EOS policy yet, so generation produces exactly the requested
number of new tokens. Results retain raw bytes and also provide
replacement-decoded UTF-8 text.

`ModelService` owns one instantiated bundle and exposes the same generation
path in process. The dependency-free Python HTTP adapter adds:

- `GET /` and `GET /chat` for a self-contained browser chat;
- `GET /health` and `GET /v1/health`;
- `POST /v1/generate`;
- request-size and generated-token limits;
- greedy decoding with `"temperature": 0`, or seeded sampling with a positive
  temperature and optional `top_k`.

The chat sends each visible message as an independent
`### User: ... ### Assistant:` request. The transcript is not put back into
the prompt because the bundled learning models have very small contexts and
the current SFT formatter is single-turn. This built-in page assumes the
`PlainChatFormatter` delimiter; the current artifact contract does not persist
an arbitrary inference chat template. The page has no external assets or
JavaScript dependencies.

The server defaults to `127.0.0.1` because it is a local learning adapter, not
a production gateway. Its threaded HTTP frontend serializes generation
through one model runtime. A loopback-bound server rejects non-loopback
`Host` headers, and JSON generation requests must use
`Content-Type: application/json`. Explicit remote binding remains a
trusted-network development option, not a production security boundary.

## Run the complete persisted example

From the repository root, install the Python package and bundled native
library:

```bash
python3 -m pip install .
```

After a release is available on PyPI, a self-contained wheel installed with
`python3 -m pip install riftco-transformer` can run the same stage commands.

Pretrain the bundled text and save the base artifact:

```bash
python3 examples/python/pretrain_stage.py \
  --file data/pretraining/tiny_corpus.txt \
  --output results/stages/tiny_pretrained.rift \
  --backend auto \
  --attention flash \
  --steps 10
```

Post-train it on the bundled instruction records and save a child artifact:

```bash
python3 examples/python/post_train_stage.py \
  --base results/stages/tiny_pretrained.rift \
  --instructions data/post_training/tiny_instructions.jsonl \
  --output results/stages/tiny_post_trained.rift \
  --backend auto \
  --attention flash \
  --steps 5
```

Start the loopback-only Python server:

```bash
python3 examples/python/serve_stage.py \
  --bundle results/stages/tiny_post_trained.rift \
  --host 127.0.0.1 \
  --port 8000 \
  --backend auto
```

Open `http://127.0.0.1:8000/` for the local chat, or in another terminal
inspect it and generate exactly 16 new tokens:

```bash
curl --fail --silent http://127.0.0.1:8000/health

curl --fail --silent \
  --header 'Content-Type: application/json' \
  --data @- \
  http://127.0.0.1:8000/v1/generate <<'JSON'
{
  "prompt": "### User:\nWhat is attention?\n### Assistant:\n",
  "max_new_tokens": 16,
  "temperature": 0
}
JSON
```

These defaults are a wiring demonstration, not a useful assistant-quality
training recipe. Pass `--help` to any stage script for its resource and
training controls. For TinyStories → Dolly rank selection and full-versus-LoRA
generalization workflows, use the commands in
[DATASETS_AND_LORA_EXPERIMENTS.md](DATASETS_AND_LORA_EXPERIMENTS.md).

## Current limits

The staged boundaries are usable, but the runtime remains intentionally
small:

- native `ModelSnapshot` has no persistent format, checksum, identity, or
  lineage;
- LoRA handoff is merge-only; adapter-only persistence and continued adapter
  training after merge are not implemented;
- native serving is in process only; local HTTP belongs to the Python layer;
- no masked response-only post-training loss;
- no pairwise preference-training objective for prepared HH-RLHF data;
- no resumable `TrainingCheckpoint`;
- prefill is one token at a time; no batched-prefill kernel;
- paged KV caching does not yet include request batching, continuous batching,
  a scheduler, or immutable-prefix sharing;
- suffix replay is required at context rollover by learned absolute positions;
- no streaming tokens or multi-model scheduling;
- no authentication, authorization, rate limiting, or TLS;
- no EOS/stop-token policy;
- no claim of production model quality from the tiny example data.

Place authentication and TLS in a real gateway before any non-local Python
deployment. Future batching, scheduling, and prefix reuse belong below the
serving interface and should not change either model/tokenizer handoff
contract.
