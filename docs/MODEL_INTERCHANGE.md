# Convert model artifacts safely

Riftco Transformer provides dependency-free adapters for complete native
models and selected industry containers. Format parsing and architecture
compatibility are separate checks: reading a tensor file never authorizes the
runtime to reinterpret weights from a different model topology.

## Format roles

| Contract | Primary role | Import | Export |
| --- | --- | --- | --- |
| `.rift` | Native inference/distribution bundle | Yes | Yes |
| SafeTensors | Named tensor container | F32 tensors | F32 tensors |
| Riftco Hugging Face-style directory | Config + tokenizer + SafeTensors for `riftco_decoder_v1` | Yes | Yes |
| GGUF v3 | Single-file model container | Riftco FP32 architecture only | Yes |
| ONNX opset 18 + Riftco sidecar | Portable inference graph | Exact Riftco exports | Yes |
| `.riftckpt` | Exact Adam-run continuation | Yes | Yes |

SafeTensors alone does not contain enough architecture and tokenizer semantics
to reconstruct a complete model. Use its lower-level API for tensors or the
Hugging Face-style directory adapter for a complete Riftco model.

## Convert a complete model

The high-level conversion API requires explicit source and destination
formats. It never guesses from a filename:

```python
from riftco_transformer.interchange import convert_model

convert_model(
    "model.rift",
    "hf-model",
    source_format="rift",
    destination_format="huggingface",
)

convert_model(
    "hf-model",
    "model.gguf",
    source_format="huggingface",
    destination_format="gguf",
    gguf_model_name="My Riftco Decoder",
)

convert_model(
    "model.rift",
    "model.onnx",
    source_format="rift",
    destination_format="onnx",
)

convert_model(
    "model.onnx",
    "restored.rift",
    source_format="onnx",
    destination_format="rift",
)
```

`load_model()` and `export_model()` accept `"rift"`, `"huggingface"`,
`"gguf"`, and `"onnx"`. ONNX import is deliberately narrower than general
ONNX execution: it accepts only an unchanged canonical Riftco export with its
required adjacent sidecar.

## Use SafeTensors directly

The zero-dependency SafeTensors adapter currently accepts finite F32 tensors:

```python
from riftco_transformer.interchange import (
    Float32Tensor,
    load_safetensors,
    save_safetensors,
)

save_safetensors(
    {"projection.weight": Float32Tensor((2, 2), (1.0, 0.0, 0.0, 1.0))},
    "weights.safetensors",
    metadata={"producer": "riftco-transformer"},
)
parsed = load_safetensors("weights.safetensors")
```

The parser validates the header and tensor-count bounds, alignment, strict
JSON, unique keys, F32 dtype, tensor shapes, exact offsets, overlap/gaps,
payload bounds, and finite values. Files written here cross-load with the reference
SafeTensors implementation, and reference-written F32 files load here.

## Hugging Face-style directory boundary

The directory contains `config.json`, `model.safetensors`, `tokenizer.json`,
and `tokenizer_config.json`. It declares the explicit architecture ID
`riftco_decoder_v1` and maps every native parameter name bijectively to a
stable external name.

This is a Hugging Face-style interchange layout, not a claim that
`AutoModelForCausalLM` already registers `RiftcoDecoderForCausalLM`. Its
tokenizer sidecar also preserves Riftco byte/BPE semantics rather than
pretending to be an incompatible external tokenizer implementation.

## GGUF boundary

GGUF export uses version 3, aligned FP32 tensor payloads, and
`general.architecture = "riftco"`. Riftco configuration, tokenizer, stage,
lineage, and JSON metadata make the file losslessly importable by this
framework. The structural reader can inspect other GGUF files, but model load
rejects foreign architectures, quantized tensor types, and schema drift.

The container is valid and parses in llama.cpp's GGUF reader. llama.cpp still
needs an implementation of the custom `riftco` architecture before it can
execute the model; GGUF syntax alone does not provide that runtime support.

## ONNX boundary

ONNX export writes a real IR 8/opset 18 inference graph with dynamic
`[batch, sequence]` INT64 token IDs and
`[batch, sequence, vocabulary]` FP32 logits. The graph contains learned
positions, a generated causal mask, separate Q/K/V/O projections,
pre-LayerNorm residual blocks, exact erf GELU, final normalization, and the
language-model head. The canonical topology is identified independently from
the package release as `riftco_decoder_v1_onnx` version 1. A producer release
change therefore does not by itself invalidate a version-1 graph; a topology
or wire-policy change requires a new canonical graph version.

The input contract is `batch >= 1`, `sequence >= 1`,
`0 <= input_ids < vocabulary_size`, and
`sequence <= maximum_context`. Standard ONNX has no portable assertion
operator. The graph feeds invalid token IDs through the indispensable
embedding `Gather` as the out-of-range `vocabulary_size` sentinel, and routes
an empty input through an out-of-range guard `Gather` before constructing
positions. External inference wrappers must still validate the contract:
optimizers and runtimes do not expose a universal assertion/error API.

The exporter also writes `<model>.onnx.riftco.json`. ONNX itself does not
define the tokenizer, original training seed, artifact stage, parent identity,
or arbitrary Riftco artifact metadata needed to reconstruct a complete
`ModelBundle`; the bounded strict-JSON sidecar preserves those fields and pins
the ONNX bytes by SHA-256. Moving an ONNX file therefore requires moving its
sidecar with it. `load_onnx(..., sidecar_path=...)` can name a deliberately
relocated sidecar. These hashes prove content consistency, not producer
authenticity: the sidecar is not signed, and `parent_artifact_id` is a lineage
identifier rather than a trust certificate.

Import reads and validates the sidecar before model bytes, derives a maximum
canonical size from its checked configuration, and parses one immutable ONNX
snapshot. The eager dependency-free reader has a 1 GiB absolute wire limit and
also caps protobuf fields, nodes, initializers, strings, attributes, and tensor
rank. The derived per-model limit is normally much smaller than 1 GiB.

Import parses and validates IR/opset, producer/domain, inputs and outputs,
Riftco architecture/topology/config metadata, every node and attribute, and
every initializer name, order, shape, dtype, payload size, and value. It then
reconstructs the bundle, verifies its artifact ID, regenerates the canonical
graph, and requires exact protobuf equality. Arbitrary ONNX, rewritten graphs,
optimizer output, and even semantically similar operator substitutions are
rejected rather than guessed.

The exporter and importer use a purpose-built standard-library Protocol
Buffers implementation; the installed package does not depend on ONNX. The
committed dependency-free suite proves deterministic encoding, strict
canonical re-import, graph/input-guard structure, malformed-input bounds,
failure rollback, conversion routing, and complete byte/BPE artifact identity.
Official ONNX checker, strict shape-inference, and ONNX Runtime numerical tests
remain external release-qualification checks until a pinned optional CI lane
is added; they are not implied by the dependency-free unit suite.

Both files are staged and fsynced before publication. An ordinary failure
during either replacement restores an existing pair, or removes both files for
a new export. No portable filesystem API atomically replaces two fixed names,
so non-cooperating concurrent readers and process/power loss between the two
renames remain outside that rollback guarantee.

## Why external Llama and Mistral checkpoints are still rejected

The framework now has a separate native dense Llama/Mistral execution path
with RMSNorm, RoPE, grouped-query attention, and SwiGLU. The interchange
formats in this guide still describe only `riftco_decoder_v1`, whose learned
absolute positions, biased LayerNorm/QKV projections, and GELU feed-forward
network have different parameter and tokenizer contracts.

Consequently, a Llama SafeTensors file is not loadable merely because its
bytes can be parsed. The Hugging Face adapter recognizes those model types and
returns an actionable architecture-incompatibility error. Loading an industry
checkpoint still requires strict config and parameter mapping, weight-tying
policy, RoPE variants, and a SentencePiece-aware tokenizer adapter; silent
renaming or transposition would produce the wrong model. See
[Dense Llama and Mistral runtime boundary](LLAMA_MISTRAL.md) for the executable
topology that is available today.

## Extension contract

Untrusted interchange metadata is allocation-bounded. A current-decoder
import accepts at most 256 blocks (4,102 native parameter descriptors), a
SafeTensors file accepts at most 8,192 tensor entries, and ONNX applies both a
config-derived byte ceiling and parser cardinality limits. These deliberately
generous teaching-framework limits prevent tiny or zero-sized attacker inputs
from expanding into unbounded Python object graphs.

New adapters should use `Float32Tensor`, declare a stable architecture ID,
map all native names bijectively, validate shapes before flattening values,
and preserve tokenizer semantics. Import must reject unknown required fields,
foreign architectures, unsupported dtypes/quantization, and extra or missing
parameters. The C++ engine remains format-neutral; Python owns external
container and naming policy.

## Related guides

- [Pipeline and model bundles](PIPELINE.md)
- [Training checkpoints](TRAINING_CHECKPOINTS.md)
- [Dense Llama and Mistral runtime](LLAMA_MISTRAL.md)
- [Transformer topology](TRANSFORMER.md)
- [Tokenization](TOKENIZATION.md)
- [API reference](API_REFERENCE.md)
