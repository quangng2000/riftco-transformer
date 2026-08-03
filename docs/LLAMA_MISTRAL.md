# Dense Llama and Mistral runtime boundary

Riftco Transformer has a separate native C++ reference runtime for the dense
topology shared by Llama- and Mistral-family decoders. It is deliberately not a
claim that every checkpoint carrying a `llama` or `mistral` label is compatible.
Configuration, tokenizer, weight names, shapes, and attention semantics must
all agree before external weights can be used.

## Support matrix

| Capability | Current status |
| --- | --- |
| RMSNorm without bias | Implemented and differentiable |
| Split-half RoPE with configurable theta | Implemented and differentiable |
| Bias-free Q/K/V/O projections | Implemented |
| Grouped-query attention (`num_key_value_heads`) | Implemented, including summed repetition VJP |
| Bias-free SwiGLU (`gate`, `up`, `down`) | Implemented with overflow-stable SiLU/VJP |
| Dense full-context causal training forward/backward | Implemented |
| Mistral sliding window covering the entire configured context | Accepted because it is exactly equivalent to dense causal attention |
| Sliding window narrower than `maximum_context` | Rejected; never silently replaced by dense attention |
| Incremental KV-cache decoding | Not implemented for this architecture |
| Flash/GQA-specialized kernels | Not implemented; the reference path uses materialized causal attention |
| Hugging Face Llama/Mistral checkpoint ingestion | Not implemented by this native milestone |
| SentencePiece tokenizer ingestion | Not implemented |
| `.rift`, GGUF, or ONNX schema for these external architectures | Not implemented |
| Python and C ABI wrapper | Implemented in ABI 2.8 for config, construction, transfer, backend query, full-sequence forward, and base parameters |
| Architecture-level LoRA/QLoRA orchestration | Not implemented |

The existing interchange adapters continue to describe
`riftco_decoder_v1`. Do not rename Llama/Mistral tensors into that schema: its
learned absolute positions, LayerNorm, GELU MLP, biases, and attention shapes
have different semantics.

## Native composition

The implementation is split into reusable modules:

- `RMSNorm` owns only a learned scale.
- `apply_rotary_position_embedding()` rotates query/key channels and supplies
  the exact transpose rotation during backward.
- `repeat_key_value_heads()` repeats each KV head contiguously and sums its
  repeated gradients.
- `GroupedQueryAttention` uses separate bias-free query, key, value, and output
  projections. Key/value projection width is
  `key_value_head_count * head_width`.
- `SwiGLU` computes `down(silu(gate(x)) * up(x))` with three bias-free
  projections.
- `LlamaMistralTransformer` composes token embeddings, pre-normalized residual
  blocks, final RMSNorm, and an untied bias-free language-model head.

```cpp
#include <riftco_transformer/model/llama_mistral_transformer.hpp>

#include <random>
#include <vector>

std::mt19937 random(42);
riftco_transformer::LlamaMistralConfig config{
    .architecture = riftco_transformer::LlamaMistralArchitecture::Llama,
    .vocabulary_size = 32'000,
    .maximum_context = 2'048,
    .model_width = 512,
    .query_head_count = 8,
    .key_value_head_count = 2,
    .block_count = 8,
    .feed_forward_width = 1'376,
    .rms_norm_epsilon = 1.0e-5F,
    .rope_theta = 10'000.0F,
};

riftco_transformer::LlamaMistralTransformer model(config, random);
const std::vector<riftco_transformer::TokenId> tokens{1, 2, 3};
riftco_transformer::Variable logits = model.forward(tokens, {1, 3});
```

The same deliberately narrow topology is available from Python:

```python
from riftco_transformer import (
    Adam,
    LlamaMistralConfig,
    LlamaMistralTransformer,
    cross_entropy,
)

config = LlamaMistralConfig(
    architecture="llama",
    vocabulary_size=32_000,
    maximum_context=2_048,
    model_width=512,
    query_head_count=8,
    key_value_head_count=2,
    block_count=8,
    feed_forward_width=1_376,
)
with LlamaMistralTransformer(config).to("cpu") as model:
    with model.parameters() as parameters, Adam(parameters) as optimizer:
        with model([[1, 2, 3]]) as logits:
            with cross_entropy(logits, [[2, 3, 4]]) as loss:
                loss.backward()
                optimizer.step()
```

ABI 2.8 exposes the equivalent `rt_llama_mistral_*` calls. It intentionally
does not attach tokenizer, artifact conversion, incremental decode, or
LoRA/QLoRA semantics to this model handle.

## Configuration contract

All dimensions must be positive. `model_width` must be divisible by
`query_head_count`; query heads must be divisible by KV heads; and the
resulting head width must be even for RoPE. RMSNorm epsilon and RoPE theta must
be finite and positive. Shape products are checked before allocation.

If `sliding_window` is present, it must be at least `maximum_context`. A real
Mistral configuration with a smaller local window is rejected today. That
guard is important: accepting the file and applying full attention would
produce a different model, even if all tensor shapes happened to fit.

The stable parameter tree is bias-free:

```text
token_embedding.weight
blocks.N.attention_norm.scale
blocks.N.attention.{query,key,value,output}.weight
blocks.N.feed_forward_norm.scale
blocks.N.feed_forward.{gate,up,down}.weight
final_norm.scale
language_model_head.weight
```

Tests cover the schema, validation failures, RMSNorm formulas and gradients,
stable SiLU at extreme values, RoPE and GQA finite differences, KV-head
repetition backward, full-model finite gradients, and exact causal isolation.
CPU is the locally verified reference. The generic storage contract keeps the
modules backend-neutral, but accelerator performance and acceptance are not
claimed for this new architecture until dedicated coverage exists.

## What external checkpoint support still requires

Executable math is only one layer of compatibility. Direct industry checkpoint
support still needs a strict Hugging Face config mapper, a SentencePiece-aware
tokenizer contract, bijective parameter-name/shape mapping, tied-embedding
policy, RoPE-scaling variants, architecture-specific artifact identity, and
numerical parity against an independent runtime. Those are separate milestones,
not safe aliases around the dense native class.
