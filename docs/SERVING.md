# Incremental Serving

Serving uses the same transformer parameters as training, but it has a
separate inference path and state contract. The ordinary training forward pass
is unchanged.

| Path | Input and output | State and gradients |
| --- | --- | --- |
| `DecoderOnlyTransformer::forward(...)` | tokens `[B,T]` → logits `[B,T,V]` | Builds an autograd graph for training |
| `DecoderOnlyTransformer::decode_token(...)` | one token → detached logits `[1,1,V]` | Appends one position to a request-local KV cache |

`decode_token` is not a second model architecture. It applies the same token
embedding, learned absolute positional embedding, transformer blocks, final
normalization, and language-model head one position at a time.

## Request flow

One generation request follows this sequence:

```text
tokenize prompt
  → create an empty request-local cache
  → prefill one prompt token at a time
  → sample one token from the latest logits
  → decode that token and append its K/V state
  → repeat
  → release or reset the cache
```

Prefill and decode currently use the same one-token operation. There is no
separate batched-prefill kernel. For each token, the cache update is
transactional across all transformer layers:

1. `begin_token()` reserves the next logical position.
2. Each layer calls `append_and_attend(...)` with query, key, and value tensors
   shaped `[1,H,1,d_h]`.
3. `commit_token()` makes the new position visible.
4. If any layer fails, `abort_token()` leaves the visible cache length
   unchanged.

The returned attention context has the same `[1,H,1,d_h]` shape. The final
model output is a detached `[1,1,V]` tensor; serving does not construct a
training graph.

## Paged KV-cache layout

Paged caching is the default. A `PagedKvCachePool` owns backend-resident key
and value storage shared by the request caches created from that native pool.
For every transformer layer, it allocates:

```text
key pages   [physical_page, head, page_offset, head_width]
value pages [physical_page, head, page_offset, head_width]
```

A request cache owns a small logical page table:

```text
logical page 0 ──→ physical page 7
logical page 1 ──→ physical page 2
logical page 2 ──→ physical page 9
```

One physical page ID addresses the matching key/value page in every layer.
Pages are leased lazily as a sequence grows and returned to the pool when the
request resets or is destroyed. The default page size is 16 tokens. In the
native `ServingConfig`, `kv_cache_block_count == 0` allocates enough physical
pages for one maximum-length context.

CPU and Metal implement the same paged-decode-attention request directly: each
reads
the logical page table and the per-layer K/V pools without first gathering the
cached sequence into a public contiguous K/V tensor. Both paths are
synchronous. This is an inference-only attention path; selectable
materialized or Flash full-sequence attention and its vector-Jacobian product
remain the training/evaluation path.

`ContiguousKvCacheFactory` is the readable reference and compatibility
strategy. It gives each request one fixed, maximum-context page and still
enters the same validated paged-decode-attention backend operation. Code above
`KeyValueCacheFactory` can swap between the contiguous and pooled strategies
without changing generation or the transformer.

The design follows the logical-to-physical paging idea described in the
[PagedAttention paper](https://arxiv.org/abs/2309.06180). The
[vLLM PagedAttention design](https://docs.vllm.ai/en/v0.10.1/design/paged_attention.html)
is a useful production-oriented reference. Riftco Transformer implements the
cache layout and direct attention operation described here, not the complete
vLLM serving system.

Paged storage is orthogonal to FlashAttention. Paging is a K/V memory-layout
policy; FlashAttention is a tiled attention-computation algorithm. The
implemented full-sequence Flash path does not change current serving because
prefill remains token-at-a-time. A future batched prefill may use
FlashAttention while writing the same paged cache that paged decode attention
reads for subsequent tokens.

## Context rollover

This model uses learned absolute positional embeddings. Position zero has a
different learned vector from position one, and so on.

When a sequence reaches `maximum_context`, generation keeps the newest suffix,
resets the cache, and replays that suffix from position zero. Simply freeing
the oldest page would not preserve the existing rolling-window result: every
remaining token would keep its old absolute position, so its hidden state and
K/V values would differ from a fresh forward pass over the cropped suffix.

Suffix replay is therefore a correctness requirement for the current model,
not a limitation of page addressing itself. A model with a different
positioning scheme could support another rollover policy behind the same cache
interface.

## Native C++ use

`ServingStack` selects paged caching by default. A caller can make the choice
explicit and size a shared pool:

```cpp
#include "transformer_lab/stages/serving/stack.hpp"

namespace serve = transformer_lab::stages::serving;

serve::ServingConfig config;
config.backend = transformer_lab::ExecutionBackend::Cpu;
config.kv_cache_kind = serve::KvCacheKind::Paged;
config.kv_cache_block_size = 16;
config.kv_cache_block_count = 0;  // one full context

serve::ServingStack serving(snapshot, config);
const auto result =
    serving.generate("Tensor:", serve::GenerationConfig{32});
```

For lower-level composition, construct a `PagedKvCachePool` or
`ContiguousKvCacheFactory` and pass it to `GenerationEngine`. Every created
cache must match the model backend, layer count, head count, head width, and
maximum context.

## Stable C and Python sessions

C ABI 1.6 introduced the opaque `tl_decode_session` lifecycle, which the
current ABI 1.8 retains:

- `tl_model_decode_session_create`
- `tl_decode_session_step`
- `tl_decode_session_reset`
- `tl_decode_session_size` and `tl_decode_session_capacity`
- `tl_decode_session_cache_kind` and `tl_decode_session_block_size`
- `tl_decode_session_release`

Initialize `tl_decode_session_options` with
`tl_decode_session_options_init`. Null options select the same defaults:
paged caching with a 16-token page. Each `step` appends exactly one token and
returns the vocabulary-sized logits that predict the next token. A raw session
reports capacity exhaustion; it does not choose a crop or replay policy for
the caller.

A session retains its model state even if the original model handle is
released. While any session derived from a model is alive, model transfer,
parameter loading, LoRA attachment or merge, and Adam updates are rejected.
The raw C ABI requires external synchronization. The Python wrapper shares the
model lock and offers an idempotent context-managed `DecodeSession`.

High-level Python generation uses that session automatically:

```python
from transformer_lab.artifacts import ModelBundle
from transformer_lab.serving import TextGenerator

bundle = ModelBundle.load("results/stages/tiny_post_trained.tlab")
with bundle.instantiate("cpu") as runtime:
    generator = TextGenerator(
        runtime.model,
        runtime.tokenizer,
        kv_cache="paged",
        kv_cache_block_size=16,
    )
    result = generator.generate("Tensor:", max_new_tokens=32)
    print(result.text)
```

For direct token-level control:

```python
with bundle.instantiate("cpu") as runtime:
    with runtime.model.decode_session(
        cache="paged",
        block_size=16,
    ) as session:
        for token in runtime.tokenizer.encode("Tensor:"):
            logits = session.step(token)
```

`TextGenerator` owns tokenization, sampling, suffix replay, and byte-safe
decoding. `DecodeSession` deliberately owns only incremental model state.

## Current scope

Paged storage does not by itself provide:

- continuous batching or a request scheduler;
- prefix-cache lookup or page sharing between matching prompts;
- copy-on-write prefix management;
- a batched prefill kernel;
- token streaming, cancellation, or admission control;
- asynchronous CPU/GPU overlap.

The native pool can back more than one request cache, but the current Python
service serializes generation through one model runtime, and the C
decode-session surface does not expose a process-wide shared pool. Continuous
batching, scheduler policy, and shared prefixes remain future serving-stack
work.
