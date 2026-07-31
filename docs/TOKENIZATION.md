# Tokenization and Next-Token Batches

The transformer consumes integer token IDs, not text:

```text
UTF-8 bytes → selected tokenizer strategy → token IDs → training batches
```

The project provides two dependency-free strategies:

| Method | Vocabulary | Main tradeoff |
| --- | --- | --- |
| `byte` | Only distinct bytes found in the construction corpus | Small vocabulary and simple behavior, but longer sequences and unseen bytes are rejected |
| `bpe` | All 256 bytes followed by learned byte-pair pieces | Encodes any byte sequence and usually shortens repeated text, at the cost of learning a larger vocabulary |

`Tokenizer(corpus)` remains the backward-compatible spelling for the byte
method. The runnable Python trainer defaults to BPE because it is closer to the
subword approach commonly used for language models.

## Byte strategy

UTF-8 stores text as bytes. An ASCII character occupies one byte, while a
character such as `é` or an emoji occupies multiple bytes. Byte mode assigns
IDs to the distinct corpus bytes in ascending unsigned-byte order:

| Corpus byte | Decimal value | Token ID |
| --- | ---: | ---: |
| newline | 10 | 0 |
| `a` | 97 | 1 |
| `b` | 98 | 2 |
| `c` | 99 | 3 |

For a corpus containing those four bytes:

```text
"cab\n" → [3, 1, 2, 0]
```

Reordering the corpus does not change the vocabulary if the set of bytes is
unchanged. A byte absent from the construction corpus cannot be encoded by this
strategy.

## Byte-pair encoding strategy

BPE begins with a universal base vocabulary:

```text
token IDs 0 ... 255 ↔ byte values 0 ... 255
```

It then repeatedly learns a new token:

1. Count adjacent token pairs in the current corpus sequence.
2. Select the most frequent pair whose count is at least
   `minimum_pair_frequency`.
3. If counts tie, select the lexicographically smaller pair of token IDs.
4. Replace non-overlapping occurrences from left to right.
5. Append the pair's concatenated byte piece to the vocabulary.

Training stops when the requested `vocabulary_size` is reached or no pair
meets the frequency threshold. Consequently, `vocabulary_size` is a maximum;
the resulting vocabulary can be smaller.

For example, learning one merge from repeated `ab` text gives:

```text
256 → b"ab"
b"abababab" → [256, 256, 256, 256]
```

Encoding starts with byte IDs and applies learned merges in their deterministic
training order. Decoding concatenates each token's byte piece. Because the
base vocabulary contains every byte, BPE can encode bytes that did not appear
in its training corpus.

Both strategies preserve every input they accept exactly:

```math
\mathrm{decode}\left(\mathrm{encode}(\mathbf{b})\right)
= \mathbf{b}
```

For BPE, $\mathbf{b}$ can be any byte sequence; byte mode requires every byte
to occur in its construction corpus. No Unicode normalization, case folding,
whitespace rewriting, or special-token insertion occurs.

## Extensible strategy selection

The tokenizer API is one context-managed facade over interchangeable native
strategies. Its options select a method, and native construction dispatches
through a factory to the matching implementation. Encoding, decoding,
vocabulary inspection, and lifecycle behavior stay behind the common
interface.

This boundary makes another method—such as Unigram or WordPiece—a focused
extension: implement the strategy contract, add one factory mapping, append a
stable C ABI method value, and expose the corresponding Python option. Model,
batch, loss, and optimizer code continue to consume ordinary token IDs and do
not depend on the tokenization algorithm.

The method values in C ABI 2.0 are stable fixed-width integers:

```text
RT_TOKENIZER_METHOD_BYTE = 0
RT_TOKENIZER_METHOD_BPE  = 1
```

## Python API

```python
from riftco_transformer import Tokenizer

corpus = "hello transformer, hello tokenizer"

with Tokenizer(
    corpus,
    method="bpe",
    vocabulary_size=272,
    minimum_pair_frequency=2,
) as tokenizer:
    token_ids = tokenizer.encode(corpus)
    assert tokenizer.decode(token_ids) == corpus

    print(tokenizer.method)      # "bpe"
    print(tokenizer.vocab_size)  # at most 272
    print(tokenizer.vocabulary)  # tuple of byte pieces
```

The configuration rules are:

- `method` is exactly `"byte"` or `"bpe"`;
- BPE defaults to a maximum vocabulary size of 512 in the general Python API;
- BPE vocabulary size must be between 256 and `2**32 - 1`;
- `minimum_pair_frequency` must be at least 1 and defaults to 2;
- `vocabulary_size` is inapplicable to byte mode;
- byte mode accepts `minimum_pair_frequency` only at its default because it
  does not learn pairs.

`encode()` and `decode()` are strict UTF-8 conveniences.
`encode_bytes()` and `decode_bytes()` preserve arbitrary binary input,
including embedded NUL bytes. `vocabulary` returns a `tuple[bytes, ...]` for
either strategy. The older `vocabulary_bytes` property is retained for byte
mode; BPE callers use `vocabulary` because its pieces have variable lengths.

These wrappers use C ABI 2.0's size-query and state-restoration functions.
Variable-length results
are validated before being copied, and a changed result size between the query
and copy passes is treated as an error.

## Training text

The example accepts literal text or a UTF-8 file:

```bash
PYTHONPATH="$PWD/python" \
python3 examples/python/train_tiny.py \
    --file data/pretraining/tiny_corpus.txt \
    --tokenizer bpe \
    --vocab-size 272 \
    --min-pair-frequency 2 \
    --context 16
```

Compare the byte strategy without changing the model or optimizer:

```bash
PYTHONPATH="$PWD/python" \
python3 examples/python/train_tiny.py \
    --file data/pretraining/tiny_corpus.txt \
    --tokenizer byte \
    --context 16
```

`--context` counts encoded tokens. BPE and byte mode can therefore produce
different numbers of valid windows from the same text.

The training example tokenizes first and then reserves a contiguous token tail
for validation. Model updates never sample that tail. The tokenizer is fitted
on the complete corpus so the corpus-derived byte strategy can encode every
held-out byte. `train_loss_average` smooths recent random mini-batches, while
the fixed validation batches make `validation_loss` directly comparable across
evaluation steps.

## Next-token windows

Given:

```text
corpus tokens = [4, 7, 2, 9, 3]
start         = 0
context       = 4
```

the batch row is:

```text
input  = [4, 7, 2, 9]
target = [7, 2, 9, 3]
```

Every target is its input shifted forward by one token:

```math
\begin{aligned}
x_{b,t} &= c_{s_b+t}, \\
y_{b,t} &= c_{s_b+t+1}.
\end{aligned}
```

Token IDs remain `std::uint32_t` rather than float tensor values. The
embedding layer maps an integer batch of shape `[batch, context]` to floating
point features of shape `[batch, context, model_width]`.

## Current limitations

- There are no beginning, end, padding, or other special tokens.
- There is no normalization or pre-tokenization layer.
- There is no standalone tokenizer-file format. `ModelBundle` persists the
  exact tokenizer state together with the model that consumes its token IDs.
- The implementation favors clarity and determinism over large-corpus
  tokenizer-training performance.

`Tokenizer.from_state(...)` reconstructs an ordered byte vocabulary or BPE
merge table. The staged pipeline uses it when loading a `ModelBundle`, so
generation sees exactly the token-to-ID mapping used during training.

The C++ tokenizer and batch tests cover deterministic learning, tie breaking,
compression, arbitrary-byte round trips, invalid options, shifted targets,
boundaries, and sampled windows. The C11 ABI and Python tests independently
cover size-query safety, lifecycle, strategy selection, and complete
text → BPE → model → loss → backward → Adam execution.
