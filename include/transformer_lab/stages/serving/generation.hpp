#pragma once

#include "transformer_lab/data/tokenizer.hpp"
#include "transformer_lab/model/decoder_only_transformer.hpp"
#include "transformer_lab/stages/serving/kv_cache.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace transformer_lab::stages::serving {

// Selects one token ID from a single vocabulary-sized logit vector.
class SamplingStrategy {
public:
    virtual ~SamplingStrategy() = default;

    [[nodiscard]] virtual TokenId sample(
        std::span<const float> logits
    ) = 0;
};

// Selects the first token with the maximum logit.
class GreedySampler final : public SamplingStrategy {
public:
    [[nodiscard]] TokenId sample(
        std::span<const float> logits
    ) override;
};

// Samples a numerically stable temperature-scaled softmax. When top_k is
// present, only the highest-scoring k tokens participate.
class TemperatureSampler final : public SamplingStrategy {
public:
    explicit TemperatureSampler(
        float temperature = 1.0F,
        std::optional<std::size_t> top_k = std::nullopt,
        std::uint64_t seed = 0
    );

    [[nodiscard]] float temperature() const noexcept;
    [[nodiscard]] std::optional<std::size_t> top_k() const noexcept;
    [[nodiscard]] std::uint64_t seed() const noexcept;

    [[nodiscard]] TokenId sample(
        std::span<const float> logits
    ) override;

private:
    float temperature_;
    std::optional<std::size_t> top_k_;
    std::uint64_t seed_;
    std::mt19937_64 random_;
};

struct GenerationConfig {
    std::size_t max_new_tokens{32};
};

struct GenerationResult {
    std::vector<TokenId> prompt_token_ids;
    std::vector<TokenId> generated_token_ids;
    // Complete prompt followed by generated tokens.
    std::vector<TokenId> token_ids;
    // Exact bytes returned by the tokenizer, including embedded NUL bytes.
    std::vector<std::uint8_t> decoded_bytes;
    // C++ strings preserve the same exact bytes; callers decide how to
    // interpret invalid UTF-8 for their presentation layer.
    std::string text;
};

// A non-owning composition point over a live model and tokenizer.
class GenerationEngine {
public:
    GenerationEngine(
        DecoderOnlyTransformer& model,
        TokenizerStrategy& tokenizer,
        std::shared_ptr<const KeyValueCacheFactory> cache_factory = {}
    );

    [[nodiscard]] const KeyValueCacheFactory&
    cache_factory() const noexcept;

    // Uses deterministic greedy sampling.
    [[nodiscard]] GenerationResult generate(
        std::string_view prompt,
        GenerationConfig config = {}
    ) const;

    // Uses the caller-owned, potentially stateful sampling policy.
    [[nodiscard]] GenerationResult generate(
        std::string_view prompt,
        SamplingStrategy& sampler,
        GenerationConfig config = {}
    ) const;

private:
    DecoderOnlyTransformer& model_;
    TokenizerStrategy& tokenizer_;
    std::shared_ptr<const KeyValueCacheFactory> cache_factory_;
};

}  // namespace transformer_lab::stages::serving
