#pragma once

#include "riftco_transformer/model/grouped_query_attention.hpp"
#include "riftco_transformer/model/swiglu.hpp"
#include "riftco_transformer/nn/embedding.hpp"
#include "riftco_transformer/nn/linear.hpp"
#include "riftco_transformer/nn/module.hpp"
#include "riftco_transformer/nn/rms_norm.hpp"

#include <cstddef>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <vector>

namespace riftco_transformer {

enum class LlamaMistralArchitecture {
    Llama,
    Mistral,
};

// Dense reference topology shared by Llama and Mistral: RMSNorm, RoPE,
// bias-free grouped-query attention, and bias-free SwiGLU. A Mistral sliding
// window is accepted only when it covers maximum_context, where dense causal
// attention is mathematically equivalent. Narrow windows are rejected until a
// window-aware attention kernel and backward pass exist.
struct LlamaMistralConfig {
    LlamaMistralArchitecture architecture =
        LlamaMistralArchitecture::Llama;
    std::size_t vocabulary_size = 0;
    std::size_t maximum_context = 0;
    std::size_t model_width = 0;
    std::size_t query_head_count = 0;
    std::size_t key_value_head_count = 0;
    std::size_t block_count = 0;
    std::size_t feed_forward_width = 0;
    float rms_norm_epsilon = 1.0e-5F;
    float rope_theta = 10000.0F;
    std::optional<std::size_t> sliding_window = std::nullopt;
};

// Returns a validated copy or throws before parameter allocation.
[[nodiscard]] LlamaMistralConfig validate_llama_mistral_config(
    LlamaMistralConfig config
);

class LlamaMistralBlock : public Module {
public:
    LlamaMistralBlock(
        const LlamaMistralConfig& config,
        std::mt19937& random
    );

    [[nodiscard]] Variable forward(const Variable& input) const;
    void to(ExecutionBackend backend) override;
    [[nodiscard]] ParameterList parameters();

private:
    RMSNorm attention_norm_;
    GroupedQueryAttention attention_;
    RMSNorm feed_forward_norm_;
    SwiGLU feed_forward_;
};

class LlamaMistralTransformer : public Module {
public:
    LlamaMistralTransformer(
        LlamaMistralConfig config,
        std::mt19937& random
    );

    [[nodiscard]] const LlamaMistralConfig& config() const noexcept;
    [[nodiscard]] ExecutionBackend backend() const noexcept;

    // token_shape must be [batch, time]. Returns
    // [batch, time, vocabulary_size]. This milestone is the full-sequence,
    // trainable dense path; incremental KV-cache decode is not yet exposed.
    [[nodiscard]] Variable forward(
        std::span<const TokenId> token_ids,
        Tensor::Shape token_shape
    ) const;
    void to(ExecutionBackend backend) override;
    [[nodiscard]] ParameterList parameters();

private:
    LlamaMistralConfig config_;
    Embedding token_embedding_;
    std::vector<std::shared_ptr<LlamaMistralBlock>> blocks_;
    ModuleList block_modules_;
    RMSNorm final_norm_;
    Linear language_model_head_;
};

}  // namespace riftco_transformer
