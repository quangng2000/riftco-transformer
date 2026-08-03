#include "riftco_transformer/model/llama_mistral_transformer.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace riftco_transformer {
namespace {

void require_positive(std::size_t value, const char* name) {
    if (value == 0) {
        throw std::invalid_argument(
            std::string("Llama/Mistral ") + name +
            " must be greater than zero"
        );
    }
}

void require_product_fits(
    std::size_t left,
    std::size_t right,
    const char* name
) {
    if (left != 0 &&
        right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::overflow_error(
            std::string("Llama/Mistral ") + name + " overflows"
        );
    }
}

std::vector<std::shared_ptr<LlamaMistralBlock>> make_blocks(
    const LlamaMistralConfig& config,
    std::mt19937& random
) {
    std::vector<std::shared_ptr<LlamaMistralBlock>> result;
    result.reserve(config.block_count);
    for (std::size_t index = 0; index < config.block_count; ++index) {
        result.push_back(
            std::make_shared<LlamaMistralBlock>(config, random)
        );
    }
    return result;
}

}  // namespace

LlamaMistralConfig validate_llama_mistral_config(
    LlamaMistralConfig config
) {
    switch (config.architecture) {
        case LlamaMistralArchitecture::Llama:
        case LlamaMistralArchitecture::Mistral:
            break;
        default:
            throw std::invalid_argument(
                "Llama/Mistral architecture is not recognized"
            );
    }
    require_positive(config.vocabulary_size, "vocabulary size");
    require_positive(config.maximum_context, "maximum context");
    require_positive(config.model_width, "model width");
    require_positive(config.query_head_count, "query head count");
    require_positive(config.key_value_head_count, "key/value head count");
    require_positive(config.block_count, "block count");
    require_positive(config.feed_forward_width, "feed-forward width");
    if (config.model_width % config.query_head_count != 0) {
        throw std::invalid_argument(
            "Llama/Mistral model width must be divisible by query head count"
        );
    }
    if (config.query_head_count % config.key_value_head_count != 0) {
        throw std::invalid_argument(
            "Llama/Mistral query head count must be divisible by key/value "
            "head count"
        );
    }
    const std::size_t head_width =
        config.model_width / config.query_head_count;
    if (head_width % 2 != 0) {
        throw std::invalid_argument(
            "Llama/Mistral head width must be even for RoPE"
        );
    }
    if (!std::isfinite(config.rms_norm_epsilon) ||
        config.rms_norm_epsilon <= 0.0F) {
        throw std::invalid_argument(
            "Llama/Mistral RMSNorm epsilon must be finite and positive"
        );
    }
    if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0F) {
        throw std::invalid_argument(
            "Llama/Mistral RoPE theta must be finite and positive"
        );
    }
    if (config.sliding_window.has_value()) {
        if (*config.sliding_window == 0) {
            throw std::invalid_argument(
                "Llama/Mistral sliding window must be greater than zero"
            );
        }
        if (*config.sliding_window < config.maximum_context) {
            throw std::invalid_argument(
                "sliding-window attention narrower than maximum_context is "
                "not implemented; refusing to substitute dense attention"
            );
        }
    }

    require_product_fits(
        config.vocabulary_size,
        config.model_width,
        "token embedding element count"
    );
    require_product_fits(
        config.model_width,
        config.model_width,
        "query projection element count"
    );
    require_product_fits(
        config.model_width,
        config.feed_forward_width,
        "feed-forward projection element count"
    );
    return config;
}

LlamaMistralBlock::LlamaMistralBlock(
    const LlamaMistralConfig& config,
    std::mt19937& random
) : attention_norm_(
        validate_llama_mistral_config(config).model_width,
        config.rms_norm_epsilon
    ),
    attention_(
        config.model_width,
        config.query_head_count,
        config.key_value_head_count,
        config.rope_theta,
        random
    ),
    feed_forward_norm_(config.model_width, config.rms_norm_epsilon),
    feed_forward_(
        config.model_width,
        config.feed_forward_width,
        random
    ) {
    register_module("attention_norm", attention_norm_);
    register_module("attention", attention_);
    register_module("feed_forward_norm", feed_forward_norm_);
    register_module("feed_forward", feed_forward_);
}

Variable LlamaMistralBlock::forward(const Variable& input) const {
    const Variable attention_state =
        input + attention_.forward(attention_norm_.forward(input));
    return attention_state + feed_forward_.forward(
        feed_forward_norm_.forward(attention_state)
    );
}

void LlamaMistralBlock::to(ExecutionBackend backend) {
    Module::to(backend);
}

ParameterList LlamaMistralBlock::parameters() {
    return Module::parameters();
}

LlamaMistralTransformer::LlamaMistralTransformer(
    LlamaMistralConfig config,
    std::mt19937& random
) : config_(validate_llama_mistral_config(std::move(config))),
    token_embedding_(
        config_.vocabulary_size,
        config_.model_width,
        random
    ),
    blocks_(make_blocks(config_, random)),
    final_norm_(config_.model_width, config_.rms_norm_epsilon),
    language_model_head_(
        config_.model_width,
        config_.vocabulary_size,
        random,
        false
    ) {
    register_module("token_embedding", token_embedding_);
    for (const auto& block : blocks_) {
        block_modules_.append(block);
    }
    register_module("blocks", block_modules_);
    register_module("final_norm", final_norm_);
    register_module("language_model_head", language_model_head_);
}

const LlamaMistralConfig&
LlamaMistralTransformer::config() const noexcept {
    return config_;
}

ExecutionBackend LlamaMistralTransformer::backend() const noexcept {
    return token_embedding_.weight().value().backend();
}

Variable LlamaMistralTransformer::forward(
    std::span<const TokenId> token_ids,
    Tensor::Shape token_shape
) const {
    if (token_shape.size() != 2) {
        throw std::invalid_argument(
            "Llama/Mistral token shape must be [batch, time]"
        );
    }
    const std::size_t batch = token_shape[0];
    const std::size_t time = token_shape[1];
    if (batch == 0 || time == 0) {
        throw std::invalid_argument(
            "Llama/Mistral batch and time must be greater than zero"
        );
    }
    require_product_fits(batch, time, "token count");
    if (batch * time != token_ids.size()) {
        throw std::invalid_argument(
            "Llama/Mistral token count must match [batch, time]"
        );
    }
    if (time > config_.maximum_context) {
        throw std::invalid_argument(
            "Llama/Mistral sequence length exceeds maximum context"
        );
    }
    for (const TokenId token : token_ids) {
        if (static_cast<std::size_t>(token) >= config_.vocabulary_size) {
            throw std::out_of_range(
                "Llama/Mistral token is outside the model vocabulary"
            );
        }
    }

    Variable hidden = token_embedding_.forward(token_ids, token_shape);
    for (const auto& block : blocks_) {
        hidden = block->forward(hidden);
    }
    return language_model_head_.forward(final_norm_.forward(hidden));
}

void LlamaMistralTransformer::to(ExecutionBackend backend) {
    Module::to(backend);
}

ParameterList LlamaMistralTransformer::parameters() {
    return Module::parameters();
}

}  // namespace riftco_transformer
