#include "riftco_transformer/stages/serving/generation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace riftco_transformer::stages::serving {
namespace {

void validate_logits(std::span<const float> logits) {
    if (logits.empty()) {
        throw std::invalid_argument("sampling logits must not be empty");
    }
    if (logits.size() - 1 >
        static_cast<std::size_t>(
            std::numeric_limits<TokenId>::max()
        )) {
        throw std::overflow_error(
            "sampling vocabulary exceeds the TokenId range"
        );
    }
    if (std::any_of(
            logits.begin(),
            logits.end(),
            [](float value) { return !std::isfinite(value); }
        )) {
        throw std::invalid_argument(
            "sampling logits must contain only finite values"
        );
    }
}

void validate_generation_capacity(
    std::size_t prompt_size,
    std::size_t maximum_new_tokens
) {
    const std::vector<TokenId> tokens;
    if (prompt_size > tokens.max_size() ||
        maximum_new_tokens > tokens.max_size() ||
        maximum_new_tokens > tokens.max_size() - prompt_size) {
        throw std::length_error(
            "generation output exceeds the supported token capacity"
        );
    }
}

std::span<const float> decoded_token_logits(
    const Tensor& output,
    std::size_t vocabulary_size
) {
    const Tensor::Shape expected_shape{
        1,
        1,
        vocabulary_size,
    };
    if (output.shape() != expected_shape) {
        throw std::runtime_error(
            "transformer decoder output has an unexpected shape"
        );
    }
    const auto values = output.data();
    if (values.size() != vocabulary_size) {
        throw std::runtime_error(
            "transformer decoder output data does not match its shape"
        );
    }
    return values;
}

bool same_dimensions(
    const TransformerDimensions& left,
    const TransformerDimensions& right
) noexcept {
    return
        left.vocabulary_size == right.vocabulary_size &&
        left.maximum_context == right.maximum_context &&
        left.model_width == right.model_width &&
        left.head_count == right.head_count &&
        left.block_count == right.block_count &&
        left.feed_forward_width == right.feed_forward_width;
}

std::vector<std::uint8_t> string_bytes(std::string_view text) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(text.size());
    for (const char character : text) {
        bytes.push_back(
            static_cast<std::uint8_t>(
                static_cast<unsigned char>(character)
            )
        );
    }
    return bytes;
}

}  // namespace

TokenId GreedySampler::sample(std::span<const float> logits) {
    validate_logits(logits);
    std::size_t selected = 0;
    for (std::size_t index = 1; index < logits.size(); ++index) {
        if (logits[index] > logits[selected]) {
            selected = index;
        }
    }
    return static_cast<TokenId>(selected);
}

TemperatureSampler::TemperatureSampler(
    float temperature,
    std::optional<std::size_t> top_k,
    std::uint64_t seed
)
    : temperature_(temperature),
      top_k_(top_k),
      seed_(seed),
      random_(seed) {
    if (!std::isfinite(temperature_) || temperature_ <= 0.0F) {
        throw std::invalid_argument(
            "sampling temperature must be finite and greater than zero"
        );
    }
    if (top_k_.has_value() && *top_k_ == 0) {
        throw std::invalid_argument(
            "sampling top_k must be greater than zero"
        );
    }
}

float TemperatureSampler::temperature() const noexcept {
    return temperature_;
}

std::optional<std::size_t> TemperatureSampler::top_k() const noexcept {
    return top_k_;
}

std::uint64_t TemperatureSampler::seed() const noexcept {
    return seed_;
}

TokenId TemperatureSampler::sample(std::span<const float> logits) {
    validate_logits(logits);
    const std::size_t candidate_count =
        top_k_.value_or(logits.size());
    if (candidate_count > logits.size()) {
        throw std::invalid_argument(
            "sampling top_k exceeds the vocabulary size"
        );
    }

    std::vector<TokenId> candidates;
    candidates.reserve(logits.size());
    for (std::size_t index = 0; index < logits.size(); ++index) {
        candidates.push_back(static_cast<TokenId>(index));
    }
    const auto better_candidate =
        [logits](TokenId left, TokenId right) {
            const float left_logit =
                logits[static_cast<std::size_t>(left)];
            const float right_logit =
                logits[static_cast<std::size_t>(right)];
            if (left_logit != right_logit) {
                return left_logit > right_logit;
            }
            return left < right;
        };
    std::partial_sort(
        candidates.begin(),
        candidates.begin() +
            static_cast<std::ptrdiff_t>(candidate_count),
        candidates.end(),
        better_candidate
    );
    candidates.resize(candidate_count);

    const double maximum_logit = static_cast<double>(
        logits[static_cast<std::size_t>(candidates.front())]
    );
    std::vector<double> weights;
    weights.reserve(candidate_count);
    double total_weight = 0.0;
    for (const TokenId candidate : candidates) {
        const double shifted =
            static_cast<double>(
                logits[static_cast<std::size_t>(candidate)]
            ) -
            maximum_logit;
        const double weight = std::exp(
            shifted / static_cast<double>(temperature_)
        );
        weights.push_back(weight);
        total_weight += weight;
    }

    std::uniform_real_distribution<double> distribution(
        0.0,
        total_weight
    );
    const double threshold = distribution(random_);
    double cumulative_weight = 0.0;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        cumulative_weight += weights[index];
        if (threshold < cumulative_weight) {
            return candidates[index];
        }
    }
    return candidates.back();
}

GenerationEngine::GenerationEngine(
    DecoderOnlyTransformer& model,
    TokenizerStrategy& tokenizer,
    std::shared_ptr<const KeyValueCacheFactory> cache_factory
)
    : model_(model),
      tokenizer_(tokenizer),
      cache_factory_(std::move(cache_factory)) {
    const auto& dimensions = model_.dimensions();
    if (dimensions.vocabulary_size == 0 ||
        dimensions.maximum_context == 0) {
        throw std::invalid_argument(
            "generation requires a nonempty model vocabulary and context"
        );
    }
    if (dimensions.vocabulary_size != tokenizer_.vocab_size()) {
        throw std::invalid_argument(
            "model and tokenizer vocabulary sizes must match"
        );
    }
    if (cache_factory_ == nullptr) {
        cache_factory_ = std::make_shared<PagedKvCachePool>(
            dimensions,
            model_.backend(),
            16
        );
    }
    if (!same_dimensions(
            cache_factory_->dimensions(),
            dimensions
        )) {
        throw std::invalid_argument(
            "generation cache dimensions must match the model"
        );
    }
    if (cache_factory_->backend() != model_.backend()) {
        throw std::invalid_argument(
            "generation cache backend must match the model"
        );
    }
}

const KeyValueCacheFactory&
GenerationEngine::cache_factory() const noexcept {
    return *cache_factory_;
}

GenerationResult GenerationEngine::generate(
    std::string_view prompt,
    GenerationConfig config
) const {
    GreedySampler sampler;
    return generate(prompt, sampler, config);
}

GenerationResult GenerationEngine::generate(
    std::string_view prompt,
    SamplingStrategy& sampler,
    GenerationConfig config
) const {
    const auto& dimensions = model_.dimensions();
    std::vector<TokenId> prompt_tokens = tokenizer_.encode(prompt);
    if (prompt_tokens.empty()) {
        throw std::invalid_argument(
            "generation prompt must encode to at least one token"
        );
    }
    for (const TokenId token : prompt_tokens) {
        if (static_cast<std::size_t>(token) >=
            dimensions.vocabulary_size) {
            throw std::out_of_range(
                "generation prompt token exceeds the model vocabulary"
            );
        }
    }
    validate_generation_capacity(
        prompt_tokens.size(),
        config.max_new_tokens
    );

    std::vector<TokenId> generated_tokens;
    generated_tokens.reserve(config.max_new_tokens);
    std::vector<TokenId> all_tokens = prompt_tokens;
    all_tokens.reserve(
        prompt_tokens.size() + config.max_new_tokens
    );

    if (config.max_new_tokens == 0) {
        GenerationResult result;
        result.prompt_token_ids = std::move(prompt_tokens);
        result.token_ids = std::move(all_tokens);
        result.text = tokenizer_.decode(result.token_ids);
        result.decoded_bytes = string_bytes(result.text);
        return result;
    }

    std::unique_ptr<DecoderKeyValueCache> cache =
        cache_factory_->create();
    if (cache == nullptr) {
        throw std::runtime_error(
            "generation cache factory returned no cache"
        );
    }
    if (cache->backend() != model_.backend() ||
        cache->layer_count() != dimensions.block_count ||
        cache->head_count() != dimensions.head_count ||
        cache->head_width() !=
            dimensions.model_width / dimensions.head_count ||
        cache->capacity() != dimensions.maximum_context) {
        throw std::invalid_argument(
            "generation cache must match the model runtime"
        );
    }
    if (cache->size() != 0) {
        throw std::logic_error(
            "generation cache factory must return an empty cache"
        );
    }

    const auto replay_rolling_suffix =
        [&]() -> Tensor {
            cache->reset();
            const std::size_t context_size = std::min(
                all_tokens.size(),
                dimensions.maximum_context
            );
            const std::size_t context_start =
                all_tokens.size() - context_size;
            std::optional<Tensor> latest_logits;
            for (std::size_t offset = 0;
                 offset < context_size;
                 ++offset) {
                latest_logits = model_.decode_token(
                    all_tokens[context_start + offset],
                    *cache
                );
            }
            if (!latest_logits.has_value()) {
                throw std::logic_error(
                    "generation rolling context must not be empty"
                );
            }
            return std::move(*latest_logits);
        };

    Tensor logits = replay_rolling_suffix();
    for (std::size_t index = 0;
         index < config.max_new_tokens;
         ++index) {
        const auto token_logits = decoded_token_logits(
            logits,
            dimensions.vocabulary_size
        );
        const TokenId next_token = sampler.sample(token_logits);
        if (static_cast<std::size_t>(next_token) >=
            dimensions.vocabulary_size) {
            throw std::out_of_range(
                "sampling strategy returned a token outside the vocabulary"
            );
        }
        generated_tokens.push_back(next_token);
        all_tokens.push_back(next_token);

        if (index + 1 == config.max_new_tokens) {
            continue;
        }
        if (cache->size() < cache->capacity()) {
            logits = model_.decode_token(next_token, *cache);
        } else {
            // Learned absolute positions are renumbered after a rolling
            // crop in the original full-forward path. Replaying the suffix
            // preserves that exact behavior; simple page eviction would not.
            logits = replay_rolling_suffix();
        }
    }

    GenerationResult result;
    result.prompt_token_ids = std::move(prompt_tokens);
    result.generated_token_ids = std::move(generated_tokens);
    result.token_ids = std::move(all_tokens);
    result.text = tokenizer_.decode(result.token_ids);
    result.decoded_bytes = string_bytes(result.text);
    return result;
}

}  // namespace riftco_transformer::stages::serving
