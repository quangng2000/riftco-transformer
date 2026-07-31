#include "transformer_lab/stages/serving/stack.hpp"

#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/data/tokenizer.hpp"
#include "transformer_lab/model/decoder_only_transformer.hpp"
#include "transformer_lab/stages/serving/kv_cache.hpp"

#include <memory>
#include <random>
#include <stdexcept>

namespace transformer_lab::stages::serving {

struct ServingStack::Implementation {
    ServingConfig config;
    std::unique_ptr<TokenizerStrategy> tokenizer;
    std::unique_ptr<DecoderOnlyTransformer> model;
    std::shared_ptr<const KeyValueCacheFactory> cache_factory;
    std::unique_ptr<GenerationEngine> engine;
};

ServingStack::ServingStack(
    const artifacts::ModelSnapshot& snapshot,
    ServingConfig config
)
    : implementation_(std::make_unique<Implementation>()) {
    config.validate();
    implementation_->config = config;
    implementation_->tokenizer =
        artifacts::restore_tokenizer(snapshot.tokenizer);
    if (implementation_->tokenizer->vocab_size() !=
        snapshot.model.dimensions.vocabulary_size) {
        throw std::invalid_argument(
            "serving model and tokenizer vocabulary sizes do not match"
        );
    }

    const ScopedExecutionBackend selected_backend(config.backend);
    std::mt19937 restoration_random(0);
    implementation_->model =
        std::make_unique<DecoderOnlyTransformer>(
            snapshot.model.dimensions,
            restoration_random,
            snapshot.model.layer_norm_epsilon
        );
    artifacts::load_model_state(
        *implementation_->model,
        snapshot.model
    );
    switch (config.kv_cache_kind) {
        case KvCacheKind::Contiguous:
            implementation_->cache_factory =
                std::make_shared<ContiguousKvCacheFactory>(
                    snapshot.model.dimensions,
                    config.backend
                );
            break;
        case KvCacheKind::Paged:
            implementation_->cache_factory =
                std::make_shared<PagedKvCachePool>(
                    snapshot.model.dimensions,
                    config.backend,
                    config.kv_cache_block_size,
                    config.kv_cache_block_count
                );
            break;
    }
    implementation_->engine = std::make_unique<GenerationEngine>(
        *implementation_->model,
        *implementation_->tokenizer,
        implementation_->cache_factory
    );
}

ServingStack::~ServingStack() = default;

const ServingConfig& ServingStack::config() const noexcept {
    return implementation_->config;
}

DecoderOnlyTransformer& ServingStack::model() noexcept {
    return *implementation_->model;
}

const DecoderOnlyTransformer&
ServingStack::model() const noexcept {
    return *implementation_->model;
}

TokenizerStrategy& ServingStack::tokenizer() noexcept {
    return *implementation_->tokenizer;
}

const TokenizerStrategy&
ServingStack::tokenizer() const noexcept {
    return *implementation_->tokenizer;
}

GenerationResult ServingStack::generate(
    std::string_view prompt,
    GenerationConfig generation
) {
    validate_request(generation);
    return implementation_->engine->generate(prompt, generation);
}

GenerationResult ServingStack::generate(
    std::string_view prompt,
    SamplingStrategy& sampler,
    GenerationConfig generation
) {
    validate_request(generation);
    return implementation_->engine->generate(
        prompt,
        sampler,
        generation
    );
}

void ServingStack::validate_request(
    const GenerationConfig& generation
) const {
    if (generation.max_new_tokens >
        implementation_->config.maximum_new_tokens) {
        throw std::invalid_argument(
            "generation max_new_tokens exceeds the serving limit"
        );
    }
}

}  // namespace transformer_lab::stages::serving
