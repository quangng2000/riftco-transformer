#include "transformer_lab/stages/serving/generation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using transformer_lab::ByteTokenizer;
using transformer_lab::DecoderOnlyTransformer;
using transformer_lab::Tensor;
using transformer_lab::TokenId;
using transformer_lab::TokenizerStrategy;
using transformer_lab::TransformerDimensions;
using transformer_lab::stages::serving::GenerationConfig;
using transformer_lab::stages::serving::GenerationEngine;
using transformer_lab::stages::serving::GreedySampler;
using transformer_lab::stages::serving::ContiguousKvCacheFactory;
using transformer_lab::stages::serving::KeyValueCacheFactory;
using transformer_lab::stages::serving::PagedKvCachePool;
using transformer_lab::stages::serving::SamplingStrategy;
using transformer_lab::stages::serving::TemperatureSampler;

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void require_throws(Function&& function, const std::string& message) {
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, message);
}

void require_tokens(
    std::span<const TokenId> actual,
    std::span<const TokenId> expected,
    const std::string& message
) {
    require(actual.size() == expected.size(), message + ": size mismatch");
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require(
            actual[index] == expected[index],
            message + " at index " + std::to_string(index)
        );
    }
}

void require_close(
    const Tensor& actual,
    std::initializer_list<float> expected,
    const std::string& message
) {
    require(
        actual.data().size() == expected.size(),
        message + ": size mismatch"
    );
    for (std::size_t index = 0; index < expected.size(); ++index) {
        const float expected_value =
            *(expected.begin() +
              static_cast<std::ptrdiff_t>(index));
        require(
            std::fabs(actual.data()[index] - expected_value) <=
                1.0e-6F,
            message + " at index " + std::to_string(index)
        );
    }
}

TransformerDimensions cache_test_dimensions() {
    return {
        .vocabulary_size = 8,
        .maximum_context = 5,
        .model_width = 2,
        .head_count = 1,
        .block_count = 2,
        .feed_forward_width = 4,
    };
}

Tensor cache_token(std::initializer_list<float> values) {
    return Tensor(
        {1, 1, 1, 2},
        std::vector<float>(values),
        transformer_lab::ExecutionBackend::Cpu
    );
}

void append_cache_token(
    transformer_lab::DecoderKeyValueCache& cache,
    std::initializer_list<float> first_layer_value,
    std::initializer_list<float> second_layer_value
) {
    const Tensor query = cache_token({0.0F, 0.0F});
    const Tensor key = cache_token({0.0F, 0.0F});
    const Tensor first_value = cache_token(first_layer_value);
    const Tensor second_value = cache_token(second_layer_value);
    cache.begin_token();
    static_cast<void>(
        cache.append_and_attend(0, query, key, first_value)
    );
    static_cast<void>(
        cache.append_and_attend(1, query, key, second_value)
    );
    cache.commit_token();
}

void exercise_cache_transaction_contract(
    const KeyValueCacheFactory& factory
) {
    auto cache = factory.create();
    require(cache != nullptr, "cache factory should create a cache");
    require(cache->size() == 0, "new cache should be empty");
    require(
        cache->capacity() ==
            factory.dimensions().maximum_context,
        "cache capacity should match maximum context"
    );

    const Tensor query = cache_token({0.0F, 0.0F});
    const Tensor key = cache_token({0.0F, 0.0F});
    const Tensor first_value = cache_token({2.0F, 4.0F});
    const Tensor second_value = cache_token({10.0F, 20.0F});
    cache->begin_token();
    const Tensor first_context =
        cache->append_and_attend(
            0,
            query,
            key,
            first_value
        );
    require_close(
        first_context,
        {2.0F, 4.0F},
        "first cached attention context"
    );
    require_throws(
        [&] {
            static_cast<void>(
                cache->append_and_attend(
                    0,
                    query,
                    key,
                    first_value
                )
            );
        },
        "a cache transaction should reject a duplicate layer"
    );
    require_throws(
        [&] { cache->commit_token(); },
        "cache commit should require every layer"
    );
    static_cast<void>(
        cache->append_and_attend(
            1,
            query,
            key,
            second_value
        )
    );
    cache->commit_token();
    require(
        cache->size() == 1,
        "cache size should advance only after commit"
    );

    cache->begin_token();
    const Tensor averaged_first =
        cache->append_and_attend(
            0,
            query,
            key,
            cache_token({6.0F, 8.0F})
        );
    const Tensor averaged_second =
        cache->append_and_attend(
            1,
            query,
            key,
            cache_token({30.0F, 40.0F})
        );
    require_close(
        averaged_first,
        {4.0F, 6.0F},
        "cached attention should include the first layer history"
    );
    require_close(
        averaged_second,
        {20.0F, 30.0F},
        "cached attention should isolate layers"
    );
    cache->abort_token();
    require(
        cache->size() == 1,
        "aborting a token should preserve committed size"
    );
    cache->reset();
    require(cache->size() == 0, "cache reset should clear size");
}

class OutOfRangeSampler final : public SamplingStrategy {
public:
    TokenId sample(std::span<const float> logits) override {
        return static_cast<TokenId>(logits.size());
    }
};

class RejectingCacheFactory final : public KeyValueCacheFactory {
public:
    explicit RejectingCacheFactory(
        TransformerDimensions dimensions
    )
        : dimensions_(dimensions) {}

    const TransformerDimensions&
    dimensions() const noexcept override {
        return dimensions_;
    }

    transformer_lab::ExecutionBackend
    backend() const noexcept override {
        return transformer_lab::ExecutionBackend::Cpu;
    }

    std::unique_ptr<transformer_lab::DecoderKeyValueCache>
    create() const override {
        throw std::runtime_error("cache creation was requested");
    }

private:
    TransformerDimensions dimensions_;
};

void test_greedy_sampler() {
    GreedySampler sampler;
    const std::vector<float> tied_logits{-2.0F, 3.0F, 3.0F};
    require(
        sampler.sample(tied_logits) == 1,
        "greedy ties should select the lowest token ID"
    );

    require_throws(
        [&] {
            static_cast<void>(
                sampler.sample(std::span<const float>{})
            );
        },
        "greedy sampling should reject empty logits"
    );
    const std::vector<float> invalid_logits{
        0.0F,
        std::numeric_limits<float>::quiet_NaN(),
    };
    require_throws(
        [&] { static_cast<void>(sampler.sample(invalid_logits)); },
        "greedy sampling should reject non-finite logits"
    );
}

void test_temperature_sampler() {
    const std::vector<float> logits{-5.0F, 7.0F, 6.0F, 4.0F};
    TemperatureSampler first(0.75F, std::size_t{3}, 17);
    TemperatureSampler second(0.75F, std::size_t{3}, 17);
    require(first.temperature() == 0.75F, "temperature introspection");
    require(first.top_k() == std::optional<std::size_t>{3}, "top_k");
    require(first.seed() == 17, "seed introspection");

    std::vector<TokenId> first_tokens;
    std::vector<TokenId> second_tokens;
    for (std::size_t index = 0; index < 32; ++index) {
        first_tokens.push_back(first.sample(logits));
        second_tokens.push_back(second.sample(logits));
    }
    require(
        first_tokens == second_tokens,
        "equal seeds should produce equal temperature samples"
    );
    require(
        std::all_of(
            first_tokens.begin(),
            first_tokens.end(),
            [](TokenId token) { return token != 0; }
        ),
        "top_k should exclude tokens outside the selected set"
    );

    TemperatureSampler top_one(100.0F, std::size_t{1}, 5);
    for (std::size_t index = 0; index < 8; ++index) {
        require(
            top_one.sample(logits) == 1,
            "top_k=1 should always select the largest logit"
        );
    }

    require_throws(
        [] { static_cast<void>(TemperatureSampler(0.0F)); },
        "zero temperature should be rejected"
    );
    require_throws(
        [] {
            static_cast<void>(TemperatureSampler(
                std::numeric_limits<float>::infinity()
            ));
        },
        "infinite temperature should be rejected"
    );
    require_throws(
        [] {
            static_cast<void>(TemperatureSampler(
                1.0F,
                std::size_t{0}
            ));
        },
        "zero top_k should be rejected"
    );

    TemperatureSampler oversized_top_k(1.0F, std::size_t{5});
    require_throws(
        [&] {
            static_cast<void>(oversized_top_k.sample(logits));
        },
        "top_k larger than the vocabulary should be rejected"
    );
}

void test_contiguous_and_paged_cache_contracts() {
    const TransformerDimensions dimensions =
        cache_test_dimensions();
    const auto backend = transformer_lab::ExecutionBackend::Cpu;

    ContiguousKvCacheFactory contiguous(dimensions, backend);
    require(
        contiguous.dimensions().maximum_context == 5 &&
            contiguous.backend() == backend,
        "contiguous cache factory introspection"
    );
    exercise_cache_transaction_contract(contiguous);

    PagedKvCachePool paged(dimensions, backend, 2);
    require(
        paged.block_size() == 2 &&
            paged.block_count() == 3 &&
            paged.blocks_per_full_cache() == 3,
        "paged cache should derive its automatic block count"
    );
    require(
        paged.free_block_count() == 3 &&
            paged.leased_block_count() == 0,
        "a new paged pool should have every block free"
    );
    exercise_cache_transaction_contract(paged);
    require(
        paged.free_block_count() == 3,
        "destroying the exercised cache should return its pages"
    );

    {
        auto cache = paged.create();
        cache->begin_token();
        require(
            paged.free_block_count() == 2,
            "beginning a page should lease it lazily"
        );
        cache->abort_token();
        require(
            paged.free_block_count() == 3 &&
                cache->size() == 0,
            "aborting a pending page should return it"
        );

        append_cache_token(
            *cache,
            {1.0F, 2.0F},
            {3.0F, 4.0F}
        );
        require(
            paged.free_block_count() == 2,
            "the first committed token should retain one page"
        );
        append_cache_token(
            *cache,
            {5.0F, 6.0F},
            {7.0F, 8.0F}
        );
        require(
            paged.free_block_count() == 2,
            "tokens in one logical block should share one page"
        );
        append_cache_token(
            *cache,
            {9.0F, 10.0F},
            {11.0F, 12.0F}
        );
        require(
            paged.free_block_count() == 1,
            "crossing a block boundary should lease one more page"
        );
        cache->reset();
        require(
            paged.free_block_count() == 3 &&
                cache->size() == 0,
            "reset should return every leased page"
        );
    }
    require(
        paged.free_block_count() == 3,
        "cache destruction should leave no page lease"
    );

    require_throws(
        [&] {
            static_cast<void>(
                PagedKvCachePool(dimensions, backend, 0)
            );
        },
        "paged caches should reject a zero block size"
    );
    require_throws(
        [&] {
            static_cast<void>(
                PagedKvCachePool(dimensions, backend, 2, 2)
            );
        },
        "a paged pool should hold at least one complete cache"
    );
}

void test_paged_pool_exhaustion_and_reuse() {
    const TransformerDimensions dimensions =
        cache_test_dimensions();
    PagedKvCachePool pool(
        dimensions,
        transformer_lab::ExecutionBackend::Cpu,
        2,
        3
    );
    auto first = pool.create();
    auto waiting = pool.create();
    for (std::size_t token = 0;
         token < dimensions.maximum_context;
         ++token) {
        append_cache_token(
            *first,
            {1.0F, 2.0F},
            {3.0F, 4.0F}
        );
    }
    require(
        pool.free_block_count() == 0,
        "one full cache should lease the entire minimal pool"
    );
    require_throws(
        [&] { waiting->begin_token(); },
        "a paged pool should report deterministic exhaustion"
    );

    first->reset();
    require(
        pool.free_block_count() == 3,
        "reset should make exhausted pages reusable"
    );
    waiting->begin_token();
    require(
        pool.free_block_count() == 2,
        "a waiting cache should lease a returned page"
    );
    waiting->abort_token();
    require(
        pool.free_block_count() == 3,
        "aborting the waiting cache should return its page"
    );
}

void test_paged_pool_keeps_sessions_isolated() {
    const TransformerDimensions dimensions =
        cache_test_dimensions();
    PagedKvCachePool pool(
        dimensions,
        transformer_lab::ExecutionBackend::Cpu,
        2,
        6
    );
    auto first = pool.create();
    auto second = pool.create();
    append_cache_token(
        *first,
        {1.0F, 3.0F},
        {10.0F, 30.0F}
    );
    append_cache_token(
        *second,
        {101.0F, 103.0F},
        {110.0F, 130.0F}
    );
    require(
        pool.leased_block_count() == 2,
        "concurrent sessions should lease distinct pages"
    );

    const Tensor query = cache_token({0.0F, 0.0F});
    const Tensor key = cache_token({0.0F, 0.0F});
    first->begin_token();
    const Tensor first_context = first->append_and_attend(
        0,
        query,
        key,
        cache_token({5.0F, 7.0F})
    );
    static_cast<void>(
        first->append_and_attend(
            1,
            query,
            key,
            cache_token({50.0F, 70.0F})
        )
    );
    first->commit_token();

    second->begin_token();
    const Tensor second_context = second->append_and_attend(
        0,
        query,
        key,
        cache_token({105.0F, 107.0F})
    );
    static_cast<void>(
        second->append_and_attend(
            1,
            query,
            key,
            cache_token({150.0F, 170.0F})
        )
    );
    second->commit_token();

    require_close(
        first_context,
        {3.0F, 5.0F},
        "the first session should attend only to its page"
    );
    require_close(
        second_context,
        {103.0F, 105.0F},
        "the second session should attend only to its page"
    );
}

std::vector<TokenId> reference_greedy_tokens(
    DecoderOnlyTransformer& model,
    TokenizerStrategy& tokenizer,
    std::string_view prompt,
    std::size_t maximum_new_tokens
) {
    std::vector<TokenId> all_tokens = tokenizer.encode(prompt);
    std::vector<TokenId> generated;
    GreedySampler sampler;
    for (std::size_t step = 0;
         step < maximum_new_tokens;
         ++step) {
        const std::size_t context_size = std::min(
            all_tokens.size(),
            model.dimensions().maximum_context
        );
        const std::size_t context_start =
            all_tokens.size() - context_size;
        const auto output = model.forward(
            std::span<const TokenId>(
                all_tokens.data() + context_start,
                context_size
            ),
            {1, context_size}
        );
        const auto values = output.value().data();
        const std::size_t vocabulary_size =
            model.dimensions().vocabulary_size;
        const std::span<const float> logits(
            values.data() +
                static_cast<std::ptrdiff_t>(
                    (context_size - 1) * vocabulary_size
                ),
            vocabulary_size
        );
        const TokenId token = sampler.sample(logits);
        generated.push_back(token);
        all_tokens.push_back(token);
    }
    return generated;
}

void test_cached_generation_matches_full_forward() {
    ByteTokenizer tokenizer("0123456789");
    std::mt19937 random(73);
    DecoderOnlyTransformer model(
        TransformerDimensions{
            .vocabulary_size = tokenizer.vocab_size(),
            .maximum_context = 3,
            .model_width = 4,
            .head_count = 2,
            .block_count = 1,
            .feed_forward_width = 8,
        },
        random
    );
    const std::vector<TokenId> expected =
        reference_greedy_tokens(model, tokenizer, "1234", 5);

    auto contiguous =
        std::make_shared<ContiguousKvCacheFactory>(
            model.dimensions(),
            transformer_lab::ExecutionBackend::Cpu
        );
    GenerationEngine contiguous_engine(
        model,
        tokenizer,
        contiguous
    );
    require(
        contiguous_engine
                .generate("1234", GenerationConfig{5})
                .generated_token_ids == expected,
        "contiguous cached generation should match full forward"
    );

    auto paged = std::make_shared<PagedKvCachePool>(
        model.dimensions(),
        transformer_lab::ExecutionBackend::Cpu,
        2
    );
    const std::size_t free_before = paged->free_block_count();
    GenerationEngine paged_engine(model, tokenizer, paged);
    require(
        paged_engine
                .generate("1234", GenerationConfig{5})
                .generated_token_ids == expected,
        "paged cached generation should match full forward across rolls"
    );
    require(
        paged->free_block_count() == free_before,
        "a completed generation request should release every page"
    );

    if (transformer_lab::execution_backend_available(
            transformer_lab::ExecutionBackend::Metal
        )) {
        model.to(transformer_lab::ExecutionBackend::Metal);
        const std::vector<TokenId> expected_metal =
            reference_greedy_tokens(
                model,
                tokenizer,
                "1234",
                5
            );
        auto metal_paged = std::make_shared<PagedKvCachePool>(
            model.dimensions(),
            transformer_lab::ExecutionBackend::Metal,
            2
        );
        GenerationEngine metal_engine(
            model,
            tokenizer,
            metal_paged
        );
        require(
            metal_engine
                    .generate("1234", GenerationConfig{5})
                    .generated_token_ids == expected_metal,
            "Metal paged generation should match Metal full forward"
        );
        require(
            metal_paged->free_block_count() ==
                metal_paged->block_count(),
            "Metal generation should release every paged block"
        );
    }
}

void test_zero_token_generation_does_not_create_a_cache() {
    ByteTokenizer tokenizer("0123456789");
    std::mt19937 random(79);
    DecoderOnlyTransformer model(
        TransformerDimensions{
            .vocabulary_size = tokenizer.vocab_size(),
            .maximum_context = 3,
            .model_width = 4,
            .head_count = 2,
            .block_count = 1,
            .feed_forward_width = 8,
        },
        random
    );
    auto rejecting = std::make_shared<RejectingCacheFactory>(
        model.dimensions()
    );
    GenerationEngine engine(model, tokenizer, rejecting);
    const auto result = engine.generate("12", GenerationConfig{0});
    require(
        result.generated_token_ids.empty() &&
            result.token_ids == result.prompt_token_ids,
        "zero-token generation should return before cache creation"
    );
    require_throws(
        [&] {
            static_cast<void>(
                engine.generate("12", GenerationConfig{1})
            );
        },
        "positive generation should request a cache"
    );
}

void test_generation_and_context_cropping() {
    ByteTokenizer tokenizer("0123456789");
    std::mt19937 random(71);
    DecoderOnlyTransformer model(
        TransformerDimensions{
            .vocabulary_size = tokenizer.vocab_size(),
            .maximum_context = 3,
            .model_width = 4,
            .head_count = 2,
            .block_count = 1,
            .feed_forward_width = 8,
        },
        random
    );
    GenerationEngine engine(model, tokenizer);

    // The prompt is longer than maximum_context. Successful generation proves
    // each model call receives only the rolling three-token suffix.
    const auto first = engine.generate("1234", GenerationConfig{3});
    const auto second = engine.generate("1234", GenerationConfig{3});
    const std::vector<TokenId> expected_prompt{1, 2, 3, 4};
    require_tokens(
        first.prompt_token_ids,
        expected_prompt,
        "generation prompt IDs"
    );
    require(
        first.generated_token_ids.size() == 3,
        "generation should produce the requested number of tokens"
    );
    require(
        first.token_ids.size() == 7,
        "complete generation should contain prompt and generated tokens"
    );
    require(
        first.generated_token_ids == second.generated_token_ids,
        "greedy generation should be deterministic"
    );
    require(
        first.text == tokenizer.decode(first.token_ids),
        "generation text should use tokenizer decoding"
    );
    require(
        first.decoded_bytes.size() == first.text.size(),
        "decoded byte and string lengths should agree"
    );
    for (std::size_t index = 0; index < first.text.size(); ++index) {
        require(
            first.decoded_bytes[index] ==
                static_cast<std::uint8_t>(
                    static_cast<unsigned char>(first.text[index])
                ),
            "decoded bytes should preserve exact tokenizer output"
        );
    }

    const auto zero = engine.generate("12", GenerationConfig{0});
    require(
        zero.generated_token_ids.empty() &&
            zero.token_ids == zero.prompt_token_ids,
        "zero-token generation should preserve only the prompt"
    );
    require_throws(
        [&] {
            static_cast<void>(
                engine.generate("", GenerationConfig{0})
            );
        },
        "an empty encoded prompt should be rejected"
    );
    require_throws(
        [&] {
            static_cast<void>(engine.generate(
                "1",
                GenerationConfig{
                    std::numeric_limits<std::size_t>::max(),
                }
            ));
        },
        "an impossible output size should be rejected before generation"
    );

    OutOfRangeSampler invalid_sampler;
    require_throws(
        [&] {
            static_cast<void>(engine.generate(
                "12",
                invalid_sampler,
                GenerationConfig{1}
            ));
        },
        "sampler tokens outside the vocabulary should be rejected"
    );

    ByteTokenizer mismatched_tokenizer("abc");
    require_throws(
        [&] {
            static_cast<void>(
                GenerationEngine(model, mismatched_tokenizer)
            );
        },
        "model/tokenizer vocabulary mismatch should be rejected"
    );
}

}  // namespace

int main() {
    try {
        test_greedy_sampler();
        test_temperature_sampler();
        test_contiguous_and_paged_cache_contracts();
        test_paged_pool_exhaustion_and_reuse();
        test_paged_pool_keeps_sessions_isolated();
        test_cached_generation_matches_full_forward();
        test_zero_token_generation_does_not_create_a_cache();
        test_generation_and_context_cropping();
        std::cout << "native serving generation tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
