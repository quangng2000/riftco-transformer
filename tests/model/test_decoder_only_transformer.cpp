#include "transformer_lab/core/backend.hpp"
#include "transformer_lab/model/decoder_only_transformer.hpp"
#include "transformer_lab/nn/loss.hpp"
#include "transformer_lab/nn/parameter.hpp"
#include "transformer_lab/optim/adam.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using transformer_lab::DecoderOnlyTransformer;
using transformer_lab::DecoderKeyValueCache;
using transformer_lab::ExecutionBackend;
using transformer_lab::FullSequenceAttentionKind;
using transformer_lab::ActivationCheckpointingKind;
using transformer_lab::NamedParameter;
using transformer_lab::Parameter;
using transformer_lab::Tensor;
using transformer_lab::TokenId;
using transformer_lab::TransformerDimensions;
using transformer_lab::Variable;
using transformer_lab::cross_entropy;
using transformer_lab::parameter_count;

constexpr TransformerDimensions kDimensions{
    5,
    4,
    4,
    2,
    2,
    6,
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void require_parameter_backend(
    const transformer_lab::ParameterList& parameters,
    ExecutionBackend backend,
    const std::string& message
) {
    require(!parameters.empty(), message + ": empty parameter list");
    for (const auto& named_parameter : parameters) {
        require(
            named_parameter.parameter != nullptr,
            message + ": null parameter " + named_parameter.name
        );
        require(
            named_parameter.parameter->value().backend() == backend,
            message + ": value backend for " + named_parameter.name
        );
        require(
            named_parameter.parameter->gradient().backend() == backend,
            message + ": gradient backend for " + named_parameter.name
        );
    }
}

void require_close(
    float actual,
    float expected,
    const std::string& message,
    float tolerance = 1.0e-5F
) {
    if (!std::isfinite(actual) ||
        !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) +
            ", got " + std::to_string(actual)
        );
    }
}

void require_tensor_close(
    const Tensor& actual,
    const Tensor& expected,
    const std::string& message,
    float tolerance = 1.0e-5F
) {
    require(
        actual.shape() == expected.shape(),
        message + ": shape mismatch"
    );
    require(
        actual.numel() == expected.numel(),
        message + ": value count mismatch"
    );
    for (std::size_t index = 0;
         index < actual.numel();
         ++index) {
        require_close(
            actual.flat(index),
            expected.flat(index),
            message + " at flat index " + std::to_string(index),
            tolerance
        );
    }
}

void require_finite_tensor(
    const Tensor& tensor,
    const std::string& message
) {
    for (std::size_t index = 0;
         index < tensor.numel();
         ++index) {
        require(
            std::isfinite(tensor.flat(index)),
            message + " at flat index " + std::to_string(index)
        );
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

class ReferenceDecoderCache final : public DecoderKeyValueCache {
public:
    ReferenceDecoderCache(
        std::size_t layer_count,
        std::size_t head_count,
        std::size_t head_width,
        std::size_t capacity,
        ExecutionBackend backend = ExecutionBackend::Cpu
    )
        : layer_count_(layer_count),
          head_count_(head_count),
          head_width_(head_width),
          capacity_(capacity),
          backend_(backend),
          keys_(layer_count),
          values_(layer_count),
          pending_keys_(layer_count),
          pending_values_(layer_count) {
        for (auto& layer : keys_) {
            layer.reserve(capacity_);
        }
        for (auto& layer : values_) {
            layer.reserve(capacity_);
        }
    }

    [[nodiscard]] ExecutionBackend backend() const noexcept override {
        return backend_;
    }

    [[nodiscard]] std::size_t layer_count() const noexcept override {
        return layer_count_;
    }

    [[nodiscard]] std::size_t head_count() const noexcept override {
        return head_count_;
    }

    [[nodiscard]] std::size_t head_width() const noexcept override {
        return head_width_;
    }

    [[nodiscard]] std::size_t capacity() const noexcept override {
        return capacity_;
    }

    [[nodiscard]] std::size_t size() const noexcept override {
        return size_;
    }

    void begin_token() override {
        if (active_) {
            throw std::logic_error("reference cache transaction is active");
        }
        if (size_ >= capacity_) {
            throw std::length_error("reference cache is full");
        }
        clear_pending();
        active_ = true;
        ++begin_count_;
    }

    [[nodiscard]] Tensor append_and_attend(
        std::size_t layer,
        const Tensor& query,
        const Tensor& key,
        const Tensor& value
    ) override {
        if (!active_) {
            throw std::logic_error(
                "reference cache has no active transaction"
            );
        }
        if (layer >= layer_count_) {
            throw std::out_of_range("reference cache layer");
        }
        if (fail_layer_.has_value() &&
            layer == *fail_layer_) {
            throw std::runtime_error("injected cache failure");
        }
        const Tensor::Shape expected_shape{
            1,
            head_count_,
            1,
            head_width_,
        };
        if (query.shape() != expected_shape ||
            key.shape() != expected_shape ||
            value.shape() != expected_shape) {
            throw std::invalid_argument(
                "reference cache tensor shape"
            );
        }
        if (query.backend() != backend_ ||
            key.backend() != backend_ ||
            value.backend() != backend_) {
            throw std::invalid_argument(
                "reference cache tensor backend"
            );
        }
        if (pending_keys_[layer].has_value()) {
            throw std::logic_error(
                "reference cache layer was appended twice"
            );
        }
        pending_keys_[layer] = key;
        pending_values_[layer] = value;

        Tensor context(expected_shape, backend_);
        const float score_scale =
            1.0F / std::sqrt(static_cast<float>(head_width_));
        std::vector<float> scores(size_ + 1, 0.0F);
        std::vector<float> probabilities(size_ + 1, 0.0F);
        for (std::size_t head = 0;
             head < head_count_;
             ++head) {
            float maximum =
                -std::numeric_limits<float>::infinity();
            for (std::size_t time = 0;
                 time <= size_;
                 ++time) {
                const Tensor& cached_key =
                    time == size_
                        ? *pending_keys_[layer]
                        : keys_[layer][time];
                float score = 0.0F;
                for (std::size_t channel = 0;
                     channel < head_width_;
                     ++channel) {
                    score +=
                        query.at({0, head, 0, channel}) *
                        cached_key.at({0, head, 0, channel});
                }
                score *= score_scale;
                scores[time] = score;
                maximum = std::max(maximum, score);
            }

            double denominator = 0.0;
            for (std::size_t time = 0;
                 time <= size_;
                 ++time) {
                const double exponential = std::exp(
                    static_cast<double>(scores[time] - maximum)
                );
                probabilities[time] =
                    static_cast<float>(exponential);
                denominator += exponential;
            }
            for (std::size_t time = 0;
                 time <= size_;
                 ++time) {
                probabilities[time] =
                    static_cast<float>(
                        static_cast<double>(probabilities[time]) /
                        denominator
                    );
            }
            for (std::size_t channel = 0;
                 channel < head_width_;
                 ++channel) {
                float total = 0.0F;
                for (std::size_t time = 0;
                     time <= size_;
                     ++time) {
                    const Tensor& cached_value =
                        time == size_
                            ? *pending_values_[layer]
                            : values_[layer][time];
                    total +=
                        probabilities[time] *
                        cached_value.at({0, head, 0, channel});
                }
                context.at({0, head, 0, channel}) = total;
            }
        }
        return context;
    }

    void commit_token() override {
        if (!active_) {
            throw std::logic_error(
                "reference cache has no transaction to commit"
            );
        }
        for (std::size_t layer = 0;
             layer < layer_count_;
             ++layer) {
            if (!pending_keys_[layer].has_value() ||
                !pending_values_[layer].has_value()) {
                throw std::logic_error(
                    "reference cache is missing a layer"
                );
            }
        }
        for (std::size_t layer = 0;
             layer < layer_count_;
             ++layer) {
            keys_[layer].push_back(
                std::move(*pending_keys_[layer])
            );
            values_[layer].push_back(
                std::move(*pending_values_[layer])
            );
        }
        ++size_;
        active_ = false;
        clear_pending();
    }

    void abort_token() noexcept override {
        clear_pending();
        active_ = false;
        ++abort_count_;
    }

    void reset() noexcept override {
        for (auto& layer : keys_) {
            layer.clear();
        }
        for (auto& layer : values_) {
            layer.clear();
        }
        clear_pending();
        size_ = 0;
        active_ = false;
    }

    void fail_on_layer(std::optional<std::size_t> layer) noexcept {
        fail_layer_ = layer;
    }

    [[nodiscard]] std::size_t begin_count() const noexcept {
        return begin_count_;
    }

    [[nodiscard]] std::size_t abort_count() const noexcept {
        return abort_count_;
    }

private:
    void clear_pending() noexcept {
        for (auto& key : pending_keys_) {
            key.reset();
        }
        for (auto& value : pending_values_) {
            value.reset();
        }
    }

    std::size_t layer_count_;
    std::size_t head_count_;
    std::size_t head_width_;
    std::size_t capacity_;
    ExecutionBackend backend_;
    std::size_t size_ = 0;
    bool active_ = false;
    std::vector<std::vector<Tensor>> keys_;
    std::vector<std::vector<Tensor>> values_;
    std::vector<std::optional<Tensor>> pending_keys_;
    std::vector<std::optional<Tensor>> pending_values_;
    std::optional<std::size_t> fail_layer_;
    std::size_t begin_count_ = 0;
    std::size_t abort_count_ = 0;
};

struct ExpectedParameter {
    std::string name;
    Tensor::Shape shape;
};

void append_expected_block_parameters(
    std::vector<ExpectedParameter>& expected,
    std::size_t block_index
) {
    const std::string prefix =
        "blocks." + std::to_string(block_index) + ".";
    expected.push_back({
        prefix + "attention_norm.scale",
        {4},
    });
    expected.push_back({
        prefix + "attention_norm.bias",
        {4},
    });
    expected.push_back({
        prefix + "attention.query.weight",
        {4, 4},
    });
    expected.push_back({
        prefix + "attention.query.bias",
        {4},
    });
    expected.push_back({
        prefix + "attention.key.weight",
        {4, 4},
    });
    expected.push_back({
        prefix + "attention.key.bias",
        {4},
    });
    expected.push_back({
        prefix + "attention.value.weight",
        {4, 4},
    });
    expected.push_back({
        prefix + "attention.value.bias",
        {4},
    });
    expected.push_back({
        prefix + "attention.output.weight",
        {4, 4},
    });
    expected.push_back({
        prefix + "attention.output.bias",
        {4},
    });
    expected.push_back({
        prefix + "feed_forward_norm.scale",
        {4},
    });
    expected.push_back({
        prefix + "feed_forward_norm.bias",
        {4},
    });
    expected.push_back({
        prefix + "feed_forward.expand.weight",
        {6, 4},
    });
    expected.push_back({
        prefix + "feed_forward.expand.bias",
        {6},
    });
    expected.push_back({
        prefix + "feed_forward.project.weight",
        {4, 6},
    });
    expected.push_back({
        prefix + "feed_forward.project.bias",
        {4},
    });
}

std::vector<ExpectedParameter> expected_parameters() {
    std::vector<ExpectedParameter> expected{
        {"token_embedding.weight", {5, 4}},
        {"position_embedding.weight", {4, 4}},
    };
    append_expected_block_parameters(expected, 0);
    append_expected_block_parameters(expected, 1);
    expected.push_back({"final_norm.scale", {4}});
    expected.push_back({"final_norm.bias", {4}});
    expected.push_back({"language_model_head.weight", {5, 4}});
    expected.push_back({"language_model_head.bias", {5}});
    return expected;
}

Parameter* find_parameter(
    const std::vector<NamedParameter>& parameters,
    const std::string& name
) {
    for (const auto& named_parameter : parameters) {
        if (named_parameter.name == name) {
            return named_parameter.parameter;
        }
    }
    throw std::runtime_error("missing parameter " + name);
}

void test_dimensions_registration_logits_and_determinism() {
    constexpr std::uint32_t seed = 1729U;
    std::mt19937 first_random(seed);
    std::mt19937 second_random(seed);
    DecoderOnlyTransformer first(kDimensions, first_random);
    DecoderOnlyTransformer second(kDimensions, second_random);

    const TransformerDimensions& dimensions = first.dimensions();
    require(
        dimensions.vocabulary_size == 5,
        "vocabulary size getter"
    );
    require(
        dimensions.maximum_context == 4,
        "maximum context getter"
    );
    require(dimensions.model_width == 4, "model width getter");
    require(dimensions.head_count == 2, "head count getter");
    require(dimensions.block_count == 2, "block count getter");
    require(
        dimensions.feed_forward_width == 6,
        "feed-forward width getter"
    );

    const auto expected = expected_parameters();
    auto first_parameters = first.parameters();
    auto second_parameters = second.parameters();
    require(expected.size() == 38, "test expects 38 tensors");
    require(
        first_parameters.size() == expected.size(),
        "registered parameter tensor count"
    );
    require(
        second_parameters.size() == expected.size(),
        "second registered parameter tensor count"
    );
    require(
        parameter_count(first_parameters) == 377,
        "registered scalar parameter count"
    );
    require(
        parameter_count(second_parameters) == 377,
        "second scalar parameter count"
    );

    std::set<const Parameter*> first_addresses;
    std::set<const Parameter*> second_addresses;
    for (std::size_t index = 0;
         index < expected.size();
         ++index) {
        const auto& first_named = first_parameters[index];
        const auto& second_named = second_parameters[index];
        require(
            first_named.name == expected[index].name,
            "parameter name/order at index " + std::to_string(index)
        );
        require(
            second_named.name == expected[index].name,
            "second parameter name/order at index " +
                std::to_string(index)
        );
        require(
            first_named.parameter != nullptr,
            "non-null parameter pointer at index " +
                std::to_string(index)
        );
        require(
            second_named.parameter != nullptr,
            "second non-null parameter pointer at index " +
                std::to_string(index)
        );
        require(
            first_addresses.insert(first_named.parameter).second,
            "unique parameter pointer at index " +
                std::to_string(index)
        );
        require(
            second_addresses.insert(second_named.parameter).second,
            "second unique parameter pointer at index " +
                std::to_string(index)
        );
        require(
            first_named.parameter->value().shape() ==
                expected[index].shape,
            "parameter shape for " + expected[index].name
        );
        require(
            second_named.parameter->value().shape() ==
                expected[index].shape,
            "second parameter shape for " + expected[index].name
        );
        require_tensor_close(
            first_named.parameter->value(),
            second_named.parameter->value(),
            "same-seed parameter " + expected[index].name,
            0.0F
        );
    }

    const std::vector<TokenId> token_ids{
        0, 1, 2,
        4, 3, 1,
    };
    const Tensor first_logits =
        first.forward(token_ids, {2, 3}).value();
    const Tensor second_logits =
        second.forward(token_ids, {2, 3}).value();
    require(
        first_logits.shape() == Tensor::Shape({2, 3, 5}),
        "decoder logits have shape [batch, time, vocabulary]"
    );
    require_tensor_close(
        first_logits,
        second_logits,
        "same-seed logits",
        0.0F
    );
}

void test_dimension_and_forward_validation() {
    {
        auto invalid = kDimensions;
        invalid.maximum_context = 0;
        std::mt19937 random(1U);
        require_throws(
            [&] {
                DecoderOnlyTransformer model(invalid, random);
            },
            "zero maximum context must be rejected"
        );
    }
    {
        auto invalid = kDimensions;
        invalid.head_count = 3;
        std::mt19937 random(2U);
        require_throws(
            [&] {
                DecoderOnlyTransformer model(invalid, random);
            },
            "non-divisible head count must be rejected"
        );
    }
    {
        std::mt19937 random(3U);
        require_throws(
            [&] {
                DecoderOnlyTransformer model(
                    kDimensions,
                    random,
                    0.0F
                );
            },
            "non-positive layer normalization epsilon must be rejected"
        );
    }

    const auto largest_token =
        static_cast<std::size_t>(
            std::numeric_limits<TokenId>::max()
        );
    if (largest_token <
        std::numeric_limits<std::size_t>::max()) {
        auto invalid = kDimensions;
        invalid.maximum_context = largest_token + std::size_t{2};
        std::mt19937 random(4U);
        require_throws(
            [&] {
                DecoderOnlyTransformer model(invalid, random);
            },
            "context that cannot fit TokenId must be rejected"
        );
    }

    std::mt19937 random(5U);
    DecoderOnlyTransformer model(kDimensions, random);
    const std::vector<TokenId> maximum_context_ids{0, 1, 2, 3};
    const Tensor maximum_context_logits =
        model.forward(maximum_context_ids, {1, 4}).value();
    require(
        maximum_context_logits.shape() ==
            Tensor::Shape({1, 4, 5}),
        "maximum context must be accepted"
    );

    const std::vector<TokenId> too_long_ids{0, 1, 2, 3, 4};
    require_throws(
        [&] {
            (void)model.forward(too_long_ids, {1, 5});
        },
        "sequence past maximum context must be rejected"
    );
    require_throws(
        [&] {
            (void)model.forward(maximum_context_ids, {4});
        },
        "rank-one token shape must be rejected"
    );
    require_throws(
        [&] {
            (void)model.forward(maximum_context_ids, {1, 2, 2});
        },
        "rank-three token shape must be rejected"
    );
    require_throws(
        [&] {
            (void)model.forward(
                std::vector<TokenId>{0, 1, 2},
                {2, 2}
            );
        },
        "token count mismatch must be rejected"
    );
    require_throws(
        [&] {
            (void)model.forward(
                std::vector<TokenId>{},
                {0, 1}
            );
        },
        "zero batch must be rejected"
    );
    require_throws(
        [&] {
            (void)model.forward(
                std::vector<TokenId>{},
                {1, 0}
            );
        },
        "zero time must be rejected"
    );
    require_throws(
        [&] {
            (void)model.forward(
                std::vector<TokenId>{0},
                {
                    std::numeric_limits<std::size_t>::max(),
                    2,
                }
            );
        },
        "overflowing token count must be rejected"
    );
    require_throws(
        [&] {
            (void)model.forward(
                std::vector<TokenId>{5},
                {1, 1}
            );
        },
        "out-of-vocabulary token must be rejected"
    );
}

void test_whole_model_causality() {
    std::mt19937 random(71U);
    DecoderOnlyTransformer model(kDimensions, random);
    const std::vector<TokenId> baseline_ids{0, 1, 2, 3};
    const std::vector<TokenId> changed_future_ids{0, 1, 2, 4};
    const Tensor baseline =
        model.forward(baseline_ids, {1, 4}).value();
    const Tensor changed =
        model.forward(changed_future_ids, {1, 4}).value();

    for (std::size_t time = 0; time < 3; ++time) {
        for (std::size_t token = 0; token < 5; ++token) {
            require_close(
                changed.at({0, time, token}),
                baseline.at({0, time, token}),
                "future token must not affect earlier logits",
                0.0F
            );
        }
    }

    bool final_position_changed = false;
    for (std::size_t token = 0; token < 5; ++token) {
        if (std::fabs(
                changed.at({0, 3, token}) -
                baseline.at({0, 3, token})
            ) > 1.0e-6F) {
            final_position_changed = true;
        }
    }
    require(
        final_position_changed,
        "changed token should affect its own logits"
    );
}

void test_whole_model_batch_isolation() {
    std::mt19937 random(79U);
    DecoderOnlyTransformer model(kDimensions, random);
    const std::vector<TokenId> baseline_ids{
        0, 1, 2,
        2, 3, 4,
    };
    const std::vector<TokenId> changed_second_batch_ids{
        0, 1, 2,
        4, 0, 1,
    };
    const Tensor baseline =
        model.forward(baseline_ids, {2, 3}).value();
    const Tensor changed =
        model.forward(changed_second_batch_ids, {2, 3}).value();

    for (std::size_t time = 0; time < 3; ++time) {
        for (std::size_t token = 0; token < 5; ++token) {
            require_close(
                changed.at({0, time, token}),
                baseline.at({0, time, token}),
                "second batch must not affect first batch",
                0.0F
            );
        }
    }

    bool second_batch_changed = false;
    for (std::size_t time = 0; time < 3; ++time) {
        for (std::size_t token = 0; token < 5; ++token) {
            if (std::fabs(
                    changed.at({1, time, token}) -
                    baseline.at({1, time, token})
                ) > 1.0e-6F) {
                second_batch_changed = true;
            }
        }
    }
    require(
        second_batch_changed,
        "changed second-batch tokens should change second-batch logits"
    );
}

void test_unused_position_rows_have_zero_gradient() {
    std::mt19937 random(103U);
    DecoderOnlyTransformer model(kDimensions, random);
    auto parameters = model.parameters();
    Parameter* position_embedding = find_parameter(
        parameters,
        "position_embedding.weight"
    );

    const std::vector<TokenId> token_ids{0, 1};
    const Variable logits = model.forward(token_ids, {1, 2});
    const Tensor seed_gradient(
        {1, 2, 5},
        {
            0.50F, -1.00F, 0.25F, 1.50F, -0.40F,
            -0.75F, 0.60F, 1.25F, -0.20F, 0.90F,
        }
    );
    logits.backward(seed_gradient);

    const Tensor& gradient = position_embedding->gradient();
    require(
        gradient.shape() == Tensor::Shape({4, 4}),
        "position embedding gradient shape"
    );
    bool used_row_has_gradient = false;
    for (std::size_t position = 0; position < 4; ++position) {
        for (std::size_t channel = 0; channel < 4; ++channel) {
            const float value = gradient.at({position, channel});
            require(
                std::isfinite(value),
                "position embedding gradient must be finite"
            );
            if (position < 2) {
                if (std::fabs(value) > 1.0e-7F) {
                    used_row_has_gradient = true;
                }
            } else {
                require_close(
                    value,
                    0.0F,
                    "unused position row gradient",
                    0.0F
                );
            }
        }
    }
    require(
        used_row_has_gradient,
        "a used position row should receive gradient"
    );
}

void test_cross_entropy_backward_reaches_every_parameter() {
    std::mt19937 random(131U);
    DecoderOnlyTransformer model(kDimensions, random);
    auto parameters = model.parameters();
    const std::vector<TokenId> token_ids{
        0, 1, 2,
        4, 3, 1,
    };
    const std::vector<TokenId> targets{
        1, 2, 3,
        3, 1, 0,
    };

    const Variable loss = cross_entropy(
        model.forward(token_ids, {2, 3}),
        targets
    );
    require(
        loss.value().shape().empty(),
        "cross entropy returns a scalar"
    );
    require(
        std::isfinite(loss.value().flat(0)),
        "cross entropy loss is finite"
    );
    loss.backward();

    require(parameters.size() == 38, "all model parameters registered");
    for (const auto& named_parameter : parameters) {
        require(
            named_parameter.parameter != nullptr,
            "backward parameter pointer is non-null"
        );
        require(
            named_parameter.parameter->gradient().shape() ==
                named_parameter.parameter->value().shape(),
            "gradient shape for " + named_parameter.name
        );
        require_finite_tensor(
            named_parameter.parameter->gradient(),
            "finite gradient for " + named_parameter.name
        );
    }
}

void test_full_sequence_attention_policy_and_parity() {
    constexpr std::uint32_t seed = 269U;
    std::mt19937 materialized_random(seed);
    std::mt19937 flash_random(seed);
    DecoderOnlyTransformer materialized(
        kDimensions,
        materialized_random
    );
    DecoderOnlyTransformer flash(
        kDimensions,
        flash_random,
        1.0e-5F,
        FullSequenceAttentionKind::Flash
    );

    require(
        materialized.full_sequence_attention_kind() ==
            FullSequenceAttentionKind::Materialized,
        "decoder should default to materialized full-sequence attention"
    );
    require(
        flash.full_sequence_attention_kind() ==
            FullSequenceAttentionKind::Flash,
        "decoder constructor should accept Flash attention"
    );

    const std::vector<TokenId> token_ids{
        0, 1, 2,
        4, 3, 1,
    };
    const std::vector<TokenId> targets{
        1, 2, 3,
        3, 1, 0,
    };
    const Variable materialized_logits =
        materialized.forward(token_ids, {2, 3});
    const Variable flash_logits =
        flash.forward(token_ids, {2, 3});
    require_tensor_close(
        flash_logits.value(),
        materialized_logits.value(),
        "Flash decoder forward parity",
        8.0e-5F
    );

    const Variable materialized_loss =
        cross_entropy(materialized_logits, targets);
    const Variable flash_loss =
        cross_entropy(flash_logits, targets);
    require_tensor_close(
        flash_loss.value(),
        materialized_loss.value(),
        "Flash decoder loss parity",
        8.0e-5F
    );
    materialized_loss.backward();
    flash_loss.backward();

    const auto materialized_parameters = materialized.parameters();
    const auto flash_parameters = flash.parameters();
    require(
        materialized_parameters.size() == flash_parameters.size(),
        "Flash decoder parameter count parity"
    );
    for (std::size_t index = 0;
         index < materialized_parameters.size();
         ++index) {
        require(
            materialized_parameters[index].name ==
                flash_parameters[index].name,
            "Flash decoder parameter name parity"
        );
        require_tensor_close(
            flash_parameters[index].parameter->gradient(),
            materialized_parameters[index].parameter->gradient(),
            "Flash decoder gradient parity for " +
                materialized_parameters[index].name,
            2.0e-4F
        );
    }

    if (transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        std::mt19937 metal_random(seed);
        DecoderOnlyTransformer metal_flash(
            kDimensions,
            metal_random,
            1.0e-5F,
            FullSequenceAttentionKind::Flash
        );
        metal_flash.to(ExecutionBackend::Metal);
        const Variable metal_logits =
            metal_flash.forward(token_ids, {2, 3});
        require(
            metal_logits.value().backend() == ExecutionBackend::Metal,
            "Flash decoder should preserve the Metal backend"
        );
        require_tensor_close(
            metal_logits.value(),
            flash_logits.value(),
            "Flash decoder Metal forward parity",
            1.0e-3F
        );
        const Variable metal_loss =
            cross_entropy(metal_logits, targets);
        require_tensor_close(
            metal_loss.value(),
            flash_loss.value(),
            "Flash decoder Metal loss parity",
            1.0e-3F
        );
        metal_loss.backward();

        const auto metal_parameters = metal_flash.parameters();
        require(
            metal_parameters.size() == flash_parameters.size(),
            "Flash decoder Metal parameter count parity"
        );
        for (std::size_t index = 0;
             index < flash_parameters.size();
             ++index) {
            require(
                metal_parameters[index].name ==
                    flash_parameters[index].name,
                "Flash decoder Metal parameter name parity"
            );
            require_tensor_close(
                metal_parameters[index].parameter->gradient(),
                flash_parameters[index].parameter->gradient(),
                "Flash decoder Metal gradient parity for " +
                    flash_parameters[index].name,
                2.0e-3F
            );
        }
    }

    materialized.set_full_sequence_attention_kind(
        FullSequenceAttentionKind::Flash
    );
    require(
        materialized.full_sequence_attention_kind() ==
            FullSequenceAttentionKind::Flash,
        "decoder setter should select Flash attention"
    );
    require_throws(
        [&] {
            materialized.set_full_sequence_attention_kind(
                static_cast<FullSequenceAttentionKind>(99)
            );
        },
        "decoder selector should reject unknown kinds"
    );
    require(
        materialized.full_sequence_attention_kind() ==
            FullSequenceAttentionKind::Flash,
        "invalid decoder attention selection preserves the prior policy"
    );
}

void test_activation_checkpointing_policy_graph_and_gradient_parity() {
    constexpr std::uint32_t seed = 277U;
    const std::vector<TokenId> token_ids{
        0, 1, 2,
        4, 3, 1,
    };
    const std::vector<TokenId> targets{
        1, 2, 3,
        3, 1, 0,
    };

    for (const auto attention_kind : {
             FullSequenceAttentionKind::Materialized,
             FullSequenceAttentionKind::Flash,
         }) {
        std::mt19937 regular_random(seed);
        std::mt19937 checkpointed_random(seed);
        DecoderOnlyTransformer regular(
            kDimensions,
            regular_random,
            1.0e-5F,
            attention_kind
        );
        DecoderOnlyTransformer checkpointed(
            kDimensions,
            checkpointed_random,
            1.0e-5F,
            attention_kind,
            ActivationCheckpointingKind::TransformerBlock
        );

        require(
            regular.activation_checkpointing_kind() ==
                ActivationCheckpointingKind::Disabled,
            "decoder should default to disabled activation checkpointing"
        );
        require(
            checkpointed.activation_checkpointing_kind() ==
                ActivationCheckpointingKind::TransformerBlock,
            "decoder constructor should select block checkpointing"
        );

        const Variable regular_logits =
            regular.forward(token_ids, {2, 3});
        const Variable checkpointed_logits =
            checkpointed.forward(token_ids, {2, 3});
        require_tensor_close(
            checkpointed_logits.value(),
            regular_logits.value(),
            "checkpointed decoder forward parity",
            8.0e-5F
        );
        const auto regular_graph = regular_logits.graph_statistics();
        const auto checkpointed_graph =
            checkpointed_logits.graph_statistics();
        require(
            checkpointed_graph.node_count < regular_graph.node_count,
            "block checkpointing should retain fewer decoder graph nodes"
        );
        require(
            checkpointed_graph.node_tensor_elements <
                regular_graph.node_tensor_elements,
            "block checkpointing should retain fewer node Tensor elements"
        );

        const Variable regular_loss =
            cross_entropy(regular_logits, targets);
        const Variable checkpointed_loss =
            cross_entropy(checkpointed_logits, targets);
        regular_loss.backward();

        // The replay must preserve the forward-time attention algorithm.
        checkpointed.set_full_sequence_attention_kind(
            attention_kind ==
                    FullSequenceAttentionKind::Materialized
                ? FullSequenceAttentionKind::Flash
                : FullSequenceAttentionKind::Materialized
        );
        checkpointed_loss.backward();

        const auto regular_parameters = regular.parameters();
        const auto checkpointed_parameters =
            checkpointed.parameters();
        require(
            regular_parameters.size() ==
                checkpointed_parameters.size(),
            "checkpointed decoder parameter count parity"
        );
        for (std::size_t index = 0;
             index < regular_parameters.size();
             ++index) {
            require(
                regular_parameters[index].name ==
                    checkpointed_parameters[index].name,
                "checkpointed decoder parameter name parity"
            );
            require_tensor_close(
                checkpointed_parameters[index].parameter->gradient(),
                regular_parameters[index].parameter->gradient(),
                "checkpointed decoder gradient parity for " +
                    regular_parameters[index].name,
                3.0e-4F
            );
        }

        transformer_lab::Adam regular_optimizer(
            regular_parameters
        );
        transformer_lab::Adam checkpointed_optimizer(
            checkpointed_parameters
        );
        static_cast<void>(regular_optimizer.step());
        static_cast<void>(checkpointed_optimizer.step());
        for (std::size_t index = 0;
             index < regular_parameters.size();
             ++index) {
            require_tensor_close(
                checkpointed_parameters[index].parameter->value(),
                regular_parameters[index].parameter->value(),
                "checkpointed Adam-step parity for " +
                    regular_parameters[index].name,
                4.0e-4F
            );
        }
    }

    std::mt19937 policy_random(seed);
    DecoderOnlyTransformer policy_model(kDimensions, policy_random);
    policy_model.set_activation_checkpointing_kind(
        ActivationCheckpointingKind::TransformerBlock
    );
    require(
        policy_model.activation_checkpointing_kind() ==
            ActivationCheckpointingKind::TransformerBlock,
        "activation checkpointing setter should update future forwards"
    );
    require_throws(
        [&] {
            policy_model.set_activation_checkpointing_kind(
                static_cast<ActivationCheckpointingKind>(99)
            );
        },
        "activation checkpointing setter should reject unknown kinds"
    );
    require(
        policy_model.activation_checkpointing_kind() ==
            ActivationCheckpointingKind::TransformerBlock,
        "invalid checkpointing selection preserves the prior policy"
    );
}

void test_activation_checkpointing_lora_and_metal_parity() {
    constexpr std::uint32_t seed = 281U;
    const transformer_lab::LoraConfig lora{
        1,
        2.0F,
        41U,
        transformer_lab::kLoraDefaultTargets,
    };
    const std::vector<TokenId> token_ids{0, 1, 2};
    const std::vector<TokenId> targets{1, 2, 3};

    std::mt19937 regular_random(seed);
    std::mt19937 checkpointed_random(seed);
    DecoderOnlyTransformer regular(
        kDimensions,
        regular_random,
        1.0e-5F,
        FullSequenceAttentionKind::Flash
    );
    DecoderOnlyTransformer checkpointed(
        kDimensions,
        checkpointed_random,
        1.0e-5F,
        FullSequenceAttentionKind::Flash,
        ActivationCheckpointingKind::TransformerBlock
    );
    regular.attach_lora(lora);
    checkpointed.attach_lora(lora);
    cross_entropy(
        regular.forward(token_ids, {1, 3}),
        targets
    ).backward();
    cross_entropy(
        checkpointed.forward(token_ids, {1, 3}),
        targets
    ).backward();

    const auto regular_adapters = regular.lora_parameters();
    const auto checkpointed_adapters =
        checkpointed.lora_parameters();
    require(
        regular_adapters.size() == checkpointed_adapters.size(),
        "checkpointed LoRA parameter count parity"
    );
    for (std::size_t index = 0;
         index < regular_adapters.size();
         ++index) {
        require(
            regular_adapters[index].name ==
                checkpointed_adapters[index].name,
            "checkpointed LoRA parameter name parity"
        );
        require_tensor_close(
            checkpointed_adapters[index].parameter->gradient(),
            regular_adapters[index].parameter->gradient(),
            "checkpointed LoRA gradient parity for " +
                regular_adapters[index].name,
            3.0e-4F
        );
    }

    if (!transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }
    for (const auto attention_kind : {
             FullSequenceAttentionKind::Materialized,
             FullSequenceAttentionKind::Flash,
         }) {
        std::mt19937 metal_regular_random(seed);
        std::mt19937 metal_checkpointed_random(seed);
        DecoderOnlyTransformer metal_regular(
            kDimensions,
            metal_regular_random,
            1.0e-5F,
            attention_kind
        );
        DecoderOnlyTransformer metal_checkpointed(
            kDimensions,
            metal_checkpointed_random,
            1.0e-5F,
            attention_kind,
            ActivationCheckpointingKind::TransformerBlock
        );
        metal_regular.to(ExecutionBackend::Metal);
        metal_checkpointed.to(ExecutionBackend::Metal);
        const Variable metal_regular_loss = cross_entropy(
            metal_regular.forward(token_ids, {1, 3}),
            targets
        );
        const Variable metal_checkpointed_loss = cross_entropy(
            metal_checkpointed.forward(token_ids, {1, 3}),
            targets
        );
        require_tensor_close(
            metal_checkpointed_loss.value(),
            metal_regular_loss.value(),
            "Metal checkpointed loss parity",
            1.0e-3F
        );
        metal_regular_loss.backward();
        metal_checkpointed.set_full_sequence_attention_kind(
            attention_kind ==
                    FullSequenceAttentionKind::Materialized
                ? FullSequenceAttentionKind::Flash
                : FullSequenceAttentionKind::Materialized
        );
        metal_checkpointed_loss.backward();

        const auto metal_regular_parameters =
            metal_regular.parameters();
        const auto metal_checkpointed_parameters =
            metal_checkpointed.parameters();
        for (std::size_t index = 0;
             index < metal_regular_parameters.size();
             ++index) {
            require_tensor_close(
                metal_checkpointed_parameters[index]
                    .parameter->gradient(),
                metal_regular_parameters[index].parameter->gradient(),
                "Metal checkpointed gradient parity for " +
                    metal_regular_parameters[index].name,
                2.0e-3F
            );
        }
    }
}

void test_activation_checkpoint_graph_lifetime_and_decode_independence() {
    std::optional<Variable> retained_loss;
    {
        std::mt19937 random(283U);
        DecoderOnlyTransformer model(
            kDimensions,
            random,
            1.0e-5F,
            FullSequenceAttentionKind::Materialized,
            ActivationCheckpointingKind::TransformerBlock
        );
        retained_loss.emplace(cross_entropy(
            model.forward(
                std::vector<TokenId>{0, 1, 2},
                {1, 3}
            ),
            std::vector<TokenId>{1, 2, 3}
        ));
    }
    retained_loss->backward();

    std::mt19937 stale_random(287U);
    DecoderOnlyTransformer stale_model(
        kDimensions,
        stale_random,
        1.0e-5F,
        FullSequenceAttentionKind::Materialized,
        ActivationCheckpointingKind::TransformerBlock
    );
    const Variable stale_loss = cross_entropy(
        stale_model.forward(
            std::vector<TokenId>{0, 1, 2},
            {1, 3}
        ),
        std::vector<TokenId>{1, 2, 3}
    );
    auto stale_parameters = stale_model.parameters();
    stale_model.attach_lora(transformer_lab::LoraConfig{
        1,
        2.0F,
        43U,
        transformer_lab::kLoraDefaultTargets,
    });
    require_throws(
        [&] { stale_loss.backward(); },
        "checkpoint replay should reject a structural LoRA change"
    );
    for (const auto& named_parameter : stale_parameters) {
        for (const float gradient :
             named_parameter.parameter->gradient().data()) {
            require_close(
                gradient,
                0.0F,
                "failed structural replay should not commit gradients",
                0.0F
            );
        }
    }

    constexpr std::uint32_t seed = 293U;
    std::mt19937 regular_random(seed);
    std::mt19937 checkpointed_random(seed);
    DecoderOnlyTransformer regular(kDimensions, regular_random);
    DecoderOnlyTransformer checkpointed(
        kDimensions,
        checkpointed_random,
        1.0e-5F,
        FullSequenceAttentionKind::Flash,
        ActivationCheckpointingKind::TransformerBlock
    );
    ReferenceDecoderCache regular_cache(
        kDimensions.block_count,
        kDimensions.head_count,
        2,
        kDimensions.maximum_context
    );
    ReferenceDecoderCache checkpointed_cache(
        kDimensions.block_count,
        kDimensions.head_count,
        2,
        kDimensions.maximum_context
    );
    for (const TokenId token : std::vector<TokenId>{0, 1, 2}) {
        require_tensor_close(
            checkpointed.decode_token(token, checkpointed_cache),
            regular.decode_token(token, regular_cache),
            "activation checkpointing must not alter incremental decode",
            2.0e-5F
        );
    }
}

std::pair<std::size_t, float> largest_gradient_entry(
    const Tensor& gradient
) {
    require(gradient.numel() > 0, "selected gradient is non-empty");
    std::size_t largest_index = 0;
    float largest_magnitude = 0.0F;
    for (std::size_t index = 0;
         index < gradient.numel();
         ++index) {
        const float value = gradient.flat(index);
        require(std::isfinite(value), "selected gradient is finite");
        const float magnitude = std::fabs(value);
        if (magnitude > largest_magnitude) {
            largest_magnitude = magnitude;
            largest_index = index;
        }
    }
    return {largest_index, largest_magnitude};
}

float evaluate_loss(
    DecoderOnlyTransformer& model,
    const std::vector<TokenId>& token_ids,
    const Tensor::Shape& token_shape,
    const std::vector<TokenId>& targets
) {
    return cross_entropy(
        model.forward(token_ids, token_shape),
        targets
    ).value().flat(0);
}

void test_selected_parameter_finite_differences() {
    std::mt19937 random(197U);
    DecoderOnlyTransformer model(kDimensions, random);
    auto parameters = model.parameters();
    const std::vector<TokenId> token_ids{0, 1, 2, 3};
    const Tensor::Shape token_shape{1, 4};
    const std::vector<TokenId> targets{1, 2, 3, 4};

    const Variable loss = cross_entropy(
        model.forward(token_ids, token_shape),
        targets
    );
    loss.backward();

    const std::vector<std::string> selected_names{
        "token_embedding.weight",
        "position_embedding.weight",
        "blocks.0.attention.value.weight",
        "blocks.1.feed_forward.project.weight",
        "final_norm.scale",
        "language_model_head.weight",
        "language_model_head.bias",
    };
    constexpr float epsilon = 1.0e-3F;
    constexpr float tolerance = 3.0e-2F;

    for (const std::string& name : selected_names) {
        Parameter* parameter = find_parameter(parameters, name);
        const Tensor original = parameter->value();
        const Tensor analytical_gradient = parameter->gradient();
        const auto [flat_index, largest_magnitude] =
            largest_gradient_entry(analytical_gradient);
        require(
            largest_magnitude > 1.0e-7F,
            "selected parameter has a meaningful gradient: " + name
        );
        const float analytical =
            analytical_gradient.flat(flat_index);

        Tensor plus = original;
        plus.flat(flat_index) += epsilon;
        parameter->set_value(std::move(plus));
        const float plus_loss = evaluate_loss(
            model,
            token_ids,
            token_shape,
            targets
        );

        Tensor minus = original;
        minus.flat(flat_index) -= epsilon;
        parameter->set_value(std::move(minus));
        const float minus_loss = evaluate_loss(
            model,
            token_ids,
            token_shape,
            targets
        );

        parameter->set_value(original);
        const float numerical =
            (plus_loss - minus_loss) / (2.0F * epsilon);
        require_close(
            analytical,
            numerical,
            "finite difference for " + name +
                " at flat index " + std::to_string(flat_index),
            tolerance
        );
    }
}

Tensor final_position_logits(
    const Tensor& full_logits,
    std::size_t position,
    std::size_t vocabulary_size
) {
    std::vector<float> values;
    values.reserve(vocabulary_size);
    for (std::size_t token = 0;
         token < vocabulary_size;
         ++token) {
        values.push_back(
            full_logits.at({0, position, token})
        );
    }
    return Tensor({1, 1, vocabulary_size}, std::move(values));
}

void test_incremental_decode_parity_and_validation() {
    std::mt19937 random(223U);
    DecoderOnlyTransformer model(kDimensions, random);
    require(
        model.backend() == ExecutionBackend::Cpu,
        "new decoder reports its CPU backend"
    );

    ReferenceDecoderCache cache(
        kDimensions.block_count,
        kDimensions.head_count,
        kDimensions.model_width / kDimensions.head_count,
        kDimensions.maximum_context
    );
    const std::vector<TokenId> tokens{0, 1, 2, 3};
    std::vector<TokenId> prefix;
    for (std::size_t position = 0;
         position < tokens.size();
         ++position) {
        prefix.push_back(tokens[position]);
        const Tensor incremental =
            model.decode_token(tokens[position], cache);
        require(
            incremental.shape() ==
                Tensor::Shape({
                    1,
                    1,
                    kDimensions.vocabulary_size,
                }),
            "incremental decoder logit shape"
        );
        const Tensor full =
            model.forward(prefix, {1, prefix.size()}).value();
        require_tensor_close(
            incremental,
            final_position_logits(
                full,
                position,
                kDimensions.vocabulary_size
            ),
            "incremental decoder matches full prefix at position " +
                std::to_string(position),
            2.0e-5F
        );
        require(
            cache.size() == position + 1,
            "successful decode commits exactly one cache position"
        );
    }
    require_throws(
        [&] {
            static_cast<void>(model.decode_token(0, cache));
        },
        "incremental decoder rejects a full cache"
    );
    require(
        cache.size() == kDimensions.maximum_context,
        "overflow rejection preserves cache size"
    );

    const auto expect_pretransaction_rejection =
        [&](ReferenceDecoderCache& invalid,
            const std::string& message) {
            require_throws(
                [&] {
                    static_cast<void>(
                        model.decode_token(0, invalid)
                    );
                },
                message
            );
            require(
                invalid.begin_count() == 0 &&
                    invalid.size() == 0,
                message + " occurs before a transaction"
            );
        };
    ReferenceDecoderCache wrong_layers(
        kDimensions.block_count + 1,
        kDimensions.head_count,
        2,
        4
    );
    expect_pretransaction_rejection(
        wrong_layers,
        "incremental decoder rejects cache layer mismatch"
    );
    ReferenceDecoderCache wrong_heads(
        kDimensions.block_count,
        kDimensions.head_count + 1,
        2,
        4
    );
    expect_pretransaction_rejection(
        wrong_heads,
        "incremental decoder rejects cache head mismatch"
    );
    ReferenceDecoderCache wrong_width(
        kDimensions.block_count,
        kDimensions.head_count,
        3,
        4
    );
    expect_pretransaction_rejection(
        wrong_width,
        "incremental decoder rejects cache head-width mismatch"
    );
    ReferenceDecoderCache zero_capacity(
        kDimensions.block_count,
        kDimensions.head_count,
        2,
        0
    );
    expect_pretransaction_rejection(
        zero_capacity,
        "incremental decoder rejects zero cache capacity"
    );
    ReferenceDecoderCache oversized_capacity(
        kDimensions.block_count,
        kDimensions.head_count,
        2,
        kDimensions.maximum_context + 1
    );
    expect_pretransaction_rejection(
        oversized_capacity,
        "incremental decoder rejects capacity beyond model context"
    );
    ReferenceDecoderCache wrong_backend(
        kDimensions.block_count,
        kDimensions.head_count,
        2,
        4,
        ExecutionBackend::Metal
    );
    expect_pretransaction_rejection(
        wrong_backend,
        "incremental decoder rejects cache backend mismatch"
    );
    ReferenceDecoderCache valid(
        kDimensions.block_count,
        kDimensions.head_count,
        2,
        4
    );
    require_throws(
        [&] {
            static_cast<void>(
                model.decode_token(
                    static_cast<TokenId>(
                        kDimensions.vocabulary_size
                    ),
                    valid
                )
            );
        },
        "incremental decoder rejects an invalid token"
    );
    require(
        valid.begin_count() == 0,
        "invalid token is rejected before a cache transaction"
    );
}

void test_incremental_decode_abort_and_lora_parity() {
    std::mt19937 random(227U);
    DecoderOnlyTransformer model(kDimensions, random);
    model.attach_lora(transformer_lab::LoraConfig{
        1,
        2.0F,
        37U,
        transformer_lab::kLoraDefaultTargets,
    });
    for (const auto& named_parameter :
         model.lora_parameters()) {
        if (named_parameter.name.ends_with("lora_b.weight")) {
            named_parameter.parameter->set_value(
                Tensor::full(
                    named_parameter.parameter->value().shape(),
                    0.05F,
                    model.backend()
                )
            );
        }
    }

    ReferenceDecoderCache cache(
        kDimensions.block_count,
        kDimensions.head_count,
        2,
        4
    );
    cache.fail_on_layer(1);
    require_throws(
        [&] {
            static_cast<void>(model.decode_token(1, cache));
        },
        "incremental decoder propagates a cache-layer failure"
    );
    require(
        cache.size() == 0 &&
            cache.abort_count() == 1,
        "cache-layer failure aborts the whole token"
    );

    cache.fail_on_layer(std::nullopt);
    const Tensor incremental = model.decode_token(1, cache);
    const Tensor full =
        model.forward(std::vector<TokenId>{1}, {1, 1}).value();
    require_tensor_close(
        incremental,
        final_position_logits(
            full,
            0,
            kDimensions.vocabulary_size
        ),
        "incremental decoder honors active LoRA projections",
        2.0e-5F
    );
    require(
        cache.size() == 1,
        "retry after abort commits one token"
    );
}

void test_model_device_transfer_and_forward() {
    const transformer_lab::ScopedExecutionBackend cpu_backend(
        ExecutionBackend::Cpu
    );
    std::mt19937 random(229U);
    DecoderOnlyTransformer model(kDimensions, random);
    const std::vector<TokenId> token_ids{0, 1, 2};
    const Tensor::Shape token_shape{1, 3};
    const Tensor expected = model.forward(
        token_ids,
        token_shape
    ).value();

    model.to(ExecutionBackend::Cpu);
    require(
        model.backend() == ExecutionBackend::Cpu,
        "CPU transfer updates the model backend getter"
    );
    require_parameter_backend(
        model.parameters(),
        ExecutionBackend::Cpu,
        "decoder-only transformer CPU transfer"
    );

    if (!transformer_lab::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    model.to(ExecutionBackend::Metal);
    require(
        model.backend() == ExecutionBackend::Metal,
        "Metal transfer updates the model backend getter"
    );
    require_parameter_backend(
        model.parameters(),
        ExecutionBackend::Metal,
        "decoder-only transformer Metal transfer"
    );
    const Tensor actual = model.forward(
        token_ids,
        token_shape
    ).value();
    require(
        actual.backend() == ExecutionBackend::Metal,
        "model forward should preserve the transferred backend"
    );
    require_tensor_close(
        actual,
        expected,
        "model transfer forward parity",
        1.0e-3F
    );

    model.to(ExecutionBackend::Cpu);
    require_parameter_backend(
        model.parameters(),
        ExecutionBackend::Cpu,
        "decoder-only transformer CPU round trip"
    );
}

}  // namespace

int main() {
    try {
        test_dimensions_registration_logits_and_determinism();
        test_dimension_and_forward_validation();
        test_whole_model_causality();
        test_whole_model_batch_isolation();
        test_unused_position_rows_have_zero_gradient();
        test_cross_entropy_backward_reaches_every_parameter();
        test_full_sequence_attention_policy_and_parity();
        test_activation_checkpointing_policy_graph_and_gradient_parity();
        test_activation_checkpointing_lora_and_metal_parity();
        test_activation_checkpoint_graph_lifetime_and_decode_independence();
        test_selected_parameter_finite_differences();
        test_incremental_decode_parity_and_validation();
        test_incremental_decode_abort_and_lora_parity();
        test_model_device_transfer_and_forward();
    } catch (const std::exception& error) {
        std::cerr
            << "decoder-only transformer test failed: "
            << error.what() << '\n';
        return 1;
    }

    std::cout << "decoder-only transformer tests passed\n";
    return 0;
}
