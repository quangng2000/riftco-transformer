#include "core/backend/adapter.hpp"

#include "riftco_transformer/core/tensor.hpp"
#include "riftco_transformer/core/tensor_ops.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using riftco_transformer::ExecutionBackend;
using riftco_transformer::Tensor;
namespace backend = riftco_transformer::backend_detail;
namespace tensor_ops = riftco_transformer::tensor_ops;

#ifndef RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL
#define RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL 0
#endif

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

void require_close(
    float actual,
    float expected,
    const std::string& message,
    float tolerance = 1.0e-5F
) {
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": expected " + std::to_string(expected) + ", got " +
            std::to_string(actual)
        );
    }
}

void require_tensor_close(
    const Tensor& actual,
    const Tensor& expected,
    const std::string& message,
    float tolerance = 3.0e-4F
) {
    require(actual.shape() == expected.shape(), message + " shape");
    for (std::size_t index = 0; index < actual.numel(); ++index) {
        const float scale = std::max(
            {1.0F,
             std::fabs(actual.flat(index)),
             std::fabs(expected.flat(index))}
        );
        if (!std::isfinite(actual.flat(index)) ||
            !std::isfinite(expected.flat(index)) ||
            std::fabs(actual.flat(index) - expected.flat(index)) >
                tolerance * scale) {
            throw std::runtime_error(
                message + " at index " + std::to_string(index)
            );
        }
    }
}

void require_tensor_equal(
    const Tensor& actual, const Tensor& expected, const std::string& message
) {
    require(actual.shape() == expected.shape(), message + " shape");
    const auto actual_values = actual.data();
    const auto expected_values = expected.data();
    for (std::size_t index = 0; index < actual_values.size(); ++index) {
        if (actual_values[index] != expected_values[index]) {
            throw std::runtime_error(
                message + " at index " + std::to_string(index)
            );
        }
    }
}

float dot(const Tensor& left, const Tensor& right) {
    require(left.shape() == right.shape(), "dot shape mismatch");
    float result = 0.0F;
    for (std::size_t index = 0; index < left.numel(); ++index) {
        result += left.flat(index) * right.flat(index);
    }
    return result;
}

struct MaterializedCausalAttentionOutput {
    Tensor probabilities;
    Tensor context;
};

MaterializedCausalAttentionOutput materialized_causal_attention_forward(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    backend::MaterializedCausalAttentionDimensions dimensions
) {
    const auto execution_backend = queries.backend();
    Tensor probabilities(
        {
            dimensions.batch,
            dimensions.heads,
            dimensions.time,
            dimensions.time,
        },
        execution_backend
    );
    Tensor context(queries.shape(), execution_backend);
    backend::dispatch_materialized_causal_attention_forward(
        execution_backend,
        {
            backend::tensor_storage(queries),
            backend::tensor_storage(keys),
            backend::tensor_storage(values),
            backend::tensor_storage(probabilities),
            backend::tensor_storage(context),
            dimensions,
        }
    );
    return {
        std::move(probabilities),
        std::move(context),
    };
}

struct FlashCausalAttentionOutput {
    Tensor row_maxima;
    Tensor row_exp_sums;
    Tensor context;
};

FlashCausalAttentionOutput flash_causal_attention_forward(
    const Tensor& queries,
    const Tensor& keys,
    const Tensor& values,
    backend::FlashCausalAttentionDimensions dimensions
) {
    const auto execution_backend = queries.backend();
    const Tensor::Shape row_shape{
        dimensions.batch,
        dimensions.heads,
        dimensions.time,
    };
    Tensor row_maxima(row_shape, execution_backend);
    Tensor row_exp_sums(row_shape, execution_backend);
    Tensor context(queries.shape(), execution_backend);
    backend::dispatch_flash_causal_attention_forward(
        execution_backend,
        {
            backend::tensor_storage(queries),
            backend::tensor_storage(keys),
            backend::tensor_storage(values),
            backend::tensor_storage(row_maxima),
            backend::tensor_storage(row_exp_sums),
            backend::tensor_storage(context),
            dimensions,
        }
    );
    return {
        std::move(row_maxima),
        std::move(row_exp_sums),
        std::move(context),
    };
}

Tensor paged_decode_attention_forward(
    const Tensor& query,
    const Tensor& key_pages,
    const Tensor& value_pages,
    std::span<const std::uint32_t> block_table,
    backend::PagedDecodeAttentionDimensions dimensions
) {
    Tensor context(query.shape(), query.backend());
    backend::dispatch_paged_decode_attention_forward(
        query.backend(),
        {
            backend::tensor_storage(query),
            backend::tensor_storage(key_pages),
            backend::tensor_storage(value_pages),
            block_table,
            backend::tensor_storage(context),
            dimensions,
        }
    );
    return context;
}

void test_layer_norm_reference() {
    const Tensor input({2, 3}, {0.2F, -0.7F, 1.3F, 2.0F, -1.0F, 0.4F});
    const Tensor scale({3}, {1.2F, 0.8F, -0.5F});
    const Tensor bias({3}, {0.1F, -0.2F, 0.3F});
    Tensor output({2, 3});
    Tensor mean({2});
    Tensor inverse_standard_deviation({2});
    backend::dispatch_layer_norm_forward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(input),
            backend::tensor_storage(scale),
            backend::tensor_storage(bias),
            backend::tensor_storage(output),
            backend::tensor_storage(mean),
            backend::tensor_storage(inverse_standard_deviation),
            2,
            3,
            1.0e-5F,
        }
    );

    const Tensor upstream({2, 3}, {0.5F, -1.0F, 0.25F, 1.5F, 0.75F, -0.4F});
    Tensor input_gradient({2, 3});
    Tensor scale_gradient({3});
    Tensor bias_gradient({3});
    backend::dispatch_layer_norm_backward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(input),
            backend::tensor_storage(scale),
            backend::tensor_storage(mean),
            backend::tensor_storage(inverse_standard_deviation),
            backend::tensor_storage(upstream),
            backend::tensor_storage(input_gradient),
            backend::tensor_storage(scale_gradient),
            backend::tensor_storage(bias_gradient),
            2,
            3,
        }
    );

    const auto evaluate = [&](const Tensor& candidate) {
        Tensor candidate_output({2, 3});
        Tensor candidate_mean({2});
        Tensor candidate_inverse_std({2});
        backend::dispatch_layer_norm_forward(
            ExecutionBackend::Cpu,
            {
                backend::tensor_storage(candidate),
                backend::tensor_storage(scale),
                backend::tensor_storage(bias),
                backend::tensor_storage(candidate_output),
                backend::tensor_storage(candidate_mean),
                backend::tensor_storage(candidate_inverse_std),
                2,
                3,
                1.0e-5F,
            }
        );
        return dot(candidate_output, upstream);
    };

    constexpr float epsilon = 1.0e-3F;
    for (std::size_t index = 0; index < input.numel(); ++index) {
        Tensor plus = input;
        Tensor minus = input;
        plus.flat(index) += epsilon;
        minus.flat(index) -= epsilon;
        require_close(
            input_gradient.flat(index),
            (evaluate(plus) - evaluate(minus)) / (2.0F * epsilon),
            "layer norm input gradient",
            2.0e-3F
        );
    }
    require_close(
        bias_gradient.flat(0),
        upstream.flat(0) + upstream.flat(3),
        "layer norm bias gradient"
    );

    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    const Tensor metal_input = input.to(ExecutionBackend::Metal);
    const Tensor metal_scale = scale.to(ExecutionBackend::Metal);
    const Tensor metal_bias = bias.to(ExecutionBackend::Metal);
    Tensor metal_output({2, 3}, ExecutionBackend::Metal);
    Tensor metal_mean({2}, ExecutionBackend::Metal);
    Tensor metal_inverse_standard_deviation({2}, ExecutionBackend::Metal);
    backend::dispatch_layer_norm_forward(
        ExecutionBackend::Metal,
        {
            backend::tensor_storage(metal_input),
            backend::tensor_storage(metal_scale),
            backend::tensor_storage(metal_bias),
            backend::tensor_storage(metal_output),
            backend::tensor_storage(metal_mean),
            backend::tensor_storage(metal_inverse_standard_deviation),
            2,
            3,
            1.0e-5F,
        }
    );
    require_tensor_close(metal_output, output, "LayerNorm Metal output");
    require_tensor_close(metal_mean, mean, "LayerNorm Metal mean");
    require_tensor_close(
        metal_inverse_standard_deviation,
        inverse_standard_deviation,
        "LayerNorm Metal inverse standard deviation"
    );

    const Tensor metal_upstream = upstream.to(ExecutionBackend::Metal);
    Tensor metal_input_gradient({2, 3}, ExecutionBackend::Metal);
    Tensor metal_scale_gradient({3}, ExecutionBackend::Metal);
    Tensor metal_bias_gradient({3}, ExecutionBackend::Metal);
    backend::dispatch_layer_norm_backward(
        ExecutionBackend::Metal,
        {
            backend::tensor_storage(metal_input),
            backend::tensor_storage(metal_scale),
            backend::tensor_storage(metal_mean),
            backend::tensor_storage(metal_inverse_standard_deviation),
            backend::tensor_storage(metal_upstream),
            backend::tensor_storage(metal_input_gradient),
            backend::tensor_storage(metal_scale_gradient),
            backend::tensor_storage(metal_bias_gradient),
            2,
            3,
        }
    );
    require_tensor_close(
        metal_input_gradient, input_gradient, "LayerNorm Metal input VJP"
    );
    require_tensor_close(
        metal_scale_gradient, scale_gradient, "LayerNorm Metal scale VJP"
    );
    require_tensor_close(
        metal_bias_gradient, bias_gradient, "LayerNorm Metal bias VJP"
    );
}

void test_cross_entropy_reference() {
    const Tensor logits({2, 3}, {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
    const std::vector<std::uint32_t> targets{0, 2};
    Tensor loss(Tensor::Shape{});
    Tensor base_gradient({2, 3});
    backend::dispatch_cross_entropy_forward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(logits),
            targets,
            backend::tensor_storage(loss),
            backend::tensor_storage(base_gradient),
            2,
            3,
        }
    );
    require_close(loss.flat(0), std::log(3.0F), "cross entropy loss");
    const std::vector<float> expected{
        -1.0F / 3.0F,
        1.0F / 6.0F,
        1.0F / 6.0F,
        1.0F / 6.0F,
        1.0F / 6.0F,
        -1.0F / 3.0F,
    };
    for (std::size_t index = 0; index < expected.size(); ++index) {
        require_close(
            base_gradient.flat(index), expected[index], "cross entropy gradient"
        );
    }
}

void test_extreme_loss_and_special_value_backend_parity() {
    const Tensor logits(
        {2, 2},
        {
            0.0F,
            -2.0e38F,
            0.0F,
            -2.0e38F,
        }
    );
    const std::vector<std::uint32_t> targets{1, 1};
    Tensor cpu_loss(Tensor::Shape{});
    Tensor cpu_gradient({2, 2});
    backend::dispatch_cross_entropy_forward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(logits),
            targets,
            backend::tensor_storage(cpu_loss),
            backend::tensor_storage(cpu_gradient),
            2,
            2,
        }
    );
    require(
        std::isfinite(cpu_loss.flat(0)),
        "extreme cross entropy mean must remain finite"
    );
    require_close(
        cpu_loss.flat(0) / 2.0e38F,
        1.0F,
        "extreme cross entropy CPU value",
        1.0e-5F
    );

    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    const Tensor metal_logits = logits.to(ExecutionBackend::Metal);
    Tensor metal_loss(Tensor::Shape{}, ExecutionBackend::Metal);
    Tensor metal_gradient({2, 2}, ExecutionBackend::Metal);
    backend::dispatch_cross_entropy_forward(
        ExecutionBackend::Metal,
        {
            backend::tensor_storage(metal_logits),
            targets,
            backend::tensor_storage(metal_loss),
            backend::tensor_storage(metal_gradient),
            2,
            2,
        }
    );
    require_tensor_close(
        metal_loss, cpu_loss, "extreme cross entropy Metal value", 2.0e-5F
    );
    require_tensor_close(
        metal_gradient, cpu_gradient, "extreme cross entropy Metal gradient"
    );

    constexpr std::size_t skewed_position_count = 1'000'000;
    std::vector<float> skewed_values(skewed_position_count * 2, 0.0F);
    std::vector<std::uint32_t> skewed_targets(skewed_position_count, 0);
    for (std::size_t position = 0; position < skewed_position_count;
         ++position) {
        skewed_values[position * 2] = 100.0F;
        skewed_values[position * 2 + 1] = -100.0F;
    }
    skewed_values[0] = 0.0F;
    skewed_values[1] = -1.0e30F;
    skewed_targets[0] = 1;
    const Tensor skewed_logits(
        {skewed_position_count, 2}, std::move(skewed_values)
    );
    Tensor skewed_cpu_loss(Tensor::Shape{});
    Tensor skewed_cpu_gradient({skewed_position_count, 2});
    backend::dispatch_cross_entropy_forward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(skewed_logits),
            skewed_targets,
            backend::tensor_storage(skewed_cpu_loss),
            backend::tensor_storage(skewed_cpu_gradient),
            skewed_position_count,
            2,
        }
    );
    const Tensor skewed_metal_logits =
        skewed_logits.to(ExecutionBackend::Metal);
    Tensor skewed_metal_loss(Tensor::Shape{}, ExecutionBackend::Metal);
    Tensor skewed_metal_gradient(
        {skewed_position_count, 2}, ExecutionBackend::Metal
    );
    backend::dispatch_cross_entropy_forward(
        ExecutionBackend::Metal,
        {
            backend::tensor_storage(skewed_metal_logits),
            skewed_targets,
            backend::tensor_storage(skewed_metal_loss),
            backend::tensor_storage(skewed_metal_gradient),
            skewed_position_count,
            2,
        }
    );
    require_close(
        skewed_metal_loss.flat(0) / skewed_cpu_loss.flat(0),
        1.0F,
        "skewed large-position Metal cross entropy mean",
        3.0e-5F
    );

    const float quiet_nan = std::numeric_limits<float>::quiet_NaN();
    const Tensor nan_input({1}, quiet_nan);
    Tensor cpu_log({1});
    backend::dispatch_unary_elementwise(
        ExecutionBackend::Cpu,
        {
            backend::UnaryOperation::Log,
            backend::tensor_storage(nan_input),
            backend::tensor_storage(cpu_log),
            1,
        }
    );
    const Tensor metal_nan_input = nan_input.to(ExecutionBackend::Metal);
    Tensor metal_log({1}, ExecutionBackend::Metal);
    backend::dispatch_unary_elementwise(
        ExecutionBackend::Metal,
        {
            backend::UnaryOperation::Log,
            backend::tensor_storage(metal_nan_input),
            backend::tensor_storage(metal_log),
            1,
        }
    );
    require(
        std::isnan(cpu_log.flat(0)) && std::isnan(metal_log.flat(0)),
        "CPU and Metal log must propagate NaN identically"
    );

    const float maximum_float = std::numeric_limits<float>::max();
    const Tensor mixed_invalid_logits(
        {2, 2},
        {
            0.0F,
            quiet_nan,
            maximum_float,
            -maximum_float,
        }
    );
    const std::vector<std::uint32_t> mixed_targets{0, 1};
    const auto throws_domain_error = [&](const Tensor& candidate,
                                         ExecutionBackend execution_backend) {
        Tensor candidate_loss(Tensor::Shape{}, execution_backend);
        Tensor candidate_gradient({2, 2}, execution_backend);
        try {
            backend::dispatch_cross_entropy_forward(
                execution_backend,
                {
                    backend::tensor_storage(candidate),
                    mixed_targets,
                    backend::tensor_storage(candidate_loss),
                    backend::tensor_storage(candidate_gradient),
                    2,
                    2,
                }
            );
        } catch (const std::domain_error&) {
            return true;
        }
        return false;
    };
    require(
        throws_domain_error(mixed_invalid_logits, ExecutionBackend::Cpu),
        "CPU mixed-invalid cross entropy status"
    );
    require(
        throws_domain_error(
            mixed_invalid_logits.to(ExecutionBackend::Metal),
            ExecutionBackend::Metal
        ),
        "Metal mixed-invalid cross entropy status priority"
    );

    const float negative_infinity = -std::numeric_limits<float>::infinity();
    const Tensor causal_scores(
        {1, 1, 2, 2}, {0.0F, negative_infinity, negative_infinity, 0.0F}
    );
    Tensor cpu_probabilities({1, 1, 2, 2});
    backend::dispatch_causal_softmax_forward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(causal_scores),
            backend::tensor_storage(cpu_probabilities),
            1,
            1,
            2,
            1.0F,
        }
    );
    const Tensor metal_scores = causal_scores.to(ExecutionBackend::Metal);
    Tensor metal_probabilities({1, 1, 2, 2}, ExecutionBackend::Metal);
    backend::dispatch_causal_softmax_forward(
        ExecutionBackend::Metal,
        {
            backend::tensor_storage(metal_scores),
            backend::tensor_storage(metal_probabilities),
            1,
            1,
            2,
            1.0F,
        }
    );
    require_tensor_close(
        metal_probabilities,
        cpu_probabilities,
        "causal softmax negative-infinity Metal parity"
    );
}

void test_metal_tensor_neural_substrate_parity() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    const Tensor input({2, 3}, {-2.0F, -0.5F, 0.0F, 0.25F, 1.0F, 2.0F});
    const Tensor other({2, 3}, {0.5F, 1.5F, 2.0F, 2.5F, 3.0F, 4.0F});
    const Tensor upstream({2, 3}, {0.2F, -0.4F, 0.6F, -0.8F, 1.0F, 1.2F});
    const Tensor metal_input = input.to(ExecutionBackend::Metal);
    const Tensor metal_other = other.to(ExecutionBackend::Metal);
    const Tensor metal_upstream = upstream.to(ExecutionBackend::Metal);

    require_tensor_close(
        tensor_ops::add(metal_input, metal_other),
        tensor_ops::add(input, other),
        "Metal elementwise add"
    );
    require_tensor_close(
        tensor_ops::multiply(metal_input, metal_other),
        tensor_ops::multiply(input, other),
        "Metal elementwise multiply"
    );
    require_tensor_close(
        tensor_ops::scale(metal_input, -0.75F),
        tensor_ops::scale(input, -0.75F),
        "Metal elementwise scale"
    );
    require_tensor_close(
        tensor_ops::erf(metal_input),
        tensor_ops::erf(input),
        "Metal erf",
        8.0e-6F
    );
    require_tensor_close(
        tensor_ops::gelu(metal_input),
        tensor_ops::gelu(input),
        "Metal fused GELU",
        8.0e-6F
    );
    require_tensor_close(
        tensor_ops::gelu_backward(metal_input, metal_upstream),
        tensor_ops::gelu_backward(input, upstream),
        "Metal fused GELU VJP",
        8.0e-6F
    );
    require_tensor_close(
        tensor_ops::sum(metal_input, 1),
        tensor_ops::sum(input, 1),
        "Metal axis reduction"
    );
    require_tensor_close(
        tensor_ops::permute(metal_input, {1, 0}),
        tensor_ops::permute(input, {1, 0}),
        "Metal permutation"
    );

    const Tensor broadcast_source({1, 3}, {0.5F, -1.0F, 2.0F});
    const Tensor metal_broadcast_source =
        broadcast_source.to(ExecutionBackend::Metal);
    const Tensor cpu_broadcast =
        tensor_ops::broadcast_to(broadcast_source, {2, 3});
    const Tensor metal_broadcast =
        tensor_ops::broadcast_to(metal_broadcast_source, {2, 3});
    require_tensor_close(metal_broadcast, cpu_broadcast, "Metal broadcast");
    require_tensor_close(
        tensor_ops::sum_to_shape(metal_broadcast, {1, 3}),
        tensor_ops::sum_to_shape(cpu_broadcast, {1, 3}),
        "Metal sum-to-shape"
    );
    require_tensor_close(
        tensor_ops::softmax(metal_input, 1),
        tensor_ops::softmax(input, 1),
        "Metal softmax"
    );

    constexpr std::size_t wide_class_count = 50'000;
    std::vector<float> wide_values(wide_class_count, std::log(0.1F));
    wide_values[0] = 0.0F;
    const Tensor wide_logits({wide_class_count}, std::move(wide_values));
    const Tensor cpu_wide_softmax = tensor_ops::softmax(wide_logits, 0);
    const Tensor metal_wide_softmax =
        tensor_ops::softmax(wide_logits.to(ExecutionBackend::Metal), 0);
    require_close(
        metal_wide_softmax.flat(0) / cpu_wide_softmax.flat(0),
        1.0F,
        "wide Metal softmax leading probability",
        3.0e-5F
    );
    double metal_probability_sum = 0.0;
    for (const float probability : metal_wide_softmax.data()) {
        metal_probability_sum += probability;
    }
    require_close(
        static_cast<float>(metal_probability_sum),
        1.0F,
        "wide Metal softmax normalization",
        3.0e-5F
    );

    const Tensor table(
        {4, 3},
        {
            0.0F,
            0.1F,
            0.2F,
            1.0F,
            1.1F,
            1.2F,
            2.0F,
            2.1F,
            2.2F,
            3.0F,
            3.1F,
            3.2F,
        }
    );
    const std::vector<std::size_t> indices{2, 1, 2};
    const Tensor metal_table = table.to(ExecutionBackend::Metal);
    const Tensor cpu_gather = tensor_ops::gather_rows(table, indices, {3});
    const Tensor metal_gather =
        tensor_ops::gather_rows(metal_table, indices, {3});
    require_tensor_close(metal_gather, cpu_gather, "Metal embedding gather");
    require_tensor_close(
        tensor_ops::scatter_add_rows(metal_gather, indices, 4),
        tensor_ops::scatter_add_rows(cpu_gather, indices, 4),
        "Metal deterministic embedding scatter-add"
    );
}

void test_metal_layout_and_scatter_scalability() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    // Both reductions visit the same flat-index subsequences. Cancellation
    // makes a changed visit order observable rather than merely approximate.
    const Tensor ordered_input(
        {2, 2, 2},
        {
            1.0e20F,
            -1.0e20F,
            -1.0e20F,
            1.0e20F,
            3.0F,
            4.0F,
            5.0F,
            6.0F,
        }
    );
    const Tensor metal_ordered_input =
        ordered_input.to(ExecutionBackend::Metal);
    for (const Tensor::Shape& output_shape :
         std::vector<Tensor::Shape>{{1, 2, 1}, {2, 1}}) {
        require_tensor_equal(
            tensor_ops::sum_to_shape(metal_ordered_input, output_shape),
            tensor_ops::sum_to_shape(ordered_input, output_shape),
            "deterministic Metal sum-to-shape order"
        );
    }

    const std::vector<std::size_t> repeated_rows{
        3,
        1,
        3,
        3,
        0,
        1,
    };
    const Tensor ordered_upstream(
        {6, 2},
        {
            1.0e20F,
            -1.0e20F,
            2.0F,
            7.0F,
            -1.0e20F,
            1.0e20F,
            3.0F,
            4.0F,
            -5.0F,
            6.0F,
            8.0F,
            -9.0F,
        }
    );
    const Tensor expected_scatter =
        tensor_ops::scatter_add_rows(ordered_upstream, repeated_rows, 5);
    const Tensor metal_ordered_upstream =
        ordered_upstream.to(ExecutionBackend::Metal);
    const Tensor first_scatter =
        tensor_ops::scatter_add_rows(metal_ordered_upstream, repeated_rows, 5);
    const Tensor second_scatter =
        tensor_ops::scatter_add_rows(metal_ordered_upstream, repeated_rows, 5);
    require_tensor_equal(
        first_scatter,
        expected_scatter,
        "grouped Metal scatter preserves source order"
    );
    require_tensor_equal(
        second_scatter, first_scatter, "grouped Metal scatter is deterministic"
    );

    // These sizes model residual-gradient reduction and vocabulary embedding
    // scatter without making the test memory-heavy. The former scan-all
    // kernels take well over a second for the reduction and hundreds of
    // milliseconds for the scatter on the reference M4 Max.
    const Tensor large_sum_input({64, 256, 64}, 0.001F);
    const Tensor expected_large_sum =
        tensor_ops::sum_to_shape(large_sum_input, {1, 256, 1});
    const Tensor metal_large_sum_input =
        large_sum_input.to(ExecutionBackend::Metal);
    const auto sum_start = std::chrono::steady_clock::now();
    const Tensor metal_large_sum =
        tensor_ops::sum_to_shape(metal_large_sum_input, {1, 256, 1});
    const auto sum_elapsed = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - sum_start
    )
                                 .count();
    require_tensor_equal(
        metal_large_sum, expected_large_sum, "realistic Metal sum-to-shape"
    );
    require(
        sum_elapsed < 1'000.0,
        "Metal sum-to-shape scalability regression: " +
            std::to_string(sum_elapsed) + " ms"
    );

    constexpr std::size_t vocabulary_rows = 100'000;
    constexpr std::size_t embedding_width = 32;
    constexpr std::size_t token_positions = 32'768;
    const Tensor large_upstream({token_positions, embedding_width}, 0.001F);
    std::vector<std::size_t> large_indices(token_positions);
    for (std::size_t position = 0; position < token_positions; ++position) {
        large_indices[position] = (position * 7'919) % vocabulary_rows;
    }
    const Tensor expected_large_scatter = tensor_ops::scatter_add_rows(
        large_upstream, large_indices, vocabulary_rows
    );
    const Tensor metal_large_upstream =
        large_upstream.to(ExecutionBackend::Metal);
    const auto scatter_start = std::chrono::steady_clock::now();
    const Tensor metal_large_scatter = tensor_ops::scatter_add_rows(
        metal_large_upstream, large_indices, vocabulary_rows
    );
    const auto scatter_elapsed =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - scatter_start
        )
            .count();
    require_tensor_equal(
        metal_large_scatter,
        expected_large_scatter,
        "realistic grouped Metal scatter"
    );
    require(
        scatter_elapsed < 500.0,
        "Metal scatter scalability regression: " +
            std::to_string(scatter_elapsed) + " ms"
    );
}

void test_attention_reference_and_gradients() {
    const backend::MaterializedCausalAttentionDimensions dimensions{
        1,
        1,
        3,
        2,
    };
    const Tensor queries({1, 1, 3, 2}, {0.2F, -0.3F, 0.5F, 0.7F, -0.4F, 0.9F});
    const Tensor keys({1, 1, 3, 2}, {-0.1F, 0.4F, 0.8F, -0.6F, 0.3F, 0.5F});
    const Tensor values({1, 1, 3, 2}, {1.0F, -0.5F, 0.2F, 0.7F, -0.3F, 1.2F});
    const Tensor context_upstream(
        {1, 1, 3, 2}, {0.5F, -1.0F, 1.5F, 0.25F, -0.75F, 0.6F}
    );

    const auto output = materialized_causal_attention_forward(
        queries,
        keys,
        values,
        dimensions
    );
    for (std::size_t query = 0; query < dimensions.time; ++query) {
        float row_sum = 0.0F;
        for (std::size_t key = 0; key < dimensions.time; ++key) {
            const float probability =
                output.probabilities.flat(query * dimensions.time + key);
            row_sum += probability;
            if (key > query) {
                require(
                    probability == 0.0F,
                    "future attention probability must be zero"
                );
            }
        }
        require_close(row_sum, 1.0F, "attention row sum");
    }

    Tensor query_gradient(queries.shape());
    Tensor key_gradient(keys.shape());
    Tensor value_gradient(values.shape());
    backend::dispatch_materialized_causal_attention_context_backward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(queries),
            backend::tensor_storage(keys),
            backend::tensor_storage(values),
            backend::tensor_storage(output.probabilities),
            backend::tensor_storage(context_upstream),
            backend::tensor_storage(query_gradient),
            backend::tensor_storage(key_gradient),
            backend::tensor_storage(value_gradient),
            dimensions,
        }
    );

    const auto evaluate_context = [&](const Tensor& candidate_queries,
                                      const Tensor& candidate_keys,
                                      const Tensor& candidate_values) {
        return dot(
            materialized_causal_attention_forward(
                candidate_queries, candidate_keys, candidate_values, dimensions
            )
                .context,
            context_upstream
        );
    };

    constexpr float epsilon = 1.0e-3F;
    constexpr float tolerance = 3.0e-3F;
    const auto check_gradient = [&](const Tensor& original,
                                    const Tensor& analytical,
                                    const auto& evaluate,
                                    const std::string& name) {
        for (std::size_t index = 0; index < original.numel(); ++index) {
            Tensor plus = original;
            Tensor minus = original;
            plus.flat(index) += epsilon;
            minus.flat(index) -= epsilon;
            require_close(
                analytical.flat(index),
                (evaluate(plus) - evaluate(minus)) / (2.0F * epsilon),
                name,
                tolerance
            );
        }
    };
    check_gradient(
        queries,
        query_gradient,
        [&](const Tensor& candidate) {
            return evaluate_context(candidate, keys, values);
        },
        "attention query gradient"
    );
    check_gradient(
        keys,
        key_gradient,
        [&](const Tensor& candidate) {
            return evaluate_context(queries, candidate, values);
        },
        "attention key gradient"
    );
    check_gradient(
        values,
        value_gradient,
        [&](const Tensor& candidate) {
            return evaluate_context(queries, keys, candidate);
        },
        "attention value gradient"
    );

    const Tensor probability_upstream(
        {1, 1, 3, 3},
        {
            0.5F,
            7.0F,
            -4.0F,
            -1.0F,
            0.25F,
            3.0F,
            1.5F,
            0.75F,
            -0.4F,
        }
    );
    Tensor probability_query_gradient(queries.shape());
    Tensor probability_key_gradient(keys.shape());
    backend::dispatch_materialized_causal_attention_probabilities_backward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(queries),
            backend::tensor_storage(keys),
            backend::tensor_storage(output.probabilities),
            backend::tensor_storage(probability_upstream),
            backend::tensor_storage(probability_query_gradient),
            backend::tensor_storage(probability_key_gradient),
            dimensions,
        }
    );
    const auto evaluate_probabilities = [&](const Tensor& candidate_queries,
                                            const Tensor& candidate_keys) {
        return dot(
            materialized_causal_attention_forward(
                candidate_queries, candidate_keys, values, dimensions
            )
                .probabilities,
            probability_upstream
        );
    };
    check_gradient(
        queries,
        probability_query_gradient,
        [&](const Tensor& candidate) {
            return evaluate_probabilities(candidate, keys);
        },
        "attention probability query gradient"
    );
    check_gradient(
        keys,
        probability_key_gradient,
        [&](const Tensor& candidate) {
            return evaluate_probabilities(queries, candidate);
        },
        "attention probability key gradient"
    );

    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    const Tensor metal_queries = queries.to(ExecutionBackend::Metal);
    const Tensor metal_keys = keys.to(ExecutionBackend::Metal);
    const Tensor metal_values = values.to(ExecutionBackend::Metal);
    const auto metal_output =
        materialized_causal_attention_forward(
            metal_queries,
            metal_keys,
            metal_values,
            dimensions
        );
    require_tensor_close(
        metal_output.probabilities,
        output.probabilities,
        "fused attention Metal probabilities"
    );
    require_tensor_close(
        metal_output.context, output.context, "fused attention Metal context"
    );

    const Tensor metal_context_upstream =
        context_upstream.to(ExecutionBackend::Metal);
    Tensor metal_query_gradient(queries.shape(), ExecutionBackend::Metal);
    Tensor metal_key_gradient(keys.shape(), ExecutionBackend::Metal);
    Tensor metal_value_gradient(values.shape(), ExecutionBackend::Metal);
    backend::dispatch_materialized_causal_attention_context_backward(
        ExecutionBackend::Metal,
        {
            backend::tensor_storage(metal_queries),
            backend::tensor_storage(metal_keys),
            backend::tensor_storage(metal_values),
            backend::tensor_storage(metal_output.probabilities),
            backend::tensor_storage(metal_context_upstream),
            backend::tensor_storage(metal_query_gradient),
            backend::tensor_storage(metal_key_gradient),
            backend::tensor_storage(metal_value_gradient),
            dimensions,
        }
    );
    require_tensor_close(
        metal_query_gradient, query_gradient, "fused attention Metal query VJP"
    );
    require_tensor_close(
        metal_key_gradient, key_gradient, "fused attention Metal key VJP"
    );
    require_tensor_close(
        metal_value_gradient, value_gradient, "fused attention Metal value VJP"
    );

    const Tensor metal_probability_upstream =
        probability_upstream.to(ExecutionBackend::Metal);
    Tensor metal_probability_query_gradient(
        queries.shape(), ExecutionBackend::Metal
    );
    Tensor metal_probability_key_gradient(
        keys.shape(), ExecutionBackend::Metal
    );
    backend::dispatch_materialized_causal_attention_probabilities_backward(
        ExecutionBackend::Metal,
        {
            backend::tensor_storage(metal_queries),
            backend::tensor_storage(metal_keys),
            backend::tensor_storage(metal_output.probabilities),
            backend::tensor_storage(metal_probability_upstream),
            backend::tensor_storage(metal_probability_query_gradient),
            backend::tensor_storage(metal_probability_key_gradient),
            dimensions,
        }
    );
    require_tensor_close(
        metal_probability_query_gradient,
        probability_query_gradient,
        "attention probability Metal query VJP"
    );
    require_tensor_close(
        metal_probability_key_gradient,
        probability_key_gradient,
        "attention probability Metal key VJP"
    );
}

void test_flash_causal_attention_reference_and_gradients() {
    constexpr std::size_t batch_count = 2;
    constexpr std::size_t head_count = 2;
    constexpr std::size_t time = 9;
    constexpr std::size_t head_width = 3;
    const backend::MaterializedCausalAttentionDimensions
        materialized_dimensions{
            batch_count,
            head_count,
            time,
            head_width,
        };
    const backend::FlashCausalAttentionDimensions flash_dimensions{
        batch_count,
        head_count,
        time,
        head_width,
    };
    const Tensor::Shape tensor_shape{
        batch_count,
        head_count,
        time,
        head_width,
    };
    const std::size_t tensor_size =
        batch_count * head_count * time * head_width;
    std::vector<float> query_values(tensor_size);
    std::vector<float> key_values(tensor_size);
    std::vector<float> value_values(tensor_size);
    std::vector<float> upstream_values(tensor_size);
    for (std::size_t index = 0; index < tensor_size; ++index) {
        query_values[index] =
            static_cast<float>(
                static_cast<int>(index % 13) - 6
            ) *
            0.071F;
        key_values[index] =
            static_cast<float>(
                static_cast<int>((index * 3) % 17) - 8
            ) *
            0.053F;
        value_values[index] =
            static_cast<float>(
                static_cast<int>((index * 5) % 19) - 9
            ) *
            0.047F;
        upstream_values[index] =
            static_cast<float>(
                static_cast<int>((index * 7) % 23) - 11
            ) *
            0.031F;
    }

    const Tensor queries(tensor_shape, query_values);
    const Tensor keys(tensor_shape, key_values);
    const Tensor values(tensor_shape, value_values);
    const Tensor upstream(tensor_shape, upstream_values);
    const Tensor original_queries = queries;
    const Tensor original_keys = keys;
    const Tensor original_values = values;
    const auto materialized = materialized_causal_attention_forward(
        queries,
        keys,
        values,
        materialized_dimensions
    );
    const auto flash = flash_causal_attention_forward(
        queries,
        keys,
        values,
        flash_dimensions
    );

    require(
        flash.row_maxima.shape() ==
            Tensor::Shape({batch_count, head_count, time}),
        "Flash attention row maxima must use linear storage"
    );
    require(
        flash.row_exp_sums.shape() ==
            Tensor::Shape({batch_count, head_count, time}),
        "Flash attention row exponential sums must use linear storage"
    );
    require_tensor_close(
        flash.context,
        materialized.context,
        "Flash attention CPU context parity",
        2.0e-5F
    );

    const float score_scale =
        1.0F / std::sqrt(static_cast<float>(head_width));
    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        for (std::size_t head = 0; head < head_count; ++head) {
            for (std::size_t query_time = 0;
                 query_time < time;
                 ++query_time) {
                const std::size_t row =
                    (batch * head_count + head) * time +
                    query_time;
                float reconstructed_sum = 0.0F;
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    float score = 0.0F;
                    for (std::size_t channel = 0;
                         channel < head_width;
                         ++channel) {
                        const std::size_t query_index =
                            (
                                (
                                    batch * head_count + head
                                ) *
                                    time +
                                query_time
                            ) *
                                head_width +
                            channel;
                        const std::size_t key_index =
                            (
                                (
                                    batch * head_count + head
                                ) *
                                    time +
                                key_time
                            ) *
                                head_width +
                            channel;
                        score +=
                            queries.flat(query_index) *
                            keys.flat(key_index);
                    }
                    score *= score_scale;
                    const float probability = static_cast<float>(
                        std::exp(
                            static_cast<double>(
                                score -
                                flash.row_maxima.flat(row)
                            )
                        ) /
                        static_cast<double>(
                            flash.row_exp_sums.flat(row)
                        )
                    );
                    const std::size_t probability_index =
                        row * time + key_time;
                    require_close(
                        probability,
                        materialized.probabilities.flat(
                            probability_index
                        ),
                        "Flash reconstructed probability",
                        2.0e-5F
                    );
                    reconstructed_sum += probability;
                }
                require_close(
                    reconstructed_sum,
                    1.0F,
                    "Flash reconstructed probability row sum",
                    2.0e-5F
                );
            }
        }
    }

    Tensor materialized_query_gradient(tensor_shape);
    Tensor materialized_key_gradient(tensor_shape);
    Tensor materialized_value_gradient(tensor_shape);
    backend::dispatch_materialized_causal_attention_context_backward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(queries),
            backend::tensor_storage(keys),
            backend::tensor_storage(values),
            backend::tensor_storage(materialized.probabilities),
            backend::tensor_storage(upstream),
            backend::tensor_storage(materialized_query_gradient),
            backend::tensor_storage(materialized_key_gradient),
            backend::tensor_storage(materialized_value_gradient),
            materialized_dimensions,
        }
    );
    Tensor flash_query_gradient(tensor_shape);
    Tensor flash_key_gradient(tensor_shape);
    Tensor flash_value_gradient(tensor_shape);
    backend::dispatch_flash_causal_attention_backward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(queries),
            backend::tensor_storage(keys),
            backend::tensor_storage(values),
            backend::tensor_storage(flash.row_maxima),
            backend::tensor_storage(flash.row_exp_sums),
            backend::tensor_storage(upstream),
            backend::tensor_storage(flash_query_gradient),
            backend::tensor_storage(flash_key_gradient),
            backend::tensor_storage(flash_value_gradient),
            flash_dimensions,
        }
    );
    require_tensor_close(
        flash_query_gradient,
        materialized_query_gradient,
        "Flash attention CPU query VJP",
        5.0e-5F
    );
    require_tensor_close(
        flash_key_gradient,
        materialized_key_gradient,
        "Flash attention CPU key VJP",
        5.0e-5F
    );
    require_tensor_close(
        flash_value_gradient,
        materialized_value_gradient,
        "Flash attention CPU value VJP",
        5.0e-5F
    );
    if (riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        const Tensor metal_queries =
            queries.to(ExecutionBackend::Metal);
        const Tensor metal_keys =
            keys.to(ExecutionBackend::Metal);
        const Tensor metal_values =
            values.to(ExecutionBackend::Metal);
        const Tensor metal_upstream =
            upstream.to(ExecutionBackend::Metal);
        const auto metal_flash = flash_causal_attention_forward(
            metal_queries,
            metal_keys,
            metal_values,
            flash_dimensions
        );
        require_tensor_close(
            metal_flash.row_maxima,
            flash.row_maxima,
            "Flash attention Metal row maxima",
            5.0e-5F
        );
        require_tensor_close(
            metal_flash.row_exp_sums,
            flash.row_exp_sums,
            "Flash attention Metal row exponential sums",
            5.0e-5F
        );
        require_tensor_close(
            metal_flash.context,
            flash.context,
            "Flash attention Metal context",
            5.0e-5F
        );

        Tensor metal_query_gradient(
            tensor_shape,
            ExecutionBackend::Metal
        );
        Tensor metal_key_gradient(
            tensor_shape,
            ExecutionBackend::Metal
        );
        Tensor metal_value_gradient(
            tensor_shape,
            ExecutionBackend::Metal
        );
        backend::dispatch_flash_causal_attention_backward(
            ExecutionBackend::Metal,
            {
                backend::tensor_storage(metal_queries),
                backend::tensor_storage(metal_keys),
                backend::tensor_storage(metal_values),
                backend::tensor_storage(metal_flash.row_maxima),
                backend::tensor_storage(metal_flash.row_exp_sums),
                backend::tensor_storage(metal_upstream),
                backend::tensor_storage(metal_query_gradient),
                backend::tensor_storage(metal_key_gradient),
                backend::tensor_storage(metal_value_gradient),
                flash_dimensions,
            }
        );
        require_tensor_close(
            metal_query_gradient,
            flash_query_gradient,
            "Flash attention Metal query VJP",
            1.0e-4F
        );
        require_tensor_close(
            metal_key_gradient,
            flash_key_gradient,
            "Flash attention Metal key VJP",
            1.0e-4F
        );
        require_tensor_close(
            metal_value_gradient,
            flash_value_gradient,
            "Flash attention Metal value VJP",
            1.0e-4F
        );
    }
    require_tensor_equal(
        queries,
        original_queries,
        "Flash attention must not modify queries"
    );
    require_tensor_equal(
        keys,
        original_keys,
        "Flash attention must not modify keys"
    );
    require_tensor_equal(
        values,
        original_values,
        "Flash attention must not modify values"
    );

    Tensor causal_upstream(tensor_shape, 0.0F);
    for (std::size_t channel = 0;
         channel < head_width;
         ++channel) {
        causal_upstream.at({0, 0, time - 2, channel}) =
            0.25F + static_cast<float>(channel);
    }
    Tensor causal_query_gradient(tensor_shape);
    Tensor causal_key_gradient(tensor_shape);
    Tensor causal_value_gradient(tensor_shape);
    backend::dispatch_flash_causal_attention_backward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(queries),
            backend::tensor_storage(keys),
            backend::tensor_storage(values),
            backend::tensor_storage(flash.row_maxima),
            backend::tensor_storage(flash.row_exp_sums),
            backend::tensor_storage(causal_upstream),
            backend::tensor_storage(causal_query_gradient),
            backend::tensor_storage(causal_key_gradient),
            backend::tensor_storage(causal_value_gradient),
            flash_dimensions,
        }
    );
    for (std::size_t channel = 0;
         channel < head_width;
         ++channel) {
        require(
            causal_key_gradient.at({
                0,
                0,
                time - 1,
                channel,
            }) == 0.0F,
            "Flash attention reached a future key"
        );
        require(
            causal_value_gradient.at({
                0,
                0,
                time - 1,
                channel,
            }) == 0.0F,
            "Flash attention reached a future value"
        );
    }

    const backend::FlashCausalAttentionDimensions finite_dimensions{
        1,
        1,
        3,
        2,
    };
    const Tensor finite_queries(
        {1, 1, 3, 2},
        {0.2F, -0.3F, 0.5F, 0.7F, -0.4F, 0.9F}
    );
    const Tensor finite_keys(
        {1, 1, 3, 2},
        {-0.1F, 0.4F, 0.8F, -0.6F, 0.3F, 0.5F}
    );
    const Tensor finite_values(
        {1, 1, 3, 2},
        {1.0F, -0.5F, 0.2F, 0.7F, -0.3F, 1.2F}
    );
    const Tensor finite_upstream(
        {1, 1, 3, 2},
        {0.5F, -1.0F, 1.5F, 0.25F, -0.75F, 0.6F}
    );
    const auto finite_forward = flash_causal_attention_forward(
        finite_queries,
        finite_keys,
        finite_values,
        finite_dimensions
    );
    Tensor finite_query_gradient(finite_queries.shape());
    Tensor finite_key_gradient(finite_keys.shape());
    Tensor finite_value_gradient(finite_values.shape());
    backend::dispatch_flash_causal_attention_backward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(finite_queries),
            backend::tensor_storage(finite_keys),
            backend::tensor_storage(finite_values),
            backend::tensor_storage(finite_forward.row_maxima),
            backend::tensor_storage(finite_forward.row_exp_sums),
            backend::tensor_storage(finite_upstream),
            backend::tensor_storage(finite_query_gradient),
            backend::tensor_storage(finite_key_gradient),
            backend::tensor_storage(finite_value_gradient),
            finite_dimensions,
        }
    );
    const auto evaluate = [&](const Tensor& candidate_queries,
                              const Tensor& candidate_keys,
                              const Tensor& candidate_values) {
        return dot(
            flash_causal_attention_forward(
                candidate_queries,
                candidate_keys,
                candidate_values,
                finite_dimensions
            )
                .context,
            finite_upstream
        );
    };
    constexpr float finite_difference_step = 1.0e-3F;
    constexpr float finite_difference_tolerance = 3.0e-3F;
    const auto check_gradient = [&](const Tensor& original,
                                    const Tensor& analytical,
                                    const auto& evaluate_candidate,
                                    const std::string& label) {
        for (std::size_t index = 0;
             index < original.numel();
             ++index) {
            Tensor plus = original;
            Tensor minus = original;
            plus.flat(index) += finite_difference_step;
            minus.flat(index) -= finite_difference_step;
            require_close(
                analytical.flat(index),
                (
                    evaluate_candidate(plus) -
                    evaluate_candidate(minus)
                ) /
                    (2.0F * finite_difference_step),
                label,
                finite_difference_tolerance
            );
        }
    };
    check_gradient(
        finite_queries,
        finite_query_gradient,
        [&](const Tensor& candidate) {
            return evaluate(
                candidate,
                finite_keys,
                finite_values
            );
        },
        "Flash attention query finite difference"
    );
    check_gradient(
        finite_keys,
        finite_key_gradient,
        [&](const Tensor& candidate) {
            return evaluate(
                finite_queries,
                candidate,
                finite_values
            );
        },
        "Flash attention key finite difference"
    );
    check_gradient(
        finite_values,
        finite_value_gradient,
        [&](const Tensor& candidate) {
            return evaluate(
                finite_queries,
                finite_keys,
                candidate
            );
        },
        "Flash attention value finite difference"
    );
}

void test_flash_causal_attention_stability_and_special_values() {
    const backend::FlashCausalAttentionDimensions tiled_dimensions{
        1,
        1,
        9,
        1,
    };
    const Tensor zero_queries({1, 1, 9, 1}, 0.0F);
    const Tensor zero_keys({1, 1, 9, 1}, 0.0F);
    const Tensor prefix_values(
        {1, 1, 9, 1},
        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F}
    );
    const auto uniform = flash_causal_attention_forward(
        zero_queries,
        zero_keys,
        prefix_values,
        tiled_dimensions
    );
    for (std::size_t query_time = 0;
         query_time < tiled_dimensions.time;
         ++query_time) {
        require_close(
            uniform.row_maxima.flat(query_time),
            0.0F,
            "Flash uniform row maximum",
            0.0F
        );
        require_close(
            uniform.row_exp_sums.flat(query_time),
            static_cast<float>(query_time + 1),
            "Flash uniform row exponential sum",
            0.0F
        );
        require_close(
            uniform.context.flat(query_time),
            (
                static_cast<float>(query_time) +
                2.0F
            ) /
                2.0F,
            "Flash uniform causal prefix mean",
            1.0e-6F
        );
    }

    const backend::FlashCausalAttentionDimensions large_dimensions{
        1,
        1,
        3,
        2,
    };
    const backend::MaterializedCausalAttentionDimensions
        large_materialized_dimensions{
            1,
            1,
            3,
            2,
        };
    const Tensor large_queries(
        {1, 1, 3, 2},
        {1000.0F, 0.0F, 1000.0F, 0.0F, 1000.0F, 0.0F}
    );
    const Tensor large_keys(
        {1, 1, 3, 2},
        {1000.0F, 0.0F, -1000.0F, 0.0F, 999.0F, 0.0F}
    );
    const Tensor large_values(
        {1, 1, 3, 2},
        {1.0F, -1.0F, 2.0F, -2.0F, 3.0F, -3.0F}
    );
    const auto large_materialized =
        materialized_causal_attention_forward(
            large_queries,
            large_keys,
            large_values,
            large_materialized_dimensions
        );
    const auto large_flash = flash_causal_attention_forward(
        large_queries,
        large_keys,
        large_values,
        large_dimensions
    );
    require_tensor_close(
        large_flash.context,
        large_materialized.context,
        "Flash large-finite-score stability",
        2.0e-5F
    );

    const backend::FlashCausalAttentionDimensions mixed_dimensions{
        1,
        1,
        2,
        1,
    };
    const backend::MaterializedCausalAttentionDimensions
        mixed_materialized_dimensions{
            1,
            1,
            2,
            1,
        };
    const Tensor mixed_queries({1, 1, 2, 1}, {1.0F, 1.0F});
    const Tensor mixed_keys(
        {1, 1, 2, 1},
        {1.0F, -std::numeric_limits<float>::infinity()}
    );
    const Tensor mixed_values({1, 1, 2, 1}, {0.5F, 9.0F});
    const auto mixed_materialized =
        materialized_causal_attention_forward(
            mixed_queries,
            mixed_keys,
            mixed_values,
            mixed_materialized_dimensions
        );
    const auto mixed_flash = flash_causal_attention_forward(
        mixed_queries,
        mixed_keys,
        mixed_values,
        mixed_dimensions
    );
    require_tensor_close(
        mixed_flash.context,
        mixed_materialized.context,
        "Flash mixed finite and negative-infinite scores",
        0.0F
    );

    const auto check_invalid_score = [&](float query_value,
                                         const std::string& label) {
        const backend::FlashCausalAttentionDimensions dimensions{
            1,
            1,
            1,
            1,
        };
        const auto check_backend =
            [&](ExecutionBackend execution_backend) {
                const Tensor queries(
                    {1, 1, 1, 1},
                    query_value,
                    execution_backend
                );
                const Tensor keys(
                    {1, 1, 1, 1},
                    1.0F,
                    execution_backend
                );
                const Tensor values(
                    {1, 1, 1, 1},
                    1.0F,
                    execution_backend
                );
                require_throws(
                    [&] {
                        static_cast<void>(
                            flash_causal_attention_forward(
                                queries,
                                keys,
                                values,
                                dimensions
                            )
                        );
                    },
                    label + " on " +
                        std::string(
                            riftco_transformer::execution_backend_name(
                                execution_backend
                            )
                        )
                );
            };
        check_backend(ExecutionBackend::Cpu);
        if (riftco_transformer::execution_backend_available(
                ExecutionBackend::Metal
            )) {
            check_backend(ExecutionBackend::Metal);
        }
    };
    check_invalid_score(
        std::numeric_limits<float>::quiet_NaN(),
        "Flash attention should reject NaN scores"
    );
    check_invalid_score(
        std::numeric_limits<float>::infinity(),
        "Flash attention should reject positive-infinite scores"
    );
    check_invalid_score(
        -std::numeric_limits<float>::infinity(),
        "Flash attention should reject all-negative-infinite scores"
    );
}

void test_flash_causal_attention_wide_metal_tiles() {
    if (!riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        return;
    }

    constexpr std::size_t time = 9;
    constexpr std::size_t head_width = 128;
    const backend::FlashCausalAttentionDimensions dimensions{
        1,
        1,
        time,
        head_width,
    };
    const Tensor::Shape shape{1, 1, time, head_width};
    std::vector<float> query_values(time * head_width);
    std::vector<float> key_values(time * head_width);
    std::vector<float> value_values(time * head_width);
    std::vector<float> upstream_values(time * head_width);
    for (std::size_t index = 0; index < query_values.size(); ++index) {
        query_values[index] =
            static_cast<float>(
                static_cast<int>(index % 17) - 8
            ) *
            0.013F;
        key_values[index] =
            static_cast<float>(
                static_cast<int>((index * 3) % 19) - 9
            ) *
            0.011F;
        value_values[index] =
            static_cast<float>(
                static_cast<int>((index * 5) % 23) - 11
            ) *
            0.017F;
        upstream_values[index] =
            static_cast<float>(
                static_cast<int>((index * 7) % 29) - 14
            ) *
            0.009F;
    }

    const Tensor queries(shape, query_values);
    const Tensor keys(shape, key_values);
    const Tensor values(shape, value_values);
    const Tensor upstream(shape, upstream_values);
    const auto cpu = flash_causal_attention_forward(
        queries,
        keys,
        values,
        dimensions
    );
    Tensor cpu_query_gradient(shape);
    Tensor cpu_key_gradient(shape);
    Tensor cpu_value_gradient(shape);
    backend::dispatch_flash_causal_attention_backward(
        ExecutionBackend::Cpu,
        {
            backend::tensor_storage(queries),
            backend::tensor_storage(keys),
            backend::tensor_storage(values),
            backend::tensor_storage(cpu.row_maxima),
            backend::tensor_storage(cpu.row_exp_sums),
            backend::tensor_storage(upstream),
            backend::tensor_storage(cpu_query_gradient),
            backend::tensor_storage(cpu_key_gradient),
            backend::tensor_storage(cpu_value_gradient),
            dimensions,
        }
    );

    const Tensor metal_queries = queries.to(ExecutionBackend::Metal);
    const Tensor metal_keys = keys.to(ExecutionBackend::Metal);
    const Tensor metal_values = values.to(ExecutionBackend::Metal);
    const Tensor metal_upstream = upstream.to(ExecutionBackend::Metal);
    const auto metal = flash_causal_attention_forward(
        metal_queries,
        metal_keys,
        metal_values,
        dimensions
    );
    require_tensor_close(
        metal.row_maxima,
        cpu.row_maxima,
        "wide Flash Metal row maxima",
        2.0e-4F
    );
    require_tensor_close(
        metal.row_exp_sums,
        cpu.row_exp_sums,
        "wide Flash Metal row exponential sums",
        2.0e-4F
    );
    require_tensor_close(
        metal.context,
        cpu.context,
        "wide Flash Metal context",
        2.0e-4F
    );

    Tensor metal_query_gradient(shape, ExecutionBackend::Metal);
    Tensor metal_key_gradient(shape, ExecutionBackend::Metal);
    Tensor metal_value_gradient(shape, ExecutionBackend::Metal);
    backend::dispatch_flash_causal_attention_backward(
        ExecutionBackend::Metal,
        {
            backend::tensor_storage(metal_queries),
            backend::tensor_storage(metal_keys),
            backend::tensor_storage(metal_values),
            backend::tensor_storage(metal.row_maxima),
            backend::tensor_storage(metal.row_exp_sums),
            backend::tensor_storage(metal_upstream),
            backend::tensor_storage(metal_query_gradient),
            backend::tensor_storage(metal_key_gradient),
            backend::tensor_storage(metal_value_gradient),
            dimensions,
        }
    );
    require_tensor_close(
        metal_query_gradient,
        cpu_query_gradient,
        "wide Flash Metal query VJP",
        2.0e-4F
    );
    require_tensor_close(
        metal_key_gradient,
        cpu_key_gradient,
        "wide Flash Metal key VJP",
        2.0e-4F
    );
    require_tensor_close(
        metal_value_gradient,
        cpu_value_gradient,
        "wide Flash Metal value VJP",
        2.0e-4F
    );

    constexpr std::size_t boundary_head_width = 192;
    const backend::FlashCausalAttentionDimensions
        boundary_dimensions{
            1,
            1,
            1,
            boundary_head_width,
        };
    const Tensor::Shape boundary_shape{
        1,
        1,
        1,
        boundary_head_width,
    };
    const Tensor boundary_queries(
        boundary_shape,
        0.01F,
        ExecutionBackend::Metal
    );
    const Tensor boundary_keys(
        boundary_shape,
        0.02F,
        ExecutionBackend::Metal
    );
    const Tensor boundary_values(
        boundary_shape,
        0.03F,
        ExecutionBackend::Metal
    );
    const Tensor boundary_upstream(
        boundary_shape,
        0.04F,
        ExecutionBackend::Metal
    );
    bool forward_completed = false;
    try {
        const auto boundary = flash_causal_attention_forward(
            boundary_queries,
            boundary_keys,
            boundary_values,
            boundary_dimensions
        );
        forward_completed = true;
        Tensor query_gradient(
            boundary_shape,
            ExecutionBackend::Metal
        );
        Tensor key_gradient(
            boundary_shape,
            ExecutionBackend::Metal
        );
        Tensor value_gradient(
            boundary_shape,
            ExecutionBackend::Metal
        );
        backend::dispatch_flash_causal_attention_backward(
            ExecutionBackend::Metal,
            {
                backend::tensor_storage(boundary_queries),
                backend::tensor_storage(boundary_keys),
                backend::tensor_storage(boundary_values),
                backend::tensor_storage(boundary.row_maxima),
                backend::tensor_storage(boundary.row_exp_sums),
                backend::tensor_storage(boundary_upstream),
                backend::tensor_storage(query_gradient),
                backend::tensor_storage(key_gradient),
                backend::tensor_storage(value_gradient),
                boundary_dimensions,
            }
        );
    } catch (const std::runtime_error& error) {
        require(
            !forward_completed,
            "Flash resource limits must be detected before forward returns"
        );
        require(
            std::string(error.what()).find(
                "requires more threadgroup memory"
            ) != std::string::npos,
            "Flash preflight should report the Metal resource limit"
        );
    }
}

void test_paged_decode_attention_reference_and_metal_parity() {
    constexpr std::size_t heads = 2;
    constexpr std::size_t time = 5;
    constexpr std::size_t width = 3;
    constexpr std::size_t block_size = 2;
    constexpr std::size_t physical_blocks = 5;
    const std::vector<std::uint32_t> block_table{3, 0, 4};

    std::vector<float> query_values(heads * time * width);
    std::vector<float> key_values(heads * time * width);
    std::vector<float> value_values(heads * time * width);
    for (std::size_t index = 0;
         index < query_values.size();
         ++index) {
        query_values[index] =
            static_cast<float>(
                static_cast<int>(index % 7) - 3
            ) * 0.17F;
        key_values[index] =
            static_cast<float>(
                static_cast<int>(index % 5) - 2
            ) * 0.23F;
        value_values[index] =
            static_cast<float>(
                static_cast<int>(index % 11) - 5
            ) * 0.11F;
    }

    const backend::MaterializedCausalAttentionDimensions causal_dimensions{
        1,
        heads,
        time,
        width,
    };
    const Tensor full_queries(
        {1, heads, time, width},
        query_values
    );
    const Tensor full_keys(
        {1, heads, time, width},
        key_values
    );
    const Tensor full_values(
        {1, heads, time, width},
        value_values
    );
    const auto causal = materialized_causal_attention_forward(
        full_queries,
        full_keys,
        full_values,
        causal_dimensions
    );

    Tensor query({1, heads, 1, width});
    Tensor key_pages(
        {physical_blocks, heads, block_size, width},
        -777.0F
    );
    Tensor value_pages(
        {physical_blocks, heads, block_size, width},
        777.0F
    );
    Tensor expected({1, heads, 1, width});
    for (std::size_t head = 0; head < heads; ++head) {
        for (std::size_t channel = 0; channel < width; ++channel) {
            const std::size_t final_index =
                (head * time + (time - 1)) * width + channel;
            query.at({0, head, 0, channel}) =
                query_values[final_index];
            expected.at({0, head, 0, channel}) =
                causal.context.at({0, head, time - 1, channel});
        }
        for (std::size_t position = 0;
             position < time;
             ++position) {
            const std::size_t physical =
                block_table[position / block_size];
            const std::size_t offset = position % block_size;
            for (std::size_t channel = 0;
                 channel < width;
                 ++channel) {
                const std::size_t source =
                    (head * time + position) * width + channel;
                key_pages.at({
                    physical,
                    head,
                    offset,
                    channel,
                }) = key_values[source];
                value_pages.at({
                    physical,
                    head,
                    offset,
                    channel,
                }) = value_values[source];
            }
        }
    }

    const backend::PagedDecodeAttentionDimensions paged_dimensions{
        heads,
        width,
        block_size,
        physical_blocks,
        time,
    };
    const Tensor cpu_context = paged_decode_attention_forward(
        query,
        key_pages,
        value_pages,
        block_table,
        paged_dimensions
    );
    require_tensor_close(
        cpu_context,
        expected,
        "paged decode attention CPU reference",
        2.0e-6F
    );

    if (riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        const Tensor metal_context = paged_decode_attention_forward(
            query.to(ExecutionBackend::Metal),
            key_pages.to(ExecutionBackend::Metal),
            value_pages.to(ExecutionBackend::Metal),
            block_table,
            paged_dimensions
        );
        require_tensor_close(
            metal_context,
            expected,
            "paged decode attention Metal parity"
        );
    }
}

void test_paged_decode_attention_contract_and_special_values() {
    Tensor query({1, 1, 1, 1}, 1.0F);
    Tensor keys({2, 1, 2, 1}, {1.0F, 2.0F, 3.0F, 4.0F});
    Tensor values({2, 1, 2, 1}, {0.5F, 1.0F, 1.5F, 2.0F});
    Tensor context({1, 1, 1, 1});
    const std::vector<std::uint32_t> one_block{0};
    const backend::PagedDecodeAttentionDimensions dimensions{
        1,
        1,
        2,
        2,
        1,
    };
    const auto dispatch = [&](
        const Tensor& candidate_query,
        const Tensor& candidate_keys,
        const Tensor& candidate_values,
        std::span<const std::uint32_t> table,
        Tensor& candidate_context,
        backend::PagedDecodeAttentionDimensions candidate_dimensions
    ) {
        backend::dispatch_paged_decode_attention_forward(
            candidate_query.backend(),
            {
                backend::tensor_storage(candidate_query),
                backend::tensor_storage(candidate_keys),
                backend::tensor_storage(candidate_values),
                table,
                backend::tensor_storage(candidate_context),
                candidate_dimensions,
            }
        );
    };

    require_throws(
        [&] {
            auto invalid = dimensions;
            invalid.heads = 0;
            dispatch(query, keys, values, one_block, context, invalid);
        },
        "paged decode attention should reject zero dimensions"
    );
    require_throws(
        [&] {
            dispatch(
                query,
                keys,
                values,
                std::span<const std::uint32_t>{},
                context,
                dimensions
            );
        },
        "paged decode attention should reject a short block table"
    );
    const std::vector<std::uint32_t> invalid_block{2};
    require_throws(
        [&] {
            dispatch(
                query,
                keys,
                values,
                invalid_block,
                context,
                dimensions
            );
        },
        "paged decode attention should reject an out-of-range block"
    );
    Tensor undersized_keys({1, 1, 2, 1});
    require_throws(
        [&] {
            dispatch(
                query,
                undersized_keys,
                values,
                one_block,
                context,
                dimensions
            );
        },
        "paged decode attention should enforce physical-pool storage size"
    );
    require_throws(
        [&] {
            backend::dispatch_paged_decode_attention_forward(
                ExecutionBackend::Cpu,
                {
                    backend::tensor_storage(query),
                    backend::tensor_storage(keys),
                    backend::tensor_storage(values),
                    one_block,
                    backend::tensor_storage(query),
                    dimensions,
                }
            );
        },
        "paged decode attention should reject output aliasing"
    );
    if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
        require_throws(
            [&] {
                auto invalid = dimensions;
                invalid.physical_block_count =
                    static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max()
                    ) +
                    2;
                dispatch(
                    query,
                    keys,
                    values,
                    one_block,
                    context,
                    invalid
                );
            },
            "paged decode attention should bound uint32 physical block IDs"
        );
    }

    const auto verify_special_values =
        [&](ExecutionBackend execution_backend) {
            const auto check = [&](float query_value,
                                   const std::string& label) {
                Tensor special_query(
                    {1, 1, 1, 1},
                    query_value,
                    execution_backend
                );
                Tensor special_keys(
                    {1, 1, 1, 1},
                    1.0F,
                    execution_backend
                );
                Tensor special_values(
                    {1, 1, 1, 1},
                    1.0F,
                    execution_backend
                );
                Tensor special_context(
                    {1, 1, 1, 1},
                    execution_backend
                );
                const backend::PagedDecodeAttentionDimensions single{
                    1,
                    1,
                    1,
                    1,
                    1,
                };
                require_throws(
                    [&] {
                        dispatch(
                            special_query,
                            special_keys,
                            special_values,
                            one_block,
                            special_context,
                            single
                        );
                    },
                    label
                );
            };
            check(
                std::numeric_limits<float>::quiet_NaN(),
                "paged decode attention should reject NaN scores"
            );
            check(
                std::numeric_limits<float>::infinity(),
                "paged decode attention should reject positive-infinite scores"
            );
            check(
                -std::numeric_limits<float>::infinity(),
                "paged decode attention should reject all-negative-infinite scores"
            );
        };

    verify_special_values(ExecutionBackend::Cpu);
    if (riftco_transformer::execution_backend_available(
            ExecutionBackend::Metal
        )) {
        verify_special_values(ExecutionBackend::Metal);
    }
}

} // namespace

int main() {
    try {
#if RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL
        require(
            riftco_transformer::execution_backend_available(
                ExecutionBackend::Metal
            ),
            "Metal is required for this NN backend test, but no Metal device "
            "is available"
        );
#endif
        test_layer_norm_reference();
        test_cross_entropy_reference();
        test_extreme_loss_and_special_value_backend_parity();
        test_metal_tensor_neural_substrate_parity();
        test_metal_layout_and_scatter_scalability();
        test_attention_reference_and_gradients();
        test_flash_causal_attention_reference_and_gradients();
        test_flash_causal_attention_stability_and_special_values();
        test_flash_causal_attention_wide_metal_tiles();
        test_paged_decode_attention_reference_and_metal_parity();
        test_paged_decode_attention_contract_and_special_values();
        std::cout << "NN backend tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test failure: " << error.what() << '\n';
        return 1;
    }
}
