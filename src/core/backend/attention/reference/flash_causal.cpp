#include "flash_causal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace transformer_lab::backend_detail {
namespace {

constexpr std::size_t kTileSize = 8;

std::size_t tensor_offset(
    const FlashCausalAttentionDimensions& dimensions,
    std::size_t batch,
    std::size_t head,
    std::size_t time,
    std::size_t channel
) {
    return (
        (
            batch * dimensions.heads + head
        ) * dimensions.time + time
    ) * dimensions.head_width + channel;
}

std::size_t row_offset(
    const FlashCausalAttentionDimensions& dimensions,
    std::size_t batch,
    std::size_t head,
    std::size_t time
) {
    return (
        batch * dimensions.heads + head
    ) * dimensions.time + time;
}

float attention_scale(
    const FlashCausalAttentionDimensions& dimensions
) {
    return 1.0F /
           std::sqrt(static_cast<float>(dimensions.head_width));
}

float attention_score(
    const FlashCausalAttentionDimensions& dimensions,
    std::span<const float> queries,
    std::span<const float> keys,
    std::size_t batch,
    std::size_t head,
    std::size_t query_time,
    std::size_t key_time,
    float scale
) {
    float score = 0.0F;
    for (std::size_t channel = 0;
         channel < dimensions.head_width;
         ++channel) {
        score +=
            queries[tensor_offset(
                dimensions,
                batch,
                head,
                query_time,
                channel
            )] *
            keys[tensor_offset(
                dimensions,
                batch,
                head,
                key_time,
                channel
            )];
    }
    score *= scale;
    if (
        std::isnan(score) ||
        score == std::numeric_limits<float>::infinity()
    ) {
        throw std::domain_error(
            "softmax rejects NaN and positive infinity"
        );
    }
    return score;
}

float reconstructed_probability(
    float score,
    float row_maximum,
    float row_exp_sum
) {
    return static_cast<float>(
        std::exp(
            static_cast<double>(score - row_maximum)
        ) /
        static_cast<double>(row_exp_sum)
    );
}

}  // namespace

void reference_flash_causal_attention_forward(
    const FlashCausalAttentionForwardRequest& request
) {
    const auto queries = request.queries.data();
    const auto keys = request.keys.data();
    const auto values = request.values.data();
    auto row_maxima = request.row_maxima.data();
    auto row_exp_sums = request.row_exp_sums.data();
    auto context = request.context.data();
    std::fill(
        row_maxima.begin(),
        row_maxima.end(),
        -std::numeric_limits<float>::infinity()
    );
    std::fill(row_exp_sums.begin(), row_exp_sums.end(), 0.0F);
    std::fill(context.begin(), context.end(), 0.0F);

    const auto& dimensions = request.dimensions;
    const float scale = attention_scale(dimensions);
    std::array<float, kTileSize> scores{};
    std::vector<double> context_numerator(
        dimensions.head_width,
        0.0
    );

    for (std::size_t batch = 0;
         batch < dimensions.batch;
         ++batch) {
        for (std::size_t head = 0;
             head < dimensions.heads;
             ++head) {
            for (std::size_t query_tile = 0;
                 query_tile < dimensions.time;
                 query_tile += kTileSize) {
                const std::size_t query_end = std::min(
                    dimensions.time,
                    query_tile + kTileSize
                );
                for (std::size_t query_time = query_tile;
                     query_time < query_end;
                     ++query_time) {
                    float running_maximum =
                        -std::numeric_limits<float>::infinity();
                    double running_exp_sum = 0.0;
                    std::fill(
                        context_numerator.begin(),
                        context_numerator.end(),
                        0.0
                    );

                    for (std::size_t key_tile = 0;
                         key_tile <= query_time;
                         key_tile += kTileSize) {
                        const std::size_t key_end = std::min(
                            query_time + 1,
                            key_tile + kTileSize
                        );
                        const std::size_t key_count =
                            key_end - key_tile;
                        float tile_maximum =
                            -std::numeric_limits<float>::infinity();
                        for (std::size_t key_index = 0;
                             key_index < key_count;
                             ++key_index) {
                            const float score = attention_score(
                                dimensions,
                                queries,
                                keys,
                                batch,
                                head,
                                query_time,
                                key_tile + key_index,
                                scale
                            );
                            scores[key_index] = score;
                            tile_maximum = std::max(
                                tile_maximum,
                                score
                            );
                        }

                        // A tile containing only -infinity has zero
                        // probability mass. Skipping it also avoids the
                        // undefined subtraction -infinity - -infinity.
                        if (
                            tile_maximum ==
                            -std::numeric_limits<float>::infinity()
                        ) {
                            continue;
                        }

                        const float updated_maximum = std::max(
                            running_maximum,
                            tile_maximum
                        );
                        const double previous_scale =
                            running_maximum ==
                                    -std::numeric_limits<float>::infinity()
                                ? 0.0
                                : std::exp(
                                      static_cast<double>(
                                          running_maximum -
                                          updated_maximum
                                      )
                                  );
                        running_exp_sum *= previous_scale;
                        for (double& element : context_numerator) {
                            element *= previous_scale;
                        }

                        for (std::size_t key_index = 0;
                             key_index < key_count;
                             ++key_index) {
                            const std::size_t key_time =
                                key_tile + key_index;
                            const double weight = std::exp(
                                static_cast<double>(
                                    scores[key_index] -
                                    updated_maximum
                                )
                            );
                            running_exp_sum += weight;
                            for (std::size_t channel = 0;
                                 channel < dimensions.head_width;
                                 ++channel) {
                                context_numerator[channel] +=
                                    weight *
                                    static_cast<double>(
                                        values[tensor_offset(
                                            dimensions,
                                            batch,
                                            head,
                                            key_time,
                                            channel
                                        )]
                                    );
                            }
                        }
                        running_maximum = updated_maximum;
                    }

                    if (
                        running_maximum ==
                        -std::numeric_limits<float>::infinity()
                    ) {
                        throw std::domain_error(
                            "softmax requires one finite value per slice"
                        );
                    }
                    if (
                        !(running_exp_sum > 0.0) ||
                        !std::isfinite(running_exp_sum)
                    ) {
                        throw std::domain_error(
                            "softmax requires a finite exponential sum"
                        );
                    }

                    const auto output_row = row_offset(
                        dimensions,
                        batch,
                        head,
                        query_time
                    );
                    const float saved_exp_sum =
                        static_cast<float>(running_exp_sum);
                    if (
                        !(saved_exp_sum > 0.0F) ||
                        !std::isfinite(saved_exp_sum)
                    ) {
                        throw std::domain_error(
                            "softmax requires a finite exponential sum"
                        );
                    }
                    row_maxima[output_row] = running_maximum;
                    row_exp_sums[output_row] = saved_exp_sum;
                    for (std::size_t channel = 0;
                         channel < dimensions.head_width;
                         ++channel) {
                        context[tensor_offset(
                            dimensions,
                            batch,
                            head,
                            query_time,
                            channel
                        )] = static_cast<float>(
                            context_numerator[channel] /
                            static_cast<double>(saved_exp_sum)
                        );
                    }
                }
            }
        }
    }
}

void reference_flash_causal_attention_backward(
    const FlashCausalAttentionBackwardRequest& request
) {
    const auto queries = request.queries.data();
    const auto keys = request.keys.data();
    const auto values = request.values.data();
    const auto row_maxima = request.row_maxima.data();
    const auto row_exp_sums = request.row_exp_sums.data();
    const auto upstream = request.upstream_context.data();
    auto query_gradient = request.query_gradient.data();
    auto key_gradient = request.key_gradient.data();
    auto value_gradient = request.value_gradient.data();
    std::fill(query_gradient.begin(), query_gradient.end(), 0.0F);
    std::fill(key_gradient.begin(), key_gradient.end(), 0.0F);
    std::fill(value_gradient.begin(), value_gradient.end(), 0.0F);

    const auto& dimensions = request.dimensions;
    const float scale = attention_scale(dimensions);
    for (std::size_t batch = 0;
         batch < dimensions.batch;
         ++batch) {
        for (std::size_t head = 0;
             head < dimensions.heads;
             ++head) {
            for (std::size_t query_tile = 0;
                 query_tile < dimensions.time;
                 query_tile += kTileSize) {
                const std::size_t query_end = std::min(
                    dimensions.time,
                    query_tile + kTileSize
                );
                for (std::size_t query_time = query_tile;
                     query_time < query_end;
                     ++query_time) {
                    const auto statistics_index = row_offset(
                        dimensions,
                        batch,
                        head,
                        query_time
                    );
                    const float row_maximum =
                        row_maxima[statistics_index];
                    const float row_exp_sum =
                        row_exp_sums[statistics_index];

                    // First pass computes the row-wise softmax Jacobian
                    // contraction D = sum_j P_j * (dO dot V_j).
                    double weighted_gradient = 0.0;
                    for (std::size_t key_tile = 0;
                         key_tile <= query_time;
                         key_tile += kTileSize) {
                        const std::size_t key_end = std::min(
                            query_time + 1,
                            key_tile + kTileSize
                        );
                        for (std::size_t key_time = key_tile;
                             key_time < key_end;
                             ++key_time) {
                            const float score = attention_score(
                                dimensions,
                                queries,
                                keys,
                                batch,
                                head,
                                query_time,
                                key_time,
                                scale
                            );
                            const float probability =
                                reconstructed_probability(
                                    score,
                                    row_maximum,
                                    row_exp_sum
                                );
                            float probability_gradient = 0.0F;
                            for (std::size_t channel = 0;
                                 channel < dimensions.head_width;
                                 ++channel) {
                                probability_gradient +=
                                    upstream[tensor_offset(
                                        dimensions,
                                        batch,
                                        head,
                                        query_time,
                                        channel
                                    )] *
                                    values[tensor_offset(
                                        dimensions,
                                        batch,
                                        head,
                                        key_time,
                                        channel
                                    )];
                            }
                            weighted_gradient +=
                                static_cast<double>(probability) *
                                static_cast<double>(
                                    probability_gradient
                                );
                        }
                    }

                    // The second pass rematerializes one probability tile at
                    // a time and writes only the linear-size Q/K/V gradients.
                    for (std::size_t key_tile = 0;
                         key_tile <= query_time;
                         key_tile += kTileSize) {
                        const std::size_t key_end = std::min(
                            query_time + 1,
                            key_tile + kTileSize
                        );
                        for (std::size_t key_time = key_tile;
                             key_time < key_end;
                             ++key_time) {
                            const float score = attention_score(
                                dimensions,
                                queries,
                                keys,
                                batch,
                                head,
                                query_time,
                                key_time,
                                scale
                            );
                            const float probability =
                                reconstructed_probability(
                                    score,
                                    row_maximum,
                                    row_exp_sum
                                );
                            float probability_gradient = 0.0F;
                            for (std::size_t channel = 0;
                                 channel < dimensions.head_width;
                                 ++channel) {
                                probability_gradient +=
                                    upstream[tensor_offset(
                                        dimensions,
                                        batch,
                                        head,
                                        query_time,
                                        channel
                                    )] *
                                    values[tensor_offset(
                                        dimensions,
                                        batch,
                                        head,
                                        key_time,
                                        channel
                                    )];
                            }
                            const float score_gradient =
                                probability *
                                (
                                    probability_gradient -
                                    static_cast<float>(
                                        weighted_gradient
                                    )
                                ) *
                                scale;

                            for (std::size_t channel = 0;
                                 channel < dimensions.head_width;
                                 ++channel) {
                                const auto query_index = tensor_offset(
                                    dimensions,
                                    batch,
                                    head,
                                    query_time,
                                    channel
                                );
                                const auto key_index = tensor_offset(
                                    dimensions,
                                    batch,
                                    head,
                                    key_time,
                                    channel
                                );
                                query_gradient[query_index] +=
                                    score_gradient * keys[key_index];
                                key_gradient[key_index] +=
                                    score_gradient *
                                    queries[query_index];
                                value_gradient[key_index] +=
                                    probability *
                                    upstream[query_index];
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace transformer_lab::backend_detail
