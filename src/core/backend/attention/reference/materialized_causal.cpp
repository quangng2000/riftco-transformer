#include "materialized_causal.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

std::size_t tensor_offset(
    const MaterializedCausalAttentionDimensions& dimensions,
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

std::size_t probability_offset(
    const MaterializedCausalAttentionDimensions& dimensions,
    std::size_t batch,
    std::size_t head,
    std::size_t query_time,
    std::size_t key_time
) {
    return (
        (
            batch * dimensions.heads + head
        ) * dimensions.time + query_time
    ) * dimensions.time + key_time;
}

float attention_scale(
    const MaterializedCausalAttentionDimensions& dimensions
) {
    return 1.0F /
           std::sqrt(static_cast<float>(dimensions.head_width));
}

}  // namespace

void reference_materialized_causal_attention_forward(
    const MaterializedCausalAttentionForwardRequest& request
) {
    const auto queries = request.queries.data();
    const auto keys = request.keys.data();
    const auto values = request.values.data();
    auto probabilities = request.probabilities.data();
    auto context = request.context.data();
    std::fill(probabilities.begin(), probabilities.end(), 0.0F);
    std::fill(context.begin(), context.end(), 0.0F);

    const auto& dimensions = request.dimensions;
    const float scale = attention_scale(dimensions);
    std::vector<float> scores(dimensions.time, 0.0F);
    for (std::size_t batch = 0;
         batch < dimensions.batch;
         ++batch) {
        for (std::size_t head = 0;
             head < dimensions.heads;
             ++head) {
            for (std::size_t query_time = 0;
                 query_time < dimensions.time;
                 ++query_time) {
                float maximum =
                    -std::numeric_limits<float>::infinity();
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
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
                    if (std::isnan(score) ||
                        score ==
                            std::numeric_limits<float>::infinity()) {
                        throw std::domain_error(
                            "softmax rejects NaN and positive infinity"
                        );
                    }
                    scores[key_time] = score;
                    maximum = std::max(maximum, score);
                }
                if (maximum ==
                    -std::numeric_limits<float>::infinity()) {
                    throw std::domain_error(
                        "softmax requires one finite value per slice"
                    );
                }

                double denominator = 0.0;
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    const double exponential = std::exp(
                        static_cast<double>(
                            scores[key_time] - maximum
                        )
                    );
                    const auto probability_index =
                        probability_offset(
                            dimensions,
                            batch,
                            head,
                            query_time,
                            key_time
                        );
                    probabilities[probability_index] =
                        static_cast<float>(exponential);
                    denominator += exponential;
                }
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    const auto probability_index =
                        probability_offset(
                            dimensions,
                            batch,
                            head,
                            query_time,
                            key_time
                        );
                    probabilities[probability_index] =
                        static_cast<float>(
                            static_cast<double>(
                                probabilities[probability_index]
                            ) / denominator
                        );
                }

                for (std::size_t channel = 0;
                     channel < dimensions.head_width;
                     ++channel) {
                    float total = 0.0F;
                    for (std::size_t key_time = 0;
                         key_time <= query_time;
                         ++key_time) {
                        total +=
                            probabilities[probability_offset(
                                dimensions,
                                batch,
                                head,
                                query_time,
                                key_time
                            )] *
                            values[tensor_offset(
                                dimensions,
                                batch,
                                head,
                                key_time,
                                channel
                            )];
                    }
                    context[tensor_offset(
                        dimensions,
                        batch,
                        head,
                        query_time,
                        channel
                    )] = total;
                }
            }
        }
    }
}

void reference_materialized_causal_attention_context_backward(
    const MaterializedCausalAttentionContextBackwardRequest& request
) {
    const auto queries = request.queries.data();
    const auto keys = request.keys.data();
    const auto values = request.values.data();
    const auto probabilities = request.probabilities.data();
    const auto upstream = request.upstream_context.data();
    auto query_gradient = request.query_gradient.data();
    auto key_gradient = request.key_gradient.data();
    auto value_gradient = request.value_gradient.data();
    std::fill(query_gradient.begin(), query_gradient.end(), 0.0F);
    std::fill(key_gradient.begin(), key_gradient.end(), 0.0F);
    std::fill(value_gradient.begin(), value_gradient.end(), 0.0F);

    const auto& dimensions = request.dimensions;
    const float scale = attention_scale(dimensions);
    std::vector<float> probability_gradient(dimensions.time, 0.0F);
    std::vector<float> score_gradient(dimensions.time, 0.0F);

    for (std::size_t batch = 0;
         batch < dimensions.batch;
         ++batch) {
        for (std::size_t head = 0;
             head < dimensions.heads;
             ++head) {
            for (std::size_t query_time = 0;
                 query_time < dimensions.time;
                 ++query_time) {
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    float total = 0.0F;
                    for (std::size_t channel = 0;
                         channel < dimensions.head_width;
                         ++channel) {
                        total +=
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
                    probability_gradient[key_time] = total;
                }

                double weighted_gradient = 0.0;
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    weighted_gradient +=
                        static_cast<double>(
                            probabilities[probability_offset(
                                dimensions,
                                batch,
                                head,
                                query_time,
                                key_time
                            )]
                        ) *
                        static_cast<double>(
                            probability_gradient[key_time]
                        );
                }
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    score_gradient[key_time] =
                        probabilities[probability_offset(
                            dimensions,
                            batch,
                            head,
                            query_time,
                            key_time
                        )] *
                        (
                            probability_gradient[key_time] -
                            static_cast<float>(weighted_gradient)
                        ) *
                        scale;
                }

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
                    float query_total = 0.0F;
                    for (std::size_t key_time = 0;
                         key_time <= query_time;
                         ++key_time) {
                        const auto key_index = tensor_offset(
                            dimensions,
                            batch,
                            head,
                            key_time,
                            channel
                        );
                        const float score_contribution =
                            score_gradient[key_time];
                        query_total +=
                            score_contribution * keys[key_index];
                        key_gradient[key_index] +=
                            score_contribution * queries[query_index];
                        value_gradient[key_index] +=
                            probabilities[probability_offset(
                                dimensions,
                                batch,
                                head,
                                query_time,
                                key_time
                            )] * upstream[query_index];
                    }
                    query_gradient[query_index] = query_total;
                }
            }
        }
    }
}

void reference_materialized_causal_attention_probabilities_backward(
    const MaterializedCausalAttentionProbabilitiesBackwardRequest& request
) {
    const auto queries = request.queries.data();
    const auto keys = request.keys.data();
    const auto probabilities = request.probabilities.data();
    const auto upstream = request.upstream_probabilities.data();
    auto query_gradient = request.query_gradient.data();
    auto key_gradient = request.key_gradient.data();
    std::fill(query_gradient.begin(), query_gradient.end(), 0.0F);
    std::fill(key_gradient.begin(), key_gradient.end(), 0.0F);

    const auto& dimensions = request.dimensions;
    const float scale = attention_scale(dimensions);
    std::vector<float> score_gradient(dimensions.time, 0.0F);

    for (std::size_t batch = 0;
         batch < dimensions.batch;
         ++batch) {
        for (std::size_t head = 0;
             head < dimensions.heads;
             ++head) {
            for (std::size_t query_time = 0;
                 query_time < dimensions.time;
                 ++query_time) {
                double weighted_gradient = 0.0;
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    const auto probability_index =
                        probability_offset(
                            dimensions,
                            batch,
                            head,
                            query_time,
                            key_time
                        );
                    weighted_gradient +=
                        static_cast<double>(
                            probabilities[probability_index]
                        ) *
                        static_cast<double>(
                            upstream[probability_index]
                        );
                }
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    const auto probability_index =
                        probability_offset(
                            dimensions,
                            batch,
                            head,
                            query_time,
                            key_time
                        );
                    score_gradient[key_time] =
                        probabilities[probability_index] *
                        (
                            upstream[probability_index] -
                            static_cast<float>(weighted_gradient)
                        ) *
                        scale;
                }

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
                    float query_total = 0.0F;
                    for (std::size_t key_time = 0;
                         key_time <= query_time;
                         ++key_time) {
                        const auto key_index = tensor_offset(
                            dimensions,
                            batch,
                            head,
                            key_time,
                            channel
                        );
                        query_total +=
                            score_gradient[key_time] * keys[key_index];
                        key_gradient[key_index] +=
                            score_gradient[key_time] *
                            queries[query_index];
                    }
                    query_gradient[query_index] = query_total;
                }
            }
        }
    }
}

}  // namespace riftco_transformer::backend_detail
