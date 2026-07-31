#include "paged_decode.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace transformer_lab::backend_detail {
namespace {

std::size_t paged_attention_offset(
    const PagedDecodeAttentionDimensions& dimensions,
    std::span<const std::uint32_t> block_table,
    std::size_t head,
    std::size_t position,
    std::size_t channel
) {
    const std::size_t logical_block =
        position / dimensions.block_size;
    const std::size_t block_offset =
        position % dimensions.block_size;
    const std::size_t physical_block =
        static_cast<std::size_t>(block_table[logical_block]);
    return (
        (
            physical_block * dimensions.heads + head
        ) * dimensions.block_size + block_offset
    ) * dimensions.head_width + channel;
}

}  // namespace

void reference_paged_decode_attention_forward(
    const PagedDecodeAttentionForwardRequest& request
) {
    const auto queries = request.queries.data();
    const auto keys = request.key_pages.data();
    const auto values = request.value_pages.data();
    auto context = request.context.data();
    std::fill(context.begin(), context.end(), 0.0F);

    const auto& dimensions = request.dimensions;
    const float scale =
        1.0F /
        std::sqrt(static_cast<float>(dimensions.head_width));
    std::vector<float> probabilities(
        dimensions.sequence_length,
        0.0F
    );

    for (std::size_t head = 0;
         head < dimensions.heads;
         ++head) {
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t position = 0;
             position < dimensions.sequence_length;
             ++position) {
            float score = 0.0F;
            for (std::size_t channel = 0;
                 channel < dimensions.head_width;
                 ++channel) {
                score +=
                    queries[head * dimensions.head_width + channel] *
                    keys[paged_attention_offset(
                        dimensions,
                        request.block_table,
                        head,
                        position,
                        channel
                    )];
            }
            score *= scale;
            if (std::isnan(score) ||
                score == std::numeric_limits<float>::infinity()) {
                throw std::domain_error(
                    "softmax rejects NaN and positive infinity"
                );
            }
            probabilities[position] = score;
            maximum = std::max(maximum, score);
        }
        if (maximum ==
            -std::numeric_limits<float>::infinity()) {
            throw std::domain_error(
                "softmax requires one finite value per slice"
            );
        }

        double denominator = 0.0;
        for (std::size_t position = 0;
             position < dimensions.sequence_length;
             ++position) {
            const double exponential = std::exp(
                static_cast<double>(
                    probabilities[position] - maximum
                )
            );
            probabilities[position] =
                static_cast<float>(exponential);
            denominator += exponential;
        }
        for (float& probability : probabilities) {
            probability = static_cast<float>(
                static_cast<double>(probability) / denominator
            );
        }

        for (std::size_t channel = 0;
             channel < dimensions.head_width;
             ++channel) {
            float total = 0.0F;
            for (std::size_t position = 0;
                 position < dimensions.sequence_length;
                 ++position) {
                total +=
                    probabilities[position] *
                    values[paged_attention_offset(
                        dimensions,
                        request.block_table,
                        head,
                        position,
                        channel
                    )];
            }
            context[head * dimensions.head_width + channel] = total;
        }
    }
}

}  // namespace transformer_lab::backend_detail
