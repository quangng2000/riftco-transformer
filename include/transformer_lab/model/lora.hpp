#pragma once

#include <cstddef>
#include <cstdint>

namespace transformer_lab {

using LoraTargetMask = std::uint64_t;

inline constexpr LoraTargetMask kLoraAttentionQuery =
    LoraTargetMask{1} << 0U;
inline constexpr LoraTargetMask kLoraAttentionKey =
    LoraTargetMask{1} << 1U;
inline constexpr LoraTargetMask kLoraAttentionValue =
    LoraTargetMask{1} << 2U;
inline constexpr LoraTargetMask kLoraAttentionOutput =
    LoraTargetMask{1} << 3U;
inline constexpr LoraTargetMask kLoraFeedForwardExpand =
    LoraTargetMask{1} << 4U;
inline constexpr LoraTargetMask kLoraFeedForwardProject =
    LoraTargetMask{1} << 5U;
inline constexpr LoraTargetMask kLoraLanguageModelHead =
    LoraTargetMask{1} << 6U;

inline constexpr LoraTargetMask kLoraDefaultTargets =
    kLoraAttentionQuery | kLoraAttentionValue;
inline constexpr LoraTargetMask kLoraAllTargets =
    kLoraAttentionQuery |
    kLoraAttentionKey |
    kLoraAttentionValue |
    kLoraAttentionOutput |
    kLoraFeedForwardExpand |
    kLoraFeedForwardProject |
    kLoraLanguageModelHead;

struct LoraConfig {
    std::size_t rank = 4;
    float alpha = 8.0F;
    std::uint32_t random_seed = 5489U;
    LoraTargetMask targets = kLoraDefaultTargets;

    [[nodiscard]] bool operator==(const LoraConfig&) const = default;
};

}  // namespace transformer_lab
