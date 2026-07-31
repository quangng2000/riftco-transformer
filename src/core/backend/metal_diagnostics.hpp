#pragma once

#include <cstdint>

namespace riftco_transformer::backend_detail {

// Private test/diagnostic counters. They count successful Adam batches, not
// individual parameter-tensor dispatches.
struct MetalAdamPathCounts {
    std::uint64_t fused_batches;
    std::uint64_t reference_batches;
};

void reset_metal_adam_path_counts() noexcept;

[[nodiscard]] MetalAdamPathCounts metal_adam_path_counts() noexcept;

}  // namespace riftco_transformer::backend_detail
