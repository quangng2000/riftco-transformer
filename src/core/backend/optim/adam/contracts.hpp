#pragma once

#include "core/backend/storage.hpp"

#include <span>

namespace riftco_transformer::backend_detail {

// One out-of-place Adam update. Inputs describe the live optimizer state;
// outputs are candidates that the public optimizer commits only after the
// complete batch succeeds.
struct AdamTensorUpdate {
    const TensorStorage& value;
    const TensorStorage& gradient;
    const TensorStorage& first_moment;
    const TensorStorage& second_moment;
    TensorStorage& next_value;
    TensorStorage& next_first_moment;
    TensorStorage& next_second_moment;
};

struct AdamUpdateRequest {
    std::span<const AdamTensorUpdate> tensors;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    double clip_scale;
    double first_correction;
    double second_correction;
};

} // namespace riftco_transformer::backend_detail
