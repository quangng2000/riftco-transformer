#pragma once

namespace riftco_transformer::backend_detail::adam_metal_detail {

inline constexpr char kAdamKernelSource[] = R"METAL(
#include <metal_stdlib>
using namespace metal;

inline bool riftco_transformer_is_subnormal(float value) {
    const uint magnitude =
        as_type<uint>(value) & 0x7fffffffu;
    return magnitude != 0u && magnitude < 0x00800000u;
}

inline bool riftco_transformer_unsafe_input(float value) {
    return !isfinite(value) ||
           riftco_transformer_is_subnormal(value);
}

inline bool riftco_transformer_unsafe_result(
    float value,
    bool exact_nonzero_expected
) {
    return !isfinite(value) ||
           riftco_transformer_is_subnormal(value) ||
           (value == 0.0f && exact_nonzero_expected);
}

inline bool riftco_transformer_ill_conditioned_sum(
    float left,
    float right,
    float result
) {
    if (left == 0.0f ||
        right == 0.0f ||
        (left < 0.0f) == (right < 0.0f)) {
        return false;
    }
    // A cancellation condition above roughly 32 magnifies ordinary float
    // rounding enough that the wide reference is the safer contract.
    const float largest = max(fabs(left), fabs(right));
    return fabs(result) < largest * 0.03125f;
}

inline void riftco_transformer_request_reference(
    device atomic_uint* reference_required
) {
    atomic_store_explicit(
        reference_required,
        1u,
        memory_order_relaxed
    );
}

kernel void riftco_transformer_adam_update(
    device const float* value [[buffer(0)]],
    device const float* gradient [[buffer(1)]],
    device const float* first_moment [[buffer(2)]],
    device const float* second_moment [[buffer(3)]],
    device float* next_value [[buffer(4)]],
    device float* next_first_moment [[buffer(5)]],
    device float* next_second_moment [[buffer(6)]],
    device atomic_uint* reference_required [[buffer(7)]],
    constant ulong& element_count [[buffer(8)]],
    constant float& learning_rate [[buffer(9)]],
    constant float& beta1 [[buffer(10)]],
    constant float& beta2 [[buffer(11)]],
    constant float& epsilon [[buffer(12)]],
    constant float& clip_mantissa [[buffer(13)]],
    constant int& clip_exponent [[buffer(14)]],
    constant float& first_correction [[buffer(15)]],
    constant float& second_correction [[buffer(16)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= element_count) {
        return;
    }

    const float parameter_value = value[index];
    const float raw_gradient = gradient[index];
    const float old_first = first_moment[index];
    const float old_second = second_moment[index];
    if (riftco_transformer_unsafe_input(parameter_value) ||
        riftco_transformer_unsafe_input(raw_gradient) ||
        riftco_transformer_unsafe_input(old_first) ||
        riftco_transformer_unsafe_input(old_second) ||
        old_second < 0.0f ||
        riftco_transformer_unsafe_input(learning_rate) ||
        riftco_transformer_unsafe_input(beta1) ||
        riftco_transformer_unsafe_input(beta2) ||
        riftco_transformer_unsafe_input(epsilon) ||
        riftco_transformer_unsafe_input(clip_mantissa) ||
        riftco_transformer_unsafe_input(first_correction) ||
        riftco_transformer_unsafe_input(second_correction)) {
        riftco_transformer_request_reference(reference_required);
        return;
    }

    float clipped_gradient = raw_gradient;
    if (clip_mantissa != 1.0f || clip_exponent != 0) {
        const float preclip = raw_gradient * clip_mantissa;
        if (riftco_transformer_unsafe_result(
                preclip,
                raw_gradient != 0.0f && clip_mantissa != 0.0f
            )) {
            riftco_transformer_request_reference(reference_required);
            return;
        }
        clipped_gradient = ldexp(preclip, clip_exponent);
        if (riftco_transformer_unsafe_result(
                clipped_gradient,
                preclip != 0.0f
            )) {
            riftco_transformer_request_reference(reference_required);
            return;
        }
    }

    const float one_minus_beta1 = 1.0f - beta1;
    const float retained_first = beta1 * old_first;
    const float gradient_first =
        one_minus_beta1 * clipped_gradient;
    if (riftco_transformer_unsafe_result(
            retained_first,
            beta1 != 0.0f && old_first != 0.0f
        ) ||
        riftco_transformer_unsafe_result(
            gradient_first,
            one_minus_beta1 != 0.0f &&
                clipped_gradient != 0.0f
        )) {
        riftco_transformer_request_reference(reference_required);
        return;
    }
    const float first = retained_first + gradient_first;
    if (riftco_transformer_unsafe_result(
            first,
            retained_first != 0.0f || gradient_first != 0.0f
        ) ||
        riftco_transformer_ill_conditioned_sum(
            retained_first,
            gradient_first,
            first
        )) {
        riftco_transformer_request_reference(reference_required);
        return;
    }

    const float gradient_square =
        clipped_gradient * clipped_gradient;
    if (riftco_transformer_unsafe_result(
            gradient_square,
            clipped_gradient != 0.0f
        )) {
        riftco_transformer_request_reference(reference_required);
        return;
    }
    const float one_minus_beta2 = 1.0f - beta2;
    const float retained_second = beta2 * old_second;
    const float gradient_second =
        one_minus_beta2 * gradient_square;
    if (riftco_transformer_unsafe_result(
            retained_second,
            beta2 != 0.0f && old_second != 0.0f
        ) ||
        riftco_transformer_unsafe_result(
            gradient_second,
            one_minus_beta2 != 0.0f &&
                gradient_square != 0.0f
        )) {
        riftco_transformer_request_reference(reference_required);
        return;
    }
    const float second = retained_second + gradient_second;
    if (riftco_transformer_unsafe_result(
            second,
            retained_second != 0.0f ||
                gradient_second != 0.0f
        ) ||
        second < 0.0f) {
        riftco_transformer_request_reference(reference_required);
        return;
    }

    const float corrected_first = first / first_correction;
    const float corrected_second = second / second_correction;
    if (riftco_transformer_unsafe_result(
            corrected_first,
            first != 0.0f
        ) ||
        riftco_transformer_unsafe_result(
            corrected_second,
            second != 0.0f
        ) ||
        corrected_second < 0.0f) {
        riftco_transformer_request_reference(reference_required);
        return;
    }

    const float root_second = sqrt(corrected_second);
    if (riftco_transformer_unsafe_result(
            root_second,
            corrected_second != 0.0f
        )) {
        riftco_transformer_request_reference(reference_required);
        return;
    }
    const float denominator = root_second + epsilon;
    if (riftco_transformer_unsafe_result(
            denominator,
            root_second != 0.0f || epsilon != 0.0f
        ) ||
        denominator <= 0.0f) {
        riftco_transformer_request_reference(reference_required);
        return;
    }
    const float scaled_first =
        learning_rate * corrected_first;
    if (riftco_transformer_unsafe_result(
            scaled_first,
            learning_rate != 0.0f &&
                corrected_first != 0.0f
        )) {
        riftco_transformer_request_reference(reference_required);
        return;
    }
    const float update = scaled_first / denominator;
    if (riftco_transformer_unsafe_result(
            update,
            scaled_first != 0.0f
        )) {
        riftco_transformer_request_reference(reference_required);
        return;
    }
    const float updated_value = parameter_value - update;
    if (riftco_transformer_unsafe_result(
            updated_value,
            parameter_value != 0.0f || update != 0.0f
        ) ||
        riftco_transformer_ill_conditioned_sum(
            parameter_value,
            -update,
            updated_value
        )) {
        riftco_transformer_request_reference(reference_required);
        return;
    }

    next_value[index] = updated_value;
    next_first_moment[index] = first;
    next_second_moment[index] = second;
}
)METAL";

}  // namespace riftco_transformer::backend_detail::adam_metal_detail
