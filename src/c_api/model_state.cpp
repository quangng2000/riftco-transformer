#include "internal/model_state.hpp"

namespace riftco_transformer::c_api::detail {

constexpr std::array<std::uint8_t, 8> kPackedModelStateMagic{
    'R', 'T', 'N', 'F', '4', 'S', '1', '\0'
};

void append_u32_le(std::vector<std::uint8_t>& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void append_u64_le(std::vector<std::uint8_t>& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<std::uint8_t>(value >> shift));
    }
}

void append_f32_le(std::vector<std::uint8_t>& output, float value) {
    append_u32_le(output, std::bit_cast<std::uint32_t>(value));
}

void append_f32_values_le(
    std::vector<std::uint8_t>& output,
    std::span<const float> values
) {
    for (const float value : values) {
        append_f32_le(output, value);
    }
}

std::vector<std::uint8_t> packed_model_state_bytes(
    const riftco_transformer::DecoderOnlyTransformer& model
) {
    const auto weights = model.packed_linear_weight_state();
    std::vector<std::uint8_t> output(
        kPackedModelStateMagic.begin(),
        kPackedModelStateMagic.end()
    );
    append_u64_le(
        output,
        checked_u64(weights.size(), "packed model weight count")
    );
    for (const auto& weight : weights) {
        const auto& payload = weight.payload;
        const bool double_quantized =
            payload.double_quantized_scales.has_value();
        const auto scale_code_count = double_quantized
                                          ? payload.double_quantized_scales
                                                ->scale_codes.size()
                                          : 0U;
        const auto second_scale_count = double_quantized
                                            ? payload
                                                  .double_quantized_scales
                                                  ->second_level_scales.size()
                                            : 0U;
        const auto scale_block_size = double_quantized
                                          ? payload.double_quantized_scales
                                                ->scale_block_size
                                          : 0U;
        const float offset = double_quantized
                                 ? payload.double_quantized_scales->offset
                                 : 0.0F;
        append_u64_le(output, checked_u64(weight.shape.size(), "NF4 rank"));
        append_u64_le(
            output,
            checked_u64(weight.block_size, "NF4 block size")
        );
        append_u64_le(
            output,
            checked_u64(payload.packed_codes.size(), "NF4 packed code count")
        );
        append_u64_le(
            output,
            checked_u64(payload.block_scales.size(), "NF4 block scale count")
        );
        append_u64_le(
            output,
            checked_u64(scale_code_count, "NF4 scale code count")
        );
        append_u64_le(
            output,
            checked_u64(second_scale_count, "NF4 second-level scale count")
        );
        append_u64_le(
            output,
            checked_u64(scale_block_size, "NF4 scale block size")
        );
        append_f32_le(output, offset);
        append_u32_le(output, double_quantized ? 1U : 0U);
        for (const std::size_t dimension : weight.shape) {
            append_u64_le(output, checked_u64(dimension, "NF4 dimension"));
        }
        output.insert(
            output.end(),
            payload.packed_codes.begin(),
            payload.packed_codes.end()
        );
        append_f32_values_le(output, payload.block_scales);
        if (double_quantized) {
            const auto& nested = *payload.double_quantized_scales;
            output.insert(
                output.end(),
                nested.scale_codes.begin(),
                nested.scale_codes.end()
            );
            append_f32_values_le(output, nested.second_level_scales);
        }
    }
    return output;
}

class PackedModelStateReader {
public:
    explicit PackedModelStateReader(std::span<const std::uint8_t> bytes)
        : remaining_(bytes) {}

    [[nodiscard]] std::span<const std::uint8_t> take(
        std::size_t count,
        const char* description
    ) {
        if (count > remaining_.size()) {
            throw std::invalid_argument(
                std::string("truncated packed model state: ") + description
            );
        }
        const auto result = remaining_.first(count);
        remaining_ = remaining_.subspan(count);
        return result;
    }

    [[nodiscard]] std::uint32_t u32(const char* description) {
        const auto bytes = take(4, description);
        std::uint32_t value = 0;
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            value |= static_cast<std::uint32_t>(bytes[index]) << (index * 8U);
        }
        return value;
    }

    [[nodiscard]] std::uint64_t u64(const char* description) {
        const auto bytes = take(8, description);
        std::uint64_t value = 0;
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
        }
        return value;
    }

    [[nodiscard]] float f32(const char* description) {
        return std::bit_cast<float>(u32(description));
    }

    [[nodiscard]] bool empty() const noexcept { return remaining_.empty(); }
    [[nodiscard]] std::size_t remaining_size() const noexcept {
        return remaining_.size();
    }

private:
    std::span<const std::uint8_t> remaining_;
};

std::vector<riftco_transformer::PackedLinearWeightState>
parse_packed_model_state(std::span<const std::uint8_t> bytes) {
    PackedModelStateReader reader(bytes);
    const auto magic = reader.take(
        kPackedModelStateMagic.size(),
        "magic"
    );
    if (!std::equal(
            magic.begin(), magic.end(), kPackedModelStateMagic.begin()
        )) {
        throw std::invalid_argument("unknown packed model state format");
    }
    const std::size_t count = checked_size(
        reader.u64("weight count"),
        "packed model weight count"
    );
    constexpr std::size_t minimum_entry_bytes = 80;
    if (count == 0 || count > reader.remaining_size() / minimum_entry_bytes) {
        throw std::invalid_argument(
            "packed model state weight count is inconsistent with its size"
        );
    }
    std::vector<riftco_transformer::PackedLinearWeightState> result;
    result.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t rank = checked_size(
            reader.u64("rank"), "NF4 rank"
        );
        if (rank != 2) {
            throw std::invalid_argument(
                "packed decoder Linear weights must have rank two"
            );
        }
        const std::size_t block_size = checked_size(
            reader.u64("block size"), "NF4 block size"
        );
        const std::size_t packed_count = checked_size(
            reader.u64("packed code count"), "NF4 packed code count"
        );
        const std::size_t block_scale_count = checked_size(
            reader.u64("block scale count"), "NF4 block scale count"
        );
        const std::size_t scale_code_count = checked_size(
            reader.u64("scale code count"), "NF4 scale code count"
        );
        const std::size_t second_scale_count = checked_size(
            reader.u64("second-level scale count"),
            "NF4 second-level scale count"
        );
        const std::size_t scale_block_size = checked_size(
            reader.u64("scale block size"), "NF4 scale block size"
        );
        const std::uint32_t offset_bits = reader.u32("scale offset");
        const float offset = std::bit_cast<float>(offset_bits);
        const std::uint32_t flags = reader.u32("flags");
        if (flags > 1U) {
            throw std::invalid_argument("packed model state has unknown flags");
        }
        const bool double_quantized = flags == 1U;
        if ((!double_quantized &&
             (scale_code_count != 0 || second_scale_count != 0 ||
              scale_block_size != 0 || offset_bits != 0U)) ||
            (double_quantized &&
             (block_scale_count != 0 || scale_block_size == 0))) {
            throw std::invalid_argument(
                "packed model state has inconsistent NF4 scale metadata"
            );
        }

        std::size_t required_payload_bytes = rank * sizeof(std::uint64_t);
        const auto add_required = [&](std::size_t value,
                                      std::size_t element_size) {
            if (value >
                (std::numeric_limits<std::size_t>::max() -
                 required_payload_bytes) /
                    element_size) {
                throw std::overflow_error(
                    "packed model state entry size exceeds addressable memory"
                );
            }
            required_payload_bytes += value * element_size;
        };
        add_required(packed_count, sizeof(std::uint8_t));
        add_required(block_scale_count, sizeof(float));
        add_required(scale_code_count, sizeof(std::uint8_t));
        add_required(second_scale_count, sizeof(float));
        if (required_payload_bytes > reader.remaining_size()) {
            throw std::invalid_argument(
                "truncated packed model state: weight payload"
            );
        }

        riftco_transformer::QuantizedWeight::Shape shape;
        shape.reserve(rank);
        for (std::size_t dimension = 0; dimension < rank; ++dimension) {
            shape.push_back(checked_size(
                reader.u64("dimension"), "NF4 dimension"
            ));
        }
        const auto packed = reader.take(packed_count, "packed codes");
        riftco_transformer::Nf4Payload payload;
        payload.packed_codes.assign(packed.begin(), packed.end());
        payload.block_scales.reserve(block_scale_count);
        for (std::size_t scale = 0; scale < block_scale_count; ++scale) {
            payload.block_scales.push_back(reader.f32("block scale"));
        }
        if (double_quantized) {
            riftco_transformer::Nf4DoubleQuantizedScales nested;
            const auto scale_codes = reader.take(
                scale_code_count, "scale codes"
            );
            nested.scale_codes.assign(scale_codes.begin(), scale_codes.end());
            nested.second_level_scales.reserve(second_scale_count);
            for (std::size_t scale = 0; scale < second_scale_count; ++scale) {
                nested.second_level_scales.push_back(
                    reader.f32("second-level scale")
                );
            }
            nested.scale_block_size = scale_block_size;
            nested.offset = offset;
            payload.double_quantized_scales.emplace(std::move(nested));
        }
        result.push_back({
            std::move(shape), block_size, std::move(payload)
        });
    }
    if (!reader.empty()) {
        throw std::invalid_argument(
            "packed model state contains trailing bytes"
        );
    }
    return result;
}

}  // namespace riftco_transformer::c_api::detail
using namespace riftco_transformer::c_api::detail;

extern "C" {

rt_status RT_CALL rt_model_packed_state_size(
    const rt_model* model,
    uint64_t* output
) {
    return guard([&] {
        require_model(model);
        if (output == nullptr) {
            throw std::invalid_argument(
                "packed model state size output must not be null"
            );
        }
        const auto bytes = packed_model_state_bytes(model->state->value);
        *output = checked_u64(bytes.size(), "packed model state size");
    });
}

rt_status RT_CALL rt_model_packed_state_copy(
    const rt_model* model,
    uint8_t* output,
    uint64_t output_size
) {
    return guard([&] {
        require_model(model);
        const auto bytes = packed_model_state_bytes(model->state->value);
        const std::size_t native_size = checked_size(
            output_size, "packed model state output size"
        );
        if (native_size != bytes.size()) {
            throw std::invalid_argument(
                "packed model state output size does not match"
            );
        }
        if (output == nullptr && native_size != 0) {
            throw std::invalid_argument(
                "packed model state output must not be null"
            );
        }
        std::copy(bytes.begin(), bytes.end(), output);
    });
}

rt_status RT_CALL rt_model_packed_state_load(
    rt_model* model,
    const uint8_t* state,
    uint64_t state_size
) {
    return guard([&] {
        require_model(model);
        const std::size_t native_size = checked_size(
            state_size, "packed model state size"
        );
        if (state == nullptr && native_size != 0) {
            throw std::invalid_argument(
                "packed model state input must not be null"
            );
        }
        if (model->state->active_variables.load(
                std::memory_order_relaxed
            ) != 0) {
            throw std::invalid_argument(
                "cannot restore packed model state while variable graphs are alive"
            );
        }
        require_no_active_decode_sessions(
            *model->state,
            "restore packed model state"
        );
        const auto parsed = parse_packed_model_state(
            std::span<const std::uint8_t>(state, native_size)
        );
        model->state->value.load_packed_linear_weight_state(parsed);
    });
}

}  // extern "C"
