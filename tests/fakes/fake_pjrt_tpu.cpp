#include "xla/pjrt/c/pjrt_c_api.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

struct FakePjrtError final : PJRT_Error {
    std::string message;
};

struct PJRT_Event {};

struct PJRT_Device {
    int identifier = 0;
};

struct PJRT_Client {};

enum class FakeOperation : std::uint8_t {
    Matmul,
    MaterializedForward,
    MaterializedContextBackward,
    MaterializedProbabilitiesBackward,
    PagedDecodeForward,
};

struct FakeTensorSpec {
    PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
    std::vector<std::int64_t> dimensions;
};

struct PJRT_LoadedExecutable {
    FakeOperation operation = FakeOperation::Matmul;
    std::vector<FakeTensorSpec> inputs;
    std::vector<FakeTensorSpec> outputs;
};

struct PJRT_Buffer {
    PJRT_Buffer_Type type = PJRT_Buffer_Type_INVALID;
    std::vector<std::int64_t> dimensions;
    std::vector<float> float_values;
    std::vector<std::int32_t> integer_values;
};

namespace {

constexpr std::uint8_t expected_compile_options[]{
    0x1A, 0x19,
    0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0x01,
    0x20, 0x01,
    0x28, 0x01,
    0x62, 0x01, 0x00,
    0x92, 0x01, 0x01, 0x00,
    0xB8, 0x01, 0x01,
};

PJRT_Device& fake_device() {
    static PJRT_Device device;
    return device;
}

PJRT_Device* const* fake_devices() {
    static PJRT_Device* devices[]{&fake_device()};
    return devices;
}

FakePjrtError* fake_error_storage(PJRT_Error* error) {
    return static_cast<FakePjrtError*>(error);
}

const FakePjrtError* fake_error_storage(const PJRT_Error* error) {
    return static_cast<const FakePjrtError*>(error);
}

void fake_error_vtable_destroy(PJRT_Error* error) {
    delete fake_error_storage(error);
}

void fake_error_vtable_message(
    const PJRT_Error* error,
    const char** message,
    std::size_t* message_size
) {
    const auto* const storage = fake_error_storage(error);
    *message = storage->message.data();
    *message_size = storage->message.size();
}

PJRT_Error_Code fake_error_vtable_code(const PJRT_Error*) {
    return PJRT_Error_Code_INVALID_ARGUMENT;
}

void fake_error_vtable_payloads(
    const PJRT_Error*,
    PJRT_Error_PayloadVisitor,
    void*
) {}

const PJRT_Error_FunctionTable& fake_error_vtable() {
    static const PJRT_Error_FunctionTable table{
        PJRT_Error_FunctionTable_STRUCT_SIZE,
        sizeof(FakePjrtError),
        nullptr,
        &fake_error_vtable_destroy,
        &fake_error_vtable_message,
        &fake_error_vtable_code,
        &fake_error_vtable_payloads,
    };
    return table;
}

PJRT_Error* make_error(std::string message) {
    auto* const error = new FakePjrtError{
        PJRT_Error{&fake_error_vtable()},
        std::move(message),
    };
    return error;
}

bool has_struct_size(std::size_t actual, std::size_t required) {
    return actual >= required;
}

struct ParsedTensorSpec {
    FakeTensorSpec spec;
    std::size_t next_offset = 0;
};

std::optional<ParsedTensorSpec> parse_tensor_spec(
    std::string_view text,
    std::size_t search_offset
) {
    constexpr std::string_view prefix = "tensor<";
    const std::size_t start = text.find(prefix, search_offset);
    if (start == std::string_view::npos) {
        return std::nullopt;
    }

    ParsedTensorSpec parsed;
    std::size_t cursor = start + prefix.size();
    while (cursor < text.size()) {
        const char* const first = text.data() + cursor;
        const char* const last = text.data() + text.size();
        std::int64_t dimension = 0;
        const auto result = std::from_chars(first, last, dimension);
        if (result.ec != std::errc{} || result.ptr == first ||
            dimension <= 0) {
            return std::nullopt;
        }
        parsed.spec.dimensions.push_back(dimension);
        cursor = static_cast<std::size_t>(result.ptr - text.data());
        if (text.substr(cursor, 5) == "xf32>") {
            parsed.spec.type = PJRT_Buffer_Type_F32;
            parsed.next_offset = cursor + 5;
            return parsed;
        }
        if (text.substr(cursor, 5) == "xi32>") {
            parsed.spec.type = PJRT_Buffer_Type_S32;
            parsed.next_offset = cursor + 5;
            return parsed;
        }
        if (cursor >= text.size() || text[cursor] != 'x') {
            return std::nullopt;
        }
        ++cursor;
    }
    return std::nullopt;
}

std::optional<std::vector<FakeTensorSpec>> parse_main_signature(
    std::string_view program,
    std::size_t expected_count
) {
    constexpr std::string_view main_marker = "func.func public @main";
    const std::size_t main_start = program.find(main_marker);
    if (main_start == std::string_view::npos) {
        return std::nullopt;
    }
    const std::size_t body_start = program.find('{', main_start);
    if (body_start == std::string_view::npos) {
        return std::nullopt;
    }
    const std::string_view signature = program.substr(
        main_start,
        body_start - main_start
    );

    std::vector<FakeTensorSpec> specs;
    specs.reserve(expected_count);
    std::size_t cursor = 0;
    while (specs.size() < expected_count) {
        const auto parsed = parse_tensor_spec(signature, cursor);
        if (!parsed.has_value()) {
            return std::nullopt;
        }
        specs.push_back(parsed->spec);
        cursor = parsed->next_offset;
    }
    if (parse_tensor_spec(signature, cursor).has_value()) {
        return std::nullopt;
    }
    return specs;
}

std::optional<std::size_t> checked_element_count(
    const std::vector<std::int64_t>& dimensions
) {
    if (dimensions.empty()) {
        return std::nullopt;
    }
    std::size_t count = 1;
    for (const std::int64_t dimension : dimensions) {
        if (dimension <= 0) {
            return std::nullopt;
        }
        const auto unsigned_dimension =
            static_cast<std::uint64_t>(dimension);
        if (unsigned_dimension > static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max()
            )) {
            return std::nullopt;
        }
        const auto converted = static_cast<std::size_t>(unsigned_dimension);
        if (count >
            std::numeric_limits<std::size_t>::max() / converted) {
            return std::nullopt;
        }
        count *= converted;
    }
    return count;
}

bool has_markers(
    std::string_view program,
    std::initializer_list<std::string_view> markers
) {
    return std::all_of(
        markers.begin(),
        markers.end(),
        [program](std::string_view marker) {
            return program.find(marker) != std::string_view::npos;
        }
    );
}

struct ProgramDefinition {
    FakeOperation operation;
    std::size_t input_count;
    std::size_t output_count;
};

std::optional<ProgramDefinition> identify_program(
    std::string_view program
) {
    if (program.find("module @riftco_tpu_matmul") !=
        std::string_view::npos) {
        if (!has_markers(
                program,
                {
                    "mhlo.num_partitions = 1",
                    "mhlo.num_replicas = 1",
                    "\"stablehlo.dot_general\"",
                    "lhs_batching_dimensions = [0]",
                    "rhs_batching_dimensions = [0]",
                    "lhs_contracting_dimensions = [2]",
                    "rhs_contracting_dimensions = [1]",
                }
            )) {
            return std::nullopt;
        }
        return ProgramDefinition{FakeOperation::Matmul, 2, 1};
    }
    if (program.find(
            "module @riftco_tpu_materialized_causal_attention_forward"
        ) != std::string_view::npos) {
        if (!has_markers(
                program,
                {
                    "\"stablehlo.dot_general\"",
                    "\"stablehlo.iota\"",
                    "\"stablehlo.compare\"",
                    "\"stablehlo.select\"",
                    "\"stablehlo.reduce\"",
                    "\"stablehlo.exponential\"",
                    "\"stablehlo.is_finite\"",
                }
            )) {
            return std::nullopt;
        }
        return ProgramDefinition{FakeOperation::MaterializedForward, 3, 3};
    }
    if (program.find(
            "module @riftco_tpu_materialized_causal_attention_context_backward"
        ) != std::string_view::npos) {
        if (!has_markers(
                program,
                {
                    "\"stablehlo.dot_general\"",
                    "\"stablehlo.reduce\"",
                    "\"stablehlo.select\"",
                }
            )) {
            return std::nullopt;
        }
        return ProgramDefinition{
            FakeOperation::MaterializedContextBackward,
            5,
            3,
        };
    }
    if (program.find(
            "module @riftco_tpu_materialized_causal_attention_probabilities_backward"
        ) != std::string_view::npos) {
        if (!has_markers(
                program,
                {
                    "\"stablehlo.dot_general\"",
                    "\"stablehlo.reduce\"",
                    "\"stablehlo.select\"",
                }
            )) {
            return std::nullopt;
        }
        return ProgramDefinition{
            FakeOperation::MaterializedProbabilitiesBackward,
            4,
            2,
        };
    }
    if (program.find(
            "module @riftco_tpu_paged_decode_attention_forward"
        ) != std::string_view::npos) {
        if (!has_markers(
                program,
                {
                    "\"stablehlo.transpose\"",
                    "\"stablehlo.reshape\"",
                    "\"stablehlo.gather\"",
                    "\"stablehlo.dot_general\"",
                    "\"stablehlo.select\"",
                    "\"stablehlo.reduce\"",
                    "\"stablehlo.exponential\"",
                    "\"stablehlo.is_finite\"",
                }
            )) {
            return std::nullopt;
        }
        return ProgramDefinition{FakeOperation::PagedDecodeForward, 5, 2};
    }
    return std::nullopt;
}

bool tensor_is(
    const FakeTensorSpec& spec,
    PJRT_Buffer_Type type,
    std::initializer_list<std::int64_t> dimensions
) {
    return spec.type == type &&
        spec.dimensions.size() == dimensions.size() &&
        std::equal(
            dimensions.begin(),
            dimensions.end(),
            spec.dimensions.begin()
        );
}

bool same_tensor_spec(
    const FakeTensorSpec& left,
    const FakeTensorSpec& right
) {
    return left.type == right.type &&
        left.dimensions == right.dimensions;
}

std::optional<std::string> validate_program_shapes(
    const PJRT_LoadedExecutable& executable
) {
    const auto& inputs = executable.inputs;
    const auto& outputs = executable.outputs;
    switch (executable.operation) {
        case FakeOperation::Matmul: {
            if (inputs[0].type != PJRT_Buffer_Type_F32 ||
                inputs[1].type != PJRT_Buffer_Type_F32 ||
                outputs[0].type != PJRT_Buffer_Type_F32 ||
                inputs[0].dimensions.size() != 3 ||
                inputs[1].dimensions.size() != 3 ||
                outputs[0].dimensions.size() != 3) {
                return "matmul tensors must be rank-3 f32";
            }
            const auto& left = inputs[0].dimensions;
            const auto& right = inputs[1].dimensions;
            const auto& output = outputs[0].dimensions;
            if (left[0] != right[0] || left[0] != output[0] ||
                left[1] != output[1] || left[2] != right[1] ||
                right[2] != output[2]) {
                return "inconsistent StableHLO matmul shapes";
            }
            return std::nullopt;
        }
        case FakeOperation::MaterializedForward: {
            if (inputs[0].type != PJRT_Buffer_Type_F32 ||
                inputs[0].dimensions.size() != 4 ||
                !same_tensor_spec(inputs[0], inputs[1]) ||
                !same_tensor_spec(inputs[0], inputs[2]) ||
                !same_tensor_spec(inputs[0], outputs[1])) {
                return "invalid materialized-attention forward tensor shapes";
            }
            const auto& shape = inputs[0].dimensions;
            if (!tensor_is(
                    outputs[0],
                    PJRT_Buffer_Type_F32,
                    {shape[0], shape[1], shape[2], shape[2]}
                ) ||
                !tensor_is(
                    outputs[2],
                    PJRT_Buffer_Type_F32,
                    {shape[0], shape[1], shape[2]}
                )) {
                return "invalid materialized-attention forward output shapes";
            }
            return std::nullopt;
        }
        case FakeOperation::MaterializedContextBackward: {
            if (inputs[0].type != PJRT_Buffer_Type_F32 ||
                inputs[0].dimensions.size() != 4 ||
                !same_tensor_spec(inputs[0], inputs[1]) ||
                !same_tensor_spec(inputs[0], inputs[2]) ||
                !same_tensor_spec(inputs[0], inputs[4]) ||
                !same_tensor_spec(inputs[0], outputs[0]) ||
                !same_tensor_spec(inputs[0], outputs[1]) ||
                !same_tensor_spec(inputs[0], outputs[2])) {
                return "invalid materialized-attention context VJP shapes";
            }
            const auto& shape = inputs[0].dimensions;
            if (!tensor_is(
                    inputs[3],
                    PJRT_Buffer_Type_F32,
                    {shape[0], shape[1], shape[2], shape[2]}
                )) {
                return "invalid materialized-attention probability shape";
            }
            return std::nullopt;
        }
        case FakeOperation::MaterializedProbabilitiesBackward: {
            if (inputs[0].type != PJRT_Buffer_Type_F32 ||
                inputs[0].dimensions.size() != 4 ||
                !same_tensor_spec(inputs[0], inputs[1]) ||
                !same_tensor_spec(inputs[0], outputs[0]) ||
                !same_tensor_spec(inputs[0], outputs[1])) {
                return "invalid materialized-attention probability VJP shapes";
            }
            const auto& shape = inputs[0].dimensions;
            if (!tensor_is(
                    inputs[2],
                    PJRT_Buffer_Type_F32,
                    {shape[0], shape[1], shape[2], shape[2]}
                ) || !same_tensor_spec(inputs[2], inputs[3])) {
                return "invalid materialized-attention probability VJP inputs";
            }
            return std::nullopt;
        }
        case FakeOperation::PagedDecodeForward: {
            if (inputs[0].type != PJRT_Buffer_Type_F32 ||
                inputs[0].dimensions.size() != 2 ||
                inputs[1].type != PJRT_Buffer_Type_F32 ||
                inputs[1].dimensions.size() != 4 ||
                !same_tensor_spec(inputs[1], inputs[2]) ||
                inputs[3].type != PJRT_Buffer_Type_S32 ||
                inputs[3].dimensions.size() != 1 ||
                inputs[4].type != PJRT_Buffer_Type_F32 ||
                inputs[4].dimensions != inputs[3].dimensions) {
                return "invalid paged-decode input tensor shapes";
            }
            const auto& query = inputs[0].dimensions;
            const auto& pages = inputs[1].dimensions;
            if (query[0] != pages[1] || query[1] != pages[3] ||
                !same_tensor_spec(inputs[0], outputs[0]) ||
                !tensor_is(
                    outputs[1],
                    PJRT_Buffer_Type_F32,
                    {query[0]}
                )) {
                return "invalid paged-decode output tensor shapes";
            }
            return std::nullopt;
        }
    }
    return "unknown fake TPU program";
}

bool has_expected_compile_options(
    const char* bytes,
    std::size_t byte_count
) {
    return bytes != nullptr &&
        byte_count == std::size(expected_compile_options) &&
        std::memcmp(
            bytes,
            expected_compile_options,
            std::size(expected_compile_options)
        ) == 0;
}

bool has_expected_host_layout(
    const PJRT_Buffer_MemoryLayout* layout,
    std::size_t rank
) {
    if (layout == nullptr ||
        !has_struct_size(
            layout->struct_size,
            PJRT_Buffer_MemoryLayout_STRUCT_SIZE
        ) ||
        layout->type != PJRT_Buffer_MemoryLayout_Type_Tiled ||
        !has_struct_size(
            layout->tiled.struct_size,
            PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE
        ) ||
        layout->tiled.minor_to_major == nullptr ||
        layout->tiled.minor_to_major_size != rank ||
        layout->tiled.num_tiles != 0) {
        return false;
    }
    for (std::size_t axis = 0; axis < rank; ++axis) {
        const auto expected = static_cast<std::int64_t>(rank - 1 - axis);
        if (layout->tiled.minor_to_major[axis] != expected) {
            return false;
        }
    }
    return true;
}

std::size_t as_size(std::int64_t value) {
    return static_cast<std::size_t>(value);
}

std::size_t attention_offset(
    std::size_t batch,
    std::size_t heads,
    std::size_t time,
    std::size_t width,
    std::size_t batch_index,
    std::size_t head,
    std::size_t time_index,
    std::size_t channel
) {
    static_cast<void>(batch);
    return (((batch_index * heads + head) * time + time_index) * width) +
        channel;
}

std::size_t probability_offset(
    std::size_t heads,
    std::size_t time,
    std::size_t batch_index,
    std::size_t head,
    std::size_t query_time,
    std::size_t key_time
) {
    return (((batch_index * heads + head) * time + query_time) * time) +
        key_time;
}

std::unique_ptr<PJRT_Buffer> make_output_buffer(
    const FakeTensorSpec& spec
) {
    auto output = std::make_unique<PJRT_Buffer>();
    output->type = spec.type;
    output->dimensions = spec.dimensions;
    const auto count = checked_element_count(spec.dimensions);
    if (!count.has_value()) {
        return nullptr;
    }
    if (spec.type == PJRT_Buffer_Type_F32) {
        output->float_values.assign(*count, 0.0F);
    } else if (spec.type == PJRT_Buffer_Type_S32) {
        output->integer_values.assign(*count, 0);
    } else {
        return nullptr;
    }
    return output;
}

using OutputBuffers = std::vector<std::unique_ptr<PJRT_Buffer>>;

std::optional<std::string> execute_matmul(
    const PJRT_LoadedExecutable& executable,
    const std::vector<const PJRT_Buffer*>& inputs,
    OutputBuffers& outputs
) {
    outputs.push_back(make_output_buffer(executable.outputs[0]));
    if (outputs.back() == nullptr) {
        return "fake TPU matmul output size overflow";
    }
    const auto& left_shape = executable.inputs[0].dimensions;
    const auto& right_shape = executable.inputs[1].dimensions;
    const std::size_t batch_count = as_size(left_shape[0]);
    const std::size_t rows = as_size(left_shape[1]);
    const std::size_t shared = as_size(left_shape[2]);
    const std::size_t columns = as_size(right_shape[2]);
    const auto& left = inputs[0]->float_values;
    const auto& right = inputs[1]->float_values;
    auto& output = outputs[0]->float_values;
    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        const std::size_t left_base = batch * rows * shared;
        const std::size_t right_base = batch * shared * columns;
        const std::size_t output_base = batch * rows * columns;
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t column = 0; column < columns; ++column) {
                float total = 0.0F;
                for (std::size_t inner = 0; inner < shared; ++inner) {
                    total += left[left_base + row * shared + inner] *
                        right[right_base + inner * columns + column];
                }
                output[output_base + row * columns + column] = total;
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> execute_materialized_forward(
    const PJRT_LoadedExecutable& executable,
    const std::vector<const PJRT_Buffer*>& inputs,
    OutputBuffers& outputs
) {
    for (const auto& spec : executable.outputs) {
        outputs.push_back(make_output_buffer(spec));
        if (outputs.back() == nullptr) {
            return "fake TPU materialized-attention output size overflow";
        }
    }

    const auto& shape = executable.inputs[0].dimensions;
    const std::size_t batch_count = as_size(shape[0]);
    const std::size_t head_count = as_size(shape[1]);
    const std::size_t time = as_size(shape[2]);
    const std::size_t width = as_size(shape[3]);
    const float scale = 1.0F / std::sqrt(static_cast<float>(width));
    const auto& queries = inputs[0]->float_values;
    const auto& keys = inputs[1]->float_values;
    const auto& values = inputs[2]->float_values;
    auto& probabilities = outputs[0]->float_values;
    auto& context = outputs[1]->float_values;
    auto& status = outputs[2]->float_values;
    std::vector<float> scores(time, 0.0F);

    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        for (std::size_t head = 0; head < head_count; ++head) {
            for (std::size_t query_time = 0;
                 query_time < time;
                 ++query_time) {
                const std::size_t row =
                    (batch * head_count + head) * time + query_time;
                float maximum = -std::numeric_limits<float>::infinity();
                bool invalid = false;
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    float score = 0.0F;
                    for (std::size_t channel = 0;
                         channel < width;
                         ++channel) {
                        score += queries[attention_offset(
                            batch_count,
                            head_count,
                            time,
                            width,
                            batch,
                            head,
                            query_time,
                            channel
                        )] * keys[attention_offset(
                            batch_count,
                            head_count,
                            time,
                            width,
                            batch,
                            head,
                            key_time,
                            channel
                        )];
                    }
                    score *= scale;
                    if (std::isnan(score) ||
                        score == std::numeric_limits<float>::infinity()) {
                        invalid = true;
                    }
                    scores[key_time] = score;
                    maximum = std::max(maximum, score);
                }
                if (invalid || maximum ==
                        -std::numeric_limits<float>::infinity()) {
                    status[row] = 1.0F;
                    continue;
                }

                double denominator = 0.0;
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    const float exponential = static_cast<float>(std::exp(
                        static_cast<double>(scores[key_time] - maximum)
                    ));
                    probabilities[probability_offset(
                        head_count,
                        time,
                        batch,
                        head,
                        query_time,
                        key_time
                    )] = exponential;
                    denominator += static_cast<double>(exponential);
                }
                if (!std::isfinite(denominator) || denominator <= 0.0) {
                    status[row] = 1.0F;
                    continue;
                }
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    auto& probability = probabilities[probability_offset(
                        head_count,
                        time,
                        batch,
                        head,
                        query_time,
                        key_time
                    )];
                    probability = static_cast<float>(
                        static_cast<double>(probability) / denominator
                    );
                }
                for (std::size_t channel = 0;
                     channel < width;
                     ++channel) {
                    float total = 0.0F;
                    for (std::size_t key_time = 0;
                         key_time <= query_time;
                         ++key_time) {
                        total += probabilities[probability_offset(
                            head_count,
                            time,
                            batch,
                            head,
                            query_time,
                            key_time
                        )] * values[attention_offset(
                            batch_count,
                            head_count,
                            time,
                            width,
                            batch,
                            head,
                            key_time,
                            channel
                        )];
                    }
                    context[attention_offset(
                        batch_count,
                        head_count,
                        time,
                        width,
                        batch,
                        head,
                        query_time,
                        channel
                    )] = total;
                }
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> execute_materialized_context_backward(
    const PJRT_LoadedExecutable& executable,
    const std::vector<const PJRT_Buffer*>& inputs,
    OutputBuffers& outputs
) {
    for (const auto& spec : executable.outputs) {
        outputs.push_back(make_output_buffer(spec));
        if (outputs.back() == nullptr) {
            return "fake TPU materialized context VJP output size overflow";
        }
    }

    const auto& shape = executable.inputs[0].dimensions;
    const std::size_t batch_count = as_size(shape[0]);
    const std::size_t head_count = as_size(shape[1]);
    const std::size_t time = as_size(shape[2]);
    const std::size_t width = as_size(shape[3]);
    const float scale = 1.0F / std::sqrt(static_cast<float>(width));
    const auto& queries = inputs[0]->float_values;
    const auto& keys = inputs[1]->float_values;
    const auto& values = inputs[2]->float_values;
    const auto& probabilities = inputs[3]->float_values;
    const auto& upstream = inputs[4]->float_values;
    auto& query_gradient = outputs[0]->float_values;
    auto& key_gradient = outputs[1]->float_values;
    auto& value_gradient = outputs[2]->float_values;
    std::vector<float> probability_gradient(time, 0.0F);
    std::vector<float> score_gradient(time, 0.0F);

    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        for (std::size_t head = 0; head < head_count; ++head) {
            for (std::size_t query_time = 0;
                 query_time < time;
                 ++query_time) {
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    float total = 0.0F;
                    for (std::size_t channel = 0;
                         channel < width;
                         ++channel) {
                        total += upstream[attention_offset(
                            batch_count,
                            head_count,
                            time,
                            width,
                            batch,
                            head,
                            query_time,
                            channel
                        )] * values[attention_offset(
                            batch_count,
                            head_count,
                            time,
                            width,
                            batch,
                            head,
                            key_time,
                            channel
                        )];
                    }
                    probability_gradient[key_time] = total;
                }
                double weighted = 0.0;
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    const std::size_t index = probability_offset(
                        head_count,
                        time,
                        batch,
                        head,
                        query_time,
                        key_time
                    );
                    weighted += static_cast<double>(probabilities[index]) *
                        static_cast<double>(probability_gradient[key_time]);
                }
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    const std::size_t index = probability_offset(
                        head_count,
                        time,
                        batch,
                        head,
                        query_time,
                        key_time
                    );
                    score_gradient[key_time] = probabilities[index] *
                        (probability_gradient[key_time] -
                         static_cast<float>(weighted)) * scale;
                }
                for (std::size_t channel = 0;
                     channel < width;
                     ++channel) {
                    const std::size_t query_index = attention_offset(
                        batch_count,
                        head_count,
                        time,
                        width,
                        batch,
                        head,
                        query_time,
                        channel
                    );
                    float query_total = 0.0F;
                    for (std::size_t key_time = 0;
                         key_time <= query_time;
                         ++key_time) {
                        const std::size_t key_index = attention_offset(
                            batch_count,
                            head_count,
                            time,
                            width,
                            batch,
                            head,
                            key_time,
                            channel
                        );
                        query_total += score_gradient[key_time] *
                            keys[key_index];
                        key_gradient[key_index] += score_gradient[key_time] *
                            queries[query_index];
                        value_gradient[key_index] += probabilities[
                            probability_offset(
                                head_count,
                                time,
                                batch,
                                head,
                                query_time,
                                key_time
                            )
                        ] * upstream[query_index];
                    }
                    query_gradient[query_index] = query_total;
                }
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> execute_materialized_probabilities_backward(
    const PJRT_LoadedExecutable& executable,
    const std::vector<const PJRT_Buffer*>& inputs,
    OutputBuffers& outputs
) {
    for (const auto& spec : executable.outputs) {
        outputs.push_back(make_output_buffer(spec));
        if (outputs.back() == nullptr) {
            return "fake TPU materialized probability VJP output overflow";
        }
    }

    const auto& shape = executable.inputs[0].dimensions;
    const std::size_t batch_count = as_size(shape[0]);
    const std::size_t head_count = as_size(shape[1]);
    const std::size_t time = as_size(shape[2]);
    const std::size_t width = as_size(shape[3]);
    const float scale = 1.0F / std::sqrt(static_cast<float>(width));
    const auto& queries = inputs[0]->float_values;
    const auto& keys = inputs[1]->float_values;
    const auto& probabilities = inputs[2]->float_values;
    const auto& upstream = inputs[3]->float_values;
    auto& query_gradient = outputs[0]->float_values;
    auto& key_gradient = outputs[1]->float_values;
    std::vector<float> score_gradient(time, 0.0F);

    for (std::size_t batch = 0; batch < batch_count; ++batch) {
        for (std::size_t head = 0; head < head_count; ++head) {
            for (std::size_t query_time = 0;
                 query_time < time;
                 ++query_time) {
                double weighted = 0.0;
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    const std::size_t index = probability_offset(
                        head_count,
                        time,
                        batch,
                        head,
                        query_time,
                        key_time
                    );
                    weighted += static_cast<double>(probabilities[index]) *
                        static_cast<double>(upstream[index]);
                }
                for (std::size_t key_time = 0;
                     key_time <= query_time;
                     ++key_time) {
                    const std::size_t index = probability_offset(
                        head_count,
                        time,
                        batch,
                        head,
                        query_time,
                        key_time
                    );
                    score_gradient[key_time] = probabilities[index] *
                        (upstream[index] - static_cast<float>(weighted)) * scale;
                }
                for (std::size_t channel = 0;
                     channel < width;
                     ++channel) {
                    const std::size_t query_index = attention_offset(
                        batch_count,
                        head_count,
                        time,
                        width,
                        batch,
                        head,
                        query_time,
                        channel
                    );
                    float query_total = 0.0F;
                    for (std::size_t key_time = 0;
                         key_time <= query_time;
                         ++key_time) {
                        const std::size_t key_index = attention_offset(
                            batch_count,
                            head_count,
                            time,
                            width,
                            batch,
                            head,
                            key_time,
                            channel
                        );
                        query_total += score_gradient[key_time] *
                            keys[key_index];
                        key_gradient[key_index] += score_gradient[key_time] *
                            queries[query_index];
                    }
                    query_gradient[query_index] = query_total;
                }
            }
        }
    }
    return std::nullopt;
}

std::optional<std::string> execute_paged_decode(
    const PJRT_LoadedExecutable& executable,
    const std::vector<const PJRT_Buffer*>& inputs,
    OutputBuffers& outputs
) {
    for (const auto& spec : executable.outputs) {
        outputs.push_back(make_output_buffer(spec));
        if (outputs.back() == nullptr) {
            return "fake TPU paged-decode output size overflow";
        }
    }

    const auto& query_shape = executable.inputs[0].dimensions;
    const auto& page_shape = executable.inputs[1].dimensions;
    const std::size_t head_count = as_size(query_shape[0]);
    const std::size_t width = as_size(query_shape[1]);
    const std::size_t physical_blocks = as_size(page_shape[0]);
    const std::size_t block_size = as_size(page_shape[2]);
    const std::size_t bucket = as_size(
        executable.inputs[3].dimensions[0]
    );
    const std::size_t flattened_capacity = physical_blocks * block_size;
    const float scale = 1.0F / std::sqrt(static_cast<float>(width));
    const auto& query = inputs[0]->float_values;
    const auto& keys = inputs[1]->float_values;
    const auto& values = inputs[2]->float_values;
    const auto& token_indices = inputs[3]->integer_values;
    const auto& valid_mask = inputs[4]->float_values;
    auto& context = outputs[0]->float_values;
    auto& status = outputs[1]->float_values;
    std::vector<float> scores(bucket, 0.0F);

    for (std::size_t head = 0; head < head_count; ++head) {
        float maximum = -std::numeric_limits<float>::infinity();
        bool invalid = false;
        std::size_t valid_count = 0;
        for (std::size_t position = 0; position < bucket; ++position) {
            if (valid_mask[position] == 0.0F) {
                continue;
            }
            const std::int32_t signed_index = token_indices[position];
            if (signed_index < 0 ||
                static_cast<std::size_t>(signed_index) >=
                    flattened_capacity) {
                invalid = true;
                continue;
            }
            const std::size_t token_index =
                static_cast<std::size_t>(signed_index);
            const std::size_t physical_block = token_index / block_size;
            const std::size_t block_offset = token_index % block_size;
            float score = 0.0F;
            for (std::size_t channel = 0; channel < width; ++channel) {
                const std::size_t page_index = (((
                    physical_block * head_count + head
                ) * block_size + block_offset) * width) + channel;
                score += query[head * width + channel] * keys[page_index];
            }
            score *= scale;
            if (std::isnan(score) ||
                score == std::numeric_limits<float>::infinity()) {
                invalid = true;
            }
            scores[position] = score;
            maximum = std::max(maximum, score);
            ++valid_count;
        }
        if (invalid || valid_count == 0 || maximum ==
                -std::numeric_limits<float>::infinity()) {
            status[head] = 1.0F;
            continue;
        }

        double denominator = 0.0;
        for (std::size_t position = 0; position < bucket; ++position) {
            if (valid_mask[position] == 0.0F) {
                continue;
            }
            scores[position] = static_cast<float>(std::exp(
                static_cast<double>(scores[position] - maximum)
            ));
            denominator += static_cast<double>(scores[position]);
        }
        if (!std::isfinite(denominator) || denominator <= 0.0) {
            status[head] = 1.0F;
            continue;
        }
        for (std::size_t position = 0; position < bucket; ++position) {
            if (valid_mask[position] == 0.0F) {
                continue;
            }
            const std::size_t token_index = static_cast<std::size_t>(
                token_indices[position]
            );
            const std::size_t physical_block = token_index / block_size;
            const std::size_t block_offset = token_index % block_size;
            const float probability = static_cast<float>(
                static_cast<double>(scores[position]) / denominator
            );
            for (std::size_t channel = 0; channel < width; ++channel) {
                const std::size_t page_index = (((
                    physical_block * head_count + head
                ) * block_size + block_offset) * width) + channel;
                context[head * width + channel] +=
                    probability * values[page_index];
            }
        }
    }
    return std::nullopt;
}

void fake_error_destroy(PJRT_Error_Destroy_Args* args) {
    if (args != nullptr &&
        has_struct_size(
            args->struct_size,
            PJRT_Error_Destroy_Args_STRUCT_SIZE
        )) {
        fake_error_vtable_destroy(args->error);
    }
}

void fake_error_message(PJRT_Error_Message_Args* args) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Error_Message_Args_STRUCT_SIZE
        )) {
        return;
    }
    if (args->error == nullptr) {
        args->message = nullptr;
        args->message_size = 0;
        return;
    }
    const auto* const storage = fake_error_storage(args->error);
    args->message = storage->message.data();
    args->message_size = storage->message.size();
}

PJRT_Error* fake_plugin_initialize(PJRT_Plugin_Initialize_Args* args) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Plugin_Initialize_Args_STRUCT_SIZE
        )) {
        return make_error("invalid plugin-initialize arguments");
    }
    return nullptr;
}

PJRT_Error* fake_event_destroy(PJRT_Event_Destroy_Args* args) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Event_Destroy_Args_STRUCT_SIZE
        )) {
        return make_error("invalid event-destroy arguments");
    }
    delete args->event;
    return nullptr;
}

PJRT_Error* fake_event_await(PJRT_Event_Await_Args* args) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Event_Await_Args_STRUCT_SIZE
        ) || args->event == nullptr) {
        return make_error("invalid event-await arguments");
    }
    return nullptr;
}

PJRT_Error* fake_client_create(PJRT_Client_Create_Args* args) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Client_Create_Args_STRUCT_SIZE
        )) {
        return make_error("invalid client-create arguments");
    }
    args->client = nullptr;
    if (args->create_options != nullptr || args->num_options != 0 ||
        args->kv_get_callback != nullptr ||
        args->kv_put_callback != nullptr ||
        args->kv_try_get_callback != nullptr) {
        return make_error("unexpected fake TPU client options");
    }
    args->client = new PJRT_Client;
    return nullptr;
}

PJRT_Error* fake_client_destroy(PJRT_Client_Destroy_Args* args) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Client_Destroy_Args_STRUCT_SIZE
        )) {
        return make_error("invalid client-destroy arguments");
    }
    delete args->client;
    return nullptr;
}

PJRT_Error* fake_client_platform_name(
    PJRT_Client_PlatformName_Args* args
) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Client_PlatformName_Args_STRUCT_SIZE
        ) || args->client == nullptr) {
        return make_error("invalid platform-name arguments");
    }
    constexpr std::string_view name = "tpu";
    args->platform_name = name.data();
    args->platform_name_size = name.size();
    return nullptr;
}

PJRT_Error* fake_client_addressable_devices(
    PJRT_Client_AddressableDevices_Args* args
) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Client_AddressableDevices_Args_STRUCT_SIZE
        ) || args->client == nullptr) {
        return make_error("invalid addressable-device arguments");
    }
    args->addressable_devices = fake_devices();
    args->num_addressable_devices = 1;
    return nullptr;
}

PJRT_Error* fake_client_compile(PJRT_Client_Compile_Args* args) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Client_Compile_Args_STRUCT_SIZE
        ) || args->client == nullptr || args->program == nullptr ||
        !has_struct_size(
            args->program->struct_size,
            PJRT_Program_STRUCT_SIZE
        )) {
        return make_error("invalid compile arguments");
    }
    args->executable = nullptr;
    if (args->program->code == nullptr ||
        args->program->format == nullptr) {
        return make_error("missing PJRT program payload");
    }
    const std::string_view format(
        args->program->format,
        args->program->format_size
    );
    if (format != "mlir") {
        return make_error("fake TPU expects the mlir PJRT format");
    }
    if (!has_expected_compile_options(
            args->compile_options,
            args->compile_options_size
        )) {
        return make_error("unexpected TPU compile-options protobuf");
    }

    const std::string_view program(
        args->program->code,
        args->program->code_size
    );
    const auto definition = identify_program(program);
    if (!definition.has_value()) {
        return make_error(
            "StableHLO program is unsupported or missing required ops"
        );
    }
    const auto specs = parse_main_signature(
        program,
        definition->input_count + definition->output_count
    );
    if (!specs.has_value()) {
        return make_error("could not parse the StableHLO main signature");
    }

    auto executable = std::make_unique<PJRT_LoadedExecutable>();
    executable->operation = definition->operation;
    executable->inputs.assign(
        specs->begin(),
        specs->begin() + static_cast<std::ptrdiff_t>(
            definition->input_count
        )
    );
    executable->outputs.assign(
        specs->begin() + static_cast<std::ptrdiff_t>(
            definition->input_count
        ),
        specs->end()
    );
    if (const auto error = validate_program_shapes(*executable);
        error.has_value()) {
        return make_error(*error);
    }
    args->executable = executable.release();
    return nullptr;
}

PJRT_Error* fake_client_buffer_from_host(
    PJRT_Client_BufferFromHostBuffer_Args* args
) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE
        )) {
        return make_error("invalid host-buffer arguments");
    }
    args->buffer = nullptr;
    args->done_with_host_buffer = nullptr;
    if (args->client == nullptr || args->data == nullptr ||
        (args->type != PJRT_Buffer_Type_F32 &&
         args->type != PJRT_Buffer_Type_S32) ||
        args->dims == nullptr || args->num_dims == 0 ||
        args->byte_strides != nullptr || args->num_byte_strides != 0 ||
        args->host_buffer_semantics !=
            PJRT_HostBufferSemantics_kImmutableOnlyDuringCall ||
        args->device != &fake_device() || args->memory != nullptr ||
        args->device_layout != nullptr) {
        return make_error("unexpected fake TPU host-buffer contract");
    }

    std::vector<std::int64_t> dimensions(
        args->dims,
        args->dims + args->num_dims
    );
    const auto element_count = checked_element_count(dimensions);
    if (!element_count.has_value()) {
        return make_error("invalid fake TPU buffer dimensions");
    }
    auto buffer = std::make_unique<PJRT_Buffer>();
    buffer->type = args->type;
    buffer->dimensions = std::move(dimensions);
    if (args->type == PJRT_Buffer_Type_F32) {
        const auto* const source = static_cast<const float*>(args->data);
        buffer->float_values.assign(source, source + *element_count);
    } else {
        const auto* const source =
            static_cast<const std::int32_t*>(args->data);
        buffer->integer_values.assign(source, source + *element_count);
    }
    auto done = std::make_unique<PJRT_Event>();
    args->buffer = buffer.release();
    args->done_with_host_buffer = done.release();
    return nullptr;
}

PJRT_Error* fake_loaded_executable_destroy(
    PJRT_LoadedExecutable_Destroy_Args* args
) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE
        )) {
        return make_error("invalid executable-destroy arguments");
    }
    delete args->executable;
    return nullptr;
}

PJRT_Error* fake_loaded_executable_addressable_devices(
    PJRT_LoadedExecutable_AddressableDevices_Args* args
) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE
        ) || args->executable == nullptr) {
        return make_error("invalid compiled-device arguments");
    }
    args->addressable_devices = fake_devices();
    args->num_addressable_devices = 1;
    return nullptr;
}

bool buffer_matches_spec(
    const PJRT_Buffer* buffer,
    const FakeTensorSpec& spec
) {
    return buffer != nullptr && buffer->type == spec.type &&
        buffer->dimensions == spec.dimensions;
}

PJRT_Error* fake_loaded_executable_execute(
    PJRT_LoadedExecutable_Execute_Args* args
) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE
        ) || args->executable == nullptr || args->options == nullptr ||
        !has_struct_size(
            args->options->struct_size,
            PJRT_ExecuteOptions_STRUCT_SIZE
        )) {
        return make_error("invalid execute arguments");
    }
    const PJRT_LoadedExecutable& executable = *args->executable;
    if (args->num_devices != 1 ||
        args->num_args != executable.inputs.size() ||
        args->argument_lists == nullptr ||
        args->argument_lists[0] == nullptr ||
        args->output_lists == nullptr || args->output_lists[0] == nullptr ||
        args->device_complete_events == nullptr ||
        args->execute_device != nullptr) {
        return make_error("unexpected fake TPU execute contract");
    }
    for (std::size_t index = 0; index < executable.outputs.size(); ++index) {
        args->output_lists[0][index] = nullptr;
    }
    args->device_complete_events[0] = nullptr;

    if (args->options->non_donatable_input_indices == nullptr ||
        args->options->num_non_donatable_input_indices !=
            executable.inputs.size()) {
        return make_error("fake TPU inputs must be non-donatable");
    }
    for (std::size_t index = 0; index < executable.inputs.size(); ++index) {
        if (args->options->non_donatable_input_indices[index] !=
            static_cast<std::int64_t>(index)) {
            return make_error("fake TPU inputs must be non-donatable");
        }
    }

    std::vector<const PJRT_Buffer*> inputs;
    inputs.reserve(executable.inputs.size());
    for (std::size_t index = 0; index < executable.inputs.size(); ++index) {
        const PJRT_Buffer* const input = args->argument_lists[0][index];
        if (!buffer_matches_spec(input, executable.inputs[index])) {
            return make_error(
                "fake TPU input shape or type does not match executable"
            );
        }
        inputs.push_back(input);
    }

    OutputBuffers outputs;
    outputs.reserve(executable.outputs.size());
    std::optional<std::string> execution_error;
    switch (executable.operation) {
        case FakeOperation::Matmul:
            execution_error = execute_matmul(executable, inputs, outputs);
            break;
        case FakeOperation::MaterializedForward:
            execution_error = execute_materialized_forward(
                executable,
                inputs,
                outputs
            );
            break;
        case FakeOperation::MaterializedContextBackward:
            execution_error = execute_materialized_context_backward(
                executable,
                inputs,
                outputs
            );
            break;
        case FakeOperation::MaterializedProbabilitiesBackward:
            execution_error = execute_materialized_probabilities_backward(
                executable,
                inputs,
                outputs
            );
            break;
        case FakeOperation::PagedDecodeForward:
            execution_error = execute_paged_decode(
                executable,
                inputs,
                outputs
            );
            break;
    }
    if (execution_error.has_value()) {
        return make_error(*execution_error);
    }
    if (outputs.size() != executable.outputs.size()) {
        return make_error("fake TPU produced the wrong number of outputs");
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        args->output_lists[0][index] = outputs[index].release();
    }
    args->device_complete_events[0] = new PJRT_Event;
    return nullptr;
}

PJRT_Error* fake_buffer_destroy(PJRT_Buffer_Destroy_Args* args) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Buffer_Destroy_Args_STRUCT_SIZE
        )) {
        return make_error("invalid buffer-destroy arguments");
    }
    delete args->buffer;
    return nullptr;
}

PJRT_Error* fake_buffer_to_host(PJRT_Buffer_ToHostBuffer_Args* args) {
    if (args == nullptr ||
        !has_struct_size(
            args->struct_size,
            PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE
        )) {
        return make_error("invalid to-host arguments");
    }
    args->event = nullptr;
    if (args->src == nullptr || !has_expected_host_layout(
            args->host_layout,
            args->src->dimensions.size()
        )) {
        return make_error("unexpected fake TPU to-host layout");
    }

    const void* source = nullptr;
    std::size_t element_count = 0;
    std::size_t element_size = 0;
    if (args->src->type == PJRT_Buffer_Type_F32) {
        source = args->src->float_values.data();
        element_count = args->src->float_values.size();
        element_size = sizeof(float);
    } else if (args->src->type == PJRT_Buffer_Type_S32) {
        source = args->src->integer_values.data();
        element_count = args->src->integer_values.size();
        element_size = sizeof(std::int32_t);
    } else {
        return make_error("unsupported fake TPU output type");
    }
    if (element_count >
        std::numeric_limits<std::size_t>::max() / element_size) {
        return make_error("fake TPU to-host byte size overflow");
    }
    const std::size_t required_bytes = element_count * element_size;
    if (args->dst == nullptr) {
        args->dst_size = required_bytes;
        return nullptr;
    }
    if (args->dst_size < required_bytes) {
        return make_error("fake TPU to-host destination is too small");
    }
    std::memcpy(args->dst, source, required_bytes);
    args->event = new PJRT_Event;
    return nullptr;
}

PJRT_Api make_api() {
    PJRT_Api api{};
    api.struct_size = PJRT_Api_STRUCT_SIZE;
    api.pjrt_api_version.struct_size = PJRT_Api_Version_STRUCT_SIZE;
    api.pjrt_api_version.major_version = PJRT_API_MAJOR;
    api.pjrt_api_version.minor_version = PJRT_API_MINOR;
    api.PJRT_Error_Destroy = &fake_error_destroy;
    api.PJRT_Error_Message = &fake_error_message;
    api.PJRT_Plugin_Initialize = &fake_plugin_initialize;
    api.PJRT_Event_Destroy = &fake_event_destroy;
    api.PJRT_Event_Await = &fake_event_await;
    api.PJRT_Client_Create = &fake_client_create;
    api.PJRT_Client_Destroy = &fake_client_destroy;
    api.PJRT_Client_PlatformName = &fake_client_platform_name;
    api.PJRT_Client_AddressableDevices =
        &fake_client_addressable_devices;
    api.PJRT_Client_Compile = &fake_client_compile;
    api.PJRT_Client_BufferFromHostBuffer =
        &fake_client_buffer_from_host;
    api.PJRT_LoadedExecutable_Destroy =
        &fake_loaded_executable_destroy;
    api.PJRT_LoadedExecutable_AddressableDevices =
        &fake_loaded_executable_addressable_devices;
    api.PJRT_LoadedExecutable_Execute =
        &fake_loaded_executable_execute;
    api.PJRT_Buffer_Destroy = &fake_buffer_destroy;
    api.PJRT_Buffer_ToHostBuffer = &fake_buffer_to_host;
    return api;
}

}  // namespace

#if defined(__GNUC__) || defined(__clang__)
#define RIFTCO_FAKE_PJRT_EXPORT __attribute__((visibility("default")))
#else
#define RIFTCO_FAKE_PJRT_EXPORT
#endif

extern "C" RIFTCO_FAKE_PJRT_EXPORT const PJRT_Api* GetPjrtApi() {
    static const PJRT_Api api = make_api();
    return &api;
}
