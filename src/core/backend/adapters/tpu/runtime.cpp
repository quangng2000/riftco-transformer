#include "core/backend/adapters/tpu/runtime.hpp"
#include "core/backend/adapters/tpu/compile_options.hpp"

#include "xla/pjrt/c/pjrt_c_api.h"

#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace riftco_transformer::backend_detail {
namespace {

constexpr int minimum_compatible_pjrt_minor = 57;
constexpr std::size_t executable_cache_capacity = 64;

using GetPjrtApi = const PJRT_Api* (*)();

void discard_error(const PJRT_Api* api, PJRT_Error* error) noexcept {
    if (api == nullptr || error == nullptr ||
        api->PJRT_Error_Destroy == nullptr) {
        return;
    }
    PJRT_Error_Destroy_Args args{};
    args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    args.error = error;
    api->PJRT_Error_Destroy(&args);
}

std::string take_error_message(const PJRT_Api* api, PJRT_Error* error) {
    if (error == nullptr) {
        return {};
    }

    std::string message = "unknown PJRT error";
    if (api != nullptr && api->PJRT_Error_Message != nullptr) {
        PJRT_Error_Message_Args args{};
        args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
        args.error = error;
        api->PJRT_Error_Message(&args);
        if (args.message != nullptr) {
            message.assign(args.message, args.message_size);
        }
    }
    discard_error(api, error);
    return message;
}

void require_success(
    const PJRT_Api* api,
    PJRT_Error* error,
    std::string_view operation) {
    if (error == nullptr) {
        return;
    }
    throw std::runtime_error(
        "TPU PJRT " + std::string(operation) +
        " failed: " + take_error_message(api, error));
}

void destroy_event(const PJRT_Api* api, PJRT_Event* event) noexcept {
    if (event == nullptr) {
        return;
    }
    PJRT_Event_Destroy_Args args{};
    args.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
    args.event = event;
    discard_error(api, api->PJRT_Event_Destroy(&args));
}

void destroy_buffer(const PJRT_Api* api, PJRT_Buffer* buffer) noexcept {
    if (buffer == nullptr) {
        return;
    }
    PJRT_Buffer_Destroy_Args args{};
    args.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    args.buffer = buffer;
    discard_error(api, api->PJRT_Buffer_Destroy(&args));
}

void destroy_loaded_executable(
    const PJRT_Api* api,
    PJRT_LoadedExecutable* executable) noexcept {
    if (executable == nullptr) {
        return;
    }
    PJRT_LoadedExecutable_Destroy_Args args{};
    args.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    args.executable = executable;
    discard_error(api, api->PJRT_LoadedExecutable_Destroy(&args));
}

template <typename Handle> class UniquePjrtHandle final {
public:
    using Deleter = void (*)(const PJRT_Api*, Handle*) noexcept;

    UniquePjrtHandle(
        const PJRT_Api* api,
        Handle* handle,
        Deleter deleter) noexcept
        : api_(api), handle_(handle), deleter_(deleter) {}

    UniquePjrtHandle(const UniquePjrtHandle&) = delete;
    UniquePjrtHandle& operator=(const UniquePjrtHandle&) = delete;

    UniquePjrtHandle(UniquePjrtHandle&& other) noexcept
        : api_(other.api_), handle_(std::exchange(other.handle_, nullptr)),
          deleter_(other.deleter_) {}

    UniquePjrtHandle& operator=(UniquePjrtHandle&& other) noexcept {
        if (this != &other) {
            reset();
            api_ = other.api_;
            handle_ = std::exchange(other.handle_, nullptr);
            deleter_ = other.deleter_;
        }
        return *this;
    }

    ~UniquePjrtHandle() { reset(); }

    [[nodiscard]] Handle* get() const noexcept { return handle_; }

    [[nodiscard]] Handle* release() noexcept {
        return std::exchange(handle_, nullptr);
    }

private:
    void reset() noexcept {
        if (handle_ != nullptr) {
            deleter_(api_, std::exchange(handle_, nullptr));
        }
    }

    const PJRT_Api* api_;
    Handle* handle_;
    Deleter deleter_;
};

using UniqueEvent = UniquePjrtHandle<PJRT_Event>;
using UniqueBuffer = UniquePjrtHandle<PJRT_Buffer>;
using UniqueLoadedExecutable = UniquePjrtHandle<PJRT_LoadedExecutable>;

void await_event(
    const PJRT_Api* api,
    PJRT_Event* event,
    std::string_view operation) {
    if (event == nullptr) {
        throw std::logic_error(
            "TPU PJRT " + std::string(operation) +
            " returned a null completion event");
    }
    PJRT_Event_Await_Args args{};
    args.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
    args.event = event;
    require_success(api, api->PJRT_Event_Await(&args), operation);
}

std::size_t element_byte_size(TpuElementType type) {
    switch (type) {
    case TpuElementType::F32:
        return sizeof(float);
    case TpuElementType::S32:
        return sizeof(std::int32_t);
    case TpuElementType::U8:
        return sizeof(std::uint8_t);
    }
    throw std::invalid_argument("unknown TPU host element type");
}

PJRT_Buffer_Type pjrt_element_type(TpuElementType type) {
    switch (type) {
    case TpuElementType::F32:
        return PJRT_Buffer_Type_F32;
    case TpuElementType::S32:
        return PJRT_Buffer_Type_S32;
    case TpuElementType::U8:
        return PJRT_Buffer_Type_U8;
    }
    throw std::invalid_argument("unknown TPU host element type");
}

std::size_t checked_tensor_byte_size(
    std::span<const std::int64_t> dimensions,
    TpuElementType element_type) {
    if (dimensions.empty()) {
        throw std::invalid_argument(
            "TPU host tensors must have at least one dimension");
    }
    std::size_t element_count = 1;
    for (const std::int64_t dimension : dimensions) {
        if (dimension <= 0) {
            throw std::invalid_argument(
                "TPU host tensor dimensions must be positive");
        }
        const auto converted = static_cast<std::uint64_t>(dimension);
        if (converted > static_cast<std::uint64_t>(
                            std::numeric_limits<std::size_t>::max())) {
            throw std::overflow_error("TPU host tensor dimension overflow");
        }
        const auto value = static_cast<std::size_t>(converted);
        if (element_count > std::numeric_limits<std::size_t>::max() / value) {
            throw std::overflow_error("TPU host tensor element-count overflow");
        }
        element_count *= value;
    }
    const std::size_t width = element_byte_size(element_type);
    if (element_count > std::numeric_limits<std::size_t>::max() / width) {
        throw std::overflow_error("TPU host tensor byte-size overflow");
    }
    return element_count * width;
}

void validate_host_input(const TpuHostInput& input) {
    if (input.data == nullptr) {
        throw std::invalid_argument("TPU host input data is null");
    }
    if (input.byte_size !=
        checked_tensor_byte_size(input.dimensions, input.element_type)) {
        throw std::logic_error(
            "TPU host input byte size does not match its shape");
    }
}

void validate_host_output(const TpuHostOutput& output) {
    if (output.data == nullptr) {
        throw std::invalid_argument("TPU host output data is null");
    }
    if (output.byte_size !=
        checked_tensor_byte_size(output.dimensions, output.element_type)) {
        throw std::logic_error(
            "TPU host output byte size does not match its shape");
    }
}

std::int64_t checked_dimension(std::size_t value) {
    if (value == 0 || value > static_cast<std::size_t>(
                                  std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(
            "TPU tensor dimension exceeds the PJRT int64 range");
    }
    return static_cast<std::int64_t>(value);
}

std::string
tensor_type(std::size_t first, std::size_t second, std::size_t third) {
    return "tensor<" + std::to_string(first) + "x" + std::to_string(second) +
           "x" + std::to_string(third) + "xf32>";
}

std::string stablehlo_matmul_program(
    std::size_t batch,
    std::size_t rows,
    std::size_t shared,
    std::size_t columns) {
    const std::string left = tensor_type(batch, rows, shared);
    const std::string right = tensor_type(batch, shared, columns);
    const std::string output = tensor_type(batch, rows, columns);

    std::ostringstream program;
    program << "module @riftco_tpu_matmul attributes {"
            << "mhlo.num_partitions = 1 : i32, "
            << "mhlo.num_replicas = 1 : i32} {\n"
            << "  func.func public @main(%lhs: " << left << ", %rhs: " << right
            << ") -> " << output << " {\n"
            << "    %result = \"stablehlo.dot_general\"(%lhs, %rhs) {\n"
            << "      dot_dimension_numbers = #stablehlo.dot<\n"
            << "        lhs_batching_dimensions = [0],\n"
            << "        rhs_batching_dimensions = [0],\n"
            << "        lhs_contracting_dimensions = [2],\n"
            << "        rhs_contracting_dimensions = [1]\n"
            << "      >,\n"
            << "      precision_config = [#stablehlo<precision HIGHEST>, "
            << "#stablehlo<precision HIGHEST>]\n"
            << "    } : (" << left << ", " << right << ") -> " << output << "\n"
            << "    func.return %result : " << output << "\n"
            << "  }\n"
            << "}\n";
    return program.str();
}

std::string selected_library_path() {
    constexpr std::array<const char*, 2> environment_names{
        "RIFTCO_TRANSFORMER_TPU_LIBRARY",
        "TPU_LIBRARY_PATH",
    };
    for (const char* name : environment_names) {
        const char* value = std::getenv(name);
        if (value != nullptr && value[0] != '\0') {
            return value;
        }
    }
    return "libtpu.so";
}

class PjrtRuntime final {
public:
    static PjrtRuntime& instance() {
        // libtpu owns process-global teardown. Keep PJRT state alive until the
        // process exits so no C API call runs after that teardown.
        static PjrtRuntime* const runtime = new PjrtRuntime;
        return *runtime;
    }

    PjrtRuntime(const PjrtRuntime&) = delete;
    PjrtRuntime& operator=(const PjrtRuntime&) = delete;

    [[nodiscard]] bool available() noexcept {
        initialize_once();
        return available_;
    }

    [[nodiscard]] std::string_view unavailability_reason() noexcept {
        initialize_once();
        if (available_) {
            return {};
        }
        return initialization_error_;
    }

    void execute(
        const TpuProgram& program,
        std::span<const TpuHostInput> inputs,
        std::span<const TpuHostOutput> outputs) {
        initialize_once();
        if (!available_) {
            throw std::runtime_error(
                "TPU PJRT runtime is unavailable: " + initialization_error_);
        }
        if (program.key.dimension_count > program.key.dimensions.size()) {
            throw std::logic_error("TPU program cache key is malformed");
        }
        if (program.stablehlo.empty() || program.operation_name.empty()) {
            throw std::logic_error("TPU program metadata is incomplete");
        }
        if (inputs.empty() || outputs.empty()) {
            throw std::invalid_argument(
                "TPU programs require at least one input and output");
        }
        for (const auto& input : inputs) {
            validate_host_input(input);
        }
        for (const auto& output : outputs) {
            validate_host_output(output);
        }

        std::scoped_lock lock(execution_mutex_);
        CacheEntry& cached = executable_for(program);

        std::vector<UniqueBuffer> input_owners;
        input_owners.reserve(inputs.size());
        for (const auto& input : inputs) {
            input_owners.push_back(upload(input, cached.device));
        }

        std::vector<PJRT_Buffer*> device_arguments;
        device_arguments.reserve(input_owners.size());
        for (const auto& input : input_owners) {
            device_arguments.push_back(input.get());
        }
        PJRT_Buffer* const* argument_lists[1]{device_arguments.data()};

        std::vector<PJRT_Buffer*> device_outputs(outputs.size(), nullptr);
        PJRT_Buffer** output_lists[1]{device_outputs.data()};
        PJRT_Event* completion_events[1]{nullptr};

        std::vector<std::int64_t> non_donatable_inputs(inputs.size());
        std::iota(
            non_donatable_inputs.begin(),
            non_donatable_inputs.end(),
            std::int64_t{0});

        PJRT_ExecuteOptions options{};
        options.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
        options.non_donatable_input_indices = non_donatable_inputs.data();
        options.num_non_donatable_input_indices = non_donatable_inputs.size();

        PJRT_LoadedExecutable_Execute_Args execute{};
        execute.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
        execute.executable = cached.executable;
        execute.options = &options;
        execute.argument_lists = argument_lists;
        execute.num_devices = 1;
        execute.num_args = inputs.size();
        execute.output_lists = output_lists;
        execute.device_complete_events = completion_events;
        execute.execute_device = nullptr;

        PJRT_Error* execute_error =
            api_->PJRT_LoadedExecutable_Execute(&execute);
        UniqueEvent completion(api_, completion_events[0], destroy_event);
        std::vector<UniqueBuffer> output_owners;
        output_owners.reserve(device_outputs.size());
        for (PJRT_Buffer* output : device_outputs) {
            output_owners.emplace_back(api_, output, destroy_buffer);
        }
        require_success(
            api_, execute_error, program.operation_name + " execution");
        for (const auto& output : output_owners) {
            if (output.get() == nullptr) {
                throw std::logic_error(
                    "TPU PJRT execute returned a null output buffer");
            }
        }
        await_event(
            api_,
            completion.get(),
            program.operation_name + " execution completion");

        // Stage every result first. User-visible host mirrors are committed
        // only after the complete multi-output operation has succeeded.
        std::vector<std::vector<std::byte>> staged_outputs;
        staged_outputs.reserve(outputs.size());
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            staged_outputs.emplace_back(outputs[index].byte_size);
            download(
                output_owners[index].get(),
                staged_outputs.back(),
                outputs[index].dimensions,
                program.operation_name);
        }
        for (std::size_t index = 0; index < outputs.size(); ++index) {
            std::memcpy(
                outputs[index].data,
                staged_outputs[index].data(),
                staged_outputs[index].size());
        }
    }

private:
    struct CacheEntry {
        TpuProgramKey key;
        PJRT_LoadedExecutable* executable;
        PJRT_Device* device;
        std::uint64_t last_used;
    };

    PjrtRuntime() = default;

    void initialize_once() noexcept {
        std::call_once(initialization_once_, [this] {
            try {
                initialize();
                available_ = true;
            } catch (const std::exception& error) {
                initialization_error_ = error.what();
                cleanup_failed_initialization();
            } catch (...) {
                initialization_error_ = "unknown initialization error";
                cleanup_failed_initialization();
            }
        });
    }

    void initialize() {
        const std::string path = selected_library_path();
        library_handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library_handle_ == nullptr) {
            const char* error = dlerror();
            throw std::runtime_error(
                "could not load " + path +
                (error == nullptr ? std::string{} : ": " + std::string(error)));
        }

        dlerror();
        void* symbol = dlsym(library_handle_, "GetPjrtApi");
        const char* symbol_error = dlerror();
        if (symbol == nullptr || symbol_error != nullptr) {
            throw std::runtime_error(
                "GetPjrtApi was not found in " + path +
                (symbol_error == nullptr ? std::string{}
                                         : ": " + std::string(symbol_error)));
        }
        GetPjrtApi get_api = nullptr;
        static_assert(sizeof(get_api) == sizeof(symbol));
        std::memcpy(&get_api, &symbol, sizeof(get_api));
        api_ = get_api();
        validate_api();

        PJRT_Plugin_Initialize_Args initialize_args{};
        initialize_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        require_success(
            api_,
            api_->PJRT_Plugin_Initialize(&initialize_args),
            "plugin initialization");

        PJRT_Client_Create_Args create{};
        create.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
        PJRT_Error* create_error = api_->PJRT_Client_Create(&create);
        client_ = create.client;
        require_success(api_, create_error, "client creation");
        if (client_ == nullptr) {
            throw std::logic_error("TPU PJRT client creation returned null");
        }

        PJRT_Client_PlatformName_Args platform{};
        platform.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
        platform.client = client_;
        require_success(
            api_, api_->PJRT_Client_PlatformName(&platform), "platform query");
        if (platform.platform_name == nullptr) {
            throw std::logic_error(
                "TPU PJRT platform query returned a null name");
        }
        const std::string_view platform_name(
            platform.platform_name, platform.platform_name_size);
        if (platform_name != "tpu") {
            throw std::runtime_error(
                "loaded PJRT plugin reports platform '" +
                std::string(platform_name) + "', expected 'tpu'");
        }

        PJRT_Client_AddressableDevices_Args devices{};
        devices.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
        devices.client = client_;
        require_success(
            api_,
            api_->PJRT_Client_AddressableDevices(&devices),
            "addressable-device query");
        if (devices.num_addressable_devices == 0 ||
            devices.addressable_devices == nullptr) {
            throw std::runtime_error(
                "libtpu found no addressable Cloud TPU device");
        }
    }

    void validate_api() const {
        if (api_ == nullptr) {
            throw std::runtime_error("GetPjrtApi returned null");
        }
        constexpr std::size_t version_end =
            offsetof(PJRT_Api, pjrt_api_version) +
            sizeof(PJRT_Api::pjrt_api_version);
        if (api_->struct_size < version_end) {
            throw std::runtime_error("PJRT API table is truncated");
        }
        if (api_->pjrt_api_version.struct_size < PJRT_Api_Version_STRUCT_SIZE) {
            throw std::runtime_error("PJRT API version record is truncated");
        }
        if (api_->pjrt_api_version.major_version != PJRT_API_MAJOR ||
            api_->pjrt_api_version.minor_version <
                minimum_compatible_pjrt_minor) {
            throw std::runtime_error(
                "incompatible PJRT API version " +
                std::to_string(api_->pjrt_api_version.major_version) + "." +
                std::to_string(api_->pjrt_api_version.minor_version) +
                "; this build uses PJRT " + std::to_string(PJRT_API_MAJOR) +
                "." + std::to_string(PJRT_API_MINOR) +
                " and requires a compatible 0.57+ table");
        }

#define RIFTCO_REQUIRE_PJRT_FUNCTION(name)                                     \
    do {                                                                       \
        constexpr std::size_t field_end =                                      \
            offsetof(PJRT_Api, name) + sizeof(api_->name);                     \
        if (api_->struct_size < field_end || api_->name == nullptr) {          \
            throw std::runtime_error(                                          \
                "TPU PJRT API does not provide required entry point " #name);  \
        }                                                                      \
    } while (false)
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Error_Destroy);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Error_Message);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Plugin_Initialize);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Event_Destroy);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Event_Await);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Client_Create);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Client_Destroy);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Client_PlatformName);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Client_AddressableDevices);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Client_Compile);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Client_BufferFromHostBuffer);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_LoadedExecutable_Destroy);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_LoadedExecutable_AddressableDevices);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_LoadedExecutable_Execute);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Buffer_Destroy);
        RIFTCO_REQUIRE_PJRT_FUNCTION(PJRT_Buffer_ToHostBuffer);
#undef RIFTCO_REQUIRE_PJRT_FUNCTION
    }

    void cleanup_failed_initialization() noexcept {
        if (api_ != nullptr && client_ != nullptr &&
            api_->PJRT_Client_Destroy != nullptr) {
            PJRT_Client_Destroy_Args args{};
            args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
            args.client = client_;
            discard_error(api_, api_->PJRT_Client_Destroy(&args));
        }
        client_ = nullptr;
    }

    CacheEntry& executable_for(const TpuProgram& requested) {
        for (auto& entry : executable_cache_) {
            if (entry.key == requested.key) {
                entry.last_used = ++use_clock_;
                return entry;
            }
        }

        std::string code = requested.stablehlo;
        constexpr std::string_view format = "mlir";
        PJRT_Program program{};
        program.struct_size = PJRT_Program_STRUCT_SIZE;
        program.code = code.data();
        program.code_size = code.size();
        program.format = format.data();
        program.format_size = format.size();

        PJRT_Client_Compile_Args compile{};
        compile.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
        compile.client = client_;
        compile.program = &program;
        compile.compile_options = reinterpret_cast<const char*>(
            single_device_compile_options_proto.data());
        compile.compile_options_size =
            single_device_compile_options_proto.size();
        PJRT_Error* compile_error = api_->PJRT_Client_Compile(&compile);
        UniqueLoadedExecutable owner(
            api_, compile.executable, destroy_loaded_executable);
        require_success(
            api_, compile_error, requested.operation_name + " compilation");
        if (owner.get() == nullptr) {
            throw std::logic_error(
                "TPU PJRT compilation returned a null executable");
        }

        PJRT_LoadedExecutable_AddressableDevices_Args devices{};
        devices.struct_size =
            PJRT_LoadedExecutable_AddressableDevices_Args_STRUCT_SIZE;
        devices.executable = owner.get();
        require_success(
            api_,
            api_->PJRT_LoadedExecutable_AddressableDevices(&devices),
            "compiled-device query");
        if (devices.num_addressable_devices != 1 ||
            devices.addressable_devices == nullptr ||
            devices.addressable_devices[0] == nullptr) {
            throw std::runtime_error(
                "the TPU backend requires exactly one addressable execution "
                "device");
        }

        if (executable_cache_.size() == executable_cache_capacity) {
            auto oldest = std::min_element(
                executable_cache_.begin(),
                executable_cache_.end(),
                [](const CacheEntry& left, const CacheEntry& right) {
                    return left.last_used < right.last_used;
                });
            destroy_loaded_executable(api_, oldest->executable);
            executable_cache_.erase(oldest);
        }
        executable_cache_.push_back({
            requested.key,
            owner.release(),
            devices.addressable_devices[0],
            ++use_clock_,
        });
        return executable_cache_.back();
    }

    UniqueBuffer upload(const TpuHostInput& input, PJRT_Device* device) const {
        PJRT_Client_BufferFromHostBuffer_Args transfer{};
        transfer.struct_size =
            PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
        transfer.client = client_;
        transfer.data = input.data;
        transfer.type = pjrt_element_type(input.element_type);
        transfer.dims = input.dimensions.data();
        transfer.num_dims = input.dimensions.size();
        transfer.host_buffer_semantics =
            PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
        transfer.device = device;
        PJRT_Error* transfer_error =
            api_->PJRT_Client_BufferFromHostBuffer(&transfer);
        UniqueEvent done(api_, transfer.done_with_host_buffer, destroy_event);
        UniqueBuffer buffer(api_, transfer.buffer, destroy_buffer);
        require_success(api_, transfer_error, "host-to-device transfer");
        if (buffer.get() == nullptr) {
            throw std::logic_error(
                "TPU PJRT host transfer returned a null buffer");
        }
        await_event(api_, done.get(), "host-to-device transfer completion");
        return buffer;
    }

    void download(
        PJRT_Buffer* source,
        std::span<std::byte> destination,
        std::span<const std::int64_t> dimensions,
        std::string_view operation) const {
        std::vector<std::int64_t> minor_to_major(dimensions.size());
        for (std::size_t index = 0; index < dimensions.size(); ++index) {
            minor_to_major[index] =
                static_cast<std::int64_t>(dimensions.size() - 1 - index);
        }

        PJRT_Buffer_MemoryLayout layout{};
        layout.struct_size = PJRT_Buffer_MemoryLayout_STRUCT_SIZE;
        layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
        layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
        layout.tiled.minor_to_major = minor_to_major.data();
        layout.tiled.minor_to_major_size = minor_to_major.size();

        PJRT_Buffer_ToHostBuffer_Args size_query{};
        size_query.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        size_query.src = source;
        size_query.host_layout = &layout;
        size_query.dst = nullptr;
        PJRT_Error* size_error = api_->PJRT_Buffer_ToHostBuffer(&size_query);
        UniqueEvent size_event(api_, size_query.event, destroy_event);
        require_success(
            api_,
            size_error,
            std::string(operation) + " device-to-host size query");
        if (size_event.get() != nullptr) {
            await_event(
                api_,
                size_event.get(),
                std::string(operation) +
                    " device-to-host size query completion");
        }
        if (size_query.dst_size != destination.size()) {
            throw std::runtime_error(
                "TPU PJRT output byte size does not match its program shape");
        }

        PJRT_Buffer_ToHostBuffer_Args transfer{};
        transfer.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
        transfer.src = source;
        transfer.host_layout = &layout;
        transfer.dst = destination.data();
        transfer.dst_size = destination.size();
        PJRT_Error* transfer_error = api_->PJRT_Buffer_ToHostBuffer(&transfer);
        UniqueEvent done(api_, transfer.event, destroy_event);
        require_success(
            api_,
            transfer_error,
            std::string(operation) + " device-to-host transfer");
        await_event(
            api_,
            done.get(),
            std::string(operation) + " device-to-host transfer completion");
    }

    std::once_flag initialization_once_;
    std::mutex execution_mutex_;
    bool available_ = false;
    std::string initialization_error_ = "not initialized";
    void* library_handle_ = nullptr;
    const PJRT_Api* api_ = nullptr;
    PJRT_Client* client_ = nullptr;
    std::vector<CacheEntry> executable_cache_;
    std::uint64_t use_clock_ = 0;
};

}  // namespace

bool tpu_runtime_available() noexcept {
    return PjrtRuntime::instance().available();
}

std::string_view tpu_runtime_unavailability_reason() noexcept {
    return PjrtRuntime::instance().unavailability_reason();
}

void tpu_runtime_execute(
    const TpuProgram& program,
    std::span<const TpuHostInput> inputs,
    std::span<const TpuHostOutput> outputs) {
    PjrtRuntime::instance().execute(program, inputs, outputs);
}

void tpu_runtime_matmul(const MatmulRequest& request) {
    const auto& dimensions = request.dimensions;
    const std::array<std::int64_t, 3> left_shape{
        checked_dimension(dimensions.batch_count),
        checked_dimension(dimensions.rows),
        checked_dimension(dimensions.shared),
    };
    const std::array<std::int64_t, 3> right_shape{
        checked_dimension(dimensions.batch_count),
        checked_dimension(dimensions.shared),
        checked_dimension(dimensions.columns),
    };
    const std::array<std::int64_t, 3> output_shape{
        checked_dimension(dimensions.batch_count),
        checked_dimension(dimensions.rows),
        checked_dimension(dimensions.columns),
    };

    TpuProgram program{};
    program.key.kind = TpuProgramKind::Matmul;
    program.key.dimensions = {
        dimensions.batch_count,
        dimensions.rows,
        dimensions.shared,
        dimensions.columns,
        0,
        0,
    };
    program.key.dimension_count = 4;
    program.stablehlo = stablehlo_matmul_program(
        dimensions.batch_count,
        dimensions.rows,
        dimensions.shared,
        dimensions.columns);
    program.operation_name = "StableHLO matmul";

    const std::array<TpuHostInput, 2> inputs{
        TpuHostInput{
            request.left.data().data(),
            request.left.data().size_bytes(),
            TpuElementType::F32,
            left_shape,
        },
        TpuHostInput{
            request.right.data().data(),
            request.right.data().size_bytes(),
            TpuElementType::F32,
            right_shape,
        },
    };
    const std::array<TpuHostOutput, 1> outputs{
        TpuHostOutput{
            request.output.data().data(),
            request.output.data().size_bytes(),
            TpuElementType::F32,
            output_shape,
        },
    };
    tpu_runtime_execute(program, inputs, outputs);
}

}  // namespace riftco_transformer::backend_detail
