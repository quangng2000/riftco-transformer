set(riftco_transformer_backend_sources
    src/core/backend/adapters/cpu/adapter.cpp
    src/core/backend/attention/dispatch.cpp
    src/core/backend/attention/reference/flash_causal.cpp
    src/core/backend/attention/reference/materialized_causal.cpp
    src/core/backend/attention/reference/paged_decode.cpp
    src/core/backend/nn/dispatch.cpp
    src/core/backend/nn/quantized_linear/dispatch.cpp
    src/core/backend/nn/quantized_linear/reference/operations.cpp
    src/core/backend/nn/reference/operations.cpp
    src/core/backend/optim/adam/dispatch.cpp
    src/core/backend/optim/adam/reference/update.cpp
    src/core/backend/registry.cpp
)

function(riftco_transformer_enable_pjrt_header_compatibility target)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # PJRT gives API-table members the same spelling as their function
        # typedefs. GCC diagnoses that valid upstream C ABI pattern in C++.
        target_compile_options(${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-changes-meaning>
        )
    endif()
endfunction()

if(APPLE)
    set(riftco_transformer_metal_default ON)
else()
    set(riftco_transformer_metal_default OFF)
endif()

option(
    RIFTCO_TRANSFORMER_ENABLE_METAL
    "Build the Metal backend adapter on Apple platforms"
    ${riftco_transformer_metal_default}
)

if(RIFTCO_TRANSFORMER_ENABLE_METAL AND NOT APPLE)
    message(FATAL_ERROR
        "RIFTCO_TRANSFORMER_ENABLE_METAL requires an Apple platform"
    )
endif()

if(RIFTCO_TRANSFORMER_ENABLE_METAL)
    enable_language(OBJCXX)
    list(APPEND riftco_transformer_backend_sources
        src/core/backend/adapters/metal/adapter.mm
        src/core/backend/adapters/metal/runtime.mm
        src/core/backend/nn/quantized_linear/metal/runtime.mm
    )
    set_source_files_properties(
        src/core/backend/adapters/metal/adapter.mm
        src/core/backend/adapters/metal/runtime.mm
        src/core/backend/nn/quantized_linear/metal/runtime.mm
        PROPERTIES
            COMPILE_OPTIONS "-fobjc-arc"
    )
else()
    list(APPEND riftco_transformer_backend_sources
        src/core/backend/adapters/metal/stub.cpp
    )
endif()

option(
    RIFTCO_TRANSFORMER_ENABLE_CUDA
    "Build the experimental CUDA backend adapter"
    OFF
)

if(RIFTCO_TRANSFORMER_ENABLE_CUDA)
    if(RIFTCO_TRANSFORMER_ENABLE_SANITIZERS)
        message(FATAL_ERROR
            "RIFTCO_TRANSFORMER_ENABLE_CUDA cannot currently be combined "
            "with RIFTCO_TRANSFORMER_ENABLE_SANITIZERS"
        )
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR
            "RIFTCO_TRANSFORMER_ENABLE_CUDA requires an NVIDIA CUDA "
            "compiler; install CUDA Toolkit 12 or newer or disable CUDA"
        )
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit 12 REQUIRED)
    set(riftco_transformer_cuda_sources
        src/core/backend/adapters/cuda/adapter.cu
        src/core/backend/attention/cuda/flash_causal.cu
        src/core/backend/attention/cuda/materialized_causal.cu
        src/core/backend/attention/cuda/paged_decode.cu
        src/core/backend/nn/cuda/elementwise.cu
        src/core/backend/nn/cuda/indexing.cu
        src/core/backend/nn/cuda/layout.cu
        src/core/backend/nn/cuda/loss.cu
        src/core/backend/nn/cuda/normalization.cu
        src/core/backend/nn/quantized_linear/cuda/operations.cu
        src/core/backend/nn/quantized_linear/cuda/storage.cu
        src/core/backend/nn/cuda/reduction.cu
        src/core/backend/nn/cuda/softmax.cu
        src/core/backend/optim/adam/cuda/update.cu
    )
else()
    list(APPEND riftco_transformer_backend_sources
        src/core/backend/adapters/cuda/stub.cpp
    )
endif()

option(
    RIFTCO_TRANSFORMER_ENABLE_TPU
    "Build the experimental Google Cloud TPU PJRT adapter"
    OFF
)

if(RIFTCO_TRANSFORMER_ENABLE_TPU)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR
            "RIFTCO_TRANSFORMER_ENABLE_TPU requires Linux and a Cloud TPU "
            "runtime; disable TPU on this platform"
        )
    endif()
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
        message(FATAL_ERROR
            "The TPU adapter supports Linux x86-64 only"
        )
    endif()
    list(APPEND riftco_transformer_backend_sources
        src/core/backend/adapters/tpu/adapter.cpp
        src/core/backend/adapters/tpu/runtime.cpp
        src/core/backend/attention/tpu/materialized_causal.cpp
        src/core/backend/attention/tpu/paged_decode.cpp
        src/core/backend/nn/quantized_linear/tpu/runtime.cpp
    )
else()
    list(APPEND riftco_transformer_backend_sources
        src/core/backend/adapters/tpu/stub.cpp
    )
endif()
