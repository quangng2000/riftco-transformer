cmake_minimum_required(VERSION 3.24)

project(
    riftco_transformer
    VERSION 0.5.0
    DESCRIPTION "An auditable dependency-free transformer framework"
    LANGUAGES C CXX
)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

set(RIFTCO_TRANSFORMER_C_ABI_VERSION 2.4.0)
string(REPLACE "." ";" riftco_transformer_c_abi_components
    "${RIFTCO_TRANSFORMER_C_ABI_VERSION}"
)
list(LENGTH riftco_transformer_c_abi_components
    riftco_transformer_c_abi_component_count
)
if(NOT riftco_transformer_c_abi_component_count EQUAL 3)
    message(FATAL_ERROR
        "RIFTCO_TRANSFORMER_C_ABI_VERSION must have major.minor.patch form"
    )
endif()
list(GET riftco_transformer_c_abi_components 0
    RIFTCO_TRANSFORMER_C_ABI_SOVERSION
)
list(GET riftco_transformer_c_abi_components 1
    riftco_transformer_c_abi_minor_version
)

file(STRINGS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/riftco_transformer/c_api.h
    riftco_transformer_c_abi_header_major
    REGEX "^#define RT_ABI_VERSION_MAJOR "
)
file(STRINGS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/riftco_transformer/c_api.h
    riftco_transformer_c_abi_header_minor
    REGEX "^#define RT_ABI_VERSION_MINOR "
)
if(NOT riftco_transformer_c_abi_header_major
       STREQUAL
       "#define RT_ABI_VERSION_MAJOR UINT32_C(${RIFTCO_TRANSFORMER_C_ABI_SOVERSION})"
   OR NOT riftco_transformer_c_abi_header_minor
       STREQUAL
       "#define RT_ABI_VERSION_MINOR UINT32_C(${riftco_transformer_c_abi_minor_version})")
    message(FATAL_ERROR
        "CMake C ABI version and c_api.h RT_ABI_VERSION must match"
    )
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(STRINGS
    ${CMAKE_CURRENT_SOURCE_DIR}/pyproject.toml
    riftco_transformer_python_version_line
    REGEX "^version = \"[0-9]+[.][0-9]+[.][0-9]+\"$"
)
if(NOT riftco_transformer_python_version_line
       STREQUAL "version = \"${PROJECT_VERSION}\"")
    message(FATAL_ERROR
        "CMake and Python package versions must match"
    )
endif()

include(cmake/RiftcoTransformerWarnings.cmake)
include(cmake/RiftcoTransformerSanitizers.cmake)
include(cmake/RiftcoTransformerBackends.cmake)

option(
    RIFTCO_TRANSFORMER_ENABLE_INSTALL
    "Generate install rules and an exported CMake package"
    ${PROJECT_IS_TOP_LEVEL}
)
option(
    RIFTCO_TRANSFORMER_BUILD_PYTHON_WHEEL
    "Build and install the native runtime for the Python wheel"
    OFF
)
add_library(riftco_transformer_library STATIC
    ${riftco_transformer_backend_sources}
    src/artifacts/state.cpp
    src/core/autograd/checkpoint.cpp
    src/core/autograd/custom_gradient.cpp
    src/core/autograd/graph.cpp
    src/core/autograd/operations.cpp
    src/core/quantization/nf4.cpp
    src/core/quantization/quantized_weight.cpp
    src/core/tensor/detail/validation.cpp
    src/core/tensor/elementwise.cpp
    src/core/tensor/indexing.cpp
    src/core/tensor/layout.cpp
    src/core/tensor/layout_ops.cpp
    src/core/tensor/matmul.cpp
    src/core/tensor/reductions.cpp
    src/core/tensor/softmax.cpp
    src/core/tensor/storage.cpp
    src/data/token_batch.cpp
    src/data/tokenizer.cpp
    src/model/causal_self_attention.cpp
    src/model/decoder_only_transformer.cpp
    src/model/feed_forward.cpp
    src/model/transformer_block.cpp
    src/optim/adam.cpp
    src/nn/activations.cpp
    src/nn/embedding.cpp
    src/nn/initialization.cpp
    src/nn/layer_norm.cpp
    src/nn/linear.cpp
    src/nn/low_rank_adapter.cpp
    src/nn/loss.cpp
    src/nn/module.cpp
    src/nn/parameter.cpp
    src/nn/quantized_linear.cpp
    src/stages/serving/config.cpp
    src/stages/serving/cache/contiguous_kv_cache.cpp
    src/stages/serving/cache/page_storage.cpp
    src/stages/serving/cache/page_table_cache.cpp
    src/stages/serving/cache/paged_kv_cache.cpp
    src/stages/serving/cache/validation.cpp
    src/stages/serving/generation.cpp
    src/stages/serving/stack.cpp
)

set_target_properties(riftco_transformer_library
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        OBJCXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME library
        OUTPUT_NAME riftco_transformer
)

add_library(riftco_transformer::library ALIAS riftco_transformer_library)

target_include_directories(riftco_transformer_library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(RIFTCO_TRANSFORMER_ENABLE_TPU)
    target_include_directories(riftco_transformer_library
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pjrt/include
    )
endif()
target_compile_features(riftco_transformer_library
    PUBLIC
        cxx_std_20
)
target_link_libraries(riftco_transformer_library
    PRIVATE
        Threads::Threads
)
if(RIFTCO_TRANSFORMER_ENABLE_TPU)
    target_link_libraries(riftco_transformer_library
        PRIVATE
            ${CMAKE_DL_LIBS}
    )
endif()

if(RIFTCO_TRANSFORMER_ENABLE_CUDA)
    add_library(riftco_transformer_cuda_objects OBJECT
        ${riftco_transformer_cuda_sources}
    )
    target_include_directories(riftco_transformer_cuda_objects
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(riftco_transformer_cuda_objects
        PRIVATE
            CUDA::cudart_static
    )
    set_target_properties(riftco_transformer_cuda_objects
        PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED YES
            CUDA_EXTENSIONS OFF
            CUDA_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES
            POSITION_INDEPENDENT_CODE ON
    )
    target_sources(riftco_transformer_library
        PRIVATE
            $<TARGET_OBJECTS:riftco_transformer_cuda_objects>
    )
    target_link_libraries(riftco_transformer_library
        PRIVATE
            CUDA::cudart_static
    )
endif()

riftco_transformer_enable_warnings(riftco_transformer_library)
if(RIFTCO_TRANSFORMER_ENABLE_TPU)
    riftco_transformer_enable_pjrt_header_compatibility(
        riftco_transformer_library
    )
endif()
riftco_transformer_enable_sanitizers(riftco_transformer_library)
riftco_transformer_propagate_sanitizer_runtime(riftco_transformer_library)

# Standard-library-only symbolic compiler. This target deliberately has no
# dependency on the tensor runtime so the boundary is enforced by the linker.
add_library(riftco_transformer_compiler STATIC
    src/compiler/cajal/checker.cpp
    src/compiler/cajal/compiler.cpp
    src/compiler/cajal/encoding.cpp
    src/compiler/cajal/expression.cpp
    src/compiler/cajal/interpreter.cpp
    src/compiler/cajal/multilinear_map.cpp
    src/compiler/cajal/type.cpp
    src/compiler/cajal/value.cpp
)
add_library(riftco_transformer::compiler ALIAS riftco_transformer_compiler)
set_target_properties(riftco_transformer_compiler
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME compiler
        OUTPUT_NAME riftco_transformer_compiler
)
target_include_directories(riftco_transformer_compiler
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(riftco_transformer_compiler
    PUBLIC
        cxx_std_20
)
riftco_transformer_enable_warnings(riftco_transformer_compiler)
riftco_transformer_enable_sanitizers(riftco_transformer_compiler)
riftco_transformer_propagate_sanitizer_runtime(riftco_transformer_compiler)

# Standard-library-only analysis algorithms. Keeping PCA and intervention
# statistics independent of Tensor makes this a stage-neutral offline tool
# rather than a hidden dependency of training or model execution.
add_library(riftco_transformer_analysis STATIC
    src/analysis/ablation.cpp
    src/analysis/intervention.cpp
    src/analysis/matrix.cpp
    src/analysis/pca.cpp
    src/analysis/representation.cpp
)
add_library(riftco_transformer::analysis ALIAS riftco_transformer_analysis)
set_target_properties(riftco_transformer_analysis
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME analysis
        OUTPUT_NAME riftco_transformer_analysis
)
target_include_directories(riftco_transformer_analysis
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(riftco_transformer_analysis
    PUBLIC
        cxx_std_20
)
riftco_transformer_enable_warnings(riftco_transformer_analysis)
riftco_transformer_enable_sanitizers(riftco_transformer_analysis)
riftco_transformer_propagate_sanitizer_runtime(riftco_transformer_analysis)

# Optional one-way bridge from the standard-library-only Cajal compiler into
# Tensor/autograd modules. Keeping it in a separate target prevents core, nn,
# and model from acquiring a reverse dependency on the symbolic compiler.
add_library(riftco_transformer_lowering STATIC
    src/lowering/cajal.cpp
    src/lowering/config.cpp
    src/lowering/module.cpp
    src/lowering/strategy.cpp
)
add_library(riftco_transformer::lowering ALIAS riftco_transformer_lowering)
set_target_properties(riftco_transformer_lowering
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME lowering
        OUTPUT_NAME riftco_transformer_lowering
)
target_include_directories(riftco_transformer_lowering
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(riftco_transformer_lowering
    PUBLIC
        cxx_std_20
)
target_link_libraries(riftco_transformer_lowering
    PUBLIC
        riftco_transformer::compiler
        riftco_transformer::library
)
riftco_transformer_enable_warnings(riftco_transformer_lowering)
riftco_transformer_enable_sanitizers(riftco_transformer_lowering)
riftco_transformer_propagate_sanitizer_runtime(riftco_transformer_lowering)

# Optional model-integration layer for placing lowered programs into sequence
# residuals. Neither the core runtime nor ordinary Transformer models depend on
# symbolic compilation or the analysis component.
add_library(riftco_transformer_programmed STATIC
    src/programmed/sequence_placement.cpp
)
add_library(riftco_transformer::programmed ALIAS riftco_transformer_programmed)
set_target_properties(riftco_transformer_programmed
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME programmed
        OUTPUT_NAME riftco_transformer_programmed
)
target_include_directories(riftco_transformer_programmed
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_features(riftco_transformer_programmed
    PUBLIC
        cxx_std_20
)
target_link_libraries(riftco_transformer_programmed
    PUBLIC
        riftco_transformer::analysis
        riftco_transformer::lowering
)
riftco_transformer_enable_warnings(riftco_transformer_programmed)
riftco_transformer_enable_sanitizers(riftco_transformer_programmed)
riftco_transformer_propagate_sanitizer_runtime(riftco_transformer_programmed)

if(RIFTCO_TRANSFORMER_ENABLE_METAL)
    target_link_libraries(riftco_transformer_library
        PUBLIC
            "-framework Foundation"
            "-framework Metal"
    )
endif()

add_library(riftco_transformer_c SHARED
    src/c_api.cpp
)

add_library(riftco_transformer::c_api ALIAS riftco_transformer_c)

target_compile_definitions(riftco_transformer_c
    PRIVATE
        RIFTCO_TRANSFORMER_C_EXPORTS
)

target_include_directories(riftco_transformer_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(riftco_transformer_c
    INTERFACE
        c_std_11
)

target_link_libraries(riftco_transformer_c
    PRIVATE
        riftco_transformer::library
)

set_target_properties(riftco_transformer_c
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        EXPORT_NAME c_api
        OUTPUT_NAME riftco_transformer_c
)

if(NOT RIFTCO_TRANSFORMER_BUILD_PYTHON_WHEEL)
    set_target_properties(riftco_transformer_c
        PROPERTIES
            VERSION ${RIFTCO_TRANSFORMER_C_ABI_VERSION}
            SOVERSION ${RIFTCO_TRANSFORMER_C_ABI_SOVERSION}
    )
endif()

riftco_transformer_enable_warnings(riftco_transformer_c)
riftco_transformer_enable_sanitizers(riftco_transformer_c)
riftco_transformer_propagate_sanitizer_runtime(riftco_transformer_c)

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
    set(riftco_transformer_tests_default ${BUILD_TESTING})
else()
    set(riftco_transformer_tests_default OFF)
endif()
option(
    RIFTCO_TRANSFORMER_BUILD_TESTS
    "Build riftco_transformer's own test suite"
    ${riftco_transformer_tests_default}
)

if(RIFTCO_TRANSFORMER_BUILD_TESTS)
    if(NOT PROJECT_IS_TOP_LEVEL)
        enable_testing()
    endif()
    add_subdirectory(tests)
    target_compile_definitions(riftco_transformer_c
        PRIVATE
            RIFTCO_TRANSFORMER_C_API_TESTING
    )
    target_compile_definitions(c_api_tests
        PRIVATE
            RIFTCO_TRANSFORMER_C_API_TESTING
    )
endif()

if(RIFTCO_TRANSFORMER_ENABLE_INSTALL OR
   RIFTCO_TRANSFORMER_BUILD_PYTHON_WHEEL)
    include(cmake/RiftcoTransformerInstall.cmake)
endif()
