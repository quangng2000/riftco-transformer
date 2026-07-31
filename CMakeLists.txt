cmake_minimum_required(VERSION 3.24)

project(
    transformer_lab
    VERSION 0.1.0
    DESCRIPTION "An auditable dependency-free transformer framework"
    LANGUAGES C CXX
)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

set(TRANSFORMER_LAB_C_ABI_VERSION 1.8.0)
string(REPLACE "." ";" transformer_lab_c_abi_components
    "${TRANSFORMER_LAB_C_ABI_VERSION}"
)
list(LENGTH transformer_lab_c_abi_components
    transformer_lab_c_abi_component_count
)
if(NOT transformer_lab_c_abi_component_count EQUAL 3)
    message(FATAL_ERROR
        "TRANSFORMER_LAB_C_ABI_VERSION must have major.minor.patch form"
    )
endif()
list(GET transformer_lab_c_abi_components 0
    TRANSFORMER_LAB_C_ABI_SOVERSION
)
list(GET transformer_lab_c_abi_components 1
    transformer_lab_c_abi_minor_version
)

file(STRINGS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/transformer_lab/c_api.h
    transformer_lab_c_abi_header_major
    REGEX "^#define TL_ABI_VERSION_MAJOR "
)
file(STRINGS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/transformer_lab/c_api.h
    transformer_lab_c_abi_header_minor
    REGEX "^#define TL_ABI_VERSION_MINOR "
)
if(NOT transformer_lab_c_abi_header_major
       STREQUAL
       "#define TL_ABI_VERSION_MAJOR UINT32_C(${TRANSFORMER_LAB_C_ABI_SOVERSION})"
   OR NOT transformer_lab_c_abi_header_minor
       STREQUAL
       "#define TL_ABI_VERSION_MINOR UINT32_C(${transformer_lab_c_abi_minor_version})")
    message(FATAL_ERROR
        "CMake C ABI version and c_api.h TL_ABI_VERSION must match"
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
    ${CMAKE_CURRENT_SOURCE_DIR}/python/pyproject.toml
    transformer_lab_python_version_line
    REGEX "^version = \"[0-9]+[.][0-9]+[.][0-9]+\"$"
)
if(NOT transformer_lab_python_version_line
       STREQUAL "version = \"${PROJECT_VERSION}\"")
    message(FATAL_ERROR
        "CMake and Python package versions must match"
    )
endif()

include(cmake/TransformerLabWarnings.cmake)
include(cmake/TransformerLabSanitizers.cmake)
include(cmake/TransformerLabBackends.cmake)

option(
    TRANSFORMER_LAB_BUILD_CLI
    "Build the transformer_lab training executable"
    ${PROJECT_IS_TOP_LEVEL}
)
option(
    TRANSFORMER_LAB_ENABLE_INSTALL
    "Generate install rules and an exported CMake package"
    ${PROJECT_IS_TOP_LEVEL}
)

add_library(transformer_lab_library STATIC
    src/config.cpp
    ${transformer_lab_backend_sources}
    src/artifacts/state.cpp
    src/core/autograd.cpp
    src/core/tensor.cpp
    src/core/tensor_ops.cpp
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
    src/stages/post_training/config.cpp
    src/stages/post_training/instruction.cpp
    src/stages/post_training/stack.cpp
    src/stages/pretraining/config.cpp
    src/stages/pretraining/stack.cpp
    src/stages/serving/config.cpp
    src/stages/serving/cache/contiguous_kv_cache.cpp
    src/stages/serving/cache/page_storage.cpp
    src/stages/serving/cache/page_table_cache.cpp
    src/stages/serving/cache/paged_kv_cache.cpp
    src/stages/serving/cache/validation.cpp
    src/stages/serving/generation.cpp
    src/stages/serving/stack.cpp
    src/training/adam_optimizer_adapter.cpp
    src/training/batch_source.cpp
    src/training/causal_language_model_trainer.cpp
)

set_target_properties(transformer_lab_library
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        OBJCXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON
        EXPORT_NAME library
        OUTPUT_NAME transformer_lab
)

add_library(transformer_lab::library ALIAS transformer_lab_library)

target_include_directories(transformer_lab_library
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_features(transformer_lab_library
    PUBLIC
        cxx_std_20
)
target_link_libraries(transformer_lab_library
    PRIVATE
        Threads::Threads
)

transformer_lab_enable_warnings(transformer_lab_library)
transformer_lab_enable_sanitizers(transformer_lab_library)
transformer_lab_propagate_sanitizer_runtime(transformer_lab_library)

if(TRANSFORMER_LAB_ENABLE_METAL)
    target_link_libraries(transformer_lab_library
        PUBLIC
            "-framework Foundation"
            "-framework Metal"
    )
endif()

add_library(transformer_lab_c SHARED
    src/c_api.cpp
)

add_library(transformer_lab::c_api ALIAS transformer_lab_c)

target_compile_definitions(transformer_lab_c
    PRIVATE
        TRANSFORMER_LAB_C_EXPORTS
)

target_include_directories(transformer_lab_c
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(transformer_lab_c
    INTERFACE
        c_std_11
)

target_link_libraries(transformer_lab_c
    PRIVATE
        transformer_lab::library
)

set_target_properties(transformer_lab_c
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        EXPORT_NAME c_api
        OUTPUT_NAME transformer_lab_c
        VERSION ${TRANSFORMER_LAB_C_ABI_VERSION}
        SOVERSION ${TRANSFORMER_LAB_C_ABI_SOVERSION}
)

transformer_lab_enable_warnings(transformer_lab_c)
transformer_lab_enable_sanitizers(transformer_lab_c)
transformer_lab_propagate_sanitizer_runtime(transformer_lab_c)

if(TRANSFORMER_LAB_BUILD_CLI)
    add_executable(transformer_lab
        apps/pretraining/train.cpp
    )

    target_link_libraries(transformer_lab
        PRIVATE
            transformer_lab::library
    )

    transformer_lab_enable_warnings(transformer_lab)
    transformer_lab_enable_sanitizers(transformer_lab)
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
    set(transformer_lab_tests_default ${BUILD_TESTING})
else()
    set(transformer_lab_tests_default OFF)
endif()
option(
    TRANSFORMER_LAB_BUILD_TESTS
    "Build transformer_lab's own test suite"
    ${transformer_lab_tests_default}
)

if(TRANSFORMER_LAB_BUILD_TESTS)
    if(NOT PROJECT_IS_TOP_LEVEL)
        enable_testing()
    endif()
    add_subdirectory(tests)
    target_compile_definitions(transformer_lab_c
        PRIVATE
            TRANSFORMER_LAB_C_API_TESTING
    )
    target_compile_definitions(c_api_tests
        PRIVATE
            TRANSFORMER_LAB_C_API_TESTING
    )
endif()

if(TRANSFORMER_LAB_ENABLE_INSTALL)
    include(cmake/TransformerLabInstall.cmake)
endif()
