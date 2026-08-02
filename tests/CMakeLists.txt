function(add_riftco_transformer_component_test
         target test_name source component)
    add_executable(${target}
        ${source}
    )

    target_link_libraries(${target}
        PRIVATE
            ${component}
    )

    riftco_transformer_enable_warnings(${target})
    riftco_transformer_enable_sanitizers(${target})

    add_test(
        NAME ${test_name}
        COMMAND ${target} ${ARGN}
    )
endfunction()

function(add_riftco_transformer_test target test_name source)
    add_riftco_transformer_component_test(
        ${target}
        ${test_name}
        ${source}
        riftco_transformer::library
        ${ARGN}
    )
endfunction()

option(
    RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL
    "Fail the NN backend test when no Metal device is available"
    OFF
)
option(
    RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA
    "Fail CUDA-aware native tests when no CUDA device is available"
    OFF
)
option(
    RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU
    "Fail TPU-aware native tests when no Cloud TPU device is available"
    OFF
)

add_executable(c_api_tests
    abi/test_c_api.c
)

target_link_libraries(c_api_tests
    PRIVATE
        riftco_transformer::c_api
)

riftco_transformer_enable_warnings(c_api_tests)
riftco_transformer_enable_sanitizers(c_api_tests)
target_compile_definitions(c_api_tests
    PRIVATE
        RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA=$<BOOL:${RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA}>
        RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU=$<BOOL:${RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU}>
)

add_test(
    NAME c_api
    COMMAND c_api_tests
)

if(WIN32)
    set_tests_properties(c_api PROPERTIES
        ENVIRONMENT_MODIFICATION
            "PATH=path_list_prepend:$<TARGET_FILE_DIR:riftco_transformer_c>"
    )
endif()

option(
    RIFTCO_TRANSFORMER_BUILD_PYTHON_TESTS
    "Build and run Python binding, workflow, and lab tests"
    ON
)
if(RIFTCO_TRANSFORMER_BUILD_PYTHON_TESTS)
    find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)
    set(python_test_environment
        RIFTCO_TRANSFORMER_LIBRARY=$<TARGET_FILE:riftco_transformer_c>
        PYTHONPATH=${PROJECT_SOURCE_DIR}/python
    )
    set(python_runtime_environment)
    set(python_binding_test_supported TRUE)

    # An ASan-instrumented shared library must load the ASan runtime before
    # the uninstrumented Python executable dlopens the binding.
    if(RIFTCO_TRANSFORMER_ENABLE_SANITIZERS OR
       CMAKE_CXX_FLAGS MATCHES "-fsanitize=[^ ]*address")
        if(APPLE)
            execute_process(
                COMMAND
                    ${CMAKE_CXX_COMPILER}
                    -print-file-name=libclang_rt.asan_osx_dynamic.dylib
                OUTPUT_VARIABLE riftco_transformer_asan_runtime
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        elseif(UNIX)
            execute_process(
                COMMAND
                    ${CMAKE_CXX_COMPILER}
                    -print-file-name=libasan.so
                OUTPUT_VARIABLE riftco_transformer_asan_runtime
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
        endif()

        if(EXISTS "${riftco_transformer_asan_runtime}")
            if(APPLE)
                list(APPEND python_runtime_environment
                    DYLD_INSERT_LIBRARIES=${riftco_transformer_asan_runtime}
                    # Homebrew's framework Python pre-binds a few libc
                    # symbols. Instrumented allocations remain checked.
                    ASAN_OPTIONS=verify_interceptors=0
                )
            elseif(UNIX)
                list(APPEND python_runtime_environment
                    LD_PRELOAD=${riftco_transformer_asan_runtime}
                )
            endif()
        elseif(UNIX)
            set(python_binding_test_supported FALSE)
            message(WARNING
                "Skipping the Python tests under AddressSanitizer: "
                "the compiler ASan runtime could not be located for preload"
            )
        endif()
    endif()

    if(python_binding_test_supported AND
       python_runtime_environment)
        execute_process(
            COMMAND
                ${CMAKE_COMMAND} -E env
                ${python_runtime_environment}
                ${Python3_EXECUTABLE}
                -c "pass"
            RESULT_VARIABLE python_preload_result
            OUTPUT_QUIET
            ERROR_VARIABLE python_preload_error
        )
        if(NOT python_preload_result EQUAL 0)
            set(python_binding_test_supported FALSE)
            string(STRIP
                "${python_preload_error}"
                python_preload_error
            )
            message(WARNING
                "Skipping the Python tests under sanitizers: "
                "the selected interpreter cannot start with the sanitizer "
                "runtime preloaded (${python_preload_error})"
            )
        endif()
    endif()

    if(python_binding_test_supported)
        list(APPEND python_test_environment
            ${python_runtime_environment}
        )
        add_test(
            NAME python_binding
            COMMAND
                ${CMAKE_COMMAND} -E env
                ${python_test_environment}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/python/test_python_binding.py
        )
        add_test(
            NAME python_generation
            COMMAND
                ${CMAKE_COMMAND} -E env
                ${python_test_environment}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/python/test_generation.py
        )
        add_test(
            NAME python_training_example
            COMMAND
                ${CMAKE_COMMAND} -E env
                ${python_test_environment}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/python/test_train_tiny.py
        )
        add_test(
            NAME python_stage_stack
            COMMAND
                ${CMAKE_COMMAND} -E env
                ${python_test_environment}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/python/test_stage_stack.py
        )
        add_test(
            NAME python_package_structure
            COMMAND
                ${CMAKE_COMMAND} -E env
                ${python_test_environment}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/python/test_package_structure.py
        )
        add_test(
            NAME python_huggingface_data
            COMMAND
                ${CMAKE_COMMAND} -E env
                ${python_test_environment}
                ${Python3_EXECUTABLE}
                ${CMAKE_CURRENT_SOURCE_DIR}/python/test_huggingface_data.py
        )
        add_test(
            NAME python_lora_rank_lab
            COMMAND
                ${CMAKE_COMMAND} -E env
                ${python_test_environment}
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/labs/lora_rank/tests/test_protocol.py
        )
        add_test(
            NAME python_fine_tuning_lab
            COMMAND
                ${CMAKE_COMMAND} -E env
                ${python_test_environment}
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/labs/fine_tuning/tests/test_protocol.py
        )
        add_test(
            NAME python_conditional_reverse_lab
            COMMAND
                ${CMAKE_COMMAND} -E env
                ${python_test_environment}
                ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/labs/conditional_reverse/tests/test_protocol.py
        )
        set_tests_properties(
            python_binding
            python_generation
            python_training_example
            python_stage_stack
            python_package_structure
            python_huggingface_data
            python_lora_rank_lab
            python_fine_tuning_lab
            python_conditional_reverse_lab
            PROPERTIES
                TIMEOUT 120
        )
    endif()
endif()

if(RIFTCO_TRANSFORMER_ENABLE_INSTALL)
    string(REPLACE ";" "\\;"
        riftco_transformer_test_cmake_prefix_path
        "${CMAKE_PREFIX_PATH}"
    )
    string(REPLACE ";" "\\;"
        riftco_transformer_test_cuda_toolkit_root
        "${CUDAToolkit_ROOT}"
    )
    add_test(
        NAME installed_package
        COMMAND
            ${CMAKE_COMMAND}
            -DRIFTCO_TRANSFORMER_BUILD_DIR=${PROJECT_BINARY_DIR}
            -DRIFTCO_TRANSFORMER_BUILD_CONFIG=$<CONFIG>
            -DRIFTCO_TRANSFORMER_CONSUMER_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/package
            -DRIFTCO_TRANSFORMER_PROJECT_VERSION=${PROJECT_VERSION}
            -DRIFTCO_TRANSFORMER_TEST_ROOT=${CMAKE_CURRENT_BINARY_DIR}/installed_package
            -DRIFTCO_TRANSFORMER_GENERATOR=${CMAKE_GENERATOR}
            "-DRIFTCO_TRANSFORMER_GENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}"
            "-DRIFTCO_TRANSFORMER_GENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}"
            "-DRIFTCO_TRANSFORMER_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
            "-DRIFTCO_TRANSFORMER_PROVIDER_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
            "-DRIFTCO_TRANSFORMER_C_COMPILER=${CMAKE_C_COMPILER}"
            "-DRIFTCO_TRANSFORMER_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
            "-DRIFTCO_TRANSFORMER_CROSSCOMPILING=${CMAKE_CROSSCOMPILING}"
            "-DRIFTCO_TRANSFORMER_CROSSCOMPILING_EMULATOR=${CMAKE_CROSSCOMPILING_EMULATOR}"
            "-DRIFTCO_TRANSFORMER_SYSTEM_NAME=${CMAKE_SYSTEM_NAME}"
            "-DRIFTCO_TRANSFORMER_SYSROOT=${CMAKE_SYSROOT}"
            "-DRIFTCO_TRANSFORMER_OSX_ARCHITECTURES=${CMAKE_OSX_ARCHITECTURES}"
            "-DRIFTCO_TRANSFORMER_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}"
            "-DRIFTCO_TRANSFORMER_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}"
            "-DRIFTCO_TRANSFORMER_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
            "-DRIFTCO_TRANSFORMER_C_FLAGS=${CMAKE_C_FLAGS}"
            "-DRIFTCO_TRANSFORMER_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
            "-DRIFTCO_TRANSFORMER_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
            "-DRIFTCO_TRANSFORMER_CMAKE_PREFIX_PATH=${riftco_transformer_test_cmake_prefix_path}"
            "-DRIFTCO_TRANSFORMER_CUDA_TOOLKIT_ROOT=${riftco_transformer_test_cuda_toolkit_root}"
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/package/test_installed_package.cmake
    )
endif()

add_riftco_transformer_test(
    tensor_tests
    tensor
    core/test_tensor.cpp
)

add_riftco_transformer_test(
    tensor_ops_tests
    tensor_ops
    core/test_tensor_ops.cpp
)

add_riftco_transformer_test(
    quantization_tests
    quantization
    core/test_quantization.cpp
)

add_riftco_transformer_test(
    backend_tests
    backend
    core/backend/test_backend.cpp
)
target_compile_definitions(backend_tests
    PRIVATE
        RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA=$<BOOL:${RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA}>
        RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU=$<BOOL:${RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU}>
)
target_include_directories(backend_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_link_libraries(c_api_tests PRIVATE Threads::Threads)
target_link_libraries(backend_tests PRIVATE Threads::Threads)

if(RIFTCO_TRANSFORMER_ENABLE_TPU)
    add_library(riftco_transformer_fake_pjrt_tpu SHARED
        fakes/fake_pjrt_tpu.cpp
    )
    target_include_directories(riftco_transformer_fake_pjrt_tpu
        PRIVATE
            ${PROJECT_SOURCE_DIR}/third_party/pjrt/include
    )
    target_compile_features(riftco_transformer_fake_pjrt_tpu
        PRIVATE
            cxx_std_20
    )
    riftco_transformer_enable_warnings(
        riftco_transformer_fake_pjrt_tpu
    )
    riftco_transformer_enable_pjrt_header_compatibility(
        riftco_transformer_fake_pjrt_tpu
    )
    riftco_transformer_enable_sanitizers(
        riftco_transformer_fake_pjrt_tpu
    )
    set_target_properties(riftco_transformer_fake_pjrt_tpu
        PROPERTIES
            CXX_VISIBILITY_PRESET hidden
            VISIBILITY_INLINES_HIDDEN YES
    )
    add_dependencies(
        backend_tests
        riftco_transformer_fake_pjrt_tpu
    )

    add_test(
        NAME backend_fake_pjrt_tpu
        COMMAND
            ${CMAKE_COMMAND} -E env
            RIFTCO_TRANSFORMER_TPU_LIBRARY=$<TARGET_FILE:riftco_transformer_fake_pjrt_tpu>
            $<TARGET_FILE:backend_tests>
            --require-tpu
    )
    set_tests_properties(backend_fake_pjrt_tpu
        PROPERTIES
            LABELS "backend;tpu;fake-pjrt"
    )
endif()

add_riftco_transformer_test(
    nn_backend_tests
    nn_backend
    core/backend/test_nn_backend.cpp
)

add_riftco_transformer_test(
    quantized_linear_backend_tests
    quantized_linear_backend
    core/backend/test_quantized_linear_backend.cpp
)
target_include_directories(quantized_linear_backend_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_compile_definitions(quantized_linear_backend_tests
    PRIVATE
        RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL=$<BOOL:${RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL}>
        RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA=$<BOOL:${RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA}>
        RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU=$<BOOL:${RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU}>
)
if(RIFTCO_TRANSFORMER_ENABLE_TPU)
    add_dependencies(
        quantized_linear_backend_tests
        riftco_transformer_fake_pjrt_tpu
    )
    add_test(
        NAME quantized_linear_backend_fake_pjrt_tpu
        COMMAND
            ${CMAKE_COMMAND} -E env
            RIFTCO_TRANSFORMER_TPU_LIBRARY=$<TARGET_FILE:riftco_transformer_fake_pjrt_tpu>
            $<TARGET_FILE:quantized_linear_backend_tests>
            --require-tpu
    )
    set_tests_properties(quantized_linear_backend_fake_pjrt_tpu
        PROPERTIES
            LABELS "backend;quantization;tpu;fake-pjrt"
    )
endif()
target_include_directories(nn_backend_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)
target_compile_definitions(nn_backend_tests
    PRIVATE
        RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL=$<BOOL:${RIFTCO_TRANSFORMER_TEST_REQUIRE_METAL}>
        RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA=$<BOOL:${RIFTCO_TRANSFORMER_TEST_REQUIRE_CUDA}>
        RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU=$<BOOL:${RIFTCO_TRANSFORMER_TEST_REQUIRE_TPU}>
)

if(RIFTCO_TRANSFORMER_ENABLE_TPU)
    add_dependencies(
        nn_backend_tests
        riftco_transformer_fake_pjrt_tpu
    )
    add_test(
        NAME nn_backend_fake_pjrt_tpu
        COMMAND
            ${CMAKE_COMMAND} -E env
            RIFTCO_TRANSFORMER_TPU_LIBRARY=$<TARGET_FILE:riftco_transformer_fake_pjrt_tpu>
            $<TARGET_FILE:nn_backend_tests>
            --require-tpu
    )
    set_tests_properties(nn_backend_fake_pjrt_tpu
        PROPERTIES
            LABELS "backend;attention;tpu;fake-pjrt"
    )
endif()

add_riftco_transformer_test(
    autograd_tests
    autograd
    core/test_autograd.cpp
)

add_riftco_transformer_component_test(
    cajal_compiler_tests
    cajal_compiler
    compiler/cajal/test_cajal.cpp
    riftco_transformer::compiler
)
set_tests_properties(cajal_compiler
    PROPERTIES
        LABELS "compiler;cajal"
)

add_riftco_transformer_component_test(
    cajal_multilinear_compiler_tests
    cajal_multilinear_compiler
    compiler/cajal/test_multilinear_compiler.cpp
    riftco_transformer::compiler
)
set_tests_properties(cajal_multilinear_compiler
    PROPERTIES
        LABELS "compiler;cajal;multilinear"
)

add_riftco_transformer_component_test(
    cajal_neural_lowering_tests
    cajal_neural_lowering
    lowering/test_cajal_neural_lowering.cpp
    riftco_transformer::lowering
)
set_tests_properties(cajal_neural_lowering
    PROPERTIES
        LABELS "compiler;cajal;neural-lowering"
)

add_riftco_transformer_component_test(
    analysis_pca_tests
    analysis_pca
    analysis/test_pca.cpp
    riftco_transformer::analysis
)
set_tests_properties(analysis_pca
    PROPERTIES
        LABELS "analysis;interpretability;pca"
)

add_riftco_transformer_component_test(
    analysis_intervention_ablation_tests
    analysis_intervention_ablation
    analysis/test_intervention_ablation.cpp
    riftco_transformer::analysis
)
set_tests_properties(analysis_intervention_ablation
    PROPERTIES
        LABELS "analysis;interpretability;intervention;ablation"
)

add_riftco_transformer_component_test(
    analysis_representation_tests
    analysis_representation
    analysis/test_representation.cpp
    riftco_transformer::analysis
)
set_tests_properties(analysis_representation
    PROPERTIES
        LABELS "analysis;interpretability;representation"
)

add_riftco_transformer_test(
    tokenizer_tests
    tokenizer
    data/test_tokenizer.cpp
    ${PROJECT_SOURCE_DIR}/data/pretraining/tiny_corpus.txt
)

add_riftco_transformer_test(
    token_batch_tests
    token_batch
    data/test_token_batch.cpp
)

add_riftco_transformer_test(
    layer_tests
    layers
    nn/test_layers.cpp
)

add_riftco_transformer_test(
    module_tests
    module
    nn/test_module.cpp
)

add_riftco_transformer_test(
    neural_ops_tests
    neural_ops
    nn/test_neural_ops.cpp
)

add_riftco_transformer_test(
    lora_layer_tests
    lora_layers
    nn/test_lora.cpp
)

add_riftco_transformer_test(
    quantized_linear_tests
    quantized_linear
    nn/test_quantized_linear.cpp
)

add_riftco_transformer_test(
    causal_self_attention_tests
    causal_self_attention
    model/test_causal_self_attention.cpp
)

add_riftco_transformer_test(
    transformer_block_tests
    transformer_block
    model/test_transformer_block.cpp
)

add_riftco_transformer_test(
    decoder_only_transformer_tests
    decoder_only_transformer
    model/test_decoder_only_transformer.cpp
)

add_riftco_transformer_test(
    lora_model_tests
    lora_model
    model/test_lora_model.cpp
)

add_riftco_transformer_test(
    qlora_model_tests
    qlora_model
    model/test_qlora_model.cpp
)

add_riftco_transformer_test(
    adam_tests
    adam
    optim/test_adam.cpp
)
target_include_directories(adam_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/src
)

if(RIFTCO_TRANSFORMER_ENABLE_TPU)
    add_dependencies(
        adam_tests
        riftco_transformer_fake_pjrt_tpu
    )
    add_test(
        NAME adam_fake_pjrt_tpu
        COMMAND
            ${CMAKE_COMMAND} -E env
            RIFTCO_TRANSFORMER_TPU_LIBRARY=$<TARGET_FILE:riftco_transformer_fake_pjrt_tpu>
            $<TARGET_FILE:adam_tests>
            --require-tpu
    )
    set_tests_properties(adam_fake_pjrt_tpu
        PROPERTIES
            LABELS "backend;optim;adam;tpu;fake-pjrt"
    )
endif()

add_riftco_transformer_test(
    artifact_state_tests
    artifact_state
    artifacts/test_state_contract.cpp
)

add_riftco_transformer_test(
    native_serving_generation_tests
    native_serving_generation
    stages/test_native_serving_generation.cpp
)
