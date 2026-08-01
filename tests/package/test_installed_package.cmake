foreach(required_variable IN ITEMS
        RIFTCO_TRANSFORMER_BUILD_DIR
        RIFTCO_TRANSFORMER_CONSUMER_SOURCE_DIR
        RIFTCO_TRANSFORMER_PROJECT_VERSION
        RIFTCO_TRANSFORMER_TEST_ROOT
        RIFTCO_TRANSFORMER_GENERATOR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

set(install_prefix "${RIFTCO_TRANSFORMER_TEST_ROOT}/install")
set(consumer_build "${RIFTCO_TRANSFORMER_TEST_ROOT}/consumer")
file(REMOVE_RECURSE
    "${install_prefix}"
    "${consumer_build}"
)

set(cmake_config_arguments)
set(ctest_config_arguments)
set(configure_config_argument)
if(DEFINED RIFTCO_TRANSFORMER_BUILD_CONFIG AND
   NOT RIFTCO_TRANSFORMER_BUILD_CONFIG STREQUAL "")
    list(APPEND cmake_config_arguments
        --config "${RIFTCO_TRANSFORMER_BUILD_CONFIG}"
    )
    list(APPEND ctest_config_arguments
        -C "${RIFTCO_TRANSFORMER_BUILD_CONFIG}"
    )
    list(APPEND configure_config_argument
        "-DCMAKE_BUILD_TYPE=${RIFTCO_TRANSFORMER_BUILD_CONFIG}"
    )
endif()

set(generator_arguments
    -G "${RIFTCO_TRANSFORMER_GENERATOR}"
)
if(DEFINED RIFTCO_TRANSFORMER_GENERATOR_PLATFORM AND
   NOT RIFTCO_TRANSFORMER_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND generator_arguments
        -A "${RIFTCO_TRANSFORMER_GENERATOR_PLATFORM}"
    )
endif()
if(DEFINED RIFTCO_TRANSFORMER_GENERATOR_TOOLSET AND
   NOT RIFTCO_TRANSFORMER_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND generator_arguments
        -T "${RIFTCO_TRANSFORMER_GENERATOR_TOOLSET}"
    )
endif()

set(toolchain_arguments)
if(DEFINED RIFTCO_TRANSFORMER_TOOLCHAIN_FILE AND
   NOT RIFTCO_TRANSFORMER_TOOLCHAIN_FILE STREQUAL "")
    set(toolchain_file "${RIFTCO_TRANSFORMER_TOOLCHAIN_FILE}")
    if(NOT IS_ABSOLUTE "${toolchain_file}")
        if(EXISTS
           "${RIFTCO_TRANSFORMER_BUILD_DIR}/${toolchain_file}")
            set(toolchain_file
                "${RIFTCO_TRANSFORMER_BUILD_DIR}/${toolchain_file}"
            )
        elseif(DEFINED RIFTCO_TRANSFORMER_PROVIDER_SOURCE_DIR AND
               EXISTS
               "${RIFTCO_TRANSFORMER_PROVIDER_SOURCE_DIR}/${toolchain_file}")
            set(toolchain_file
                "${RIFTCO_TRANSFORMER_PROVIDER_SOURCE_DIR}/${toolchain_file}"
            )
        else()
            message(FATAL_ERROR
                "cannot resolve provider toolchain file: "
                "${RIFTCO_TRANSFORMER_TOOLCHAIN_FILE}"
            )
        endif()
    endif()
    cmake_path(NORMAL_PATH toolchain_file)
    list(APPEND toolchain_arguments
        "-DCMAKE_TOOLCHAIN_FILE=${toolchain_file}"
    )
else()
    foreach(variable IN ITEMS C_COMPILER CXX_COMPILER)
        if(DEFINED RIFTCO_TRANSFORMER_${variable} AND
           NOT RIFTCO_TRANSFORMER_${variable} STREQUAL "")
            string(REPLACE ";" "\\;" escaped_value
                "${RIFTCO_TRANSFORMER_${variable}}"
            )
            list(APPEND toolchain_arguments
                "-DCMAKE_${variable}=${escaped_value}"
            )
        endif()
    endforeach()
endif()

foreach(variable IN ITEMS
        OSX_ARCHITECTURES
        OSX_DEPLOYMENT_TARGET
        OSX_SYSROOT
        MSVC_RUNTIME_LIBRARY)
    if(DEFINED RIFTCO_TRANSFORMER_${variable} AND
       NOT RIFTCO_TRANSFORMER_${variable} STREQUAL "")
        string(REPLACE ";" "\\;" escaped_value
            "${RIFTCO_TRANSFORMER_${variable}}"
        )
        list(APPEND toolchain_arguments
            "-DCMAKE_${variable}=${escaped_value}"
        )
    endif()
endforeach()

if(RIFTCO_TRANSFORMER_CROSSCOMPILING)
    foreach(variable IN ITEMS SYSTEM_NAME SYSROOT)
        if(DEFINED RIFTCO_TRANSFORMER_${variable} AND
           NOT RIFTCO_TRANSFORMER_${variable} STREQUAL "")
            string(REPLACE ";" "\\;" escaped_value
                "${RIFTCO_TRANSFORMER_${variable}}"
            )
            list(APPEND toolchain_arguments
                "-DCMAKE_${variable}=${escaped_value}"
            )
        endif()
    endforeach()
endif()
if(RIFTCO_TRANSFORMER_CROSSCOMPILING AND
   DEFINED RIFTCO_TRANSFORMER_CROSSCOMPILING_EMULATOR AND
   NOT RIFTCO_TRANSFORMER_CROSSCOMPILING_EMULATOR STREQUAL "")
    string(REPLACE ";" "\\;"
        crosscompiling_emulator_escaped
        "${RIFTCO_TRANSFORMER_CROSSCOMPILING_EMULATOR}"
    )
    list(APPEND toolchain_arguments
        "-DCMAKE_CROSSCOMPILING_EMULATOR=${crosscompiling_emulator_escaped}"
    )
endif()
foreach(language IN ITEMS C CXX)
    if(DEFINED RIFTCO_TRANSFORMER_${language}_FLAGS AND
       NOT RIFTCO_TRANSFORMER_${language}_FLAGS STREQUAL "")
        string(REPLACE ";" "\\;" escaped_flags
            "${RIFTCO_TRANSFORMER_${language}_FLAGS}"
        )
        list(APPEND toolchain_arguments
            "-DCMAKE_${language}_FLAGS=${escaped_flags}"
        )
    endif()
endforeach()
if(DEFINED RIFTCO_TRANSFORMER_EXE_LINKER_FLAGS AND
   NOT RIFTCO_TRANSFORMER_EXE_LINKER_FLAGS STREQUAL "")
    string(REPLACE ";" "\\;" escaped_linker_flags
        "${RIFTCO_TRANSFORMER_EXE_LINKER_FLAGS}"
    )
    list(APPEND toolchain_arguments
        "-DCMAKE_EXE_LINKER_FLAGS=${escaped_linker_flags}"
    )
endif()
if(DEFINED RIFTCO_TRANSFORMER_CUDA_TOOLKIT_ROOT AND
   NOT RIFTCO_TRANSFORMER_CUDA_TOOLKIT_ROOT STREQUAL "")
    list(APPEND toolchain_arguments
        "-DCUDAToolkit_ROOT=${RIFTCO_TRANSFORMER_CUDA_TOOLKIT_ROOT}"
    )
endif()

set(consumer_prefix_path "${install_prefix}")
if(DEFINED RIFTCO_TRANSFORMER_CMAKE_PREFIX_PATH AND
   NOT RIFTCO_TRANSFORMER_CMAKE_PREFIX_PATH STREQUAL "")
    list(APPEND consumer_prefix_path
        ${RIFTCO_TRANSFORMER_CMAKE_PREFIX_PATH}
    )
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
        --install "${RIFTCO_TRANSFORMER_BUILD_DIR}"
        --prefix "${install_prefix}"
        ${cmake_config_arguments}
    RESULT_VARIABLE install_result
    OUTPUT_VARIABLE install_output
    ERROR_VARIABLE install_error
)
if(NOT install_result EQUAL 0)
    message(FATAL_ERROR
        "framework install failed with ${install_result}\n"
        "stdout:\n${install_output}\n"
        "stderr:\n${install_error}"
    )
endif()

file(GLOB_RECURSE installed_package_configs
    LIST_DIRECTORIES FALSE
    "${install_prefix}/*/riftco_transformerConfig.cmake"
)
list(LENGTH installed_package_configs installed_package_config_count)
if(NOT installed_package_config_count EQUAL 1)
    message(FATAL_ERROR
        "expected exactly one installed package config, found "
        "${installed_package_config_count}: ${installed_package_configs}"
    )
endif()
list(GET installed_package_configs 0 installed_package_config)
get_filename_component(
    installed_package_config_dir
    "${installed_package_config}"
    DIRECTORY
)

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
        -S "${RIFTCO_TRANSFORMER_CONSUMER_SOURCE_DIR}"
        -B "${consumer_build}"
        ${generator_arguments}
        "-DCMAKE_PREFIX_PATH=${consumer_prefix_path}"
        "-DRIFTCO_TRANSFORMER_REQUIRED_VERSION=${RIFTCO_TRANSFORMER_PROJECT_VERSION}"
        "-Driftco_transformer_DIR=${installed_package_config_dir}"
        ${configure_config_argument}
        ${toolchain_arguments}
    RESULT_VARIABLE configure_result
    OUTPUT_VARIABLE configure_output
    ERROR_VARIABLE configure_error
)
if(NOT configure_result EQUAL 0)
    message(FATAL_ERROR
        "consumer configure failed with ${configure_result}\n"
        "stdout:\n${configure_output}\n"
        "stderr:\n${configure_error}"
    )
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
        --build "${consumer_build}"
        ${cmake_config_arguments}
    RESULT_VARIABLE build_result
    OUTPUT_VARIABLE build_output
    ERROR_VARIABLE build_error
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR
        "consumer build failed with ${build_result}\n"
        "stdout:\n${build_output}\n"
        "stderr:\n${build_error}"
    )
endif()

if(NOT RIFTCO_TRANSFORMER_CROSSCOMPILING OR
   (DEFINED RIFTCO_TRANSFORMER_CROSSCOMPILING_EMULATOR AND
    NOT RIFTCO_TRANSFORMER_CROSSCOMPILING_EMULATOR STREQUAL ""))
    execute_process(
        COMMAND
            "${CMAKE_CTEST_COMMAND}"
            --test-dir "${consumer_build}"
            --output-on-failure
            ${ctest_config_arguments}
        RESULT_VARIABLE test_result
        OUTPUT_VARIABLE test_output
        ERROR_VARIABLE test_error
    )
    if(NOT test_result EQUAL 0)
        message(FATAL_ERROR
            "consumer tests failed with ${test_result}\n"
            "stdout:\n${test_output}\n"
            "stderr:\n${test_error}"
        )
    endif()
else()
    message(STATUS
        "consumer executables built; execution skipped while "
        "cross-compiling without an emulator"
    )
endif()
