foreach(required_variable IN ITEMS
        TRANSFORMER_LAB_BUILD_DIR
        TRANSFORMER_LAB_CONSUMER_SOURCE_DIR
        TRANSFORMER_LAB_TEST_ROOT
        TRANSFORMER_LAB_GENERATOR)
    if(NOT DEFINED ${required_variable})
        message(FATAL_ERROR "${required_variable} is required")
    endif()
endforeach()

set(install_prefix "${TRANSFORMER_LAB_TEST_ROOT}/install")
set(consumer_build "${TRANSFORMER_LAB_TEST_ROOT}/consumer")
file(REMOVE_RECURSE
    "${install_prefix}"
    "${consumer_build}"
)

set(cmake_config_arguments)
set(ctest_config_arguments)
set(configure_config_argument)
if(DEFINED TRANSFORMER_LAB_BUILD_CONFIG AND
   NOT TRANSFORMER_LAB_BUILD_CONFIG STREQUAL "")
    list(APPEND cmake_config_arguments
        --config "${TRANSFORMER_LAB_BUILD_CONFIG}"
    )
    list(APPEND ctest_config_arguments
        -C "${TRANSFORMER_LAB_BUILD_CONFIG}"
    )
    list(APPEND configure_config_argument
        "-DCMAKE_BUILD_TYPE=${TRANSFORMER_LAB_BUILD_CONFIG}"
    )
endif()

set(generator_arguments
    -G "${TRANSFORMER_LAB_GENERATOR}"
)
if(DEFINED TRANSFORMER_LAB_GENERATOR_PLATFORM AND
   NOT TRANSFORMER_LAB_GENERATOR_PLATFORM STREQUAL "")
    list(APPEND generator_arguments
        -A "${TRANSFORMER_LAB_GENERATOR_PLATFORM}"
    )
endif()
if(DEFINED TRANSFORMER_LAB_GENERATOR_TOOLSET AND
   NOT TRANSFORMER_LAB_GENERATOR_TOOLSET STREQUAL "")
    list(APPEND generator_arguments
        -T "${TRANSFORMER_LAB_GENERATOR_TOOLSET}"
    )
endif()

set(toolchain_arguments)
if(DEFINED TRANSFORMER_LAB_TOOLCHAIN_FILE AND
   NOT TRANSFORMER_LAB_TOOLCHAIN_FILE STREQUAL "")
    set(toolchain_file "${TRANSFORMER_LAB_TOOLCHAIN_FILE}")
    if(NOT IS_ABSOLUTE "${toolchain_file}")
        if(EXISTS
           "${TRANSFORMER_LAB_BUILD_DIR}/${toolchain_file}")
            set(toolchain_file
                "${TRANSFORMER_LAB_BUILD_DIR}/${toolchain_file}"
            )
        elseif(DEFINED TRANSFORMER_LAB_PROVIDER_SOURCE_DIR AND
               EXISTS
               "${TRANSFORMER_LAB_PROVIDER_SOURCE_DIR}/${toolchain_file}")
            set(toolchain_file
                "${TRANSFORMER_LAB_PROVIDER_SOURCE_DIR}/${toolchain_file}"
            )
        else()
            message(FATAL_ERROR
                "cannot resolve provider toolchain file: "
                "${TRANSFORMER_LAB_TOOLCHAIN_FILE}"
            )
        endif()
    endif()
    cmake_path(NORMAL_PATH toolchain_file)
    list(APPEND toolchain_arguments
        "-DCMAKE_TOOLCHAIN_FILE=${toolchain_file}"
    )
else()
    foreach(variable IN ITEMS C_COMPILER CXX_COMPILER)
        if(DEFINED TRANSFORMER_LAB_${variable} AND
           NOT TRANSFORMER_LAB_${variable} STREQUAL "")
            string(REPLACE ";" "\\;" escaped_value
                "${TRANSFORMER_LAB_${variable}}"
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
    if(DEFINED TRANSFORMER_LAB_${variable} AND
       NOT TRANSFORMER_LAB_${variable} STREQUAL "")
        string(REPLACE ";" "\\;" escaped_value
            "${TRANSFORMER_LAB_${variable}}"
        )
        list(APPEND toolchain_arguments
            "-DCMAKE_${variable}=${escaped_value}"
        )
    endif()
endforeach()

if(TRANSFORMER_LAB_CROSSCOMPILING)
    foreach(variable IN ITEMS SYSTEM_NAME SYSROOT)
        if(DEFINED TRANSFORMER_LAB_${variable} AND
           NOT TRANSFORMER_LAB_${variable} STREQUAL "")
            string(REPLACE ";" "\\;" escaped_value
                "${TRANSFORMER_LAB_${variable}}"
            )
            list(APPEND toolchain_arguments
                "-DCMAKE_${variable}=${escaped_value}"
            )
        endif()
    endforeach()
endif()
if(TRANSFORMER_LAB_CROSSCOMPILING AND
   DEFINED TRANSFORMER_LAB_CROSSCOMPILING_EMULATOR AND
   NOT TRANSFORMER_LAB_CROSSCOMPILING_EMULATOR STREQUAL "")
    string(REPLACE ";" "\\;"
        crosscompiling_emulator_escaped
        "${TRANSFORMER_LAB_CROSSCOMPILING_EMULATOR}"
    )
    list(APPEND toolchain_arguments
        "-DCMAKE_CROSSCOMPILING_EMULATOR=${crosscompiling_emulator_escaped}"
    )
endif()
foreach(language IN ITEMS C CXX)
    if(DEFINED TRANSFORMER_LAB_${language}_FLAGS AND
       NOT TRANSFORMER_LAB_${language}_FLAGS STREQUAL "")
        string(REPLACE ";" "\\;" escaped_flags
            "${TRANSFORMER_LAB_${language}_FLAGS}"
        )
        list(APPEND toolchain_arguments
            "-DCMAKE_${language}_FLAGS=${escaped_flags}"
        )
    endif()
endforeach()
if(DEFINED TRANSFORMER_LAB_EXE_LINKER_FLAGS AND
   NOT TRANSFORMER_LAB_EXE_LINKER_FLAGS STREQUAL "")
    string(REPLACE ";" "\\;" escaped_linker_flags
        "${TRANSFORMER_LAB_EXE_LINKER_FLAGS}"
    )
    list(APPEND toolchain_arguments
        "-DCMAKE_EXE_LINKER_FLAGS=${escaped_linker_flags}"
    )
endif()

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
        --install "${TRANSFORMER_LAB_BUILD_DIR}"
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

execute_process(
    COMMAND
        "${CMAKE_COMMAND}"
        -S "${TRANSFORMER_LAB_CONSUMER_SOURCE_DIR}"
        -B "${consumer_build}"
        ${generator_arguments}
        "-DCMAKE_PREFIX_PATH=${install_prefix}"
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

if(NOT TRANSFORMER_LAB_CROSSCOMPILING OR
   (DEFINED TRANSFORMER_LAB_CROSSCOMPILING_EMULATOR AND
    NOT TRANSFORMER_LAB_CROSSCOMPILING_EMULATOR STREQUAL ""))
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
