cmake_minimum_required(VERSION 3.24)

project(riftco_transformer_package_consumer LANGUAGES C CXX)

if(NOT DEFINED RIFTCO_TRANSFORMER_REQUIRED_VERSION)
    message(FATAL_ERROR "RIFTCO_TRANSFORMER_REQUIRED_VERSION is required")
endif()
find_package(
    riftco_transformer
    ${RIFTCO_TRANSFORMER_REQUIRED_VERSION}
    EXACT
    CONFIG
    REQUIRED
)

add_executable(cpp_package_consumer
    cpp_consumer.cpp
)
target_link_libraries(cpp_package_consumer
    PRIVATE
        riftco_transformer::library
        riftco_transformer::lowering
)

add_executable(cpp_compiler_package_consumer
    compiler_consumer.cpp
)
target_link_libraries(cpp_compiler_package_consumer
    PRIVATE
        riftco_transformer::compiler
)

add_executable(cpp_analysis_package_consumer
    analysis_consumer.cpp
)
target_link_libraries(cpp_analysis_package_consumer
    PRIVATE
        riftco_transformer::analysis
)

add_executable(c_package_consumer
    c_consumer.c
)
target_link_libraries(c_package_consumer
    PRIVATE
        riftco_transformer::c_api
)

add_executable(c_abi_2_0_package_consumer
    abi_2_0_consumer.c
)
target_link_libraries(c_abi_2_0_package_consumer
    PRIVATE
        riftco_transformer::c_api
)

add_executable(c_abi_2_1_package_consumer
    abi_2_1_consumer.c
)
target_link_libraries(c_abi_2_1_package_consumer
    PRIVATE
        riftco_transformer::c_api
)

add_executable(c_abi_2_3_package_consumer
    abi_2_3_consumer.c
)
target_link_libraries(c_abi_2_3_package_consumer
    PRIVATE
        riftco_transformer::c_api
)

add_executable(cpp_c_api_package_consumer
    cpp_c_api_consumer.cpp
)
target_link_libraries(cpp_c_api_package_consumer
    PRIVATE
        riftco_transformer::c_api
)

if(WIN32)
    foreach(consumer IN ITEMS
            c_package_consumer
            c_abi_2_0_package_consumer
            c_abi_2_1_package_consumer
            c_abi_2_3_package_consumer
            cpp_c_api_package_consumer)
        add_custom_command(
            TARGET ${consumer}
            POST_BUILD
            COMMAND
                ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:riftco_transformer::c_api>
                $<TARGET_FILE_DIR:${consumer}>
            VERBATIM
        )
    endforeach()
endif()

enable_testing()
add_test(NAME cpp_package_consumer COMMAND cpp_package_consumer)
add_test(
    NAME cpp_compiler_package_consumer
    COMMAND cpp_compiler_package_consumer
)
add_test(
    NAME cpp_analysis_package_consumer
    COMMAND cpp_analysis_package_consumer
)
add_test(NAME c_package_consumer COMMAND c_package_consumer)
add_test(
    NAME c_abi_2_0_package_consumer
    COMMAND c_abi_2_0_package_consumer
)
add_test(
    NAME c_abi_2_1_package_consumer
    COMMAND c_abi_2_1_package_consumer
)
add_test(
    NAME c_abi_2_3_package_consumer
    COMMAND c_abi_2_3_package_consumer
)
add_test(
    NAME cpp_c_api_package_consumer
    COMMAND cpp_c_api_package_consumer
)
