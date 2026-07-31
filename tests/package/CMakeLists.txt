cmake_minimum_required(VERSION 3.24)

project(transformer_lab_package_consumer LANGUAGES C CXX)

find_package(transformer_lab 0.1 CONFIG REQUIRED)

add_executable(cpp_package_consumer
    cpp_consumer.cpp
)
target_link_libraries(cpp_package_consumer
    PRIVATE
        transformer_lab::library
)

add_executable(c_package_consumer
    c_consumer.c
)
target_link_libraries(c_package_consumer
    PRIVATE
        transformer_lab::c_api
)

add_executable(cpp_c_api_package_consumer
    cpp_c_api_consumer.cpp
)
target_link_libraries(cpp_c_api_package_consumer
    PRIVATE
        transformer_lab::c_api
)

if(WIN32)
    foreach(consumer IN ITEMS
            c_package_consumer
            cpp_c_api_package_consumer)
        add_custom_command(
            TARGET ${consumer}
            POST_BUILD
            COMMAND
                ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:transformer_lab::c_api>
                $<TARGET_FILE_DIR:${consumer}>
            VERBATIM
        )
    endforeach()
endif()

enable_testing()
add_test(NAME cpp_package_consumer COMMAND cpp_package_consumer)
add_test(NAME c_package_consumer COMMAND c_package_consumer)
add_test(
    NAME cpp_c_api_package_consumer
    COMMAND cpp_c_api_package_consumer
)
