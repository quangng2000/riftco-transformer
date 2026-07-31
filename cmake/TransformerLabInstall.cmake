if(TRANSFORMER_LAB_BUILD_PYTHON_WHEEL)
    install(
        TARGETS transformer_lab_c
        LIBRARY
            DESTINATION transformer_lab/.libs
            COMPONENT PythonWheel
        RUNTIME
            DESTINATION transformer_lab/.libs
            COMPONENT PythonWheel
    )
    return()
endif()

include(CMakePackageConfigHelpers)

# Multi-config generators can install several configurations into one prefix.
# Give every non-Release artifact a distinct name so one configuration cannot
# silently overwrite another while both imported-target fragments remain.
if(CMAKE_CONFIGURATION_TYPES)
    set(transformer_lab_installable_targets
        transformer_lab_library
        transformer_lab_c
    )
    if(TARGET transformer_lab)
        list(APPEND transformer_lab_installable_targets
            transformer_lab
        )
    endif()

    foreach(configuration IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${configuration}" configuration_upper)
        if(NOT configuration_upper STREQUAL "RELEASE")
            string(TOLOWER "${configuration}" configuration_lower)
            set_target_properties(
                ${transformer_lab_installable_targets}
                PROPERTIES
                    "${configuration_upper}_POSTFIX"
                    "_${configuration_lower}"
            )
        endif()
    endforeach()
endif()

set(transformer_lab_cmake_install_dir
    ${CMAKE_INSTALL_LIBDIR}/cmake/transformer_lab
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/transformer_labConfig.cmake.in
    ${PROJECT_BINARY_DIR}/transformer_labConfig.cmake
    INSTALL_DESTINATION ${transformer_lab_cmake_install_dir}
)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/transformer_labConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(
    TARGETS
        transformer_lab_library
        transformer_lab_c
    EXPORT transformer_labTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

if(TARGET transformer_lab)
    install(
        TARGETS transformer_lab
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    EXPORT transformer_labTargets
    FILE transformer_labTargets.cmake
    NAMESPACE transformer_lab::
    DESTINATION ${transformer_lab_cmake_install_dir}
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/transformer_labConfig.cmake
        ${PROJECT_BINARY_DIR}/transformer_labConfigVersion.cmake
    DESTINATION ${transformer_lab_cmake_install_dir}
)
