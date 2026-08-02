if(RIFTCO_TRANSFORMER_BUILD_PYTHON_WHEEL)
    install(
        TARGETS riftco_transformer_c
        LIBRARY
            DESTINATION riftco_transformer/.libs
            COMPONENT PythonWheel
        RUNTIME
            DESTINATION riftco_transformer/.libs
            COMPONENT PythonWheel
    )
    return()
endif()

include(CMakePackageConfigHelpers)

# Multi-config generators can install several configurations into one prefix.
# Give every non-Release artifact a distinct name so one configuration cannot
# silently overwrite another while both imported-target fragments remain.
if(CMAKE_CONFIGURATION_TYPES)
    set(riftco_transformer_installable_targets
        riftco_transformer_library
        riftco_transformer_compiler
        riftco_transformer_analysis
        riftco_transformer_lowering
        riftco_transformer_programmed
        riftco_transformer_c
    )
    foreach(configuration IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER "${configuration}" configuration_upper)
        if(NOT configuration_upper STREQUAL "RELEASE")
            string(TOLOWER "${configuration}" configuration_lower)
            set_target_properties(
                ${riftco_transformer_installable_targets}
                PROPERTIES
                    "${configuration_upper}_POSTFIX"
                    "_${configuration_lower}"
            )
        endif()
    endforeach()
endif()

set(riftco_transformer_cmake_install_dir
    ${CMAKE_INSTALL_LIBDIR}/cmake/riftco_transformer
)

configure_package_config_file(
    ${CMAKE_CURRENT_LIST_DIR}/riftco_transformerConfig.cmake.in
    ${PROJECT_BINARY_DIR}/riftco_transformerConfig.cmake
    INSTALL_DESTINATION ${riftco_transformer_cmake_install_dir}
)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/riftco_transformerConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(
    TARGETS
        riftco_transformer_library
        riftco_transformer_compiler
        riftco_transformer_analysis
        riftco_transformer_lowering
        riftco_transformer_programmed
        riftco_transformer_c
    EXPORT riftco_transformerTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
    FILES ${PROJECT_SOURCE_DIR}/LICENSE
    DESTINATION ${CMAKE_INSTALL_DATADIR}/licenses/riftco_transformer
)

install(
    EXPORT riftco_transformerTargets
    FILE riftco_transformerTargets.cmake
    NAMESPACE riftco_transformer::
    DESTINATION ${riftco_transformer_cmake_install_dir}
)

install(
    FILES
        ${PROJECT_BINARY_DIR}/riftco_transformerConfig.cmake
        ${PROJECT_BINARY_DIR}/riftco_transformerConfigVersion.cmake
    DESTINATION ${riftco_transformer_cmake_install_dir}
)
