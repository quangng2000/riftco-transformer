set(transformer_lab_backend_sources
    src/core/backend/adam_reference.cpp
    src/core/backend/attention/dispatch.cpp
    src/core/backend/attention/reference/flash_causal.cpp
    src/core/backend/attention/reference/materialized_causal.cpp
    src/core/backend/attention/reference/paged_decode.cpp
    src/core/backend/nn_reference.cpp
    src/core/backend/registry.cpp
    src/core/backend/cpu_adapter.cpp
)

if(APPLE)
    set(transformer_lab_metal_default ON)
else()
    set(transformer_lab_metal_default OFF)
endif()

option(
    TRANSFORMER_LAB_ENABLE_METAL
    "Build the Metal backend adapter on Apple platforms"
    ${transformer_lab_metal_default}
)

if(TRANSFORMER_LAB_ENABLE_METAL AND NOT APPLE)
    message(FATAL_ERROR
        "TRANSFORMER_LAB_ENABLE_METAL requires an Apple platform"
    )
endif()

if(TRANSFORMER_LAB_ENABLE_METAL)
    enable_language(OBJCXX)
    list(APPEND transformer_lab_backend_sources
        src/core/backend/metal_adapter.mm
        src/core/backend/metal_nn_runtime.mm
    )
    set_source_files_properties(
        src/core/backend/metal_adapter.mm
        src/core/backend/metal_nn_runtime.mm
        PROPERTIES
            COMPILE_OPTIONS "-fobjc-arc"
    )
else()
    list(APPEND transformer_lab_backend_sources
        src/core/backend/metal_adapter_stub.cpp
    )
endif()
