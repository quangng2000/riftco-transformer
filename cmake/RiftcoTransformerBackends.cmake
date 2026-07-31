set(riftco_transformer_backend_sources
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
    set(riftco_transformer_metal_default ON)
else()
    set(riftco_transformer_metal_default OFF)
endif()

option(
    RIFTCO_TRANSFORMER_ENABLE_METAL
    "Build the Metal backend adapter on Apple platforms"
    ${riftco_transformer_metal_default}
)

if(RIFTCO_TRANSFORMER_ENABLE_METAL AND NOT APPLE)
    message(FATAL_ERROR
        "RIFTCO_TRANSFORMER_ENABLE_METAL requires an Apple platform"
    )
endif()

if(RIFTCO_TRANSFORMER_ENABLE_METAL)
    enable_language(OBJCXX)
    list(APPEND riftco_transformer_backend_sources
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
    list(APPEND riftco_transformer_backend_sources
        src/core/backend/metal_adapter_stub.cpp
    )
endif()
