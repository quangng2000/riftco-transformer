option(
    RIFTCO_TRANSFORMER_ENABLE_SANITIZERS
    "Instrument riftco_transformer with AddressSanitizer and UndefinedBehaviorSanitizer"
    OFF
)

if(RIFTCO_TRANSFORMER_ENABLE_SANITIZERS)
    if(MSVC OR
       NOT CMAKE_C_COMPILER_ID MATCHES
           "^(AppleClang|Clang|GNU)$" OR
       NOT CMAKE_CXX_COMPILER_ID MATCHES
           "^(AppleClang|Clang|GNU)$")
        message(FATAL_ERROR
            "RIFTCO_TRANSFORMER_ENABLE_SANITIZERS requires a GNU-like "
            "Clang, AppleClang, or GCC toolchain"
        )
    endif()

    include(CMakePushCheckState)
    include(CheckCSourceCompiles)
    include(CheckCXXSourceCompiles)
    cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_FLAGS
        "-fsanitize=address,undefined -fno-omit-frame-pointer"
    )
    set(CMAKE_REQUIRED_LINK_OPTIONS
        -fsanitize=address,undefined
    )
    check_c_source_compiles(
        "int main(void) { return 0; }"
        RIFTCO_TRANSFORMER_C_SANITIZERS_SUPPORTED
    )
    check_cxx_source_compiles(
        "int main() { return 0; }"
        RIFTCO_TRANSFORMER_CXX_SANITIZERS_SUPPORTED
    )
    cmake_pop_check_state()

    if(NOT RIFTCO_TRANSFORMER_C_SANITIZERS_SUPPORTED OR
       NOT RIFTCO_TRANSFORMER_CXX_SANITIZERS_SUPPORTED)
        message(FATAL_ERROR
            "The selected C/C++ toolchain cannot compile and link with "
            "AddressSanitizer and UndefinedBehaviorSanitizer. Install "
            "the compiler sanitizer runtimes or disable "
            "RIFTCO_TRANSFORMER_ENABLE_SANITIZERS."
        )
    endif()
endif()

function(riftco_transformer_enable_sanitizers target)
    if(NOT RIFTCO_TRANSFORMER_ENABLE_SANITIZERS)
        return()
    endif()

    target_compile_options(${target}
        PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
    )
    target_link_options(${target}
        PRIVATE
            -fsanitize=address,undefined
    )
endfunction()

# A consumer that links an instrumented static library must also link the
# sanitizer runtime. The requirement therefore belongs to the exported target
# when this non-production build mode is enabled.
function(riftco_transformer_propagate_sanitizer_runtime target)
    if(NOT RIFTCO_TRANSFORMER_ENABLE_SANITIZERS)
        return()
    endif()

    target_link_options(${target}
        INTERFACE
            -fsanitize=address,undefined
    )
endfunction()
