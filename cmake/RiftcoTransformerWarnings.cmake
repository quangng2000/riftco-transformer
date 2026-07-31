function(riftco_transformer_enable_warnings target)
    if(MSVC)
        target_compile_options(${target}
            PRIVATE
                /W4
                $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        )
    else()
        target_compile_options(${target}
            PRIVATE
                -Wall
                -Wextra
                -Wpedantic
                -Wconversion
                -Wshadow
        )
    endif()
endfunction()
