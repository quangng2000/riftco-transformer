function(riftco_transformer_enable_warnings target)
    if(MSVC)
        target_compile_options(${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CXX>:/permissive->
        )
    else()
        target_compile_options(${target}
            PRIVATE
                $<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-Wall>
                $<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-Wextra>
                $<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-Wpedantic>
                $<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-Wconversion>
                $<$<COMPILE_LANGUAGE:C,CXX,OBJCXX>:-Wshadow>
        )
    endif()
endfunction()
