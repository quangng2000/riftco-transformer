if(NOT DEFINED TRANSFORMER_LAB_EXECUTABLE)
    message(FATAL_ERROR "TRANSFORMER_LAB_EXECUTABLE is required")
endif()
if(NOT DEFINED CONFIG_PATH)
    message(FATAL_ERROR "CONFIG_PATH is required")
endif()
if(NOT DEFINED METRICS_PATH)
    message(FATAL_ERROR "METRICS_PATH is required")
endif()

file(REMOVE "${METRICS_PATH}")

execute_process(
    COMMAND
        "${TRANSFORMER_LAB_EXECUTABLE}"
        --config "${CONFIG_PATH}"
        --steps 3
        --metrics "${METRICS_PATH}"
        --backend cpu
        --attention flash
        --activation-checkpointing block
    RESULT_VARIABLE training_result
    OUTPUT_VARIABLE training_output
    ERROR_VARIABLE training_error
)

if(NOT training_result EQUAL 0)
    message(FATAL_ERROR
        "training CLI exited with ${training_result}\n"
        "stdout:\n${training_output}\n"
        "stderr:\n${training_error}"
    )
endif()

string(FIND
    "${training_output}"
    "Training loop: complete."
    completion_marker
)
if(completion_marker EQUAL -1)
    message(FATAL_ERROR
        "training CLI output is missing its completion marker\n"
        "stdout:\n${training_output}"
    )
endif()

string(REGEX MATCH
    "Backend:[^\n]*cpu"
    backend_marker
    "${training_output}"
)
if(backend_marker STREQUAL "")
    message(FATAL_ERROR
        "training CLI output did not confirm the CPU backend\n"
        "stdout:\n${training_output}"
    )
endif()

string(REGEX MATCH
    "Attention:[^\\n]*flash"
    attention_marker
    "${training_output}"
)
if(attention_marker STREQUAL "")
    message(FATAL_ERROR
        "training CLI output did not confirm Flash attention\\n"
        "stdout:\\n${training_output}"
    )
endif()

string(REGEX MATCH
    "Checkpointing:[^\\n]*block"
    checkpointing_marker
    "${training_output}"
)
if(checkpointing_marker STREQUAL "")
    message(FATAL_ERROR
        "training CLI output did not confirm block checkpointing\\n"
        "stdout:\\n${training_output}"
    )
endif()

if(NOT EXISTS "${METRICS_PATH}")
    message(FATAL_ERROR
        "training CLI did not create metrics file: ${METRICS_PATH}"
    )
endif()

file(STRINGS "${METRICS_PATH}" metric_rows)
list(LENGTH metric_rows metric_row_count)
if(NOT metric_row_count EQUAL 4)
    message(FATAL_ERROR
        "expected one CSV header and three metric rows, got "
        "${metric_row_count}\n${metric_rows}"
    )
endif()

list(GET metric_rows 0 metric_header)
if(NOT metric_header STREQUAL "step,loss,gradient_norm,clip_scale")
    message(FATAL_ERROR
        "unexpected metrics CSV header: ${metric_header}"
    )
endif()

foreach(expected_step RANGE 1 3)
    list(GET metric_rows ${expected_step} metric_row)
    if(NOT metric_row MATCHES "^${expected_step},")
        message(FATAL_ERROR
            "expected metrics row ${expected_step} to begin with "
            "'${expected_step},', got: ${metric_row}"
        )
    endif()

    string(REPLACE "," ";" metric_columns "${metric_row}")
    list(LENGTH metric_columns metric_column_count)
    if(NOT metric_column_count EQUAL 4)
        message(FATAL_ERROR
            "expected four metrics columns, got "
            "${metric_column_count}: ${metric_row}"
        )
    endif()
    foreach(metric_column IN LISTS metric_columns)
        if(NOT metric_column MATCHES
           "^-?[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$")
            message(FATAL_ERROR
                "metrics fields must be finite numbers: ${metric_row}"
            )
        endif()
    endforeach()
endforeach()
