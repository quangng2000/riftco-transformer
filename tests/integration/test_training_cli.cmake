if(NOT DEFINED RIFTCO_TRANSFORMER_EXECUTABLE)
    message(FATAL_ERROR "RIFTCO_TRANSFORMER_EXECUTABLE is required")
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
        "${RIFTCO_TRANSFORMER_EXECUTABLE}"
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

# CUDA is a stable CLI value even in builds that compile the unavailable
# adapter. A real CUDA device runs a one-step smoke test; otherwise the CLI
# must reach backend selection and report unavailability rather than reject
# the spelling as an unknown option value.
set(cuda_metrics_path "${METRICS_PATH}.cuda")
file(REMOVE "${cuda_metrics_path}")
execute_process(
    COMMAND
        "${RIFTCO_TRANSFORMER_EXECUTABLE}"
        --config "${CONFIG_PATH}"
        --steps 1
        --metrics "${cuda_metrics_path}"
        --backend cuda
        --attention materialized
        --activation-checkpointing disabled
    RESULT_VARIABLE cuda_result
    OUTPUT_VARIABLE cuda_output
    ERROR_VARIABLE cuda_error
)

if(cuda_result EQUAL 0)
    string(REGEX MATCH
        "Backend:[^\n]*cuda"
        cuda_backend_marker
        "${cuda_output}"
    )
    if(cuda_backend_marker STREQUAL "" OR
       NOT EXISTS "${cuda_metrics_path}")
        message(FATAL_ERROR
            "CUDA CLI run succeeded without confirming CUDA or writing "
            "metrics\nstdout:\n${cuda_output}"
        )
    endif()
else()
    string(FIND
        "${cuda_error}"
        "cuda execution backend is unavailable"
        cuda_unavailable_marker
    )
    if(cuda_unavailable_marker EQUAL -1)
        message(FATAL_ERROR
            "CUDA CLI value was not executed or failed unexpectedly\n"
            "stdout:\n${cuda_output}\n"
            "stderr:\n${cuda_error}"
        )
    endif()
endif()

# TPU is also a stable CLI value. Standard builds carry its unavailable stub;
# a TPU-enabled Cloud TPU build runs this as a one-step end-to-end smoke test.
set(tpu_metrics_path "${METRICS_PATH}.tpu")
file(REMOVE "${tpu_metrics_path}")
execute_process(
    COMMAND
        "${RIFTCO_TRANSFORMER_EXECUTABLE}"
        --config "${CONFIG_PATH}"
        --steps 1
        --metrics "${tpu_metrics_path}"
        --backend tpu
        --attention materialized
        --activation-checkpointing disabled
    RESULT_VARIABLE tpu_result
    OUTPUT_VARIABLE tpu_output
    ERROR_VARIABLE tpu_error
)

if(tpu_result EQUAL 0)
    string(REGEX MATCH
        "Backend:[^\n]*tpu"
        tpu_backend_marker
        "${tpu_output}"
    )
    if(tpu_backend_marker STREQUAL "" OR
       NOT EXISTS "${tpu_metrics_path}")
        message(FATAL_ERROR
            "TPU CLI run succeeded without confirming TPU or writing "
            "metrics\nstdout:\n${tpu_output}"
        )
    endif()
else()
    string(FIND
        "${tpu_error}"
        "tpu execution backend is unavailable"
        tpu_unavailable_marker
    )
    if(tpu_unavailable_marker EQUAL -1)
        message(FATAL_ERROR
            "TPU CLI value was not executed or failed unexpectedly\n"
            "stdout:\n${tpu_output}\n"
            "stderr:\n${tpu_error}"
        )
    endif()
endif()
