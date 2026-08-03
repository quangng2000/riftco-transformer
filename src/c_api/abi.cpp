#include "internal/error.hpp"

using namespace riftco_transformer::c_api::detail;

extern "C" {

uint32_t RT_CALL rt_abi_version(void) {
    return RT_ABI_VERSION;
}

const char* RT_CALL rt_status_string(rt_status status) {
    switch (status) {
        case RT_STATUS_OK:
            return "ok";
        case RT_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case RT_STATUS_OUT_OF_RANGE:
            return "out of range";
        case RT_STATUS_OVERFLOW:
            return "overflow";
        case RT_STATUS_BACKEND_UNAVAILABLE:
            return "backend unavailable";
        case RT_STATUS_OUT_OF_MEMORY:
            return "out of memory";
        case RT_STATUS_RUNTIME_ERROR:
            return "runtime error";
        case RT_STATUS_UNKNOWN_ERROR:
            return "unknown error";
    }
    return "unrecognized status";
}

const char* RT_CALL rt_last_error(void) {
    return last_error_message();
}

}  // extern "C"
