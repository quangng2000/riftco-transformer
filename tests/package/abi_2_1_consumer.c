#include "abi_2_1_client.h"

#include <stdlib.h>
#include <string.h>

int main(void) {
    const uint32_t runtime_version = rt_abi_version();
    if ((runtime_version >> 16) != RT_2_1_ABI_VERSION_MAJOR ||
        runtime_version < RT_2_1_ABI_VERSION ||
        strcmp(rt_status_string(RT_2_1_STATUS_OK), "ok") != 0) {
        return EXIT_FAILURE;
    }

    int32_t cpu_available = 0;
    int32_t cuda_available = 0;
    if (rt_backend_is_available(
            RT_2_1_BACKEND_CPU,
            &cpu_available
        ) != RT_2_1_STATUS_OK ||
        cpu_available == 0 ||
        rt_backend_is_available(
            RT_2_1_BACKEND_CUDA,
            &cuda_available
        ) != RT_2_1_STATUS_OK) {
        return EXIT_FAILURE;
    }

    rt_context* context = NULL;
    rt_backend backend = (rt_backend)-1;
    if (rt_context_create(
            RT_2_1_BACKEND_CPU,
            &context
        ) != RT_2_1_STATUS_OK ||
        context == NULL ||
        rt_context_backend(context, &backend) != RT_2_1_STATUS_OK ||
        backend != RT_2_1_BACKEND_CPU) {
        rt_context_release(context);
        return EXIT_FAILURE;
    }
    rt_context_release(context);
    return EXIT_SUCCESS;
}
