#ifndef RIFTCO_TRANSFORMER_TEST_ABI_2_1_CLIENT_H
#define RIFTCO_TRANSFORMER_TEST_ABI_2_1_CLIENT_H

#include <stdint.h>

#if defined(_WIN32)
#define RT_2_1_API __declspec(dllimport)
#define RT_2_1_CALL __cdecl
#else
#define RT_2_1_API __attribute__((visibility("default")))
#define RT_2_1_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Frozen subset of the published 2.1 header. It deliberately excludes the
// current header so ABI 2.2 cannot rewrite this older client's declarations.
#define RT_2_1_ABI_VERSION_MAJOR UINT32_C(2)
#define RT_2_1_ABI_VERSION_MINOR UINT32_C(1)
#define RT_2_1_ABI_VERSION \
    ((RT_2_1_ABI_VERSION_MAJOR << 16) | RT_2_1_ABI_VERSION_MINOR)

typedef struct rt_context rt_context;
typedef int32_t rt_status;
#define RT_2_1_STATUS_OK ((rt_status)0)

typedef int32_t rt_backend;
#define RT_2_1_BACKEND_CPU ((rt_backend)0)
#define RT_2_1_BACKEND_METAL ((rt_backend)1)
#define RT_2_1_BACKEND_CUDA ((rt_backend)2)

RT_2_1_API uint32_t RT_2_1_CALL rt_abi_version(void);
RT_2_1_API const char* RT_2_1_CALL rt_status_string(rt_status status);
RT_2_1_API rt_status RT_2_1_CALL rt_backend_is_available(
    rt_backend backend,
    int32_t* available
);
RT_2_1_API rt_status RT_2_1_CALL rt_context_create(
    rt_backend backend,
    rt_context** output
);
RT_2_1_API void RT_2_1_CALL rt_context_release(rt_context* context);
RT_2_1_API rt_status RT_2_1_CALL rt_context_backend(
    const rt_context* context,
    rt_backend* output
);

#ifdef __cplusplus
}
#endif

#endif
