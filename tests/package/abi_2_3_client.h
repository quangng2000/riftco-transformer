#ifndef RIFTCO_TRANSFORMER_TEST_ABI_2_3_CLIENT_H
#define RIFTCO_TRANSFORMER_TEST_ABI_2_3_CLIENT_H

#include <stdint.h>

#if defined(_WIN32)
#define RT_2_3_API __declspec(dllimport)
#define RT_2_3_CALL __cdecl
#else
#define RT_2_3_API __attribute__((visibility("default")))
#define RT_2_3_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Frozen subset of the published 2.3 header. It deliberately excludes the
// current header so ABI 2.4 cannot silently rewrite this client's 32-byte
// Adam options layout.
#define RT_2_3_ABI_VERSION_MAJOR UINT32_C(2)
#define RT_2_3_ABI_VERSION_MINOR UINT32_C(3)
#define RT_2_3_ABI_VERSION \
    ((RT_2_3_ABI_VERSION_MAJOR << 16) | RT_2_3_ABI_VERSION_MINOR)

typedef int32_t rt_2_3_status;
#define RT_2_3_STATUS_OK ((rt_2_3_status)0)

typedef struct rt_2_3_adam_options {
    uint64_t struct_size;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float maximum_gradient_norm;
    uint32_t reserved;
} rt_2_3_adam_options;

RT_2_3_API uint32_t RT_2_3_CALL rt_abi_version(void);
RT_2_3_API rt_2_3_status RT_2_3_CALL rt_adam_options_init(
    rt_2_3_adam_options* options,
    uint64_t options_size
);

#ifdef __cplusplus
}
#endif

#endif
