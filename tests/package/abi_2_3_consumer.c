#include "abi_2_3_client.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct guarded_adam_options {
    rt_2_3_adam_options value;
    uint64_t canary[2];
} guarded_adam_options;

_Static_assert(
    sizeof(rt_2_3_adam_options) == 32,
    "published ABI 2.3 Adam options layout"
);

int main(void) {
    const uint32_t runtime_version = rt_abi_version();
    if ((runtime_version >> 16) != RT_2_3_ABI_VERSION_MAJOR ||
        runtime_version < RT_2_3_ABI_VERSION) {
        return EXIT_FAILURE;
    }

    const uint64_t first_canary = UINT64_C(0x0123456789abcdef);
    const uint64_t second_canary = UINT64_C(0xfedcba9876543210);
    guarded_adam_options options;
    memset(&options, 0xa5, sizeof(options));
    options.canary[0] = first_canary;
    options.canary[1] = second_canary;

    if (rt_adam_options_init(
            &options.value,
            (uint64_t)sizeof(options.value)
        ) != RT_2_3_STATUS_OK ||
        options.value.struct_size != (uint64_t)sizeof(options.value) ||
        !isfinite(options.value.learning_rate) ||
        options.value.learning_rate <= 0.0F ||
        options.value.reserved != 0 ||
        options.canary[0] != first_canary ||
        options.canary[1] != second_canary) {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
