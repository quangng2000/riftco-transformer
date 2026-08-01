#pragma once

#include <array>
#include <cstdint>

namespace riftco_transformer::backend_detail {

// Serialized xla::CompileOptions{}.ToProto() from the OpenXLA revision pinned
// in third_party/pjrt/README.md. The outer message contains field 3
// (ExecutableBuildOptionsProto). Its serialized wire fields are:
//
//   device_ordinal = -1
//   num_replicas = 1
//   num_partitions = 1
//   allow_spmd_sharding_propagation_to_output = [false]
//   allow_spmd_sharding_propagation_to_parameters = [false]
//   process_count = 1
//
// Do not replace this with an empty or partial message. OpenXLA's proto3
// deserializer assigns absent scalar fields as zero, which would silently turn
// device_ordinal from -1 to 0 and process_count from 1 to 0.
inline constexpr std::array<std::uint8_t, 27>
    single_device_compile_options_proto{
        0x1A, 0x19,
        0x08, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
        0xFF, 0xFF, 0xFF, 0x01,
        0x20, 0x01,
        0x28, 0x01,
        0x62, 0x01, 0x00,
        0x92, 0x01, 0x01, 0x00,
        0xB8, 0x01, 0x01,
    };

}  // namespace riftco_transformer::backend_detail
