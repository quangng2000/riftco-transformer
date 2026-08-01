# Pinned PJRT C API declaration

This directory contains the official OpenXLA `pjrt_c_api.h` from commit
`edf55400a224ce46ac3305754a03af6419831ef6` (PJRT API 0.114).

Only the Apache-2.0 C declaration is vendored. The project does not include,
link, or redistribute `libtpu.so`; an opted-in TPU build discovers the user's
Google-provided runtime dynamically. See `LICENSE.openxla` for the vendored
header's license.

The upstream API table gives function-pointer members the same names as their
function typedefs. GNU C++ diagnoses that C ABI pattern with
`-Wchanges-meaning`, so TPU-enabled GNU builds apply
`-Wno-changes-meaning` only to targets that consume this header. The vendored
declaration remains byte-for-byte upstream.

`src/core/backend/adapters/tpu/compile_options.hpp` pins the 27-byte protobuf
serialization produced by `xla::CompileOptions{}.ToProto()` at the same
revision. The full default matters: OpenXLA's deserializer assigns absent
proto3 scalars as zero, while the C++ defaults include `device_ordinal = -1`
and `process_count = 1`. A native backend regression test locks the bytes.
