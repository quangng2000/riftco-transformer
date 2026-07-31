#pragma once

#include "transformer_lab/model/decoder_kv_cache.hpp"

#include <memory>

namespace transformer_lab::stages::serving::cache_detail {

class PageStorage;

[[nodiscard]] std::unique_ptr<DecoderKeyValueCache>
make_fixed_page_table_cache(std::shared_ptr<PageStorage> storage);

[[nodiscard]] std::unique_ptr<DecoderKeyValueCache>
make_leased_page_table_cache(std::shared_ptr<PageStorage> storage);

}  // namespace transformer_lab::stages::serving::cache_detail
