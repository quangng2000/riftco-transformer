#include "riftco_transformer/stages/serving/kv_cache.hpp"

#include "stages/serving/cache/detail/page_storage.hpp"
#include "stages/serving/cache/detail/page_table_cache.hpp"
#include "stages/serving/cache/detail/validation.hpp"

#include <memory>
#include <utility>

namespace riftco_transformer::stages::serving {

struct ContiguousKvCacheFactory::Implementation {
    TransformerDimensions dimensions;
    ExecutionBackend backend;
};

ContiguousKvCacheFactory::ContiguousKvCacheFactory(
    TransformerDimensions dimensions,
    ExecutionBackend backend
)
    : implementation_(std::make_unique<Implementation>(
          Implementation{
              cache_detail::checked_dimensions(dimensions),
              cache_detail::checked_backend(backend),
          }
      )) {}

ContiguousKvCacheFactory::~ContiguousKvCacheFactory() = default;

const TransformerDimensions&
ContiguousKvCacheFactory::dimensions() const noexcept {
    return implementation_->dimensions;
}

ExecutionBackend
ContiguousKvCacheFactory::backend() const noexcept {
    return implementation_->backend;
}

std::unique_ptr<DecoderKeyValueCache>
ContiguousKvCacheFactory::create() const {
    auto storage = std::make_shared<cache_detail::PageStorage>(
        dimensions(),
        backend(),
        dimensions().maximum_context,
        1,
        cache_detail::PageAllocation::Fixed
    );
    return cache_detail::make_fixed_page_table_cache(
        std::move(storage)
    );
}

}  // namespace riftco_transformer::stages::serving
