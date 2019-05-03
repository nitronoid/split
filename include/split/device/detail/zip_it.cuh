#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_ZIP_IT)
#define SPLIT_DEVICE_INCLUDED_DETAIL_ZIP_IT

#include "split/detail/internal.h"
#include <thrust/iterator/zip_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
/// @brief A wrapper for creating thrust zip iterators, regular syntax is pretty
/// exhausting
template <typename... Args>
auto zip_it(Args&&... args) -> decltype(
  thrust::make_zip_iterator(thrust::make_tuple(std::forward<Args>(args)...)))
{
  return thrust::make_zip_iterator(
    thrust::make_tuple(std::forward<Args>(args)...));
}
}

SPLIT_DEVICE_NAMESPACE_END

#endif // SPLIT_DEVICE_INCLUDED_DETAIL_ZIP_IT
