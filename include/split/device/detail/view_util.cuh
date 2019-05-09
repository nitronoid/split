#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_VIEW_UTIL)
#define SPLIT_DEVICE_INCLUDED_DETAIL_VIEW_UTIL

#include "split/detail/internal.h"
#include <thrust/iterator/zip_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
/// @brief A wrapper for creating thrust zip iterators, regular syntax is pretty
/// exhausting
template <typename T,
          typename ConstArrayView2D = typename T::container::const_view>
ConstArrayView2D make_const_array2d_view(const T& array_view)
{
  return ConstArrayView2D(array_view.num_rows,
                          array_view.num_cols,
                          array_view.num_cols,
                          array_view.values);
}
}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_VIEW_UTIL

