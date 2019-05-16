#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_VIEW_UTIL)
#define SPLIT_DEVICE_INCLUDED_DETAIL_VIEW_UTIL

#include "split/detail/internal.h"
#include <thrust/iterator/zip_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
/// @brief Converts a mutable array2d_view to a const view
template <typename T,
          typename ConstArrayView2D = typename T::container::const_view>
ConstArrayView2D make_const_array2d_view(const T& array_view)
{
  return ConstArrayView2D(array_view.num_rows,
                          array_view.num_cols,
                          array_view.num_cols,
                          array_view.values);
}
/// @brief Converts a mutable array1d_view to a const view
template <typename T,
          typename ConstArrayView1D = typename T::container::const_view>
ConstArrayView1D make_const_array1d_view(const T& array_view)
{
  return ConstArrayView1D(array_view);
}
}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_VIEW_UTIL

