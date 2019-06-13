#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_MAP_UNTIL_CONVERGED)
#define SPLIT_DEVICE_INCLUDED_DETAIL_MAP_UNTIL_CONVERGED

#include "split/detail/internal.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
/// @brief A function for removing cycles within a mapping
template <typename ValueIterator, typename MapIterator, typename BufferIterator>
void map_until_converged(ValueIterator&& values_begin,
                         ValueIterator&& values_end,
                         MapIterator&& map_begin,
                         BufferIterator&& buffer_begin)
{
  const int nvalues = values_end - values_begin;
  // We have converged when there is no change in labeling after an iteration
  const auto has_converged = [=] {
    return thrust::equal(values_begin, values_end, buffer_begin);
  };
  // Loop until convergence
  while (!has_converged())
  {
    thrust::copy(values_begin, values_end, buffer_begin);
    thrust::gather(
      buffer_begin, buffer_begin + nvalues, map_begin, values_begin);
  }
}

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_MAP_UNTIL_CONVERGED

