#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_AVERAGE_BY_KEY)
#define SPLIT_DEVICE_INCLUDED_DETAIL_AVERAGE_BY_KEY

#include "split/detail/internal.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/iterator_traits.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
/// @brief A function for averaging values over a segmented range.
/// e.g. 6 6 6, 5 5, 13, 4 4 4   ->   0 0 0, 1 1, 2, 3 3 3
template <typename KeyBeginIterator,
          typename KeyEndIterator,
          typename ValueIterator,
          typename LengthIterator,
          typename KeyOutIterator,
          typename AverageIterator,
          typename IndexT,
          // Extract the value type from the iterator
          typename value_type =
            typename thrust::iterator_traits<ValueIterator>::value_type>
void average_by_key(KeyBeginIterator&& keys_in_begin,
                    KeyEndIterator&& keys_in_end,
                    ValueIterator&& values_begin,
                    LengthIterator&& length_begin,
                    KeyOutIterator&& keys_out_begin,
                    AverageIterator&& average_begin,
                    IndexT&& nsegments,
                    value_type = {})
{
  // Compute the segment sizes
  detail::segment_length(keys_in_begin, keys_in_end, length_begin);
  // Reduce all segments to get their totals
  auto total_end = thrust::reduce_by_key(
    keys_in_begin, keys_in_end, values_begin, keys_out_begin, average_begin);
  // Divide through by the lengths to get the averages
  thrust::transform(average_begin,
                    total_end.second,
                    length_begin,
                    average_begin,
                    thrust::divides<value_type>());
}

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_AVERAGE_BY_KEY

