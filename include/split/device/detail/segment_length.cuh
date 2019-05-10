#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_SEGMENT_LENGTH)
#define SPLIT_DEVICE_INCLUDED_DETAIL_SEGMENT_LENGTH

#include "split/detail/internal.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
/// @brief A function for calculating the length and cumulative length of each
/// segment in an input list. Useful for calculating valences, CSR matrix
/// offsets etc.
template <typename InputIterator,
          typename IndexT,
          typename CumulativeLengthIterator,
          typename LengthIterator>
void segment_length(InputIterator&& input_begin,
                    InputIterator&& input_end,
                    IndexT num_segments,
                    CumulativeLengthIterator&& cumulative_length,
                    LengthIterator&& length)
{
  auto count = thrust::make_counting_iterator(0);
  // Find the cumulative length of each segment
  thrust::upper_bound(
    input_begin, input_end, count, count + num_segments, cumulative_length);
  // Compute the length of each segment
  thrust::adjacent_difference(
    cumulative_length, cumulative_length + num_segments, length);
}

/// @brief A wrapper for the function above, where only the length is requested,
/// it is safe to use the same length array for the two arguments
template <typename InputIterator,
          typename IndexT,
          typename LengthIterator>
void segment_length(InputIterator&& input_begin,
                    InputIterator&& input_end,
                    IndexT num_segments,
                    LengthIterator&& length)
{
  segment_length(input_begin, input_end, num_segments, length, length);
}

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_SEGMENT_LENGTH

