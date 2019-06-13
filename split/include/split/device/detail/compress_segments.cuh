#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_COMPRESS_SEGMENTS)
#define SPLIT_DEVICE_INCLUDED_DETAIL_COMPRESS_SEGMENTS

#include "split/detail/internal.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
/// @brief A function for compressing a segment sequence, into it's strictly
/// ascending equivalent.
/// e.g. 6 6 6, 5 5, 13, 4 4 4   ->   0 0 0, 1 1, 2, 3 3 3
template <typename BeginIterator,
          typename EndIterator,
          typename OutputIterator>
void compress_segments(BeginIterator&& input_begin,
                       EndIterator&& input_end,
                       OutputIterator&& output_begin)
{
  // Get the length of the input sequence
  const int length = input_end - input_begin;
  // Deduce the value type in the segmented sequence
  using value_type = typename std::remove_cv<
    typename std::remove_reference<decltype(*input_begin)>::type>::type;
  // Create the correct comparison op
  thrust::not_equal_to<value_type> binary_op;
  // Locate the boundaries between segments and write a 1 into the output
  thrust::adjacent_difference(input_begin, input_end, output_begin, binary_op);
  // Set the first to a zero as it's ignored by adjacent_difference
  output_begin[0] = 0;
  // Scan the results to produce our new segments
  thrust::inclusive_scan(output_begin, output_begin + length, output_begin);
}

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_COMPRESS_SEGMENTS

