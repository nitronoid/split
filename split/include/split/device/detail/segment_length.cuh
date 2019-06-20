#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_SEGMENT_LENGTH)
#define SPLIT_DEVICE_INCLUDED_DETAIL_SEGMENT_LENGTH

#include "split/detail/internal.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
/// @brief A function for calculating the length of each segment in an input
/// list.
template <typename InputBeginT, typename InputEndT, typename LengthIterator>
void segment_length(InputBeginT&& input_begin,
                    InputEndT&& input_end,
                    LengthIterator&& length)
{
  const auto discard_it = thrust::make_discard_iterator();
  const auto one = thrust::make_constant_iterator(1);
  // Add up the constant iterator per segment to get the length
  thrust::reduce_by_key(input_begin, input_end, one, discard_it, length);
}

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_SEGMENT_LENGTH

