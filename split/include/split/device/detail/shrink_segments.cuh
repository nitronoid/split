#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_SHRINK_SEGMENTS)
#define SPLIT_DEVICE_INCLUDED_DETAIL_SHRINK_SEGMENTS

#include "split/detail/internal.h"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/segment_length.cuh"
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
template <typename KeyBegin, typename KeyEnd, typename ValueBegin>
int shrink_segments(KeyBegin&& key_begin,
                    KeyEnd&& key_end,
                    ValueBegin&& value_begin,
                    const int n_segments,
                    const int max_length)
{
  const int n_data = key_end - key_begin;
  // Find the length of each segment
  cusp::array1d<int, cusp::device_memory> segment_lengths(n_segments + 1);
  detail::segment_length(key_begin, key_end, segment_lengths.begin());
  // Calculate the number of elements to be removed per segment
  cusp::array1d<int, cusp::device_memory> segment_removal(n_segments);
  thrust::transform(segment_lengths.begin(),
                    segment_lengths.end(),
                    segment_removal.begin(),
                    detail::unary_minus<int>(max_length));
  // Scan the lengths to get indices
  thrust::exclusive_scan(segment_lengths.begin(),
                         segment_lengths.end(),
                         segment_lengths.begin());
  // Reset the keys
  thrust::fill(key_begin, key_end, 0);
  // Write ones in the scanned positions
  const auto one = thrust::make_constant_iterator(1);
  const auto scatter_keys =
    thrust::make_permutation_iterator(key_begin, segment_lengths.begin());
  thrust::copy_n(one, n_segments, scatter_keys);
  auto key_view = cusp::make_array1d_view(key_begin, key_end);
  // Write ones in the new end positions
  using ivec2 = thrust::tuple<int, int>;
  const auto scatter_ends = thrust::make_permutation_iterator(
    key_begin,
    thrust::make_transform_iterator(
      detail::zip_it(segment_removal.begin(), segment_lengths.begin() + 1),
      [] __host__ __device__(ivec2 seg_pair) {
        return seg_pair.get<1>() - max(0, seg_pair.get<0>());
      }));
  thrust::transform(
    one, one + n_segments, scatter_ends, scatter_ends, thrust::plus<int>());
  // Scan the markers to get even or odd keys
  thrust::inclusive_scan(key_begin, key_end, key_begin);
  // Remove all even keys
  auto key_value_begin = detail::zip_it(key_begin, value_begin);
  auto key_value_end = key_value_begin + n_data;
  auto new_end =
    thrust::remove_if(key_value_begin,
                      key_value_end,
                      [] __host__ __device__(ivec2 key_value_pair) {
                        return !(key_value_pair.get<0>() & 1);
                      });

  return new_end - key_value_begin;
}

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_SHRINK_SEGMENTS



