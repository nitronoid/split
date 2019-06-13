#include "split/device/morph/erode.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/transposed_copy.cuh"
#include <cusp/print.h>
#include <thrust/iterator/transform_output_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace morph
{
namespace
{
struct tuple_add
{
  using vec2 = thrust::tuple<int, int>;
  __host__ __device__ int operator()(vec2 x) const
  {
    return x.get<0>() + x.get<1>();
  }
};

struct FindKey
{
  __host__ __device__ thrust::tuple<int, int>
  operator()(const thrust::tuple<int, int>& lhs,
             const thrust::tuple<int, int>& rhs)
  {
    return thrust::make_tuple(
      NULL, (lhs.get<0>() != rhs.get<0>()) || (lhs.get<1>() != rhs.get<1>()));
  }
};

template <typename EntryIt, typename KeyIt>
void make_gather_indices(EntryIt&& entry_begin,
                         EntryIt&& entry_end,
                         KeyIt&& key_begin)
{
  // Discard iterator
  auto discard_it = thrust::make_discard_iterator();
  // Mark neighbors with different labels, or columns
  thrust::adjacent_difference(
    entry_begin, entry_end, detail::zip_it(discard_it, key_begin), FindKey{});
  // First should always be a one
  key_begin[0] = 1;
  // Find the number of keys
  const int nkeys = entry_end - entry_begin;
  // Prefix-sum the marked locations to provide lookup indices
  thrust::inclusive_scan(key_begin,
                         key_begin + nkeys,
                         thrust::make_transform_output_iterator(
                           key_begin, detail::unary_minus<int>(1)));
}
}

SPLIT_API void erode_horizontal_sweep(
  cusp::array2d<int, cusp::device_memory>::view dio_labels,
  const int niterations)
{
  // Get height and width of the label matrix
  const int npoints = dio_labels.num_entries;

  const auto labels_begin = dio_labels.values.begin();
  const auto labels_end = dio_labels.values.end();
  // Useful placeholders
  const auto one = thrust::make_constant_iterator(1);
  const auto discard = thrust::make_discard_iterator();
  // Create a buffer the same size as the label matrix
  cusp::array1d<int, cusp::device_memory> d_segments(npoints);
  // Convert the labels, to rowise ascending segments
  auto row_indices = detail::make_row_iterator(dio_labels.num_cols);
  auto row_begin = detail::zip_it(row_indices, dio_labels.values.begin());
  auto row_end = row_begin + npoints;
  make_gather_indices(row_begin, row_end, d_segments.begin());

  // Create an array to store the size of each rowwise segment, now we know 
  // how many exist
  cusp::array1d<int, cusp::device_memory> d_segment_sizes(d_segments.back() + 1);
  cusp::array1d<int, cusp::device_memory> d_cum_segment_sizes(d_segments.back() + 1);
  auto segment_sizes_begin = d_segment_sizes.begin();
  auto segment_sizes_end = d_segment_sizes.end();
  auto cum_segment_sizes_begin = d_cum_segment_sizes.begin();
  // Reduce these rowwise segments to find sizes
  thrust::reduce_by_key(
    d_segments.begin(), d_segments.end(), one, discard, segment_sizes_begin);
  // Get the cumulative sizes
  thrust::exclusive_scan(
    segment_sizes_begin, segment_sizes_end, cum_segment_sizes_begin);

  // Compute the new sizes of each segment post erosion, clamp above zero
  thrust::transform(
    segment_sizes_begin,
    segment_sizes_end,
    thrust::make_transform_output_iterator(
      segment_sizes_begin, detail::unary_max<int>(0)),
    detail::unary_minus<int>(2 * niterations));
  
  // Remove segment sizes that are zero
  auto size_pair_begin = detail::zip_it(
    segment_sizes_begin, cum_segment_sizes_begin);
  auto size_pair_end = size_pair_begin + d_segment_sizes.size();
  using ivec2 = thrust::tuple<int, int>;
  size_pair_end = thrust::remove_if(size_pair_begin,
                                    size_pair_end, 
                                    [] __host__ __device__ (const ivec2& pair)
                                    {
                                      return pair.get<0>() == 0;
                                    });

  // Reset the segments as we'll be recalculating them
  thrust::fill(d_segments.begin(), d_segments.end(), 0);

  // Now scatter 1's to the base positions 
  // i.e. remaining cumulative sizes + #No. iterations
  const auto base_positions = thrust::make_transform_iterator(
    cum_segment_sizes_begin, detail::unary_plus<int>(niterations));
  thrust::copy_n(
    thrust::make_constant_iterator(1),
    size_pair_end - size_pair_begin,
    thrust::make_permutation_iterator(d_segments.begin(), base_positions));

  // Next scatter 1's to the end positions 
  // i.e. base positions plus the segment size
  const auto end_positions = thrust::make_transform_iterator(
    detail::zip_it(base_positions, segment_sizes_begin), tuple_add());
  thrust::copy_n(
    thrust::make_constant_iterator(1),
    size_pair_end - size_pair_begin,
    thrust::make_permutation_iterator(d_segments.begin(), end_positions));

  // Now we can scan the 1's to get the new segments
  thrust::inclusive_scan(
    d_segments.begin(), d_segments.end(), d_segments.begin());
  // Finally, convert all even segments to -1 sentinel labels
  const auto pred = [] __host__ __device__ (int x) { return (x & 1) == 0; };
  thrust::replace_if(labels_begin, labels_end, d_segments.begin(), pred, -1);
}

SPLIT_API void erode(
  cusp::array2d<int, cusp::device_memory>::view dio_labels,
  const int niterations)
{
  // Allocate room to transpose the label, with reversed dimensions
  cusp::array2d<int, cusp::device_memory> d_labels_transposed(
    dio_labels.num_cols, dio_labels.num_rows, dio_labels.num_entries);
  // Transpose the labels
  detail::transposed_copy(dio_labels.num_cols, 
                          dio_labels.num_rows, 
                          dio_labels.values,
                          d_labels_transposed.values);
  // Perform an erosion sweep on the transposed labels
  erode_horizontal_sweep(d_labels_transposed, niterations);
  // Transpose the new labels back
  detail::transposed_copy(d_labels_transposed.num_cols, 
                          d_labels_transposed.num_rows, 
                          d_labels_transposed.values,
                          dio_labels.values);
  // Perform the second sweep
  erode_horizontal_sweep(dio_labels, niterations);
}

}  // namespace morph

SPLIT_DEVICE_NAMESPACE_END


