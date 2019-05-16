#include "split/device/ccl/merge_small_segments.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/detail/segment_length.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/average_by_key.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include "split/device/detail/fix_map_cycles.cuh"
#include "split/device/detail/map_until_converged.cuh"
#include "split/device/detail/view_util.cuh"
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
namespace
{
struct ChominanceDistance2
{
  __host__ __device__ thrust::tuple<real, int>
  operator()(const thrust::tuple<thrust::tuple<real, real>,
                                 thrust::tuple<real, real>,
                                 int>& tup) const
  {
    // distance in x
    const real x = tup.get<0>().get<0>() - tup.get<1>().get<0>();
    // distance in y
    const real y = tup.get<0>().get<1>() - tup.get<1>().get<1>();
    // return squared distance and the id
    return thrust::make_tuple(x * x + y * y, tup.get<2>());
  }
};

// Struct that defines the temporary memory partition types
struct Workspace
{
  using IntArrayT = cusp::array1d<int, cusp::device_memory>::view;
  using RealMatrixT =
    cusp::array2d_view<cusp::array1d_view<thrust::device_ptr<real>>>;
  using TupleT =
    std::tuple<RealMatrixT, IntArrayT, IntArrayT, IntArrayT, IntArrayT>;
};

Workspace::TupleT partition_workspace(const int i_npoints,
                                      const int i_nsegments,
                                      thrust::device_ptr<void> di_temp)
{
  // Read the temp memory as integer storage
  auto itemp =
    thrust::device_pointer_cast(reinterpret_cast<int*>(di_temp.get()));
  // Partition all of our simple integer arrays
  auto d_segment_sizes =
    cusp::make_array1d_view(itemp, itemp + i_nsegments + 1);
  auto d_indices = cusp::make_array1d_view(d_segment_sizes.end(),
                                           d_segment_sizes.end() + i_npoints);
  auto d_label_copy =
    cusp::make_array1d_view(d_indices.end(), d_indices.end() + i_npoints);
  auto d_targets = cusp::make_array1d_view(d_label_copy.end(),
                                           d_label_copy.end() + i_nsegments);

  // Read the temp memory as real storage
  auto rtemp =
    thrust::device_pointer_cast(reinterpret_cast<real*>(d_targets.end().get()));
  // Partition the temp memory as real storage
  auto d_average_chroma = cusp::make_array2d_view(
    2,
    i_nsegments,
    i_nsegments,
    cusp::make_array1d_view(rtemp, rtemp + i_nsegments * 2),
    cusp::row_major{});

  // Return the struct
  return Workspace::TupleT{
    d_average_chroma, d_segment_sizes, d_indices, d_label_copy, d_targets};
}

}  // namespace

SPLIT_API std::size_t merge_small_segments_workspace(const int i_npoints,
                                                     const int i_nsegments)
{
  // Calculate the size of all the arrays we require
  const std::size_t chroma_averages = 2 * i_nsegments * sizeof(real);
  const std::size_t segment_sizes = (i_nsegments + 1) * sizeof(int);
  const std::size_t indices = i_npoints * sizeof(int);
  const std::size_t labels_copy = i_npoints * sizeof(int);
  const std::size_t targets = i_nsegments * sizeof(int);
  // Return the sum
  return chroma_averages + segment_sizes + indices + labels_copy + targets;
}

SPLIT_API void merge_small_segments(
  cusp::array2d<real, cusp::device_memory>::const_view di_chroma,
  cusp::array2d<int, cusp::device_memory>::view dio_segment_labels,
  thrust::device_ptr<void> do_temp,
  const int P)
{
  // Get these sizes upfront
  const int npoints = dio_segment_labels.num_entries;
  assert(npoints == di_chroma.num_cols);

  // Calculate the segment adjacency
  cusp::array1d<int, cusp::device_memory> d_segment_adjacency_keys(npoints * 8);
  cusp::array1d<int, cusp::device_memory> d_segment_adjacency(npoints * 8);
  const int nadjacency =
    segment_adjacency(detail::make_const_array2d_view(dio_segment_labels),
                      d_segment_adjacency_keys,
                      d_segment_adjacency);
  const int nsegments = d_segment_adjacency_keys[nadjacency - 1] + 1;

  Workspace::IntArrayT d_segment_sizes, d_indices, d_label_copy, d_targets;
  Workspace::RealMatrixT d_average_chroma;
  // Partition our temporary workspace
  std::tie(
    d_average_chroma, d_segment_sizes, d_indices, d_label_copy, d_targets) =
    partition_workspace(npoints, nsegments, do_temp);

  // TODO: Could pack this into one kernel?
  // Target array contains the target segment to join with, initially no change
  thrust::sequence(d_targets.begin(), d_targets.end());
  // Initialize the indices with a standard sequence
  thrust::sequence(d_indices.begin(), d_indices.end());
  // Copy our labels for sorting
  thrust::copy(dio_segment_labels.values.begin(),
               dio_segment_labels.values.end(),
               d_label_copy.begin());
  // Sort the indices using the labels
  thrust::sort_by_key(
    d_label_copy.begin(), d_label_copy.end(), d_indices.begin());
  // Make this once
  auto discard_it = thrust::make_discard_iterator();

  {
    // Cycle back for all dimensions of the data
    auto sorted_cycle = detail::make_cycle_iterator(d_indices.begin(), npoints);
    auto label_cycle_begin =
      detail::make_cycle_iterator(d_label_copy.begin(), npoints);
    auto label_cycle_end = label_cycle_begin + npoints * 2;
    // Average the chroma over each segment
    detail::average_by_key(
      label_cycle_begin,
      label_cycle_end,
      thrust::make_permutation_iterator(di_chroma.values.begin(), sorted_cycle),
      detail::make_cycle_iterator(d_segment_sizes.begin(), nsegments),
      discard_it,
      d_average_chroma.values.begin(),
      nsegments);
  }

  {
    // Iterator to access the average chroma of each segment
    auto average_chroma = detail::zip_it(d_average_chroma.row(0).begin(),
                                         d_average_chroma.row(1).begin());
    // Get matrix values as squared distance in chroma space
    auto entry_it = thrust::make_transform_iterator(
      detail::zip_it(thrust::make_permutation_iterator(
                       average_chroma, d_segment_adjacency_keys.begin()),
                     thrust::make_permutation_iterator(
                       average_chroma, d_segment_adjacency.begin()),
                     d_segment_adjacency.begin()),
      ChominanceDistance2{});

    // Reduce by column to find the lowest distance, and hence nearest in
    // chroma space to our segment, this is the segment we want to merge
    // with.
    thrust::reduce_by_key(d_segment_adjacency_keys.begin(),
                          d_segment_adjacency_keys.begin() + nadjacency,
                          entry_it,
                          discard_it,
                          detail::zip_it(discard_it, d_targets.begin()),
                          thrust::equal_to<int>(),
                          thrust::minimum<thrust::tuple<real, int>>());
  }
  // Cull mappings from segments that are large enough
  {
    // Make a counting iterator
    const auto count = thrust::make_counting_iterator(0);
    // Iterate over the source regions and their sizes simultaneously
    auto map_begin = detail::zip_it(count, d_targets.begin());
    thrust::transform(
      map_begin,
      map_begin + nsegments,
      d_segment_sizes.begin(),
      d_targets.begin(),
      [=] __device__(const thrust::tuple<int, int>& map, int size) {
        return size < P ? map.get<1>() : map.get<0>();
      });
  }
  // Remove any cyclic mappings, as they would cause oscillations
  detail::fix_map_cycles(d_targets.begin(), d_targets.end());
  // Map everything to it's target
  detail::map_until_converged(dio_segment_labels.values.begin(),
                              dio_segment_labels.values.end(),
                              d_targets.begin(),
                              d_label_copy.begin());
}

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

