#include "split/device/ccl/merge_insignificant.cuh"
#include <cusp/graph/connected_components.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
namespace
{
struct AddPair
{
  using pair = thrust::tuple<real, real>;

  __host__ __device__ pair operator()(const pair& lhs, const pair& rhs) const
  {
    const real x = lhs.get<0>() + rhs.get<0>();
    const real y = lhs.get<1>() + rhs.get<1>();
    // Add both channels
    return thrust::make_tuple(x, y);
  }
};

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

struct MergeSegment
{
  __host__ __device__ int operator()(const thrust::tuple<int, int>& map,
                                     int target_target) const
  {
    const int current = map.get<0>();
    const int target = map.get<1>();
    // if the target is attempting to merge into us simultaneously, we select
    // the segment with the larger index to prevent oscillations
    if (current == target_target)
      return max(current, target);
    return target;
  }
};

struct TargetMap
{
  const int thresh;

  // Packed => { target, current, size }
  __host__ __device__ int operator()(const thrust::tuple<int, int, int>& tup)
  {
    return tup.get<2>() > thresh ? tup.get<1>() : tup.get<0>();
  }
};

template <typename... Args>
auto zip_it(Args&&... args) -> decltype(
  thrust::make_zip_iterator(thrust::make_tuple(std::forward<Args>(args)...)))
{
  return thrust::make_zip_iterator(
    thrust::make_tuple(std::forward<Args>(args)...));
}

}  // namespace

SPLIT_API void merge_insignificant(
  cusp::array1d<real, cusp::device_memory>::view di_chrominance,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_labels,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_adjacency,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_size,
  int P)
{
  // Get these sizes upfront
  const int nsegments = dio_segment_adjacency_keys.back() + 1;
  const int npoints = dio_segment_labels.size();
  // Push these into temp storage param eventually
  cusp::array1d<real, cusp::device_memory> total_chrominance(nsegments * 2);
  cusp::array1d<int, cusp::device_memory> indices(npoints);
  cusp::array1d<int, cusp::device_memory> labels(npoints);
  // Target array contains the target segment to join with, initially no change
  cusp::array1d<int, cusp::device_memory> d_targets = dio_segment_labels;

  // Useful iterators

  // Initialize the indices with a standard sequence
  thrust::sequence(indices.begin(), indices.end());
  // Copy our labels for sorting
  thrust::copy(
    dio_segment_labels.begin(), dio_segment_labels.end(), labels.begin());
  // Sort the indices using the labels
  thrust::sort_by_key(labels.begin(), labels.begin(), indices.begin());

  // Counting iterator
  auto count = thrust::make_counting_iterator(0);
  // Segment size range iterators
  auto size_begin = dio_segment_size.begin();
  auto size_end = dio_segment_size.end();
  // Compute the segment sizes
  thrust::upper_bound(
    count, count + nsegments, labels.begin(), labels.end(), size_begin);
  thrust::adjacent_difference(size_begin, size_end, size_begin);

  // Access the chrominance using the sorted indices
  auto value_it = thrust::make_permutation_iterator(
    zip_it(di_chrominance.begin(), di_chrominance.begin() + npoints),
    indices.begin());
  auto total_it =
    zip_it(total_chrominance.begin(), total_chrominance.begin() + nsegments);
  auto discard_it = thrust::make_discard_iterator();
  // Reduce all segments to get their total chrominance
  thrust::reduce_by_key(labels.begin(),
                        labels.end(),
                        value_it,
                        discard_it,
                        total_it,
                        thrust::equal_to<int>(),
                        AddPair{});

  // Iterator to access the average chrominance of each segment
  auto average_chrominance = thrust::make_transform_iterator(
    zip_it(total_chrominance.begin(),
           total_chrominance.begin() + nsegments,
           size_begin),
    [] __device__(const thrust::tuple<real, real, int>& tc) {
      return thrust::make_tuple(tc.get<0>() / tc.get<2>(),
                                tc.get<1>() / tc.get<2>());
    });
  // Get matrix values as squared distance in chrominance space
  auto entry_it = thrust::make_transform_iterator(
    zip_it(thrust::make_permutation_iterator(
             average_chrominance, dio_segment_adjacency_keys.begin()),
           thrust::make_permutation_iterator(average_chrominance,
                                             dio_segment_adjacency.begin()),
           dio_segment_adjacency.begin()),
    ChominanceDistance2{});

  // Reduce by column to find the lowest distance, and hence nearest in
  // chrominance space to our segment, this is the segment we want to merge
  // with.
  thrust::reduce_by_key(dio_segment_adjacency_keys.begin(),
                        dio_segment_adjacency_keys.end(),
                        entry_it,
                        discard_it,
                        zip_it(discard_it, d_targets.begin()),
                        thrust::equal_to<int>(),
                        thrust::minimum<thrust::tuple<real, int>>());

  // We have converged if all segments have size greater than or equal to P
  auto has_converged = [old_labels = labels.begin(),
                        new_labels = dio_segment_labels.begin(),
                        npoints = npoints] {
    return thrust::equal(new_labels, new_labels + npoints, old_labels);
  };
  // Iterate over the target and current labels, with the current segment size,
  // using a transform functor to decide the final target to write
  auto target_it = thrust::make_transform_iterator(
    thrust::make_permutation_iterator(
      zip_it(d_targets.begin(), dio_segment_labels.begin(), size_begin),
      dio_segment_labels.begin()),
    TargetMap{P});
  // An iterator that provides a mapping from current to target labels
  auto map_it = zip_it(dio_segment_labels.begin(), target_it);
  // Loop until convergence
  while (!has_converged())
  {
    thrust::copy_n(dio_segment_labels.begin(), npoints, labels.begin());
    // Merge segments by replacing their labels with the target labels, if the
    // segment is small (size < P)
    thrust::transform(
      map_it,
      map_it + nsegments,
      // Targets targets
      thrust::make_permutation_iterator(d_targets.begin(), target_it),
      dio_segment_labels.begin(),
      MergeSegment{});
  }
}

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

