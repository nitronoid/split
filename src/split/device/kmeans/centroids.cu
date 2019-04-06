#include "split/device/kmeans/centroids.cuh"
#include "split/device/cuda_raii.cuh"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <algorithm>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
SPLIT_API void calculate_centroids(
  cusp::array1d<int, cusp::device_memory>::const_view di_labels,
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::view
    do_centroids,
  cusp::array1d<int, cusp::device_memory>::view do_temp)
{
  // Grab these sizes upfront with more readable names
  const int nlabels = di_labels.size();
  const int ndimensions = di_points.num_rows;

  // FIXME: VLA's are not allowed in C++
  ScopedCuStream s[std::max(2, ndimensions)];
  using thrust::cuda::par;

  // The first chunk of memory will be used to store labels
  auto labels = do_temp.subarray(0, nlabels);
  // Make a copy of the labels for sorting
  thrust::copy(
    par.on(s[0]), di_labels.begin(), di_labels.end(), labels.begin());

  // The next chunk of temp storage will be used to store indices
  auto indices = do_temp.subarray(nlabels, nlabels);
  // Initialize the indices as a sequence
  thrust::sequence(par.on(s[1]), indices.begin(), indices.end());

  // We need to complete the copy and sequence before sorting
  s[0].join();
  s[1].join();
  // Sort the indices by label
  thrust::sort_by_key(
    par.on(s[0]), labels.begin(), labels.end(), indices.begin());

  // Leave the valence uninitialized for now
  auto valence = do_temp.subarray(nlabels * 2, do_centroids.num_rows);
  // Calculate a dense histogram to find the cumulative valence
  // Find the indices where a segment boundary occurs
  thrust::upper_bound(par.on(s[0]),
                      labels.begin(),
                      labels.end(),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator<int>(valence.size()),
                      valence.begin());
  // Calculate the non-cumulative valence by subtracting neighboring elements
  thrust::adjacent_difference(
    par.on(s[0]), valence.begin(), valence.end(), valence.begin());

  s[0].join();
  // Launch a reduction and a transform for each dimension of our data
  // asynchronously. This divides the workload and is flexible for N dimensions.
  for (int i = 0; i < ndimensions; ++i)
  {
    // Get the reordered point iterator for this dimension
    auto point_it = thrust::make_permutation_iterator(di_points.row(i).begin(),
                                                      indices.begin());
    // Get the centroid iterators for this dimension
    auto centroid_begin = do_centroids.column(i).begin();
    auto centroid_end = do_centroids.column(i).end();

    // Grab a handle to this dimensions task
    // Reduce by label to find the totals of all points for each cluster
    auto discard_it = thrust::make_discard_iterator();
    thrust::reduce_by_key(par.on(s[i]),
                          labels.begin(),
                          labels.end(),
                          point_it,
                          discard_it,
                          centroid_begin);
    // Divide through by the number of contributions to each new centroid,
    // but actually multiply by the reciprocal through a transform iterator.
    thrust::transform(
      par.on(s[i]),
      centroid_begin,
      centroid_end,
      thrust::make_transform_iterator(
        valence.begin(),
        [] __host__ __device__(int x) -> real { return 1.f / x; }),
      centroid_begin,
      thrust::multiplies<real>());
  }
  // Wait for all of our dimension tasks to complete
  std::for_each(s, s + ndimensions, [](ScopedCuStream& s) { s.join(); });
}
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

