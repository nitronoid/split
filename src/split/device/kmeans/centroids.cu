#include "split/device/kmeans/centroids.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>
#include <algorithm>
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
SPLIT_API std::size_t calculate_centroids_workspace(const int i_npoints,
                                                    const int i_ncentroids)
{
  return i_npoints * 2 * sizeof(int) + i_ncentroids * sizeof(real);
}

SPLIT_API void calculate_centroids(
  cusp::array1d<int, cusp::device_memory>::const_view di_labels,
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array2d<real, cusp::device_memory>::view do_centroids,
  thrust::device_ptr<void> do_temp)
{
  // Store the number of points
  const int npoints = di_points.num_cols;
  // Store the number of centroids
  const int ncentroids = do_centroids.num_cols;

  // Cast our temp memory to integer storage
  auto itemp = thrust::device_pointer_cast(static_cast<int*>(do_temp.get()));
  // Cast our temp memory to real storage
  auto rtemp = thrust::device_pointer_cast(static_cast<real*>(do_temp.get()));
  // Make a copy of our input labels, storing it in the temp memory provided
  auto labels_copy = cusp::make_array1d_view(itemp, itemp + npoints);
  thrust::copy(di_labels.begin(), di_labels.end(), labels_copy.begin());
  // Initialize the point indices to a standard sequence, for sorting later
  auto indices = cusp::make_array1d_view(itemp + npoints, itemp + npoints * 2);
  thrust::sequence(indices.begin(), indices.end());
  // Create a new sub view into our temp memory for storing cluster valence
  auto rvalence = cusp::make_array1d_view(rtemp + npoints * 2,
                                          rtemp + npoints * 2 + ncentroids);

  // Sort by label, we use the copy here to avoid modifying the input labels
  thrust::sort_by_key(labels_copy.begin(), labels_copy.end(), indices.begin());

  // Create a counting iter to output the index values from the upper_bound
  auto search_begin = thrust::make_counting_iterator(0);
  // Calculate a dense histogram to find the cumulative valence
  thrust::upper_bound(labels_copy.begin(),
                      labels_copy.end(),
                      search_begin,
                      search_begin + rvalence.size(),
                      rvalence.begin());
  // Calculate the non-cumulative valence by subtracting neighboring elements,
  // and then write out the reciprocal using a transform functor
  thrust::adjacent_difference(rvalence.begin(),
                              rvalence.end(),
                              thrust::make_transform_output_iterator(
                                rvalence.begin(), detail::reciprocal<real>()));

  // Initialize the centroids to the origin, in-case of no points belonging to
  // that cluster, we have zero rather than an uninitialized value
  thrust::fill(do_centroids.values.begin(), do_centroids.values.end(), 0.f);

  // Iterate over all channels at once
  auto centroid_it = detail::zip_it(do_centroids.row(0).begin().base(),
                                    do_centroids.row(1).begin().base(),
                                    do_centroids.row(2).begin().base());
  auto pixel_it = detail::zip_it(di_points.row(0).begin(),
                                 di_points.row(1).begin(),
                                 di_points.row(2).begin());

  // Reduce all points by their cluster label, to produce a total
  thrust::reduce_by_key(
    labels_copy.begin(),
    labels_copy.end(),
    thrust::make_permutation_iterator(pixel_it, indices.begin()),
    thrust::make_discard_iterator(),
    centroid_it,
    thrust::equal_to<int>(),
    [] __device__(const thrust::tuple<real, real, real>& lhs,
                  const thrust::tuple<real, real, real>& rhs) {
      return thrust::make_tuple(lhs.get<0>() + rhs.get<0>(),
                                lhs.get<1>() + rhs.get<1>(),
                                lhs.get<2>() + rhs.get<2>());
    });

  // Loop over the reciprocal valence for each dimension of the input data
  auto rvalence_it =
    detail::make_cycle_iterator(rvalence.begin(), rvalence.size());
  // Divide each dimension of the totals, by their number of contributions
  thrust::transform(do_centroids.values.begin(),
                    do_centroids.values.end(),
                    rvalence_it,
                    do_centroids.values.begin(),
                    thrust::multiplies<real>());
}
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

