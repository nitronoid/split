#include "split/device/kmeans/centroids.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/cycle_iterator.cuh"
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
  // Store the number of points
  const int npoints = di_points.num_cols;

  // Make a copy of our input labels, storing it in the temp memory provided
  auto labels_copy = do_temp.subarray(0, npoints);
  thrust::copy(di_labels.begin(), di_labels.end(), labels_copy.begin());

  // Initialize the point indices to a standard sequence, for sorting later
  auto indices = do_temp.subarray(npoints, npoints);
  thrust::sequence(indices.begin(), indices.end());

  // Sort by label, we use the copy here to avoid modifying the input labels
  thrust::sort_by_key(labels_copy.begin(), labels_copy.end(), indices.begin());

  // Create a new sub view into our temp memory for storing cluster valence
  auto valence = do_temp.subarray(npoints * 2, labels_copy.back() + 1);
  // Create a counting iter to output the index values from the upper_bound
  auto search_begin = thrust::make_counting_iterator(0);
  // Calculate a dense histogram to find the cumulative valence
  thrust::upper_bound(labels_copy.begin(),
                      labels_copy.end(),
                      search_begin,
                      search_begin + valence.size(),
                      valence.begin());
  // Calculate the non-cumulative valence by subtracting neighboring elements
  thrust::adjacent_difference(valence.begin(), valence.end(), valence.begin());

  // Initialize the centroids to the origin, in-case of no points belonging to
  // that cluster, we have zero rather than an uninitialized value
  thrust::fill(do_centroids.values.begin(), do_centroids.values.end(), 0.f);

  // Iterate over all channels at once
  auto centroid_it = detail::zip_it(do_centroids.column(0).begin().base(),
                                    do_centroids.column(1).begin().base(),
                                    do_centroids.column(2).begin().base());
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

  // auto valence_it = thrust::make_transform_iterator(
  //  thrust::make_permutation_iterator(
  //    valence.begin(), detail::make_cycle_iterator(valence.size())),
  //  [] __host__ __device__(int x) -> real { return 1.f / x; });

  // std::cout << valence_it[0] << ' ' << valence_it[1] << '\n';
  // valence_it += valence.size();
  // std::cout << valence_it[0] << ' ' << valence_it[1] << '\n';
  // valence_it += valence.size();
  // std::cout << valence_it[0] << ' ' << valence_it[1] << '\n';

  // thrust::transform(do_centroids.values.begin(),
  //                  do_centroids.values.end(),
  //                  valence_it,
  //                  do_centroids.values.begin(),
  //                  thrust::multiplies<real>());

  thrust::transform(
    centroid_it,
    centroid_it + do_centroids.num_rows,
    thrust::make_transform_iterator(
      valence.begin(),
      [] __host__ __device__(int x) -> real { return 1.f / x; }),
    centroid_it,
    [] __device__(const thrust::tuple<real, real, real>& v, real c) {
      return thrust::make_tuple(v.get<0>() * c, v.get<1>() * c, v.get<2>() * c);
    });
}
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

