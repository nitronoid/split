#include "split/device/kmeans/centroids.cuh"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
SPLIT_API void calculate_centroids(
  cusp::array1d<int, cusp::device_memory>::const_view di_cluster_labels,
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::view
    do_centroids,
  cusp::array1d<int, cusp::device_memory>::view do_temp)
{
  const int nlabels = di_cluster_labels.size();

  auto labels_copy = do_temp.subarray(0, nlabels);
  thrust::copy(
    di_cluster_labels.begin(), di_cluster_labels.end(), labels_copy.begin());

  auto indices = do_temp.subarray(nlabels, nlabels);
  thrust::sequence(indices.begin(), indices.end());

  // Iterate over rgb channels at once
  thrust::fill(do_centroids.values.begin(), do_centroids.values.end(), 0.f);
  auto centroid_it = thrust::make_zip_iterator(
    thrust::make_tuple(do_centroids.column(0).begin(),
                       do_centroids.column(1).begin(),
                       do_centroids.column(2).begin()));
  auto pixel_it =
    thrust::make_zip_iterator(thrust::make_tuple(di_points.row(0).begin(),
                                                 di_points.row(1).begin(),
                                                 di_points.row(2).begin()));

  // Sort by label
  thrust::sort_by_key(labels_copy.begin(), labels_copy.end(), indices.begin());

  cusp::array1d<int, cusp::device_memory> valence(labels_copy.back() + 1);
  // Calculate a dense histogram to find the cumulative valence
  // Create a counting iter to output the index values from the upper_bound
  thrust::counting_iterator<int> search_begin(0);
  thrust::upper_bound(labels_copy.begin(),
                      labels_copy.end(),
                      search_begin,
                      search_begin + valence.size(),
                      valence.begin());
  // Calculate the non-cumulative valence by subtracting neighbouring elements
  thrust::adjacent_difference(valence.begin(), valence.end(), valence.begin());

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

