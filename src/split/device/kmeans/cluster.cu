#include "split/device/kmeans/cluster.cuh"
#include <cusp/print.h>
#include "split/device/kmeans/centroids.cuh"
#include "split/device/kmeans/label.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/view_util.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
SPLIT_API std::size_t
cluster_workspace(const int i_npoints, const int i_nclusters, const int i_dim)
{
  // Calculate how much memory is required by the two functions we'll call
  const std::size_t labeling_workspace =
    label_points_workspace(i_npoints, i_nclusters);
  const std::size_t centroid_workspace =
    calculate_centroids_workspace(i_npoints, i_nclusters);
  // We also store old centroids, so need to account for that.
  const std::size_t this_workspace = i_nclusters * i_dim * sizeof(real);
  // We'll never use the labeling workspace at the same time as the other two,
  // so we can simply take which ever is bigger, and use it for both
  return std::max(this_workspace + centroid_workspace, labeling_workspace);
}

SPLIT_API void
cluster(cusp::array2d<real, cusp::device_memory>::const_view di_points,
        cusp::array2d<real, cusp::device_memory>::view dio_centroids,
        cusp::array1d<int, cusp::device_memory>::view do_cluster_labels,
        thrust::device_ptr<void> do_temp,
        int i_max_iter,
        real i_threshold)
{
  const int npoints = di_points.num_cols;
  const int nclusters = dio_centroids.num_cols;
  const int dim = dio_centroids.num_rows;

  // Organize the provided temporary memory into something usable
  auto d_rtemp = thrust::device_pointer_cast(static_cast<real*>(do_temp.get()) +
                                             npoints * 2);
  // Reuse the temporary storage again here to store our old centroids
  // NOTE: we offset the beginning of the sub-array as the start of our temp
  // memory is used for a radix sort every iteration.
  auto d_old_centroids = cusp::make_array2d_view(
    dim,
    nclusters,
    nclusters,
    cusp::make_array1d_view(d_rtemp, d_rtemp + nclusters * dim),
    cusp::row_major{});

  // Pair of iterators that will be used to check convergence
  auto squared_diff_begin = thrust::make_transform_iterator(
    detail::zip_it(dio_centroids.values.begin(),
                   d_old_centroids.values.begin()),
    [] __host__ __device__(const thrust::tuple<real, real>& pair) {
      const real diff = pair.get<0>() - pair.get<1>();
      return diff * diff;
    });
  auto squared_diff_end = squared_diff_begin + dio_centroids.num_entries;

  // Need to explicitly create an immutable view over the centroids
  auto d_const_centroids = detail::make_const_array2d_view(dio_centroids);

  // We'll use this to store th shift in centroids
  real delta = 0.f;
  real old_delta = delta + i_threshold + 1.f;
  int iter = 0;

  // Check for convergence
  while ((iter < i_max_iter) && (abs(delta - old_delta) > i_threshold))
  {
    ++iter;
    // Copy our centroids before calculating the new ones
    thrust::copy(dio_centroids.values.begin(),
                 dio_centroids.values.end(),
                 d_old_centroids.values.begin());
    // Assign each pixel to it's nearest centroid
    split::device::kmeans::label_points(
      d_const_centroids, di_points, do_cluster_labels, do_temp);
    // Calculate the new centroids by averaging all points in every centroid
    split::device::kmeans::calculate_centroids(
      do_cluster_labels, di_points, dio_centroids, do_temp);
    // Save the old delta
    old_delta = delta;
    // Calculate the total squared shift in centroids this iteration
    delta = std::sqrt(thrust::reduce(squared_diff_begin, squared_diff_end));
    std::cout<< delta<<'\n';
  }

  printf(
    "Achieved delta of: %f, in %d iterations.\n", abs(delta - old_delta), iter);
}
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

