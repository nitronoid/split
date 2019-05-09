#include "split/device/kmeans/cluster.cuh"
#include <cusp/print.h>
#include "split/device/kmeans/centroids.cuh"
#include "split/device/kmeans/label.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/view_util.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
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

  // Reuse the temporary storage again here to store our old centroids
  // NOTE: we offset the beginning of the sub-array as the start of our temp
  // memory is used for a radix sort every iteration.
  auto d_rtemp = thrust::device_pointer_cast(
    static_cast<real*>(do_temp.get()) + npoints * 2);
  auto d_old_centroids = cusp::make_array2d_view(
    dio_centroids.num_rows,
    dio_centroids.num_cols,
    dio_centroids.num_cols,
    cusp::make_array1d_view(d_rtemp, d_rtemp + nclusters * dim),
    cusp::row_major{});

  // Need to explicitly create an immutable view over the centroids
  auto d_const_centroids = detail::make_const_array2d_view(dio_centroids);

  auto centroid_it = detail::zip_it(dio_centroids.row(0).begin(),
                                    dio_centroids.row(1).begin(),
                                    dio_centroids.row(2).begin());
  auto old_centroid_it = detail::zip_it(d_old_centroids.row(0).begin(),
                                        d_old_centroids.row(1).begin(),
                                        d_old_centroids.row(2).begin());

  real old_delta, delta = 1.f;
  int iter = 0;
  do
  {
    ++iter;
    // Assign each pixel to it's nearest centroid
    split::device::kmeans::label_points(
      d_const_centroids, di_points, do_cluster_labels, do_temp);
    // Copy our centroids before calculating the new ones
    thrust::copy(dio_centroids.values.begin(),
                 dio_centroids.values.end(),
                 d_old_centroids.values.begin());
    // Calculate the new centroids by averaging all points in every centroid
    split::device::kmeans::calculate_centroids(
      do_cluster_labels, di_points, dio_centroids, do_temp);
    // Calculate the total squared shift in centroids this iteration
    old_delta = delta;
    delta = thrust::inner_product(
      centroid_it,
      centroid_it + nclusters,
      old_centroid_it,
      0.f,
      thrust::plus<real>(),
      [] __device__(const thrust::tuple<real, real, real>& a,
                    const thrust::tuple<real, real, real>& b) -> real {
        thrust::tuple<real, real, real> v;
        v.get<0>() = a.get<0>() - b.get<0>();
        v.get<1>() = a.get<1>() - b.get<1>();
        v.get<2>() = a.get<2>() - b.get<2>();
        return sqrt(v.get<0>() * v.get<0>() + v.get<1>() * v.get<1>() +
                    v.get<2>() * v.get<2>());
      });
  } while ((iter < i_max_iter) && ((delta / old_delta) > i_threshold));

  printf(
    "Achieved variance of: %f, in %d iterations.\n", (delta / old_delta), iter);
}
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

