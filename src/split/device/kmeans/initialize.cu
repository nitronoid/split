#include "split/device/kmeans/initialize.cuh"
#include <thrust/random.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace
{
struct RandomPointSelector
{
  thrust::default_random_engine rng;
  thrust::uniform_int_distribution<int> dist;
  thrust::device_ptr<const real> points;
  int num_pixels;

  __host__ __device__ RandomPointSelector(
    cusp::array2d<real, cusp::device_memory>::const_view di_points, int seed)
    : rng(seed)
    , dist(0, di_points.num_cols)
    , points(di_points.values.begin().base())
    , num_pixels(di_points.num_cols)
  {
  }

  __host__ __device__ thrust::tuple<real, real, real> operator()(int n)
  {
    rng.discard(n);
    const int id = dist(rng);
    return thrust::make_tuple(points[num_pixels * 0 + id],
                              points[num_pixels * 1 + id],
                              points[num_pixels * 2 + id]);
  }
};
}  // namespace

namespace kmeans
{
SPLIT_API void initialize_centroids(
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::view
    do_centroids)
{
  auto centroid_it = thrust::make_zip_iterator(
    thrust::make_tuple(do_centroids.column(0).begin(),
                       do_centroids.column(1).begin(),
                       do_centroids.column(2).begin()));
  thrust::tabulate(centroid_it,
                   centroid_it + do_centroids.num_rows,
                   RandomPointSelector(di_points, time(NULL)));
}
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END
