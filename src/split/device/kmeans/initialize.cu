#include "split/device/kmeans/initialize.cuh"
#include <thrust/random.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
SPLIT_API void initialize_centroids(
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::view
    do_centroids)
{
  // Get these up front
  const int ndimensions = di_points.num_rows;
  const int npoints = di_points.num_cols;
  const int ncentroids = do_centroids.num_rows;

  // Pointer to the centroids
  auto centroid_ptr = do_centroids.values.begin().base().get();
  // Pointer to the points
  auto point_ptr = di_points.values.begin().base().get();

  // For selecting the random points as initial centroids
  thrust::default_random_engine rng(time(NULL));
  thrust::uniform_int_distribution<int> dist(0, npoints);

  auto count = thrust::make_counting_iterator(0);

  thrust::for_each_n(count, ncentroids, [=] __device__(int x) mutable {
    // Get the random point index
    rng.discard(x);
    const int rand_index = dist(rng);
    // For each dimension of the data, copy from the random
    // index
    for (int i = 0; i < ndimensions; ++i)
    {
      centroid_ptr[ncentroids * i + x] = point_ptr[npoints * i + rand_index];
    }
  });
}
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END
