#include "split/device/kmeans/uniform_random_initialize.cuh"
#include "split/device/detail/zip_it.cuh"
#include <thrust/random.h>
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
SPLIT_API void uniform_random_initialize(
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array2d<real, cusp::device_memory>::view do_centroids)
{
  // Get these up front
  const int ndimensions = di_points.num_rows;
  const int npoints = di_points.num_cols;
  const int ncentroids = do_centroids.num_cols;

  // Pointer to the centroids
  auto centroid_ptr = do_centroids.values.begin().base().get();
  // Pointer to the points
  auto point_ptr = di_points.values.begin().base().get();

  auto point_begin = detail::zip_it(di_points.row(0).begin(),
                                    di_points.row(1).begin(),
                                    di_points.row(2).begin());
  auto point_end = point_begin + di_points.num_cols;
  // cusp::array2d<real, cusp::device_memory> unique_points(di_points.num_rows,
  //                                                       di_points.num_cols);
  // auto unique_begin = detail::zip_it(unique_points.row(0).begin(),
  //                                   unique_points.row(1).begin(),
  //                                   unique_points.row(2).begin());
  // auto unique_end = thrust::unique_copy(point_begin, point_end,
  // unique_begin); auto uniquep_ptr = unique_points.values.begin().base().get();
  // const int nunique = unique_end - unique_begin;
  // std::cout<<nunique<<'\n';

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
