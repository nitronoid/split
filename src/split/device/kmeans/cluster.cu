#include "split/device/kmeans/cluster.cuh"
#include "split/device/kmeans/centroids.cuh"
#include "split/device/kmeans/label.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
SPLIT_API void
cluster(cusp::array2d<real, cusp::device_memory>::const_view di_points,
        cusp::array2d<real, cusp::device_memory, cusp::column_major>::view
          dio_centroids,
        cusp::array1d<int, cusp::device_memory>::view do_cluster_labels,
        thrust::device_ptr<void> do_temp,
        int i_max_iter,
        real i_threshold)
{
  const int ndimensions = di_points.num_rows;
  cusp::array1d<ScopedCuStream, cusp::host_memory> streams(ndimensions + 1);
  cluster(streams,
          di_points,
          dio_centroids,
          do_cluster_labels,
          do_temp,
          i_max_iter,
          i_threshold);
}

SPLIT_API void
cluster(cusp::array1d<ScopedCuStream, cusp::host_memory>::view io_streams,
        cusp::array2d<real, cusp::device_memory>::const_view di_points,
        cusp::array2d<real, cusp::device_memory, cusp::column_major>::view
          dio_centroids,
        cusp::array1d<int, cusp::device_memory>::view do_cluster_labels,
        thrust::device_ptr<void> do_temp,
        int i_max_iter,
        real i_threshold)
{
  const int npoints = di_points.num_cols;
  const int nclusters = dio_centroids.num_rows;
  const int ndimensions = di_points.num_rows;
  assert(io_streams.size() >= (ndimensions + 1) &&
         "Insufficient number of streams");
  using thrust::cuda::par;

  // Divide the supplied temporary memory into an integer part and a real part
  // The integer part is used for a radix sort by key
  auto d_itemp_ptr =
    thrust::device_pointer_cast<int>(static_cast<int*>(do_temp.get()));
  // Create a view over the integer part
  auto d_itemp =
    cusp::make_array1d_view(d_itemp_ptr, d_itemp_ptr + npoints * 2 + nclusters);

  // Offset the real part pointer by the size of the integer part
  auto d_rtemp_ptr =
    thrust::device_pointer_cast<real>(static_cast<real*>(do_temp.get()));
  // Create a view over the real part
  cusp::array1d<real, cusp::device_memory>::view d_rtemp(
    d_rtemp_ptr, d_rtemp_ptr + npoints * nclusters);

  // Create a dense, 2D, column major matrix view over the temp storage
  auto d_temp_mat = cusp::make_array2d_view(
    npoints, nclusters, 1, d_rtemp, cusp::column_major{});

  // Reuse the temporary storage again here to store our old centroids
  auto d_old_centroids = cusp::make_array2d_view(
    dio_centroids.num_rows,
    dio_centroids.num_cols,
    1,
    d_rtemp.subarray(d_itemp.size(), nclusters * dio_centroids.num_cols),
    cusp::column_major{});

  // Need to explicitly create an immutable view over the centroids
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::const_view
    d_const_centroids(
      dio_centroids.num_rows, dio_centroids.num_cols, 1, dio_centroids.values);

  auto centroid_it = thrust::make_zip_iterator(
    thrust::make_tuple(dio_centroids.column(0).begin(),
                       dio_centroids.column(1).begin(),
                       dio_centroids.column(2).begin()));
  auto old_centroid_it = thrust::make_zip_iterator(
    thrust::make_tuple(d_old_centroids.column(0).begin(),
                       d_old_centroids.column(1).begin(),
                       d_old_centroids.column(2).begin()));

  real old_delta, delta = 1.f;
  int iter = 0;
  do
  {
    ++iter;
    // Copy our centroids before calculating the new ones
    thrust::copy(par.on(io_streams[0]),
                 dio_centroids.values.begin(),
                 dio_centroids.values.end(),
                 d_old_centroids.values.begin());
    // Assign each pixel to it's nearest centroid
    split::device::kmeans::label_points(io_streams[1],
                                        d_const_centroids,
                                        di_points,
                                        do_cluster_labels,
                                        d_temp_mat);
    // Wait for our labeling to complete
    io_streams[1].join();
    // Calculate the new centroids by averaging all points in every centroid,
    // skip the first stream being used for the copy
    split::device::kmeans::calculate_centroids(
      io_streams.subarray(1, ndimensions),
      do_cluster_labels,
      di_points,
      dio_centroids,
      d_itemp);
    // Wait for the new centroids, and the old ones getting copied
    std::for_each(io_streams.begin(), io_streams.end(), [](ScopedCuStream& s) {
      s.join();
    });
    // Calculate the total squared shift in centroids this iteration
    old_delta = delta;
    delta = thrust::inner_product(
      par.on(io_streams[0]),
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
  } while ((iter < i_max_iter) && ((delta / old_delta) < (1.f + i_threshold)));

  std::cout << "Achieved variance of: " << (delta / old_delta) << ", in "
            << iter << " iterations\n";
}
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

