#include "split/device/kmeans/propagation.cuh"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/gather.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
SPLIT_API void propagate_centroids(
  cusp::array1d<int, cusp::device_memory>::const_view di_cluster_labels,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::const_view
    di_centroids,
  cusp::array2d<real, cusp::device_memory>::view do_points)
{
  const int ndimensions = do_points.num_rows;
  cusp::array1d<ScopedCuStream, cusp::host_memory> streams(ndimensions);
  propagate_centroids(streams, di_cluster_labels, di_centroids, do_points);
  // Wait for all of our dimension tasks to complete
  std::for_each(
    streams.begin(), streams.end(), [](ScopedCuStream& s) { s.join(); });
}

SPLIT_API void propagate_centroids(
  cusp::array1d<ScopedCuStream, cusp::host_memory>::view io_streams,
  cusp::array1d<int, cusp::device_memory>::const_view di_cluster_labels,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::const_view
    di_centroids,
  cusp::array2d<real, cusp::device_memory>::view do_points)
{
  const int ndimensions = di_centroids.num_cols;
  assert(io_streams.size() >= ndimensions && "Insufficient number of streams");
  using thrust::cuda::par;

  // Launch a kernel for each dimension on a new thread
  for (int i = 0; i < ndimensions; ++i)
  {
    auto centroid_begin = di_centroids.column(i).begin();
    auto point_begin = do_points.row(i).begin();
    thrust::gather(par.on(io_streams[i]),
                   di_cluster_labels.begin(),
                   di_cluster_labels.end(),
                   centroid_begin,
                   point_begin);
  }
}

}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

