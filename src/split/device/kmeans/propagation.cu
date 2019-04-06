#include "split/device/kmeans/propagation.cuh"
#include "split/device/cuda_raii.cuh"
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
  const int ndimensions = di_centroids.num_rows;
  // FIXME: VLA's not allowed in C++
  ScopedCuStream s[ndimensions];
  using thrust::cuda::par;

  for (int i = 0; i < ndimensions; ++i)
  {
    auto centroid_begin = di_centroids.column(i).begin();
    auto point_begin = do_points.row(i).begin();
    thrust::gather(par.on(s[i]),
                   di_cluster_labels.begin(),
                   di_cluster_labels.end(),
                   centroid_begin,
                   point_begin);
  }
  // Wait for all of our dimension tasks to complete
  std::for_each(s, s + ndimensions, [](ScopedCuStream& s) { s.join(); });
}

}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

