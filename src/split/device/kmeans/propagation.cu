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
  // Iterate over rgb channels at once
  auto centroid_it = thrust::make_zip_iterator(
    thrust::make_tuple(di_centroids.column(0).begin(),
                       di_centroids.column(1).begin(),
                       di_centroids.column(2).begin()));
  auto pixel_it =
    thrust::make_zip_iterator(thrust::make_tuple(do_points.row(0).begin(),
                                                 do_points.row(1).begin(),
                                                 do_points.row(2).begin()));

  thrust::gather(
    di_cluster_labels.begin(), di_cluster_labels.end(), centroid_it, pixel_it);
}

}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

