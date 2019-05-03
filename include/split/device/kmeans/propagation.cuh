#if !defined(SPLIT_DEVICE_INCLUDED_KMEANS_PROPAGATION)
#define SPLIT_DEVICE_INCLUDED_KMEANS_PROPAGATION

#include "split/detail/internal.h"
#include "split/device/cuda_raii.cuh"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
/***
   @brief Duplicates the centroid values to each of it's corresponding labels.
   @param di_cluster_labels A dense array containing the labels for each point,
   indicating which cluster the point belongs to. This array is stored in device
   memory.
   @param di_centroids A 2D, dense, column major matrix containing the centroids
   of each cluster. This matrix is stored in device memory.
   @param do_points A 2D, dense, row major matrix containing the propagated,
   centroid value as dictated by the corresponding label.
   This matrix is stored in device memory.
   ***/
SPLIT_API void propagate_centroids(
  cusp::array1d<int, cusp::device_memory>::const_view di_cluster_labels,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::const_view
    di_centroids,
  cusp::array2d<real, cusp::device_memory>::view do_points);
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_KMEANS_PROPAGATION

