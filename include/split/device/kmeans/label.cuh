#if !defined(SPLIT_DEVICE_INCLUDED_KMEANS_LABEL)
#define SPLIT_DEVICE_INCLUDED_KMEANS_LABEL

#include "split/detail/internal.h"
#include "split/device/cuda_raii.cuh"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
/***
   @brief Calculates the centroids of each labeled cluster.
   @param di_centroids A 2D, dense, column major matrix containing the centroids
   of each cluster. This matrix is stored in device
   memory.
   @param di_points A 2D, dense, row major matrix containing the points to
   cluster. This data is stored with each dimension of the data point being
   represented by a new row. This matrix is stored in device memory.
   @param do_cluster_labels A dense array containing the labels calculated for
   each point, indicating which cluster the point belongs to. This array is
   stored in device memory.
   @param do_temp A 2D, dense, column major matrix used for temporary storage.
   The memory is used to store the squared distances from each point to each
   centroid. This matrix is stored in device memory.
   ***/
SPLIT_API void label_points(
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::const_view
    di_centroids,
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array1d<int, cusp::device_memory>::view do_cluster_labels,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::view do_temp);
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_KMEANS_LABEL

