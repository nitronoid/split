#if !defined(SPLIT_DEVICE_INCLUDED_KMEANS_CENTROIDS)
#define SPLIT_DEVICE_INCLUDED_KMEANS_CENTROIDS

#include "split/detail/internal.h"
#include "split/device/cuda_raii.cuh"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
/***
   @brief Calculates the centroids of each labeled cluster.
   @param di_labels A dense array containing the labels for each point,
   indicating which cluster the point belongs to. This array is stored in device
   memory.
   @param di_points A 2D, dense, row major matrix containing the points to
   cluster. This data is stored with each dimension of the data point being
   represented by a new row. This matrix is stored in device memory.
   @param do_centroids A 2D, dense, row major matrix containing the centroids
   calculated using the labeled clusters. This matrix is stored in device
   memory.
   @param do_temp A pointer to temporary storage, so that we can perform a radix
   sort without allocation. This memory is stored on the device.
   ***/
SPLIT_API void calculate_centroids(
  cusp::array1d<int, cusp::device_memory>::const_view di_labels,
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array2d<real, cusp::device_memory>::view do_centroids,
  thrust::device_ptr<void> do_temp);

/***
   @brief Calculates the amount of temporary memory required by the
   calculate_centroids function.
   @param i_npoints The number of data points to be supplied to
 calculate_centroids.
   @param i_ncentroids The number of centroids to be calculated.
 ***/
SPLIT_API std::size_t calculate_centroids_workspace(const int i_npoints,
                                                    const int i_ncentroids);

}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_KMEANS_CENTROIDS

