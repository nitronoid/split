#if !defined(SPLIT_DEVICE_INCLUDED_KMEANS_CLUSTER)
#define SPLIT_DEVICE_INCLUDED_KMEANS_CLUSTER

#include "split/detail/internal.h"
#include "split/device/cuda_raii.cuh"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
/***
   @brief Performs the clustering of our points, by iteratively labeling and
   recalculating the centroids. Stopping criteria is supplied through setting
   the maximum iterations and threshold.
   @param di_points A 2D, dense, row major matrix containing the points to
   cluster. This data is stored with each dimension of the data point being
   represented by a new row. This matrix is stored in device memory.
   @param dio_centroids A 2D, dense, row major matrix containing the
   initial seeded centroids. The centroids are recalculated using the labeled
   clusters, every iteration and written back here. This matrix is stored in
   device memory.
   @param do_cluster_labels A dense array containing the labels for each point,
   indicating which cluster the point belongs to. This array is stored in device
   memory.
   @param do_temp A pointer to temporary workspace storage.
   @param i_max_iter The maximum number of iterations to perform if convergence
   isn't found.
   @param i_threshold The threshold for error when checking for convergence.
   ***/
SPLIT_API void
cluster(cusp::array2d<real, cusp::device_memory>::const_view di_points,
        cusp::array2d<real, cusp::device_memory>::view dio_centroids,
        cusp::array1d<int, cusp::device_memory>::view do_cluster_labels,
        thrust::device_ptr<void> do_temp,
        int i_max_iter,
        real i_threshold = 1.f);

/***
   @brief Calculates the amount of temporary memory required by the
   cluster function.
   @param i_npoints The number of data points to be supplied to cluster.
   @param i_nclusters The number of clusters to be produced.
   @param i_dim The dimension of the data.
 ***/
SPLIT_API std::size_t
cluster_workspace(const int i_npoints, const int i_nclusters, const int i_dim);

}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_KMEANS_CLUSTER

