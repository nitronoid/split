#if !defined(SPLIT_DEVICE_INCLUDED_KMEANS_INITIALIZE)
#define SPLIT_DEVICE_INCLUDED_KMEANS_INITIALIZE

#include "split/detail/internal.h"
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
/***
   @brief Initializes centroids using randomly selected sample points.
   @param di_points A 2D, dense, row major matrix containing the points to
   cluster. This data is stored with each dimension of the data point being
   represented by a new row. This matrix is stored in device memory.
   @param do_centroids A 2D, dense, row major matrix containing the randomly
   selected points, acting as the initial seeding for clustering. This matrix is
   stored in device memory.
   ***/
SPLIT_API void initialize_centroids(
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array2d<real, cusp::device_memory>::view do_centroids);
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_KMEANS_INITIALIZE
