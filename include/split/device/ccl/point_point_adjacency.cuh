#if !defined(SPLIT_DEVICE_INCLUDED_CCL_POINT_POINT_ADJACENCY)
#define SPLIT_DEVICE_INCLUDED_CCL_POINT_POINT_ADJACENCY

#include "split/detail/internal.h"
#include "split/device/cuda_raii.cuh"
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
/***
   @brief Builds a sparse matrix of adjacent points based on their 
   8-neighborhood and cluster labels.
   @param di_labels A dense array containing the cluster labels for
   each point, indicating which cluster the point belongs to.
   @param do_adjacency A sparse matrix, #Npoints x #Npoints with #Npoints*8 
   entries. This matrix is stored in device memory.
   ***/
SPLIT_API int point_point_adjacency(
  cusp::array2d<int, cusp::device_memory>::const_view di_labels,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_adjacency);

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_CCL_POINT_POINT_ADJACENCY

