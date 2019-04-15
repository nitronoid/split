#if !defined(SPLIT_DEVICE_INCLUDED_CCL_CONNECTED_COMPONENTS)
#define SPLIT_DEVICE_INCLUDED_CCL_CONNECTED_COMPONENTS

#include "split/detail/internal.h"
#include "split/device/cuda_raii.cuh"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
/***
   @brief Builds a list of labels, representing isolated segments within the
   clustered data. A segment is isolated when it has a closed border with 
   clusters it doesn't belong to.
   @param di_labels A dense, 2D, row major matrix containing the cluster labels
   for each point. This array is stored in device memory.
   @param dio_temp A pointer to a memory block for use in this function.
   @param do_labels A dense array containing the segment labels for each point,
   indicating which segment the point belongs to. This array is stored in device
   memory.
   @param i_max_iterations The maximum number of iterations that should be
   performed when convergence isn't reached.
   ***/
SPLIT_API void connected_components(
  cusp::array2d<int, cusp::device_memory>::const_view di_labels,
  thrust::device_ptr<void> dio_temp,
  cusp::array1d<int, cusp::device_memory>::view do_labels,
  int i_max_iterations = 10);

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_CCL_CONNECTED_COMPONENTS

