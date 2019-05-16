#if !defined(SPLIT_DEVICE_INCLUDED_CCL_MERGE_SMOOTH_BOUNDARIES)
#define SPLIT_DEVICE_INCLUDED_CCL_MERGE_SMOOTH_BOUNDARIES

#include "split/detail/internal.h"
#include "split/device/cuda_raii.cuh"
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
/***
   @brief Merges any segments which share a smooth boundary. A smooth boundary
   is defined by the average difference across the shared boundary of two 
   segments. If the shared difference < D*max(pixel) then we merge.
   @param dio_segment_labels A dense array containing the segment labels for
   each point, indicating which segment the point belongs to.
   This array is stored in device memory.
   ***/
SPLIT_API void merge_smooth_boundaries(
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  const int i_nsegments,
  cusp::array2d<int, cusp::device_memory>::view dio_labels,
  thrust::device_ptr<void> do_temp,
  const real i_threshold = 0.01f);

SPLIT_API std::size_t merge_smooth_boundaries_workspace(const int i_npoints,
                                                        const int i_nsegments,
                                                        const int i_nedges);

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_CCL_MERGE_SMOOTH_BOUNDARIES

