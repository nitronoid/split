#if !defined(SPLIT_DEVICE_INCLUDED_CCL_MERGE_SMALL_SEGMENTS)
#define SPLIT_DEVICE_INCLUDED_CCL_MERGE_SMALL_SEGMENTS

#include "split/detail/internal.h"
#include "split/device/cuda_raii.cuh"
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
/***
   @brief Merges any small segments (size < P), with their nearest neighbor,
   measured via closest average chrominance.
   ***/
SPLIT_API void merge_small_segments(
  cusp::array2d<real, cusp::device_memory>::const_view di_chrominance,
  cusp::array1d<int, cusp::device_memory>::const_view di_segment_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::const_view di_segment_adjacency,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_labels,
  thrust::device_ptr<void> do_temp,
  int P = 10);

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_CCL_MERGE_SMALL_SEGMENTS

