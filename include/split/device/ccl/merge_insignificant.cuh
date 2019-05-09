#if !defined(SPLIT_DEVICE_INCLUDED_CCL_MERGE_INSIGNIFICANT)
#define SPLIT_DEVICE_INCLUDED_CCL_MERGE_INSIGNIFICANT

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
SPLIT_API void merge_insignificant(
  cusp::array2d<real, cusp::device_memory>::view di_chrominance,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_labels,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_adjacency,
  int P = 10);

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_CCL_MERGE_INSIGNIFICANT

