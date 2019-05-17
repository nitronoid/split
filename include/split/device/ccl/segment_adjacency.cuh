#if !defined(SPLIT_DEVICE_INCLUDED_CCL_SEGMENT_ADJACENCY)
#define SPLIT_DEVICE_INCLUDED_CCL_SEGMENT_ADJACENCY

#include "split/detail/internal.h"
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
/***
   @brief Builds an edge list of pixels that connect two segments.
   ***/
SPLIT_API int segment_adjacency_edges(
  cusp::array2d<int, cusp::device_memory>::const_view di_labels,
  cusp::array1d<int, cusp::device_memory>::view do_edges);

/***
   @brief Builds adjacency lists of the segment to segment adjacency.
   ***/
SPLIT_API int segment_adjacency(
  cusp::array2d<int, cusp::device_memory>::const_view di_labels,
  cusp::array1d<int, cusp::device_memory>::view do_segment_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view do_segment_adjacency);

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_CCL_SEGMENT_ADJACENCY

