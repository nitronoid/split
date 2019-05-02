#include "split/device/ccl/merge_smooth_boundaries.cuh"
#include <cusp/graph/connected_components.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
namespace
{
}

SPLIT_API void merge_smooth_boundaries(
  cusp::array2d<int, cusp::device_memory>::view dio_segment_labels,
  real D)
{
}

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END


