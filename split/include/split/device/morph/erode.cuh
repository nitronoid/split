#if !defined(SPLIT_DEVICE_INCLUDED_MORPH_ERODE)
#define SPLIT_DEVICE_INCLUDED_MORPH_ERODE

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace morph
{
SPLIT_API void erode_horizontal_sweep(
  cusp::array2d<int, cusp::device_memory>::view dio_labels,
  const int niterations = 1);

SPLIT_API void erode(
  cusp::array2d<int, cusp::device_memory>::view dio_labels,
  const int niterations = 1);

}  // namespace morph

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_MORPH_ERODE

