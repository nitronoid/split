#if !defined(SPLIT_DEVICE_INCLUDED_INTRINSIC_ESTIMATE_ALBEDO_INTENSITY)
#define SPLIT_DEVICE_INCLUDED_INTRINSIC_ESTIMATE_ALBEDO_INTENSITY

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace intrinsic
{
/***
 ***/
SPLIT_API void estimate_albedo_intensity(
  cusp::array2d<real, cusp::device_memory>::const_view di_intensity,
  cusp::array2d<real, cusp::device_memory>::const_view di_chroma,
  cusp::array1d<real, cusp::device_memory>::view dio_albedo_intensity,
  const int i_nslots = 20,
  const int i_niterations = 5);

}  // namespace intrinsic

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_INTRINSIC_ESTIMATE_ALBEDO_INTENSITY

