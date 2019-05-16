#if !defined(SPLIT_DEVICE_INCLUDED_SEPARATION_LUMINANCE_CONTINUITY)
#define SPLIT_DEVICE_INCLUDED_SEPARATION_LUMINANCE_CONTINUITY

#include "split/detail/internal.h"
#include "split/device/cuda_raii.cuh"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace separation
{
/***
   @brief Computes the left hand side A matrix, and the right hand side b
   vector, for use in solving an intrinsic separation via luminance continuity
   equations. These equations are log space values.
   ***/
SPLIT_API std::tuple<cusp::array2d<real, cusp::device_memory>,
                     cusp::array1d<real, cusp::device_memory>>
luminance_continuity(
  cusp::array2d<int, cusp::device_memory>::const_view di_labels,
  cusp::array1d<real, cusp::device_memory>::const_view di_luminance,
  thrust::device_ptr<void> dio_temp);

}  // namespace separation

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_SEPARATION_LUMINANCE_CONTINUITY

