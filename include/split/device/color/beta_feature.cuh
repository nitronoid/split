#if !defined(SPLIT_DEVICE_INCLUDED_COLOR_BETA_FEATURE)
#define SPLIT_DEVICE_INCLUDED_COLOR_BETA_FEATURE

#include "split/detail/internal.h"
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace color
{
SPLIT_API void beta_feature(
    cusp::array2d<real, cusp::device_memory>::const_view di_lab_points,
    cusp::array1d<real, cusp::device_memory>::view do_beta,
    const real mu = 1e+5);



}  // namespace color

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_COLOR_BETA_FEATURE


