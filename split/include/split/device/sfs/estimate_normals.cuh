#if !defined(SPLIT_DEVICE_INCLUDED_SFS_ESTIMATE_NORMALS)
#define SPLIT_DEVICE_INCLUDED_SFS_ESTIMATE_NORMALS

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace sfs
{
/***
   @brief Estimates a normal map using a lighting direction and know intensity.
   ***/
SPLIT_API void estimate_normals(
  cusp::array2d<real, cusp::device_memory>::const_view di_shading_intensity,
  const float3 i_light_vector,
  cusp::array2d<real, cusp::device_memory>::view do_normals,
  const real i_smoothness_weight);

/***
   @brief Calculates the number of entries in a discrete Poisson matrix of the
   given dimensions.
   ***/
int n_poisson_entries(const int m, const int n);
}  // namespace sfs

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_SFS_ESTIMATE_NORMALS

