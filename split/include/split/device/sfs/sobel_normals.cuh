#if !defined(SPLIT_DEVICE_INCLUDED_SFS_SOBEL_NORMALS)
#define SPLIT_DEVICE_INCLUDED_SFS_SOBEL_NORMALS

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace sfs
{
/***
   @brief Applies a Sobel filter over over a height map to obtain surface
   derivatives. Final normals are calculated from the surface derivatives and
   the provided depth.
   ***/
SPLIT_API void
sobel_normals(cusp::array2d<real, cusp::device_memory>::const_view di_heights,
              cusp::array2d<real, cusp::device_memory>::view do_normals,
              const real i_depth);

/***
   @brief Applies a Sobel filter in one dimension over a provided height map and
   calculates the surface derivative.
   ***/
SPLIT_API void sobel_derivative(
  cusp::array2d<real, cusp::device_memory>::const_view di_heights,
  cusp::array1d<real, cusp::device_memory>::view do_derivative);

}  // namespace sfs

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_SFS_SOBEL_NORMALS

