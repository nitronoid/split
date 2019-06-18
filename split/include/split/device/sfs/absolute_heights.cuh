#if !defined(SPLIT_DEVICE_INCLUDED_SFS_ABSOLUTE_HEIGHTS)
#define SPLIT_DEVICE_INCLUDED_SFS_ABSOLUTE_HEIGHTS

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace sfs
{
/***
   @brief Calculates the relative heights between points by fitting an
   osculating arc through their normals
   ***/
SPLIT_API void absolute_heights(
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view
    di_relative_heights,
  cusp::array2d<real, cusp::device_memory>::view do_absolute_heights,
  const int i_max_iterations);
}  // namespace sfs

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_SFS_ABSOLUTE_HEIGHTS

