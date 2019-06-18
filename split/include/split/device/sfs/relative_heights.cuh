#if !defined(SPLIT_DEVICE_INCLUDED_SFS_RELATIVE_HEIGHTS)
#define SPLIT_DEVICE_INCLUDED_SFS_RELATIVE_HEIGHTS

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
SPLIT_API void relative_heights(
  const int m,
  const int n,
  cusp::array2d<real, cusp::device_memory>::const_view di_normals,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_heights);

/***
   @brief Calculates the number of entries in a discrete grid matrix of the
   given dimensions.
   ***/
int n_grid_entries(const int m, const int n);

}  // namespace sfs

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_SFS_RELATIVE_HEIGHTS

