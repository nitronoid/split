#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_APPLY_SOR)
#define SPLIT_DEVICE_INCLUDED_DETAIL_APPLY_SOR

#include "split/detail/internal.h"
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
void apply_sor(
  cusp::csr_matrix<int, real, cusp::device_memory>::const_view di_A,
  cusp::array1d<real, cusp::device_memory>::const_view di_b,
  cusp::array1d<real, cusp::device_memory>::view do_x,
  const real i_w,
  const real i_tol,
  const int i_max_iter,
  const bool verbose);

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_APPLY_SOR


