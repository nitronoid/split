#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_NORMALIZE_VECTORS)
#define SPLIT_DEVICE_INCLUDED_DETAIL_NORMALIZE_VECTORS

#include "split/detail/internal.h"
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
void normalize_vectors(
  cusp::array2d<real, cusp::device_memory>::view dio_vectors);

void normalize_vectors(
  cusp::array2d<real, cusp::device_memory>::const_view di_vectors,
  cusp::array2d<real, cusp::device_memory>::view do_vectors);
}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_NORMALIZE_VECTORS

