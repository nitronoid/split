#if !defined(SPLIT_DEVICE_INCLUDED_PROBABILITY_SET_PROBABILITY)
#define SPLIT_DEVICE_INCLUDED_PROBABILITY_SET_PROBABILITY

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace probability
{
SPLIT_API void set_probability(
  cusp::array2d<real, cusp::device_memory>::const_view di_albedo,
  cusp::array1d<int, cusp::device_memory>::const_view di_set_labels,
  cusp::array1d<int, cusp::device_memory>::const_view dio_set_ids,
  cusp::array1d<real, cusp::device_memory>::view do_probability);

}  // namespace probability

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_PROBABILITY_SET_PROBABILITY


