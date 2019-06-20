#if !defined(SPLIT_DEVICE_INCLUDED_PROBABILITY_REMOVE_SET_OUTLIERS)
#define SPLIT_DEVICE_INCLUDED_PROBABILITY_REMOVE_SET_OUTLIERS

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace probability
{
SPLIT_API int remove_set_outliers(
  cusp::array2d<real, cusp::device_memory>::const_view di_albedo,
  cusp::array1d<int, cusp::device_memory>::view dio_set_ids,
  cusp::array1d<int, cusp::device_memory>::view dio_set_labels,
  thrust::device_ptr<void> dio_temp);

}  // namespace probability

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_PROBABILITY_REMOVE_SET_OUTLIERS

