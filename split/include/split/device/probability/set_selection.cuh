#if !defined(SPLIT_DEVICE_INCLUDED_PROBABILITY_SET_SELECTION)
#define SPLIT_DEVICE_INCLUDED_PROBABILITY_SET_SELECTION

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>


SPLIT_DEVICE_NAMESPACE_BEGIN

namespace probability
{

SPLIT_API int set_selection(
  cusp::array2d<real, cusp::device_memory>::const_view di_albedo,
  cusp::array2d<real, cusp::device_memory>::const_view di_centroids,
  cusp::array1d<int, cusp::device_memory>::const_view di_set_labels,
  cusp::array1d<int, cusp::device_memory>::view dio_set_ids,
  const int i_nsets);

}  // namespace probability

SPLIT_DEVICE_NAMESPACE_END


#endif  // SPLIT_DEVICE_INCLUDED_PROBABILITY_SET_SELECTION

