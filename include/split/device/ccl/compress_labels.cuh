#if !defined(SPLIT_DEVICE_INCLUDED_CCL_COMPRESS_LABELS)
#define SPLIT_DEVICE_INCLUDED_CCL_COMPRESS_LABELS

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
/***
   @brief Modifies a list of labels which contain arbitrary unique keys, and 
   produces a list of labels with the same unique properties, but with the
   all labels in the range [0 =< L < #No.labels]
   @param dio_labels A dense array containing the cluster labels for each point.
   This list will be compressed in place. This array is stored in device memory.
   @param dio_temp A pointer to a memory block for use in this function.
   @return The number of unique labels after compression.
   ***/
SPLIT_API int compress_labels(
  cusp::array1d<int, cusp::device_memory>::view dio_labels,
  thrust::device_ptr<void> dio_temp);

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_CCL_COMPRESS_LABELS


