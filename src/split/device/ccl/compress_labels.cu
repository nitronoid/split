#include "split/device/ccl/compress_labels.cuh"
#include <thrust/iterator/transform_output_iterator.h>
#include <cusp/transpose.h>
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
SPLIT_API int
compress_labels(cusp::array1d<int, cusp::device_memory>::view dio_labels,
                thrust::device_ptr<void> dio_temp)
{
  const int nlabels = dio_labels.size();
  // Convert our temporary storage pointer to an int pointer
  auto itemp_ptr =
    thrust::device_pointer_cast(static_cast<int*>(dio_temp.get()));
  // Copy our labels into the temp memory
  auto d_labels = cusp::make_array1d_view(itemp_ptr, itemp_ptr + nlabels);
  thrust::copy(dio_labels.begin(), dio_labels.end(), d_labels.begin());
  // Sort our copy
  thrust::sort(d_labels.begin(), d_labels.end());
  // Once sorted, we can create a unique list
  auto unique_label_end = thrust::unique(d_labels.begin(), d_labels.end());
  // Finally we need to binary search to map from a unique labels index to that
  // label
  thrust::lower_bound(d_labels.begin(),
                      unique_label_end,
                      dio_labels.begin(),
                      dio_labels.end(),
                      dio_labels.begin());
  // Return the number of unique labels
  return *(unique_label_end - 1);
}

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END
