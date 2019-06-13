#if !defined(SPLIT_DEVICE_INCLUDED_CCL_LINKS)
#define SPLIT_DEVICE_INCLUDED_CCL_LINKS

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{

int cluster_segment_map(
  cusp::array1d<int, cusp::device_memory>::view dio_cluster_labels,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_labels);

int n_inter_cluster_combinations(
  cusp::array1d<int, cusp::device_memory>::const_view di_cluster_map,
  cusp::array1d<int, cusp::device_memory>::view do_lengths);

cusp::array1d<int, cusp::device_memory> inter_cluster_links(
  cusp::array1d<int, cusp::device_memory>::view dio_cluster_labels,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_labels,
  const int i_nclusters,
  const int i_nsegments,
  thrust::device_ptr<void> do_temp);

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_CCL_LINKS



