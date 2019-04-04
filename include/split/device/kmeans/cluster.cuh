#if !defined(SPLIT_DEVICE_INCLUDED_KMEANS_CLUSTER)
#define SPLIT_DEVICE_INCLUDED_KMEANS_CLUSTER

#include "split/detail/internal.h"
#include <cusp/array1d.h>
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
SPLIT_API void
cluster(cusp::array2d<real, cusp::device_memory>::const_view di_points,
        cusp::array2d<real, cusp::device_memory, cusp::column_major>::view
          dio_centroids,
        cusp::array1d<int, cusp::device_memory>::view do_cluster_labels,
        thrust::device_ptr<void> do_temp,
        int i_max_iter,
        real i_threshold = 1.f);
}

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_KMEANS_CLUSTER

