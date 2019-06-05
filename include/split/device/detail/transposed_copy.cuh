#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_TRANSPOSED_COPY)
#define SPLIT_DEVICE_INCLUDED_DETAIL_TRANSPOSED_COPY

#include "split/detail/internal.h"
#include "split/device/detail/matrix_functional.cuh"
#include <thrust/scatter.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
inline void transposed_copy(const int m,
                            const int n,
                            cusp::array1d<int, cusp::device_memory>::view A,
                            cusp::array1d<int, cusp::device_memory>::view At)
{
  auto indices = make_transpose_iterator(m, n);
  thrust::scatter(A.begin(), A.end(), indices, At.begin());
}
}

SPLIT_DEVICE_NAMESPACE_END

#endif // SPLIT_DEVICE_INCLUDED_DETAIL_TRANSPOSED_COPY
