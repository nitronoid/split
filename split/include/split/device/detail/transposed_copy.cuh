#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_TRANSPOSED_COPY)
#define SPLIT_DEVICE_INCLUDED_DETAIL_TRANSPOSED_COPY

#include "split/detail/internal.h"
#include "split/device/detail/matrix_functional.cuh"
#include <thrust/scatter.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
template <typename T>
inline void
transposed_copy(const int m,
                const int n,
                typename cusp::array1d<T, cusp::device_memory>::const_view A,
                typename cusp::array1d<T, cusp::device_memory>::view At)
{
  auto indices = make_transpose_iterator(m, n);
  thrust::scatter(A.begin(), A.end(), indices, At.begin());
}
}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_TRANSPOSED_COPY
