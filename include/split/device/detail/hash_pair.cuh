#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_HASH_PAIR)
#define SPLIT_DEVICE_INCLUDED_DETAIL_HASH_PAIR

#include "split/detail/internal.h"
#include <thrust/tuple.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
struct hash_pair
{
  __host__ __device__ int64_t
  operator()(const thrust::tuple<int, int>& pair) const
  {
    const int64_t long_copy = pair.get<0>();
    return (long_copy << 31) + pair.get<1>();
  }
};
}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_HASH_PAIR
