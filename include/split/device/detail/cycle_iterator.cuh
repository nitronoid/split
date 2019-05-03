#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_CYCLE_ITERATOR)
#define SPLIT_DEVICE_INCLUDED_DETAIL_CYCLE_ITERATOR

#include "split/detail/internal.h"
#include "split/device/detail/unary_functional.cuh"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
template <typename IndexT>
auto make_cycle_iterator(const IndexT length)
  -> decltype(thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0), unary_modulo<int>(length)))
{
  return thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0), unary_modulo<int>(length));
}
}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_CYCLE_ITERATOR
