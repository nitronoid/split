#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_NORMALIZE)
#define SPLIT_DEVICE_INCLUDED_DETAIL_NORMALIZE

#include "split/detail/internal.h"
#include <cusp/array2d.h>
#include <thrust/extrema.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
template <typename BeginInIteratorT,
          typename EndInIteratorT,
          typename OutputIteratorT>
void normalize(BeginInIteratorT&& range_begin,
               EndInIteratorT&& range_end,
               OutputIteratorT&& out_begin)
{
  const int n_data = range_end - range_begin;
  // Subtract the minimum value
  const real min = *thrust::min_element(range_begin, range_end);
  const detail::unary_minus<real> subf(min);
  thrust::transform(range_begin, range_end, out_begin, subf);
  // Divide by the maximum value
  const real scale = 1.f / *thrust::max_element(range_begin, range_end);
  const detail::unary_multiplies<real> mulf(scale);
  thrust::transform(out_begin, out_begin + n_data, out_begin, mulf);
}

template <typename BeginIteratorT, typename EndIteratorT>
void normalize(BeginIteratorT&& range_begin, EndIteratorT&& range_end)
{
  normalize(range_begin, range_end, range_begin);
}
}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_NORMALIZE

