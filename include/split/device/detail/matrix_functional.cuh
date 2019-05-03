#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_MATRIX_FUNCTIONAL)
#define SPLIT_DEVICE_INCLUDED_DETAIL_MATRIX_FUNCTIONAL

#include "split/detail/internal.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
struct transpose_index : public thrust::unary_function<int, int>
{
  const int height;
  const int width;

  transpose_index(int height, int width) : height(height), width(width)
  {
  }

  __host__ __device__ int operator()(int i) const
  {
    const int x = i / width;
    const int y = i % width;
    return y * height + x;
  }
};
template <typename IndexT>
auto make_transpose_iterator(IndexT i_height, IndexT i_width) -> decltype(
  thrust::make_transform_iterator(thrust::make_counting_iterator<IndexT>(0),
                                  transpose_index(i_height, i_width)))
{
  return thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0),
    transpose_index(i_height, i_width));
}

// convert a linear index to a row index
struct row_index : public thrust::unary_function<int, int>
{
  const int n;

  __host__ __device__ row_index(int _n) : n(_n)
  {
  }

  __host__ __device__ int operator()(int i)
  {
    return i / n;
  }
};

template <typename IndexT>
auto make_row_iterator(IndexT i_width)
  -> decltype(thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0), row_index(i_width)))
{
  return thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0), row_index(i_width));
}

// convert a linear index to a column index
struct column_index : public thrust::unary_function<int, int>
{
  const int m;

  __host__ __device__ column_index(int _m) : m(_m)
  {
  }

  __host__ __device__ int operator()(int i)
  {
    return i % m;
  }
};

template <typename IndexT>
auto make_column_iterator(IndexT i_width)
  -> decltype(thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0), column_index(i_width)))
{
  return thrust::make_transform_iterator(
    thrust::make_counting_iterator<IndexT>(0), column_index(i_width));
}
}

SPLIT_DEVICE_NAMESPACE_END

#endif // SPLIT_DEVICE_INCLUDED_DETAIL_MATRIX_FUNCTIONAL
