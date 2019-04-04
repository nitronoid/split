#include "split/device/kmeans/label.cuh"
#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace
{
template <typename T>
struct unary_divide : public thrust::unary_function<T, T>
{
  __host__ __device__ unary_divide(T denominator = 1) : denominator(denominator)
  {
  }

  const T denominator;
  __host__ __device__ T operator()(T x)
  {
    return x / denominator;
  }
};

template <typename T>
struct unary_modulus : public thrust::unary_function<T, T>
{
  __host__ __device__ unary_modulus(T denominator = 1)
    : denominator(denominator)
  {
  }

  const T denominator;
  __host__ __device__ T operator()(T x)
  {
    return x % denominator;
  }
};
}  // namespace

namespace kmeans
{
SPLIT_API void label_points(
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::const_view
    di_centroids,
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array1d<int, cusp::device_memory>::view do_cluster_labels,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::view do_temp)
{
  cusp::multiply(di_centroids,
                 di_points,
                 do_temp,
                 cusp::constant_functor<real>(),
                 // Calculate distances from all centroids to all pixels
                 [] __host__ __device__(real lhs, real rhs) -> real {
                   const real diff = lhs - rhs;
                   return diff * diff;
                 },
                 thrust::plus<real>());

  // Converts a 1D index into a row index
  auto row_indices = thrust::make_transform_iterator(
    thrust::counting_iterator<int>(0), unary_modulus<int>(do_temp.num_rows));

  // Converts a 1D index into a column index
  auto col_indices = thrust::make_transform_iterator(
    thrust::counting_iterator<int>(0), unary_divide<int>(do_temp.num_rows));

  // Reduce each column, by finding the smallest distance contained, and writing
  // it's row index as the label
  thrust::reduce_by_key(
    col_indices,
    col_indices + do_temp.num_entries,
    thrust::make_zip_iterator(
      thrust::make_tuple(do_temp.values.begin(), row_indices)),
    thrust::make_discard_iterator(),
    thrust::make_zip_iterator(thrust::make_tuple(
      thrust::make_discard_iterator(), do_cluster_labels.begin())),
    thrust::equal_to<int>(),
    thrust::minimum<thrust::tuple<real, int>>());
}

}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END


