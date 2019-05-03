#include "split/device/kmeans/label.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/zip_it.cuh"
#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

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
  auto row_indices = detail::make_row_iterator(do_temp.num_rows);
  // Converts a 1D index into a column index
  auto col_indices = detail::make_column_iterator(do_temp.num_rows);

  // Reduce each row, by finding the smallest distance contained, and writing
  // it's column index as the label
  thrust::reduce_by_key(
    row_indices,
    row_indices + do_temp.num_entries,
    detail::zip_it(do_temp.values.begin(), col_indices),
    thrust::make_discard_iterator(),
    detail::zip_it(thrust::make_discard_iterator(), do_cluster_labels.begin()),
    thrust::equal_to<int>(),
    thrust::minimum<thrust::tuple<real, int>>());
}

}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

