#include "split/device/sfs/sobel_normals.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include "split/device/detail/transposed_copy.cuh"
#include "split/device/detail/view_util.cuh"
#include "split/device/detail/normalize_vectors.cuh"

#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace sfs
{
SPLIT_API void sobel_derivative(
  cusp::array2d<real, cusp::device_memory>::const_view di_heights,
  cusp::array1d<real, cusp::device_memory>::view do_derivative)
{
  // How many points we're working with
  const int n_data = di_heights.num_entries;
  const int width = di_heights.num_cols;
  const int height = di_heights.num_rows;
  // Convenience
  auto height_begin = di_heights.values.begin();
  const auto count = thrust::make_counting_iterator(0);
  const thrust::plus<real> add;
  const thrust::minus<real> subtract;
  // Allocate two buffers for intermediate results
  cusp::array1d<real, cusp::device_memory> d_ping(n_data + 1);
  // Actually re-use the output memory as the second buffer
  auto& d_pong = do_derivative;
  // Insert a sentinel value here, useful later
  d_ping[0] = 0.f;
  // Ignore the sentinel value for the most part
  auto ping_begin = d_ping.begin() + 1;

  // Scan each row of the data, make sure to enforce the row boundaries
  // ---------------------------------------------------------------------------
  const auto rows_begin = detail::make_row_iterator(width);
  const auto rows_end = rows_begin + n_data;
  thrust::inclusive_scan_by_key(rows_begin, rows_end, height_begin, ping_begin);

  // Subtract the i - 3 neighbor from each value, and shift results one left
  // ---------------------------------------------------------------------------
  // If we're past the initial 2 values we can look up a previous neighbor,
  // otherwise we would be out of bounds or looking up a previous row.
  // Instead we index a sentinel value inserted at the start of the range.
  const auto base_op = [=] __host__ __device__(int i) {
    return (i % width) >= 2 ? i - 2 : -1;
  };
  const auto base_it = thrust::make_permutation_iterator(
    ping_begin, thrust::make_transform_iterator(count, base_op));
  // Calculate a capped index, which allows us to shift one left and duplicate
  // the final row element. Cap our value at the end of this row
  const auto cap_op = [=] __host__ __device__(int i) {
    return min(i + 1, (i / width + 1) * width - 1);
  };
  const auto cap_it = thrust::make_permutation_iterator(
    ping_begin, thrust::make_transform_iterator(count, cap_op));
  // Subtract the base value from each scanned value
  thrust::transform(cap_it, cap_it + n_data, base_it, d_pong.begin(), subtract);
  // Add the original values
  thrust::transform(
    d_pong.begin(), d_pong.end(), height_begin, ping_begin, add);

  // Now subtract columns either side of a value to obtain it's derivative
  // ---------------------------------------------------------------------------
  // Clamp the row and column indices inside the image
  const auto bottom_op = [=] __host__ __device__(int i) {
    const int r = min(i / width + 1, height - 1);
    return r * width + (i % width);
  };
  const auto bottom_it = thrust::make_permutation_iterator(
    ping_begin, thrust::make_transform_iterator(count, bottom_op));
  const auto top_op = [=] __host__ __device__(int i) {
    const int r = max(i / width - 1, 0);
    return r * width + (i % width);
  };
  const auto top_it = thrust::make_permutation_iterator(
    ping_begin, thrust::make_transform_iterator(count, top_op));
  thrust::transform(
    bottom_it, bottom_it + n_data, top_it, do_derivative.begin(), subtract);
}

SPLIT_API void sobel_normals(
  cusp::array2d<real, cusp::device_memory>::const_view di_heights,
  cusp::array2d<real, cusp::device_memory>::view do_normals,
  const real i_depth)
{
  // Clearer references
  auto d_Xderivative = do_normals.row(0);
  auto d_Yderivative = do_normals.row(1);
  auto d_Z = do_normals.row(2);

  // Transpose the image
  cusp::array2d<real, cusp::device_memory> d_height_transposed(
    di_heights.num_cols, di_heights.num_rows, di_heights.num_entries);
  cusp::array1d<real, cusp::device_memory> d_Xderivative_transposed(
    d_Xderivative.size());
  detail::transposed_copy<real>(di_heights.num_cols,
                                di_heights.num_rows,
                                di_heights.values,
                                d_height_transposed.values);

  // Compute the Y derivative using the original image
  sobel_derivative(di_heights, d_Yderivative);
  // Compute the X derivative using the transposed image
  sobel_derivative(d_height_transposed, d_Xderivative_transposed);
  // Transpose the X derivatives back
  detail::transposed_copy<real>(d_height_transposed.num_cols,
                                d_height_transposed.num_rows,
                                d_Xderivative_transposed,
                                d_Xderivative);
  // Fill the Z with the provided depth
  thrust::fill(d_Z.begin(), d_Z.end(), 1.f / std::max(0.001f, i_depth));
  // Finally normalize the vectors
  detail::normalize_vectors(do_normals);
}

}  // namespace sfs

SPLIT_DEVICE_NAMESPACE_END

