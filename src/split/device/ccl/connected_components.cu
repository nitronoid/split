#include "split/device/ccl/connected_components.cuh"
#include <thrust/iterator/transform_output_iterator.h>
#include <cusp/transpose.h>
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
namespace
{
template <typename... Args>
auto zip_it(Args&&... args) -> decltype(
  thrust::make_zip_iterator(thrust::make_tuple(std::forward<Args>(args)...)))
{
  return thrust::make_zip_iterator(
    thrust::make_tuple(std::forward<Args>(args)...));
}

struct transpose_index : public thrust::unary_function<int, int>
{
  const int width;
  const int height;

  transpose_index(int width, int height) : width(width), height(height)
  {
  }

  __host__ __device__ int operator()(int i) const
  {
    const int x = i % height;
    const int y = i / height;
    return x * width + y;
  }
};

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

struct Decrement
{
  __host__ __device__ int operator()(int x)
  {
    return x - 1;
  }
};

struct TupleEqual
{
  __host__ __device__ bool operator()(const thrust::tuple<int, int>& lhs,
                                      const thrust::tuple<int, int>& rhs)
  {
    return (lhs.get<0>() == rhs.get<0>()) && (lhs.get<1>() == rhs.get<1>());
  }
};

struct FindKey
{
  __host__ __device__ thrust::tuple<int, int>
  operator()(const thrust::tuple<int, int>& lhs,
             const thrust::tuple<int, int>& rhs)
  {
    return thrust::make_tuple(
      NULL, (lhs.get<0>() != rhs.get<0>()) || (lhs.get<1>() != rhs.get<1>()));
  }
};
}  // namespace

SPLIT_API void connected_components(
  cusp::array2d<int, cusp::device_memory>::const_view di_labels,
  thrust::device_ptr<void> dio_temp,
  cusp::array1d<int, cusp::device_memory>::view do_labels,
  int i_max_iterations)
{
  const int npoints = di_labels.num_entries;
  const int width = di_labels.num_cols;
  const int height = di_labels.num_rows;
  // Iteratively find the max value in every row, then column until convergence
  // Initialize the new labels to the point indices
  thrust::sequence(do_labels.begin(), do_labels.end());

  // Convert our temporary storage pointer to an int pointer
  auto itemp_ptr =
    thrust::device_pointer_cast<int>(static_cast<int*>(dio_temp.get()));
  // Temp storage for the row and column maximum indices, used to scatter back
  auto row_keys = cusp::make_array1d_view(itemp_ptr, itemp_ptr + npoints);
  auto col_keys =
    cusp::make_array1d_view(itemp_ptr + npoints, itemp_ptr + npoints * 2);
  // Iterate over the row index and the labels simultaneously
  auto row_indices = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), row_index{width});
  auto row_begin = zip_it(row_indices, di_labels.values.begin());
  auto row_end = row_begin + npoints;
  // Iterate over the column index and the labels simultaneously, using a
  // transposed view of the matrix
  auto col_indices = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), column_index{width});
  auto transposed_indices = thrust::make_transform_iterator(
    thrust::make_counting_iterator(0), transpose_index{width, height});
  auto col_begin = thrust::make_permutation_iterator(
    zip_it(col_indices, di_labels.values.begin()), transposed_indices);
  auto col_end = col_begin + npoints;

  auto discard_it = thrust::make_discard_iterator();
  // Calculate the gather indices
  // Row gather indices
  thrust::adjacent_difference(
    row_begin, row_end, zip_it(discard_it, row_keys.begin()), FindKey{});
  thrust::inclusive_scan(
    row_keys.begin(),
    row_keys.end(),
    thrust::make_transform_output_iterator(row_keys.begin(), Decrement{}));
  // Column gather indices
  thrust::adjacent_difference(
    col_begin, col_end, zip_it(discard_it, col_keys.begin()), FindKey{});
  thrust::inclusive_scan(
    col_keys.begin(),
    col_keys.end(),
    thrust::make_transform_output_iterator(col_keys.begin(), Decrement{}));
  // We can transpose this upfront to avoid a permutation iterator later
  {
    auto col_keys_2d =
      cusp::make_array2d_view(height, width, 1, col_keys, cusp::row_major{});
    cusp::transpose(col_keys_2d, col_keys_2d);
  }

  const int nmaximums = max(col_keys.back(), row_keys.back()) + 1;
  // Temp storage for the row and column maximums
  auto max_ptr = col_keys.end();
  auto d_maximums = cusp::make_array1d_view(max_ptr, max_ptr + nmaximums);
  auto d_old_maximums =
    cusp::make_array1d_view(max_ptr + nmaximums, max_ptr + nmaximums * 2);

  // Make copies of these so we don't copy the entire arrays through the capture
  auto maximums_begin = d_maximums.begin();
  auto maximums_end = d_maximums.end();
  auto old_maximums_begin = d_old_maximums.begin();
  // Set this to ensure we fail on first convergence check
  *old_maximums_begin = -1;
  // We've converged if our maximums for each column are the same as last time
  auto has_converged = [=] {
    return thrust::equal(maximums_begin, maximums_end, old_maximums_begin);
  };
  // Iterate until convergence
  for (int iter = 0; iter < i_max_iterations && !has_converged(); ++iter)
  {
    // Store the previous maximums to later check for convergence
    thrust::copy(maximums_begin, maximums_end, old_maximums_begin);
    // Reduce by row to find the maximum index of every cluster
    thrust::reduce_by_key(row_begin,
                          row_end,
                          do_labels.begin(),
                          discard_it,
                          maximums_begin,
                          TupleEqual{},
                          thrust::maximum<int>());
    // Now repeat for the columns
    // Reduce by column to find the maximum index of every cluster, use a
    // permutation iterator to read the row maximums as the input labels
    thrust::reduce_by_key(
      col_begin,
      col_end,
      thrust::make_permutation_iterator(
        thrust::make_permutation_iterator(d_maximums.begin(), row_keys.begin()),
        transposed_indices),
      discard_it,
      d_maximums.begin(),
      TupleEqual{},
      thrust::maximum<int>());
    // Gather the maximums as the new labels
    thrust::gather(
      col_keys.begin(), col_keys.end(), d_maximums.begin(), do_labels.begin());
  }
}

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

