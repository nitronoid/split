#include "split/device/ccl/connected_components.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/transposed_copy.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/zip_it.cuh"
#include <thrust/iterator/transform_output_iterator.h>
#include <cusp/transpose.h>
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
namespace
{
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

template <typename EntryIt, typename KeyIt>
void make_gather_indices(EntryIt&& entry_begin,
                         EntryIt&& entry_end,
                         KeyIt&& key_begin)
{
  // Discard iterator
  auto discard_it = thrust::make_discard_iterator();
  // Mark neighbors with different labels, or columns
  thrust::adjacent_difference(
    entry_begin, entry_end, detail::zip_it(discard_it, key_begin), FindKey{});
  // First should always be a one
  key_begin[0] = 1;
  // Find the number of keys
  const int nkeys = entry_end - entry_begin;
  // Prefix-sum the marked locations to provide lookup indices
  thrust::inclusive_scan(key_begin,
                         key_begin + nkeys,
                         thrust::make_transform_output_iterator(
                           key_begin, detail::unary_minus<int>(1)));
}

struct ComponentConvergence
{
  ComponentConvergence(thrust::device_ptr<int> i_max_begin,
                       thrust::device_ptr<int> i_max_end,
                       thrust::device_ptr<int> i_old_max_begin)
    : maximums_begin(i_max_begin)
    , maximums_end(i_max_end)
    , old_maximums_begin(i_old_max_begin)
  {
  }

  thrust::device_ptr<int> maximums_begin;
  thrust::device_ptr<int> maximums_end;
  thrust::device_ptr<int> old_maximums_begin;

  bool operator()() const
  {
    return thrust::equal(maximums_begin, maximums_end, old_maximums_begin);
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

  std::cout<<"DEBUG wxh: "<<width<<'x'<<height<<'\n';

#if 1
  cusp::array1d<int, cusp::device_memory> row_keys(npoints);
  cusp::array1d<int, cusp::device_memory> col_keys(npoints);
#else
  // Convert our temporary storage pointer to an int pointer
  auto itemp_ptr =
    thrust::device_pointer_cast(static_cast<int*>(dio_temp.get()));
  // Temp storage for the row and column maximum indices, used to scatter back
  auto row_keys = cusp::make_array1d_view(itemp_ptr, itemp_ptr + npoints);
  auto col_keys =
    cusp::make_array1d_view(itemp_ptr + npoints, itemp_ptr + npoints * 2);
#endif

  // Iterate over the row index and the labels simultaneously
  auto row_indices = detail::make_row_iterator(width);
  auto row_begin = detail::zip_it(row_indices, di_labels.values.begin());
  auto row_end = row_begin + npoints;

  // Iterate over the column index and the labels simultaneously, using a
  // transposed view of the matrix
  auto col_indices = detail::make_column_iterator(width);
  auto transposed_indices = detail::make_transpose_iterator(height, width);
  auto col_begin = thrust::make_permutation_iterator(
    detail::zip_it(col_indices, di_labels.values.begin()), transposed_indices);
  auto col_end = col_begin + npoints;

  // Calculate the gather indices
  // Row gather indices
  make_gather_indices(row_begin, row_end, row_keys.begin());
  // Column gather indices
  make_gather_indices(col_begin, col_end, col_keys.begin());
  // We can transpose this upfront to avoid a permutation iterator later
  // Note we use the label memory here to avoid an extra allocation
  detail::transposed_copy(height, width, col_keys, do_labels);
  // Copy back to the column keys
  thrust::copy(do_labels.begin(), do_labels.end(), col_keys.begin());


  // Initialize the new labels to the point indices
  thrust::sequence(do_labels.begin(), do_labels.end());

  const int nmaximums = max(col_keys.back(), row_keys.back()) + 1;
  // Temp storage for the row and column maximums
  // auto max_ptr = col_keys.end();
  // auto d_maximums = cusp::make_array1d_view(max_ptr, max_ptr + nmaximums);
  // auto d_maximum_buffer = max_ptr + nmaximums;
  // auto d_old_maximums = max_ptr + nmaximums * 2;
  // Set this to ensure we fail on first convergence check
  cusp::array1d<int, cusp::device_memory> d_maximums(nmaximums);
  cusp::array1d<int, cusp::device_memory> d_maximum_buffer(nmaximums);
  cusp::array1d<int, cusp::device_memory> d_old_maximums(nmaximums);
  d_old_maximums[0] = -1;

  // We've converged if our maximums for each column are the same as last time
  ComponentConvergence has_converged(d_maximums.begin().base(),
                                     d_maximums.end().base(),
                                     d_old_maximums.begin().base());
  // Iterate until convergence
  int iter;
  for (iter = 0; iter < i_max_iterations && !has_converged(); ++iter)
  {
    // Store the previous maximums to later check for convergence
    thrust::copy(d_maximums.begin(), d_maximums.end(), d_old_maximums.begin());
    // Reduce by row to find the maximum index of every cluster
    thrust::reduce_by_key(row_begin,
                          row_end,
                          do_labels.begin(),
                          thrust::make_discard_iterator(),
                          d_maximum_buffer.begin(),
                          TupleEqual{},
                          thrust::maximum<int>());
    // Now repeat for the columns
    // Reduce by column to find the maximum index of every cluster, use a
    // permutation iterator to read the row maximums as the input labels
    thrust::reduce_by_key(col_begin,
                          col_end,
                          thrust::make_permutation_iterator(
                            thrust::make_permutation_iterator(
                              d_maximum_buffer.begin(), row_keys.begin()),
                            transposed_indices),
                          thrust::make_discard_iterator(),
                          d_maximums.begin(),
                          TupleEqual{},
                          thrust::maximum<int>());
    // Gather the maximums as the new labels
    thrust::gather(
      col_keys.begin(), col_keys.end(), d_maximums.begin(), do_labels.begin());
  }
  std::cout << "Segmentation performed in " << iter << " iterations.\n";
}

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

