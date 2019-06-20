#include "split/device/sfs/estimate_normals.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include "split/device/detail/apply_sor.cuh"
#include "split/device/detail/normalize.cuh"
#include <cusp/gallery/poisson.h>
#include <cusp/convert.h>
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace sfs
{
// Alias the sparse matrix types
using CooMatrix = cusp::coo_matrix<int, real, cusp::device_memory>;
using CsrMatrix = cusp::csr_matrix<int, real, cusp::device_memory>;
using Vec2 = thrust::tuple<real, real>;

SPLIT_API void absolute_heights(
  cusp::coo_matrix<int, real, cusp::device_memory>::const_view
    di_relative_heights,
  cusp::array2d<real, cusp::device_memory>::view do_absolute_heights,
  const int i_max_iterations)
{
  const int height = do_absolute_heights.num_rows;
  const int width = do_absolute_heights.num_cols;
  const int n_normals = do_absolute_heights.num_entries;

  // We can assemble a poisson problem to solve the absolute heights
  // The A matrix is a discrete Poisson matrix in CSR format
  CsrMatrix d_A(n_normals, n_normals, n_poisson_entries(height, width));
  cusp::gallery::poisson5pt(d_A, width, height);
  // Calculate the COO row indices
  cusp::array1d<real, cusp::device_memory> d_row_indices(d_A.num_entries);
  cusp::offsets_to_indices(d_A.row_offsets, d_row_indices);
  // Iterate over the A matrix entries using the row indices
  auto A_begin = detail::zip_it(
    d_row_indices.begin(), d_A.column_indices.begin(), d_A.values.begin());

  using tup3 = thrust::tuple<int, int, real>;
  const auto fix_bounds = [=] __host__ __device__(tup3 entry) {
    // Fix boundary cell diagonals
    real val = entry.get<2>();
    if (entry.get<0>() == entry.get<1>())
    {
      const int r = entry.get<0>() / width;
      const int c = entry.get<0>() % width;
      // If we're in a boundary cell we subtract one from the valence
      val -= (r == 0 || r == (height - 1));
      val -= (c == 0 || c == (width - 1));
    }
    return val;
  };
  // Fix the boundary cell diagonals
  thrust::transform(
    A_begin, A_begin + d_A.num_entries, d_A.values.begin(), fix_bounds);

  // The b vector is the relative height from a normal to all it's neighbors
  cusp::array1d<real, cusp::device_memory> d_b(n_normals);
  thrust::reduce_by_key(di_relative_heights.row_indices.begin(),
                        di_relative_heights.row_indices.end(),
                        di_relative_heights.values.begin(),
                        thrust::make_discard_iterator(),
                        d_b.begin());

  // Set our initial guess to a constant
  auto d_h = do_absolute_heights.values.subarray(0, n_normals);
  thrust::fill_n(d_h.begin(), n_normals, 0.5f);

  // To get a result we need to "pin" the solution by setting an arbitrary
  // value to some constant. I use the first height.
  // Make the first equation a trivial solution 1*h0 = x
  d_A.values.begin()[0] = 1.f;
  d_A.values.begin()[1] = 0.f;
  d_A.values.begin()[2] = 0.f;
  d_A.values.begin()[3] = 0.f;
  d_A.values.begin()[4 * width - 2] = 0.f;
  // Need to replace any references to the final solution with constants in b
  d_b[0] = 0.5f;
  d_b.begin()[1] += d_b[0];
  d_b.begin()[width] += d_b[0];
  // Set the first result to a trivial solution to pin the results
  d_h[0] = d_b[0];

  // Solve with Successive over relaxation
  detail::apply_sor(d_A, d_b, d_h, 0.9f, 1e-4f, i_max_iterations, false);

  // Normalize the final heights
  detail::normalize(d_h.begin(), d_h.end());
}
}  // namespace sfs

SPLIT_DEVICE_NAMESPACE_END

