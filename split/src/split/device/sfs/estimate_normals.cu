#include "split/device/sfs/estimate_normals.cuh"
#include "split/device/detail/cu_raii.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include <cusp/gallery/poisson.h>
#include <cusp/convert.h>
#include <cusp/relaxation/sor.h>
#include <cusp/monitor.h>
#include <cusparse_v2.h>

#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace sfs
{
// Alias the sparse matrix types
using CooMatrix = cusp::coo_matrix<int, real, cusp::device_memory>;
using CsrMatrix = cusp::csr_matrix<int, real, cusp::device_memory>;

namespace
{
static const char* _cusparseGetErrorEnum(cusparseStatus_t error)
{
  switch (error)
  {
  case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";

  case CUSPARSE_STATUS_NOT_INITIALIZED:
    return "CUSPARSE_STATUS_NOT_INITIALIZED";

  case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";

  case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";

  case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";

  case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";

  case CUSPARSE_STATUS_EXECUTION_FAILED:
    return "CUSPARSE_STATUS_EXECUTION_FAILED";

  case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";

  case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
    return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

  case CUSPARSE_STATUS_ZERO_PIVOT: return "CUSPARSE_STATUS_ZERO_PIVOT";
  }

  return "<unknown>";
}
inline void
__cusparseSafeCall(cusparseStatus_t err, const char* file, const int line)
{
  if (CUSPARSE_STATUS_SUCCESS != err)
  {
    fprintf(stderr,
            "CUSPARSE error in file '%s', line %d, error %s\nterminating!\n",
            __FILE__,
            __LINE__,
            _cusparseGetErrorEnum(err));
    assert(0);
  }
}

extern "C" void cusparseSafeCall(cusparseStatus_t err)
{
  __cusparseSafeCall(err, __FILE__, __LINE__);
}

template <typename T>
constexpr T sqr(T val) noexcept
{
  return val * val;
}

void build_M(float3 L, CooMatrix::view do_M)
{
  const int n_normals = do_M.num_rows / 3;
  // Outer product of the lighting direction
  // clang-format off
  const std::array<real, 9> LLT = {{L.x * L.x, L.x * L.y, L.x * L.z,
                                    L.x * L.y, L.y * L.y, L.y * L.z,
                                    L.x * L.z, L.y * L.z, L.z * L.z}};
  // clang-format on
  // Copy to the device
  thrust::device_vector<real> d_LLT(LLT.begin(), LLT.end());
  // Perform a kronecker product of LLT with the Identity matrix
  // We want to iterate over each row of LLT, No. Normals times
  const auto LLT_row = detail::make_row_iterator(n_normals * 3);
  // We want to iterate over each column of LLT, in a repeating cycle for each n
  const auto LLT_col = detail::make_column_iterator(3);
  // Use the combined look up index to get the real value from LLT
  const auto LLT_v = thrust::make_permutation_iterator(
    d_LLT.begin(),
    thrust::make_transform_iterator(
      detail::zip_it(LLT_row, LLT_col),
      [=] __host__ __device__(const thrust::tuple<int, int>& coord) {
        return coord.get<0>() * 3 + coord.get<1>();
      }));
  // Copy the values across to M
  thrust::copy_n(LLT_v, n_normals * 9, do_M.values.begin());
  // The row keys will be i / 3, as we only have 3 values per row and column
  const auto count = thrust::make_counting_iterator(0);
  thrust::transform(count,
                    count + n_normals * 9,
                    do_M.row_indices.begin(),
                    detail::unary_divides<int>(3));
  // To write the column keys we need a repeating sequence of 0, 1, 2 * n to
  // give 0, n, 2n, and then we offset by the row % n
  thrust::transform(LLT_col,
                    LLT_col + n_normals * 9,
                    do_M.row_indices.begin(),
                    do_M.column_indices.begin(),
                    [=] __host__ __device__(int s, int r) {
                      return (r % n_normals) + s * n_normals;
                    });
}


void build_B(const int m, const int n, CooMatrix::view do_B)
{
  // Get the number of pixel adjacencies, including self adjacency
  const int n_entries = n_poisson_entries(m, n);
  // Get the number of normals
  const int n_normals = m * n;
  // Iterate over the coordinates and values of B simultaneously
  auto entry_it = detail::zip_it(
    do_B.row_indices.begin(), do_B.column_indices.begin(), do_B.values.begin());
  // Build the discrete Poisson problem matrix
  // FIXME START
  CooMatrix d_temp;
  cusp::gallery::poisson5pt(d_temp, n, m);
  const auto temp_begin = detail::zip_it(d_temp.row_indices.begin(),
                                         d_temp.column_indices.begin(),
                                         d_temp.values.begin());
  thrust::copy_n(temp_begin, n_entries, entry_it);
  // FIXME END

  // Correct the boundaries which don't have valence of 4
  using tup3 = thrust::tuple<int, int, real>;
  const auto fix_boundaries = [=] __host__ __device__(tup3 entry) {
    // Fix boundary cell diagonals
    if (entry.get<0>() == entry.get<1>())
    {
      const int r = entry.get<0>() / n;
      const int c = entry.get<0>() % n;
      // If we're in a boundary cell we subtract one from the valence
      entry.get<2>() -= (r == 0 || r == (m - 1));
      entry.get<2>() -= (c == 0 || c == (n - 1));
    }
    return entry;
  };
  thrust::transform(entry_it, entry_it + n_entries, entry_it, fix_boundaries);

  // Copy sB 3 times, offsetting by the number of normals for each new copy
  const auto offset = [=] __host__ __device__(tup3 entry, int count) {
    // Work out what channel we're in
    const int channel = count / n_entries;
    // Offset for the channel
    entry.get<0>() += channel * n_normals;
    entry.get<1>() += channel * n_normals;
    return entry;
  };
  // Iterate over the original entries
  auto entry_og = detail::make_cycle_iterator(entry_it, n_entries);
  const auto count = thrust::make_counting_iterator(n_entries);
  thrust::transform(
    entry_og, entry_og + n_entries * 2, count, entry_it + n_entries, offset);
}

CsrMatrix cusparse_add(CooMatrix::const_view di_A,
                       CooMatrix::const_view di_B,
                       const real i_lambda)
{
  cusp::array1d<int, cusp::device_memory> A_row_offsets(di_A.num_rows + 1);
  cusp::indices_to_offsets(di_A.row_indices, A_row_offsets);
  cusp::array1d<int, cusp::device_memory> B_row_offsets(di_B.num_rows + 1);
  cusp::indices_to_offsets(di_B.row_indices, B_row_offsets);

  using namespace detail;
  // Create a cuSparse handle
  cu_raii::sparse::Handle handle;
  // Create descriptions of our 3 matrices
  cu_raii::sparse::MatrixDescription A_desc, B_desc, C_desc;

  int dummy;
  cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));
  cusp::array1d<int, cusp::device_memory> C_row_offsets(di_A.num_rows + 1);

  cusparseSafeCall(cusparseXcsrgeamNnz(handle,
                                       di_A.num_rows,
                                       di_A.num_cols,
                                       A_desc,
                                       di_A.num_entries,
                                       A_row_offsets.begin().base().get(),
                                       di_A.column_indices.begin().base().get(),
                                       B_desc,
                                       di_B.num_entries,
                                       B_row_offsets.begin().base().get(),
                                       di_B.column_indices.begin().base().get(),
                                       C_desc,
                                       C_row_offsets.begin().base().get(),
                                       &dummy));
  // Calculate the number of entries in the resulting C matrix
  const int C_nnz = C_row_offsets.back() - C_row_offsets.front();

  // Allocate for the final matrix
  CsrMatrix do_C(di_A.num_rows, di_A.num_cols, C_nnz);
  // Move the row offsets that we already calculated to this matrix
  do_C.row_offsets = std::move(C_row_offsets);

  // A should be unscaled, B should be scaled by lambda
  const real one = 1.f;
  // Now actually do the add
  cusparseSafeCall(cusparseScsrgeam(handle,
                                    di_A.num_rows,
                                    di_A.num_cols,
                                    &one,
                                    A_desc,
                                    di_A.num_entries,
                                    di_A.values.begin().base().get(),
                                    A_row_offsets.begin().base().get(),
                                    di_A.column_indices.begin().base().get(),
                                    &i_lambda,
                                    B_desc,
                                    di_B.num_entries,
                                    di_B.values.begin().base().get(),
                                    B_row_offsets.begin().base().get(),
                                    di_B.column_indices.begin().base().get(),
                                    C_desc,
                                    do_C.values.begin().base().get(),
                                    do_C.row_offsets.begin().base().get(),
                                    do_C.column_indices.begin().base().get()));
  return do_C;
}

void build_L(const float3 i_light_vector,
             cusp::array1d<real, cusp::device_memory>::const_view di_S,
             cusp::array1d<real, cusp::device_memory>::view do_L_star)
{
  const int n_normals = do_L_star.size() / 3;
  // Copy L to the device
  thrust::device_vector<real> d_L(&i_light_vector.x, (&i_light_vector.x) + 3);
  // The b vector of the system is (shading intensity * L), where L repeats
  // Iterate over one component of L per channel of the normals
  const auto cyclic_L = thrust::make_permutation_iterator(
    d_L.begin(), detail::make_row_iterator(n_normals));
  // Loop over for each dimension of the normals
  const auto cyclic_i = detail::make_cycle_iterator(di_S.begin(), n_normals);
  // Write the results
  const thrust::multiplies<real> mul;
  thrust::transform(
    cyclic_i, cyclic_i + do_L_star.size(), cyclic_L, do_L_star.begin(), mul);
}

void apply_sor(
  cusp::csr_matrix<int, real, cusp::device_memory>::const_view di_A,
  cusp::array1d<real, cusp::device_memory>::const_view di_b,
  cusp::array1d<real, cusp::device_memory>::view do_x,
  const real i_w,
  const real i_tol,
  const int i_max_iter,
  const bool verbose)
{
  // Linear SOR operator
  cusp::relaxation::sor<real, cusp::device_memory> M(di_A, i_w);
  // Array to store the residual
  cusp::array1d<real, cusp::device_memory> d_r(di_b.size());
  // Compute the initial residual
  const auto compute_residual = [&] __host__ {
    cusp::multiply(di_A, do_x, d_r);
    cusp::blas::axpy(di_b, d_r, -1.f);
  };
  compute_residual();
  // Monitor the convergence
  cusp::monitor<real> monitor(di_b, i_max_iter, i_tol, 0, verbose);
  // Iterate until convergence criteria is met
  for (; !monitor.finished(d_r); ++monitor)
  {
    // Apply the SOR linear operator to iterate on our solution
    M(di_A, di_b, do_x);
    // Compute the residual
    compute_residual();
  }
}

}  // namespace

SPLIT_API void estimate_normals(
  cusp::array2d<real, cusp::device_memory>::const_view di_shading_intensity,
  const float3 i_light_vector,
  cusp::array2d<real, cusp::device_memory>::view do_normals,
  const real i_smoothness_weight)
{
  // 3 channels per normals
  const int n_normals = di_shading_intensity.num_entries;
  const int n_unknowns = n_normals * 3;
  const int height = di_shading_intensity.num_rows;
  const int width = di_shading_intensity.num_cols;
  // Normalize the light vector
  auto L = i_light_vector;
  {
    const real rmag = (1.f / std::sqrt(L.x * L.x + L.y * L.y + L.z * L.z));
    L.x *= rmag;
    L.y *= rmag;
    L.z *= rmag;
  }

  // First calculate M which is the kronecker product of (LL^T) and I_n
  CooMatrix d_M(n_unknowns, n_unknowns, n_unknowns * 3);
  build_M(L, d_M);

  // Next calculate B as the discrete Poisson matrix, with adjusted boundaries
  CooMatrix d_B(n_unknowns, n_unknowns, 3 * n_poisson_entries(height, width));
  build_B(height, width, d_B);

  // Subtract B scaled by lambda from M to get the final A matrix
  auto d_A = cusparse_add(d_M, d_B, i_smoothness_weight * 2.f);

  // Calculate L^* as the b vector for our system AN = L^*
  cusp::array1d<real, cusp::device_memory> d_L_star(n_unknowns);
  build_L(L, di_shading_intensity.values, d_L_star);

  // Set our initial guess for the normals to the Z vector
  auto d_x = do_normals.values.subarray(0, n_unknowns);
  thrust::tabulate(
    d_x.begin(), d_x.end(), [=] __host__ __device__(int x) -> real {
      return x >= n_normals * 2;
    });

  // Now we can solve for the relative normals via SOR
  apply_sor(d_A, d_L_star, d_x, 1.f, 1e-5f, 1500, true);

  // Normalize the resulting solution
  using vec3 = thrust::tuple<real, real, real>;
  const auto normalize_vec = [] __host__ __device__(vec3 v) {
    const real rmag =
      1.f / std::sqrt(sqr(v.get<0>()) + sqr(v.get<1>()) + sqr(v.get<2>()));
    v.get<0>() *= rmag;
    v.get<1>() *= rmag;
    v.get<2>() = std::abs(v.get<2>()) * rmag;
    return v;
  };
  // Iterate over the 3 dimensional normals
  auto norm_begin = detail::zip_it(do_normals.row(0).begin(),
                                   do_normals.row(1).begin(),
                                   do_normals.row(2).begin());
  auto norm_end = norm_begin + n_normals;
  thrust::transform(norm_begin, norm_end, norm_begin, normalize_vec);
}

int n_poisson_entries(const int m, const int n)
{
  return m * (5 * n - 2) - 2 * n;
}

}  // namespace sfs

SPLIT_DEVICE_NAMESPACE_END
