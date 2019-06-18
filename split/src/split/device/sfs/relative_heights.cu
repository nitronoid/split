#include "split/device/sfs/estimate_normals.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include <cusp/gallery/grid.h>
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace sfs
{
// Alias the sparse matrix types
using CooMatrix = cusp::coo_matrix<int, real, cusp::device_memory>;
using CsrMatrix = cusp::csr_matrix<int, real, cusp::device_memory>;
using Vec2 = thrust::tuple<real, real>;

namespace
{
struct relative_height_from_normals
{
  using vec2 = thrust::tuple<real, real>;

  __host__ __device__ real dot(const vec2& n1, const vec2& n2) const noexcept
  {
    return n1.get<0>() * n2.get<0>() + n1.get<1>() * n2.get<1>();
  }

  __host__ __device__ vec2 normalize(const vec2& n) const noexcept
  {
    const auto norm = std::sqrt(dot(n, n));
    return thrust::make_tuple(n.get<0>() / norm, n.get<1>() / norm);
  }

  __host__ __device__ real operator()(vec2 n1, vec2 n2) const noexcept
  {
    // Normalize n1 and n2
    n1 = normalize(n1);
    n2 = normalize(n2);
    const real x = n1.get<0>() - n2.get<0>();
    const real y = n1.get<1>() - n2.get<1>();

    real q;
    constexpr float epsilon = 0.0000001f;
    if (std::abs(x) > epsilon)
    {
      q = y / x;
    }
    else
    {
      const auto inf = std::numeric_limits<real>::infinity();
      const real g1 = n1.get<0>() == 0.f ? inf : n1.get<1>() / n1.get<0>();
      if (g1 == inf)
        q = 0.f;
      else if (g1 == 0.f)
        q = 1.f / epsilon;
      else
        q = 1.f / g1;
    }

    return q;
  }
};

}  // namespace

SPLIT_API void relative_heights(
  const int m,
  const int n,
  cusp::array2d<real, cusp::device_memory>::const_view di_normals,
  cusp::coo_matrix<int, real, cusp::device_memory>::view do_heights)
{
  const int n_heights = do_heights.num_entries;
  // FIXME begin
  // Initialize a grid matrix using CUSP
  CooMatrix d_temp;
  cusp::gallery::grid2d(d_temp, n, m);

  const auto temp_begin = detail::zip_it(d_temp.row_indices.begin(),
                                         d_temp.column_indices.begin(),
                                         d_temp.values.begin());
  auto entry_it = detail::zip_it(do_heights.row_indices.begin(),
                                 do_heights.column_indices.begin(),
                                 do_heights.values.begin());
  thrust::copy_n(temp_begin, n_heights, entry_it);
  // FIXME end

  // Iterate over the normals with their index
  const auto count = thrust::make_counting_iterator(0);
  const auto normal_begin = detail::zip_it(di_normals.row(0).begin(),
                                           di_normals.row(1).begin(),
                                           di_normals.row(2).begin(),
                                           count);
  // Iterate over pairs of normals using the matrix coordinates
  const auto n1_begin = thrust::make_permutation_iterator(
    normal_begin, do_heights.row_indices.begin());
  const auto n2_begin = thrust::make_permutation_iterator(
    normal_begin, do_heights.column_indices.begin());
  const auto n1_end = n1_begin + n_heights;

  using vec = thrust::tuple<real, real, real, int>;
  const auto calc_height = [] __host__ __device__(const vec& i_n1,
                                                  const vec& i_n2) {
    // Check whether these normals are vertical or horizontal
    // neighbors and project the normals accordingly
    auto n1 = thrust::make_tuple(0.f, i_n1.get<2>());
    auto n2 = thrust::make_tuple(0.f, i_n2.get<2>());
    if (std::abs(i_n1.get<3>() - i_n2.get<3>()) == 1)
    {
      n1.get<0>() = i_n1.get<0>();
      n2.get<0>() = i_n2.get<0>();
    }
    else
    {
      n1.get<0>() = i_n1.get<1>();
      n2.get<0>() = i_n2.get<1>();
    }
    // in lower triangle
    const bool lower = i_n1.get<3>() > i_n2.get<3>();
    const real q = relative_height_from_normals{}(n1, n2);
    return lower ? -q : q;
  };
  thrust::transform(
    n1_begin, n1_end, n2_begin, do_heights.values.begin(), calc_height);
}

int n_grid_entries(const int m, const int n)
{
  return m * (4 * n - 2) - 2 * n;
}

}  // namespace sfs

SPLIT_DEVICE_NAMESPACE_END

