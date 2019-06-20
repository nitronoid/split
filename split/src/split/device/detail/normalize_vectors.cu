#include "split/device/detail/normalize_vectors.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/view_util.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
void normalize_vectors(
  cusp::array2d<real, cusp::device_memory>::view dio_vectors)
{
  normalize_vectors(make_const_array2d_view(dio_vectors), dio_vectors);
}

void normalize_vectors(
  cusp::array2d<real, cusp::device_memory>::const_view di_vectors,
  cusp::array2d<real, cusp::device_memory>::view do_vectors)
{
  // Iterate in tuples of 3
  auto norm_out = detail::zip_it(do_vectors.row(0).begin(),
                                 do_vectors.row(1).begin(),
                                 do_vectors.row(2).begin());
  auto norm_begin = detail::zip_it(di_vectors.row(0).begin(),
                                   di_vectors.row(1).begin(),
                                   di_vectors.row(2).begin());
  auto norm_end = norm_begin + di_vectors.num_cols;

  // Normalize our resulting normals
  using vec3 = thrust::tuple<real, real, real>;
  const auto sqr = [] __host__ __device__(real x) { return x * x; };
  thrust::transform(
    norm_begin, norm_end, norm_out, [=] __host__ __device__(vec3 normal) {
      const real rmag =
        1.f / std::sqrt(sqr(normal.get<0>()) + sqr(normal.get<1>()) +
                        sqr(normal.get<2>()));
      normal.get<0>() *= rmag;
      normal.get<1>() *= rmag;
      normal.get<2>() = std::abs(normal.get<2>()) * rmag;
      return normal;
    });
}

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

