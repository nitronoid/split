#include "split/device/color/beta_feature.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace color
{
namespace
{
struct lab_to_beta
{
  lab_to_beta(real lambda, real Lmin, real Lmax, real mu)
    : lambda(std::move(lambda))
    , Lmin(std::move(Lmin))
    , Lmax(std::move(Lmax))
    , mu(std::move(mu))
  {
  }

  real lambda;
  real Lmin;
  real Lmax;
  real mu;

  __host__ __device__ real
  operator()(const thrust::tuple<real, real, real>& lab) const
  {
    const real L = lab.get<0>();
    const real a = abs(lab.get<1>());
    const real b = abs(lab.get<2>());

    real beta = 0.f;

    if (a < lambda && b < lambda)
    {
      if (L < Lmin)
        beta = -mu;
      else if (L > Lmax)
        beta = mu;
    }

    return beta;
  }
};
}  // namespace

SPLIT_API void
beta_feature(cusp::array2d<real, cusp::device_memory>::const_view di_lab_points,
             cusp::array1d<real, cusp::device_memory>::view do_beta,
             const real mu)
{
  const int npoints = di_lab_points.num_cols;
  // Find the maximum luminance
  const auto luminance = di_lab_points.row(0);
  const auto luminance_max =
    *thrust::max_element(luminance.begin(), luminance.end());

  // Read the absolute chroma values
  const auto chroma_begin = thrust::make_transform_iterator(
    di_lab_points.row(1).begin(), detail::unary_abs<real>());
  const auto chroma_end = chroma_begin + npoints;
  // Find the lambda value, as the maximum absolute chroma * 0.2
  const auto lambda = 0.2f * (*thrust::max_element(chroma_begin, chroma_end));

  // Compute the upper and lower bounds for the beta feature
  const real lo = luminance_max * 0.20f;
  const real hi = luminance_max * 0.95f;

  // Iterate over the input Lab tuples
  auto lab_begin = detail::zip_it(di_lab_points.row(0).begin(),
                                  di_lab_points.row(1).begin(),
                                  di_lab_points.row(2).begin());
  auto lab_end = lab_begin + npoints;

  thrust::transform(
    lab_begin, lab_end, do_beta.begin(), lab_to_beta(lambda, lo, hi, mu));
}

}  // namespace color

SPLIT_DEVICE_NAMESPACE_END

