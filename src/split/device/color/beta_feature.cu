#include "split/device/color/beta_feature.cuh"
#include "split/device/detail/zip_it.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace color
{
namespace
{
struct lab_to_beta
{
  lab_to_beta(real Lmin, real Lmax, real mu)
    : Lmin(std::move(Lmin)), Lmax(std::move(Lmax)), mu(std::move(mu))
  {
  }

  real Lmin;
  real Lmax;
  real mu;

  __host__ __device__ real
  operator()(const thrust::tuple<real, real, real>& lab) const
  {
    const real L = lab.get<0>();
    const real a = abs(lab.get<1>());
    const real b = abs(lab.get<2>());
    const real lambda = 0.2f * max(a, b);

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
  // Find the maximum luminance
  const auto luminance = di_lab_points.row(0);
  const auto luminance_max =
    *thrust::max_element(luminance.begin(), luminance.end());
  // Compute the upper and lower bounds for the beta feature
  const real lo = luminance_max * 0.20f;
  const real hi = luminance_max * 0.95f;

  // Iterate over the input Lab tuples
  auto lab_begin = detail::zip_it(di_lab_points.row(0).begin(),
                                  di_lab_points.row(1).begin(),
                                  di_lab_points.row(2).begin());
  auto lab_end = lab_begin + di_lab_points.num_cols;

  thrust::transform(
    lab_begin, lab_end, do_beta.begin(), lab_to_beta(lo, hi, mu));
}

}  // namespace color

SPLIT_DEVICE_NAMESPACE_END

