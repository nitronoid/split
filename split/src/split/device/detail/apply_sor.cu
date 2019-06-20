#include "split/device/detail/apply_sor.cuh"
#include <cusp/relaxation/sor.h>
#include <cusp/monitor.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
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

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END


