#include "split/device/detail/cu_raii.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{

namespace cu_raii
{

Stream::Stream()
{
  status = cudaStreamCreate(&handle);
}

Stream::~Stream()
{
  join();
  cudaStreamDestroy(handle);
}

Stream::operator cudaStream_t() const noexcept
{
  return handle;
}

void Stream::join() noexcept
{
  status = cudaStreamSynchronize(handle);
}

namespace blas
{
Handle::Handle()
{
  status = cublasCreate(&handle);
}

Handle::~Handle()
{
  cublasDestroy(handle);
}

Handle::operator cublasHandle_t() const noexcept
{
  return handle;
}
}

namespace solver
{
SolverDn::SolverDn()
{
  status = cusolverDnCreate(&handle);
}

SolverDn::~SolverDn()
{
  cusolverDnDestroy(handle);
}

SolverDn::operator cusolverDnHandle_t() const noexcept
{
  return handle;
}

SolverSp::SolverSp()
{
  status = cusolverSpCreate(&handle);
}

SolverSp::~SolverSp()
{
  cusolverSpDestroy(handle);
}

SolverSp::operator cusolverSpHandle_t() const noexcept
{
  return handle;
}
}  // namespace solver

namespace sparse
{
Handle::Handle()
{
  status = cusparseCreate(&handle);
}

Handle::~Handle()
{
  cusparseDestroy(handle);
}

Handle::operator cusparseHandle_t() const noexcept
{
  return handle;
}

MatrixDescription::MatrixDescription()
{
    cusparseCreateMatDescr(&description);
}

MatrixDescription::MatrixDescription(cusparseStatus_t* io_status)
{
    *io_status = cusparseCreateMatDescr(&description);
}

MatrixDescription::~MatrixDescription()
{
    cusparseDestroyMatDescr(description);
}

MatrixDescription::operator cusparseMatDescr_t() const noexcept
{
    return description;
}
}  // namespace sparse
}
}

SPLIT_DEVICE_NAMESPACE_END
