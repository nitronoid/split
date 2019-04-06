#include "split/device/cuda_raii.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

ScopedCuStream::ScopedCuStream()
{
  status = cudaStreamCreate(&handle);
}

ScopedCuStream::~ScopedCuStream()
{
  join();
  status = cudaStreamDestroy(handle);
}

ScopedCuStream::operator cudaStream_t() const noexcept
{
  return handle;
}

void ScopedCuStream::join() noexcept
{
  status = cudaStreamSynchronize(handle);
}

SPLIT_DEVICE_NAMESPACE_END
