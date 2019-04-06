#if !defined(SPLIT_DEVICE_INCLUDED_CUDA_RAII)
#define SPLIT_DEVICE_INCLUDED_CUDA_RAII

#include "split/detail/internal.h"

SPLIT_DEVICE_NAMESPACE_BEGIN

/***
   @brief An RAII wrapper for a CUDA stream.
   ***/
struct ScopedCuStream
{
  ///@brief Public access to the managed stream.
  cudaStream_t handle;
  ///@brief Public access to the stream errors.
  cudaError_t status;
  ///@brief Constructor that creates the CUDA stream.
  ScopedCuStream();
  ///@brief Destructor that destroys the CUDA stream.
  ~ScopedCuStream();
  ///@brief Implicit conversion to the managed stream.
  operator cudaStream_t() const noexcept;
  ///@brief Blocks until all tasks on the managed stream have completed.
  void join() noexcept;
  ///@brief Disable copy construction.
  ScopedCuStream(const ScopedCuStream&) = delete;
  ///@brief Disable copy assignment.
  ScopedCuStream& operator=(const ScopedCuStream&) = delete;
  ///@brief Default move construction.
  ScopedCuStream(ScopedCuStream&&) = default;
  ///@brief Default move assignment.
  ScopedCuStream& operator=(ScopedCuStream&&) = default;
};

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_CUDA_RAII
