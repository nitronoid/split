#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_CU_RAII)
#define SPLIT_DEVICE_INCLUDED_DETAIL_CU_RAII

#include "split/detail/internal.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>
#include <cusolverDn.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{

namespace cu_raii
{
  /// @brief RAII wrapper for a cuda stream
  struct Stream
  {
    cudaStream_t handle;
    cudaError_t status;

    Stream();
    ~Stream();

    operator cudaStream_t() const noexcept;
    void join() noexcept;
  };

  namespace blas
  {
  /// @brief RAII wrapper for a cublas instance
  struct Handle
  {
    cublasHandle_t handle;
    cublasStatus_t status;

    Handle();
    ~Handle();

    operator cublasHandle_t() const noexcept;
  };
  }

  namespace solver
  {
  /// @brief RAII wrapper for a cuda dense solver instance
  struct SolverDn
  {
    cusolverDnHandle_t handle;
    cusolverStatus_t status;

    SolverDn();
    ~SolverDn();

    operator cusolverDnHandle_t() const noexcept;
  };

  /// @brief RAII wrapper for a cuda sparse solver instance
  struct SolverSp
  {
    cusolverSpHandle_t handle;
    cusolverStatus_t status;

    SolverSp();
    ~SolverSp();

    operator cusolverSpHandle_t() const noexcept;
  };
  }  // namespace solver

  namespace sparse
  {
  /// @brief RAII wrapper for a cuda sparse instance
  struct Handle
  {
    cusparseHandle_t handle;
    cusparseStatus_t status;

    Handle();
    ~Handle();

    operator cusparseHandle_t() const noexcept;
  };

  /// @brief RAII wrapper for a cuda sparse matrix description
  struct MatrixDescription
  {
    cusparseMatDescr_t description;

    MatrixDescription();
    MatrixDescription(cusparseStatus_t* io_status);
    ~MatrixDescription();

    operator cusparseMatDescr_t() const noexcept;
  };
  }  // namespace sparse
}
}

SPLIT_DEVICE_NAMESPACE_END

#endif // SPLIT_DEVICE_INCLUDED_DETAIL_CU_RAII
