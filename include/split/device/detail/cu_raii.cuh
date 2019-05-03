#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_CU_RAII)
#define SPLIT_DEVICE_INCLUDED_DETAIL_CU_RAII

#include "split/detail/internal.h"
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cusolverSp.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
FLO_DEVICE_NAMESPACE_BEGIN

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

  namespace solver
  {
  /// @brief RAII wrapper for a cuda sparse solver instance
  struct SolverSp
  {
    cusolverSpHandle_t handle;
    cusolverStatus_t status;

    SolverSp();
    ~SolverSp();

    operator cusolverSpHandle_t() const noexcept;
    bool error_check(int line = -1) const noexcept;
    void error_assert(int line = -1) const noexcept;
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
    bool error_check(int line = -1) const noexcept;
    void error_assert(int line = -1) const noexcept;
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
