#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL)
#define SPLIT_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL

#include "split/detail/internal.h"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
template <typename T>
struct unary_modulo
{
  unary_modulo(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs % rhs;
  }
};

template <typename T>
struct unary_divides
{
  unary_divides(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs / rhs;
  }
};

template <typename T>
struct unary_multiplies
{
  unary_multiplies(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs * rhs;
  }
};

template <typename T>
struct unary_minus
{
  unary_minus(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs - rhs;
  }
};

template <typename T>
struct unary_plus
{
  unary_plus(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs + rhs;
  }
};
}

SPLIT_DEVICE_NAMESPACE_END

#endif // SPLIT_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL
