#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL)
#define SPLIT_DEVICE_INCLUDED_DETAIL_UNARY_FUNCTIONAL

#include "split/detail/internal.h"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
template <typename T>
struct constant
{
  constant(T i_val) : value(i_val)
  {
  }
  T value;

  template <typename... Args>
  __host__ __device__ T operator()(Args&&...) const
  {
    return value;
  }
};

template <typename T>
struct unary_max
{
  unary_max(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return max(i_lhs, rhs);
  }
};

template <typename T>
struct unary_min
{
  unary_min(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return min(i_lhs, rhs);
  }
};

template <typename T>
struct unary_equal
{
  unary_equal(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs == rhs;
  }
};

template <typename T>
struct unary_not_equal
{
  unary_not_equal(T i_rhs) : rhs(i_rhs)
  {
  }
  T rhs;

  __host__ __device__ T operator()(T i_lhs) const
  {
    return i_lhs != rhs;
  }
};

template <typename T>
struct reciprocal
{
  __host__ __device__ T operator()(T i_rhs) const
  {
    return T(1) / i_rhs;
  }
};

template <typename T>
struct unary_abs
{
  __host__ __device__ T operator()(T i_x) const
  {
    return abs(i_x);
  }
};

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
struct unary_pow
{
  unary_pow(T i_exponent) : exponent(i_exponent)
  {
  }
  T exponent;

  __host__ __device__ T operator()(T i_x) const
  {
    return pow(i_x, exponent);
  }
};

template <typename T>
struct unary_exp
{
  __host__ __device__ T operator()(T i_x) const
  {
    return exp(i_x);
  }
};

template <typename T>
struct unary_log
{
  __host__ __device__ T operator()(T i_x) const
  {
    return log(i_x);
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
