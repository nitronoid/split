#if !defined(SPLIT_DEVICE_INCLUDED_COLOR_CONVERSION)
#define SPLIT_DEVICE_INCLUDED_COLOR_CONVERSION

#include "split/detail/internal.h"
#include "split/device/detail/zip_it.cuh"
#include <cusp/array2d.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace color
{
///@brief Type alias for a color
using Color = thrust::tuple<real, real, real>;

///@brief Function object to convert RGB to CIE XYZ color space
struct rgb_to_xyz
{
  __host__ __device__ Color operator()(const Color& rgb) const;
};

///@brief Function object to convert CIE XYZ to CIE Lab color space
struct xyz_to_lab
{
  __host__ __device__ Color operator()(const Color& xyz) const;
};

///@brief Function object to convert CIE XYZ to RGB color space
struct xyz_to_rgb
{
  __host__ __device__ Color operator()(const Color& xyz) const;
};

///@brief Function object to convert CIE Lab to CIE XYZ color space
struct lab_to_xyz
{
  __host__ __device__ Color operator()(const Color& lab) const;
};

///@brief Function object to convert RGB to CIE Lab color space
struct rgb_to_lab
{
  __host__ __device__ Color operator()(const Color& rgb) const;
};

///@brief Function object to convert CIE Lab to RGB color space
struct lab_to_rgb
{
  __host__ __device__ Color operator()(const Color& lab) const;
};

/***
   @brief Transforms a set of values from one color space to another via a
   converter
   ***/
template <typename Converter>
SPLIT_API void
convert_color_space(cusp::array2d<real, cusp::device_memory>::const_view di_in,
                    cusp::array2d<real, cusp::device_memory>::view do_out,
                    const Converter& di_converter)
{
  // Iterate over the input range
  auto in_begin = detail::zip_it(
    di_in.row(0).begin(), di_in.row(1).begin(), di_in.row(2).begin());
  auto in_end = in_begin + di_in.num_cols;
  // Iterate over the output range
  auto out_begin = detail::zip_it(
    do_out.row(0).begin(), do_out.row(1).begin(), do_out.row(2).begin());

  thrust::transform(in_begin, in_end, out_begin, di_converter);
}
}  // namespace color

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_COLOR_CONVERSION

