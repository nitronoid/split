#include "split/device/color/conversion.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

/// These conversions were implemented, following the OpenCV specification:
/// https://docs.opencv.org/3.1.0/de/d25/imgproc_color_conversions.html

namespace color
{
__host__ __device__ Color rgb_to_xyz::operator()(const Color& rgb) const
{
  real r = rgb.get<0>();
  real g = rgb.get<1>();
  real b = rgb.get<2>();

  const real r1 = 1.f / 1.055f;
  const real r2 = 1.f / 12.92f;

  r = ((r > 0.04045f) ? pow((r + 0.055f) * r1, 2.4f) : (r * r2)) * 100.0f;
  g = ((g > 0.04045f) ? pow((g + 0.055f) * r1, 2.4f) : (g * r2)) * 100.0f;
  b = ((b > 0.04045f) ? pow((b + 0.055f) * r1, 2.4f) : (b * r2)) * 100.0f;

  return thrust::make_tuple(r * 0.4124564f + g * 0.3575761f + b * 0.1804375f,
                            r * 0.2126729f + g * 0.7151522f + b * 0.0721750f,
                            r * 0.0193339f + g * 0.1191920f + b * 0.9503041f);
}

__host__ __device__ Color xyz_to_rgb::operator()(const Color& xyz) const
{
  const real r1 = 1.f / 100.f;
  const real x = xyz.get<0>() * r1;
  const real y = xyz.get<1>() * r1;
  const real z = xyz.get<2>() * r1;

  real r = x * 3.2404542f + y * -1.5371385f + z * -0.4985314f;
  real g = x * -0.9692660f + y * 1.8760108f + z * 0.0415560f;
  real b = x * 0.0556434f + y * -0.2040259f + z * 1.0572252f;

  const real r2 = 1.f / 2.4f;
  return thrust::make_tuple(
    ((r > 0.0031308f) ? (1.055f * pow(r, r2) - 0.055f) : (12.92f * r)),
    ((g > 0.0031308f) ? (1.055f * pow(g, r2) - 0.055f) : (12.92f * g)),
    ((b > 0.0031308f) ? (1.055f * pow(b, r2) - 0.055f) : (12.92f * b)));
}

__host__ __device__ Color xyz_to_lab::operator()(const Color& xyz) const
{
  const real r1 = 1.f / 95.047f;
  const real r2 = 0.01f;
  const real r3 = 1.f / 108.883f;
  const real r4 = 16.f / 116.f;

  real x = xyz.get<0>() * r1;
  real y = xyz.get<1>() * r2;
  real z = xyz.get<2>() * r3;

  x = (x > 0.008856f) ? cbrt(x) : (7.787f * x + r4);
  y = (y > 0.008856f) ? cbrt(y) : (7.787f * y + r4);
  z = (z > 0.008856f) ? cbrt(z) : (7.787f * z + r4);

  return thrust::make_tuple(
    (116.f * y) - 16.f, 500.f * (x - y), 200.f * (y - z));
}

__host__ __device__ Color lab_to_xyz::operator()(const Color& lab) const
{
  const real r1 = 1.f / 116.f;
  const real r2 = 1.f / 500.f;
  const real r3 = 1.f / 200.f;
  const real r4 = 1.f / 7.787f;

  real y = (lab.get<0>() + 16.f) * r1;
  real x = lab.get<1>() * r2 + y;
  real z = y - lab.get<2>() * r3;

  const real x3 = x * x * x;
  const real y3 = y * y * y;
  const real z3 = z * z * z;

  return thrust::make_tuple(
    ((x3 > 0.008856f) ? x3 : ((x - 16.0 * r1) * r4)) * 95.047f,
    ((y3 > 0.008856f) ? y3 : ((y - 16.0 * r1) * r4)) * 100.0f,
    ((z3 > 0.008856f) ? z3 : ((z - 16.0 * r1) * r4)) * 108.883f);
}

__host__ __device__ Color rgb_to_lab::operator()(const Color& rgb) const
{
  return xyz_to_lab{}(rgb_to_xyz{}(rgb));
}

__host__ __device__ Color lab_to_rgb::operator()(const Color& lab) const
{
  return xyz_to_rgb{}(lab_to_xyz{}(lab));
}

__host__ __device__ Color rgb_to_ic::operator()(const Color& rgb) const
{
  constexpr real third = 1.f / 3.f;
  const real intensity = (rgb.get<0>() + rgb.get<1>() + rgb.get<2>()) * third;
  const real ri = 1.f / intensity;
  const real cr = rgb.get<0>() * ri;
  const real cg = rgb.get<1>() * ri;
  return thrust::make_tuple(intensity, cr, cg);
}

}  // namespace color

SPLIT_DEVICE_NAMESPACE_END

