#include "split/host/stbi/stbi_raii.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

SPLIT_HOST_NAMESPACE_BEGIN

namespace
{
template <typename T>
T clamp(T v, T lo, T hi)
{
  return std::max(std::min(v, hi), lo);
}
}  // namespace

namespace stbi
{
struct ScopedImage::ScopedImageImpl
{
  ScopedImageImpl(int i_width, int i_height, int i_nchannels)
    : width(i_width), height(i_height), nchannels(i_nchannels)
  {
    const int ndata = i_width * i_height * i_nchannels;
    image.resize(ndata);
  }

  ScopedImageImpl(real* i_image_ptr, int i_width, int i_height, int i_nchannels)
    : width(i_width), height(i_height), nchannels(i_nchannels)
  {
    const int ndata = i_width * i_height * i_nchannels;
    image.reserve(ndata);
    std::move(i_image_ptr, i_image_ptr + ndata, std::back_inserter(image));
    stbi_image_free(i_image_ptr);
  }

  ScopedImageImpl() = default;
  ScopedImageImpl(const ScopedImageImpl&) = default;
  ScopedImageImpl& operator=(const ScopedImageImpl&) = default;
  ScopedImageImpl(ScopedImageImpl&&) = default;
  ScopedImageImpl& operator=(ScopedImageImpl&&) = default;
  ~ScopedImageImpl() = default;

  std::vector<real> image;
  int width;
  int height;
  int nchannels;
};

ScopedImage::ScopedImage(int i_width, int i_height, int i_nchannels)
  : m_impl(nonstd::make_value<ScopedImageImpl>(i_width, i_height, i_nchannels))
{
}
ScopedImage::ScopedImage(gsl::not_null<real*> i_image_ptr,
                         int i_width,
                         int i_height,
                         int i_nchannels)
  : m_impl(nonstd::make_value<ScopedImageImpl>(
      i_image_ptr, i_width, i_height, i_nchannels))
{
}

ScopedImage::ScopedImage() = default;
ScopedImage::ScopedImage(const ScopedImage&) = default;
ScopedImage& ScopedImage::operator=(const ScopedImage&) = default;
ScopedImage::ScopedImage(ScopedImage&&) = default;
ScopedImage& ScopedImage::operator=(ScopedImage&&) = default;
ScopedImage::~ScopedImage() = default;

real* ScopedImage::release() noexcept
{
  // Move the image data to a raw buffer
  auto raw_image = new real[m_impl->image.size()];
  std::move(m_impl->image.begin(), m_impl->image.end(), raw_image);
  // Copy an empty implementation to reset the internals
  m_impl = {};
  // Return the raw buffer
  return raw_image;
}

real* ScopedImage::get() noexcept
{
  return m_impl->image.data();
}

const real* ScopedImage::get() const noexcept
{
  return m_impl->image.data();
}
int ScopedImage::width() const noexcept
{
  return m_impl->width;
}

int ScopedImage::height() const noexcept
{
  return m_impl->height;
}

int ScopedImage::n_channels() const noexcept
{
  return m_impl->nchannels;
}

int ScopedImage::n_pixels() const noexcept
{
  return m_impl->height * m_impl->width;
}
int ScopedImage::n_pixel_data() const noexcept
{
  return m_impl->height * m_impl->width * m_impl->nchannels;
}

ScopedImage loadf(gsl::czstring i_path, int i_desired_channels)
{
  int width, height, nchannels;
  auto raw_image =
    stbi_loadf(i_path, &width, &height, &nchannels, i_desired_channels);
  return ScopedImage(raw_image,
                     width,
                     height,
                     i_desired_channels ? i_desired_channels : nchannels);
}

std::vector<uint8_t> quantize(const ScopedImage& i_image)
{
  std::vector<uint8_t> quantized(i_image.n_pixel_data());
  std::transform(i_image.get(),
                 i_image.get() + i_image.n_pixel_data(),
                 quantized.begin(),
                 [=](real v) { return clamp(v, 0.f, 1.f) * 255.0f + 0.5f; });
  return quantized;
}

void gamma_correct(gsl::span<const uint8_t> i_image,
                   gsl::not_null<uint8_t*> o_image,
                   const real gamma)
{
  std::transform(i_image.begin(), i_image.end(), o_image.get(), [=](uint8_t v) {
    return 255.f * pow(v / 255.f, gamma);
  });
}

void gamma_correct(const ScopedImage& i_image,
                   gsl::not_null<real*> o_image,
                   const real gamma)
{
  std::transform(i_image.get(),
                 i_image.get() + i_image.n_pixel_data(),
                 o_image.get(),
                 [=](real v) { return pow(v, gamma); });
}

void writef(gsl::czstring i_path, const ScopedImage& i_image)
{
  auto quantized = quantize(i_image);
  stbi_write_png(i_path,
                 i_image.width(),
                 i_image.height(),
                 i_image.n_channels(),
                 quantized.data(),
                 i_image.n_channels() * i_image.width());
}
}  // namespace stbi

SPLIT_HOST_NAMESPACE_END
