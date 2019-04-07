#include "split/device/kmeans/kmeans.cuh"
#include "split/host/stbi/stbi_raii.hpp"

#include <cusp/print.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <chrono>

#define real split::real

template <typename T>
void strided_copy(const T* i_src,
                  T* i_dest,
                  int src_stride,
                  int dest_stride,
                  int n,
                  cudaMemcpyKind i_kind)
{
  cudaMemcpy2D(i_dest,
               sizeof(T) * dest_stride,
               i_src,
               sizeof(T) * src_stride,
               sizeof(T),
               n,
               i_kind);
}

void make_device_image(gsl::not_null<const real*> h_image,
                       cusp::array2d<real, cusp::device_memory>::view d_image)
{
  const int npixels = d_image.num_cols;
  const int nchannels = d_image.num_rows;
  for (int c = 0; c < nchannels; ++c)
  {
    auto d_image_channel = d_image.values.begin().base().get() + npixels * c;
    const auto h_image_channel = h_image.get() + c;
    strided_copy(h_image_channel,
                 d_image_channel,
                 nchannels,
                 1,
                 npixels,
                 cudaMemcpyHostToDevice);
  }
}

void make_host_image(cusp::array2d<real, cusp::device_memory>::view d_image,
                     gsl::not_null<real*> h_image)
{
  const int npixels = d_image.num_cols;
  const int nchannels = d_image.num_rows;
  for (int c = 0; c < nchannels; ++c)
  {
    auto d_image_channel = d_image.values.begin().base().get() + npixels * c;
    const auto h_image_channel = h_image.get() + c;
    strided_copy(d_image_channel,
                 h_image_channel,
                 1,
                 nchannels,
                 npixels,
                 cudaMemcpyDeviceToHost);
  }
}

int main()
{
  auto h_image = split::host::stbi::loadf("assets/images/rust.png", 3);
  printf("Loaded image with dim: %dx%dx%d\n",
         h_image.width(),
         h_image.height(),
         h_image.n_channels());

  cusp::array2d<real, cusp::device_memory> d_image(h_image.n_channels(),
                                                   h_image.n_pixels());
  make_device_image(h_image.get(), d_image);
  std::fill_n(h_image.get(), h_image.n_pixel_data(), 0.5f);

  // Create initial means
  const int nclusters = 3;
  cusp::array1d<split::device::ScopedCuStream, cusp::host_memory> streams(4);

  cusp::array2d<real, cusp::device_memory, cusp::column_major> d_centroids(
    nclusters, h_image.n_channels());
  std::cout << "Generating centroids\n";
  split::device::kmeans::initialize_centroids(d_image, d_centroids);
  cusp::print(d_centroids);
  std::cout << "Done\n";

  cusp::array1d<int, cusp::device_memory> d_cluster_labels(h_image.n_pixels());

  // Allocate temporary memory
  thrust::device_vector<uint8_t> d_temp(h_image.n_pixels() * nclusters *
                                        sizeof(real));

  thrust::device_ptr<void> d_temp_ptr{static_cast<void*>(d_temp.data().get())};

  split::device::kmeans::cluster(
    streams, d_image, d_centroids, d_cluster_labels, d_temp_ptr, 100, 5e-1);

  std::cout << "Finalizing cluster colors\n";
  split::device::kmeans::propagate_centroids(
    streams, d_cluster_labels, d_centroids, d_image);
  std::cout << "Done\n";

  make_host_image(d_image, h_image.get());

  split::host::stbi::writef("assets/images/out.png", h_image);

  return 0;
}

