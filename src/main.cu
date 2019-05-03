#include "split/device/kmeans/kmeans.cuh"
#include "split/device/ccl/point_point_adjacency.cuh"
#include "split/device/ccl/connected_components.cuh"
#include "split/device/ccl/compress_labels.cuh"
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

  // Create initial means
  const int nclusters = 10;
  const int npixels = h_image.n_pixels();
  cusp::array1d<split::device::ScopedCuStream, cusp::host_memory> streams(4);

  cusp::array2d<real, cusp::device_memory, cusp::column_major> d_centroids(
    nclusters, h_image.n_channels());
  split::device::kmeans::initialize_centroids(d_image, d_centroids);

  cusp::array2d<int, cusp::device_memory> d_cluster_labels(h_image.height(),
                                                           h_image.width());
  cusp::array2d<int, cusp::device_memory> d_segment_labels(h_image.height(),
                                                           h_image.width());

  // Allocate temporary memory
  struct TempMemory
  {
    TempMemory(std::size_t N) : m_ptr(thrust::device_malloc(N))
    {
    }

    ~TempMemory()
    {
      thrust::device_free(m_ptr);
    }

    thrust::device_ptr<void> get() const
    {
      return m_ptr;
    }

  private:
    thrust::device_ptr<void> m_ptr;
  } d_temp(h_image.n_pixels() * nclusters * sizeof(real));

  // K-means cluster the image
  split::device::kmeans::cluster(
    d_image, d_centroids, d_cluster_labels.values, d_temp.get(), 100, 5e-1);

  // Obtain isolated segments from our initial clustering
  split::device::ccl::connected_components(
    d_cluster_labels, d_temp.get(), d_segment_labels.values, 100);
  // Compress the segment labels to produce a contiguous sequence
  const int nsegments =
    split::device::ccl::compress_labels(d_segment_labels.values, d_temp.get());
  // Re-calculate the centroids using the segment labels
  cusp::array2d<real, cusp::device_memory, cusp::column_major> d_seg_centroids(
    nsegments, h_image.n_channels());
  // Re-calculate the centroids using the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_segment_labels.values, d_image, d_seg_centroids, d_temp.get());
  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_segment_labels.values, d_seg_centroids, d_image);

  make_host_image(d_image, h_image.get());
  split::host::stbi::writef("assets/images/segments.png", h_image);

  return 0;
}

