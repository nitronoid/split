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
  std::fill_n(h_image.get(), h_image.n_pixel_data(), 0.5f);

  // Create initial means
  const int nclusters = 10;
  const int npixels = h_image.n_pixels();
  cusp::array1d<split::device::ScopedCuStream, cusp::host_memory> streams(4);

  cusp::array2d<real, cusp::device_memory, cusp::column_major> d_centroids(
    nclusters, h_image.n_channels());
  std::cout << "Generating centroids\n";
  split::device::kmeans::initialize_centroids(d_image, d_centroids);
  cusp::print(d_centroids);
  std::cout << "Done\n";

  cusp::array2d<int, cusp::device_memory> d_cluster_labels(h_image.height(),
                                                           h_image.width());
  cusp::array2d<int, cusp::device_memory> d_segment_labels(h_image.height(),
                                                           h_image.width());
  // Allocate temporary memory
  thrust::device_vector<uint8_t> d_temp(h_image.n_pixels() * nclusters *
                                        sizeof(real));
  auto d_temp_ptr =
    thrust::device_pointer_cast(static_cast<void*>(d_temp.data().get()));
  auto d_itemp_ptr =
    thrust::device_pointer_cast(static_cast<int*>(d_temp_ptr.get()));
  // Create a view over the integer part
  auto d_itemp =
    cusp::make_array1d_view(d_itemp_ptr, d_itemp_ptr + npixels * 2 + nclusters);

  // K-means cluster the image
  split::device::kmeans::cluster(streams,
                                 d_image,
                                 d_centroids,
                                 d_cluster_labels.values,
                                 d_temp_ptr,
                                 100,
                                 5e-1);

  // Obtain isolated segments from our initial clustering

  std::cout << "Finalizing cluster colors\n";
  split::device::kmeans::propagate_centroids(
    streams, d_cluster_labels.values, d_centroids, d_image);
  std::cout << "Done\n";
  make_host_image(d_image, h_image.get());
  split::host::stbi::writef("assets/images/clusters.png", h_image);

#if 0
  std::vector<int> l = {
    3, 7, 7, 7, 7, 7, 7, 0, 6, 6, 2, 2, 2, 2, 7, 7, 
    2, 3, 7, 7, 7, 7, 7, 7, 7, 7, 0, 2, 8, 2, 0, 7, 
    2, 3, 7, 7, 7, 7, 7, 7, 7, 7, 0, 2, 8, 8, 2, 0, 
    2, 2, 3, 0, 7, 7, 7, 7, 7, 7, 0, 0, 8, 8, 2, 3, 
    2, 2, 2, 2, 2, 0, 7, 7, 7, 7, 7, 0, 2, 2, 2, 2, 
    8, 2, 2, 2, 2, 3, 7, 7, 7, 7, 7, 7, 2, 2, 2, 2, 
    8, 8, 2, 2, 2, 2, 0, 4, 7, 7, 7, 3, 2, 7, 7, 2, 
    8, 8, 2, 2, 2, 2, 3, 7, 7, 7, 7, 2, 2, 0, 7, 7, 
    8, 8, 8, 2, 2, 2, 4, 4, 7, 7, 7, 8, 2, 2, 4, 7, 
    8, 8, 8, 2, 2, 2, 2, 2, 7, 7, 7, 8, 8, 2, 3, 3, 
    8, 8, 8, 2, 2, 2, 2, 2, 7, 7, 2, 8, 8, 2, 2, 3, 
    8, 2, 2, 2, 2, 2, 2, 2, 7, 7, 2, 8, 8, 2, 2, 2, 
    8, 2, 4, 3, 2, 2, 2, 2, 7, 7, 2, 8, 8, 2, 2, 2, 
    8, 8, 2, 2, 8, 2, 2, 3, 7, 7, 8, 8, 8, 2, 2, 2, 
    8, 2, 8, 2, 2, 2, 2, 3, 7, 7, 8, 8, 8, 2, 2, 2, 
    8, 2, 8, 2, 7, 4, 2, 7, 7, 7, 8, 8, 8, 8, 2, 8};

  cusp::array2d<int, cusp::device_memory> d_cl(16,16);
  thrust::copy(l.begin(), l.end(), d_cl.values.begin());
  cusp::array2d<int, cusp::device_memory> d_sl(16,16);

  split::device::ccl::connected_components(d_cl, d_temp_ptr, d_sl.values, 1);
  split::device::ccl::compress_labels(d_sl.values, d_temp_ptr);
  //cusp::print(d_sl);

  const int nsegments = (*thrust::max_element(d_sl.values.begin(),
                                              d_sl.values.end())) +
                        1;
  std::cout << nsegments << '\n';
#else
  split::device::ccl::connected_components(
      d_cluster_labels, d_temp_ptr, d_segment_labels.values, 10);

  split::device::ccl::compress_labels(d_segment_labels.values, d_temp_ptr);
  const int nsegments = (*thrust::max_element(d_segment_labels.values.begin(),
                                              d_segment_labels.values.end())) +
                        1;
  std::cout << nsegments << '\n';
  // Re-calculate the centroids using the segment labels
  cusp::array2d<real, cusp::device_memory, cusp::column_major> d_seg_centroids(
    nsegments, h_image.n_channels());
  split::device::kmeans::calculate_centroids(
    d_segment_labels.values, d_image, d_seg_centroids, d_itemp);
  split::device::kmeans::propagate_centroids(
    streams, d_segment_labels.values, d_seg_centroids, d_image);

  make_host_image(d_image, h_image.get());
  split::host::stbi::writef("assets/images/segments.png", h_image);
#endif

  return 0;
}

