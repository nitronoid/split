#include "split/device/kmeans/kmeans.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/ccl/connected_components.cuh"
#include "split/device/ccl/compress_labels.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/ccl/merge_small_segments.cuh"
#include "split/device/ccl/merge_smooth_boundaries.cuh"
#include "split/device/color/conversion.cuh"
#include "split/device/color/beta_feature.cuh"
#include "split/device/separation/luminance_continuity.cuh"
#include "split/device/separation/similar_reflectance.cuh"
#include "split/device/detail/view_util.cuh"
#include "split/host/stbi/stbi_raii.hpp"

#include <cusp/print.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <chrono>

#define real split::real

void sandbox()
{
  printf("\n\n\n\n\n SANDBOX\n\n");
  std::vector<int> labels = {0, 0, 0, 1, 1, 4, 5, 0, 0, 0, 1, 1, 4, 5, 0, 0, 0,
                             0, 1, 4, 4, 2, 2, 1, 1, 1, 4, 4, 2, 2, 1, 1, 1, 3,
                             3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3};

  const int npixels = labels.size();
  int nsegments = 5;

  cusp::array2d<int, cusp::device_memory> d_labels(7, 7);
  thrust::copy_n(labels.begin(), npixels, d_labels.values.begin());
  cusp::array1d<int, cusp::device_memory> d_connections(16 * npixels);

  cusp::array2d<real, cusp::device_memory> d_chroma(3, npixels);
  thrust::tabulate(d_chroma.values.begin(),
                   d_chroma.values.end(),
                   [=] __device__(int) { return 0.5f; });
  cusp::array2d<real, cusp::device_memory> d_rgb_image(3, npixels);
  thrust::tabulate(d_rgb_image.values.begin(),
                   d_rgb_image.values.end(),
                   [=] __device__(int i) {
                     int x = i % npixels;
                     int y = i / npixels;

                     return x * (0.5f / npixels) + y * (0.5f / 3.f);
                   });

  thrust::device_ptr<void> d_temp = thrust::device_malloc(
    split::device::ccl::merge_small_segments_workspace(npixels, 5));

  for (int i = 0; i < 1; ++i)
  {
    std::cout << "Merging small clusters\n";
    split::device::ccl::merge_small_segments(d_chroma, d_labels, d_temp);
    nsegments = split::device::ccl::compress_labels(d_labels.values, d_temp);
  }
  std::cout << "Post small merging\n";
  for (int i = 0; i < d_labels.num_rows; ++i)
  {
    for (int j = 0; j < d_labels.num_cols; ++j)
    {
      std::cout << d_labels(i, j) << ' ';
    }
    std::cout << '\n';
  }

  for (int i = 0; i < 1; ++i)
  {
    std::cout << "Merging smooth boundaries\n";
    split::device::ccl::merge_smooth_boundaries(
      d_rgb_image, nsegments, d_labels, d_temp);
    nsegments = split::device::ccl::compress_labels(d_labels.values, d_temp);
  }
  std::cout << "Post smooth merging\n";
  for (int i = 0; i < d_labels.num_rows; ++i)
  {
    for (int j = 0; j < d_labels.num_cols; ++j)
    {
      std::cout << d_labels(i, j) << ' ';
    }
    std::cout << '\n';
  }

  thrust::device_free(d_temp);
}

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
};

int main(int argc, char* argv[])
{
#if 1
  assert(argc == 2);
  auto h_image = split::host::stbi::loadf(argv[1], 3);
  printf("Loaded image with dim: %dx%dx%d\n",
         h_image.width(),
         h_image.height(),
         h_image.n_channels());

  cusp::array2d<real, cusp::device_memory> d_rgb_image(h_image.n_channels(),
                                                       h_image.n_pixels());
  cusp::array2d<real, cusp::device_memory> d_lab_image(h_image.n_channels(),
                                                       h_image.n_pixels());
  make_device_image(h_image.get(), d_rgb_image);

  // Convert the input linear RGB image into L*a*b color space
  split::device::color::convert_color_space(
    d_rgb_image, d_lab_image, split::device::color::rgb_to_lab());
  // Make a copy of the image luminance before writing the beta feature
  cusp::array1d<real, cusp::device_memory> d_luminance(h_image.n_pixels());
  thrust::copy(
    d_lab_image.row(0).begin(), d_lab_image.row(0).end(), d_luminance.begin());
  // Compute the beta feature from our lab image, and write it in place of L
  split::device::color::beta_feature(d_lab_image, d_lab_image.row(0));

  // Create initial means
  const int nclusters = 10;
  const int npixels = h_image.n_pixels();
  cusp::array1d<split::device::ScopedCuStream, cusp::host_memory> streams(4);

  cusp::array2d<real, cusp::device_memory> d_centroids(h_image.n_channels(),
                                                       nclusters);
  split::device::kmeans::uniform_random_initialize(d_lab_image, d_centroids);

  cusp::array2d<int, cusp::device_memory> d_cluster_labels(h_image.height(),
                                                           h_image.width());
  cusp::array2d<int, cusp::device_memory> d_segment_labels(h_image.height(),
                                                           h_image.width());

  // Allocate temporary memory
  TempMemory d_temp(split::device::kmeans::cluster_workspace(
    npixels, nclusters, h_image.n_channels()));

  // K-means cluster the image
  split::device::kmeans::cluster(
    d_lab_image, d_centroids, d_cluster_labels.values, d_temp.get(), 100, 1e-3);

  // Obtain isolated segments from our initial clustering
  split::device::ccl::connected_components(
    d_cluster_labels, d_temp.get(), d_segment_labels.values, 100);
  // Compress the segment labels to produce a contiguous sequence
  int nsegments =
    split::device::ccl::compress_labels(d_segment_labels.values, d_temp.get());
  std::cout << "Segmented into " << nsegments << " connected components.\n";
  // Re-calculate the centroids using the segment labels
  cusp::array2d<real, cusp::device_memory> d_seg_centroids(h_image.n_channels(),
                                                           nsegments);

  // Re-calculate the centroids using the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_cluster_labels.values, d_rgb_image, d_centroids, d_temp.get());
  //---------------------------------------------------------------------------
  auto d_chrominance = split::device::detail::make_const_array2d_view(
    cusp::make_array2d_view(2,
                            npixels,
                            npixels,
                            d_lab_image.values.subarray(npixels, npixels * 2),
                            cusp::row_major{}));

  for (int i = 0; i < 10 && (nsegments > 1000 || i < 2); ++i)
  {
    TempMemory d_temp(
      split::device::ccl::merge_small_segments_workspace(npixels, nsegments));
    std::cout << "Merging small clusters\n";
    split::device::ccl::merge_small_segments(
      d_chrominance, d_segment_labels, d_temp.get(), 10 * (i + 1));
    nsegments = split::device::ccl::compress_labels(d_segment_labels.values,
                                                    d_temp.get());
  }
  printf("Number of segments post merge: %d\n", nsegments);

  for (int i = 0; i < 0; ++i)
  {
    TempMemory d_temp(split::device::ccl::merge_smooth_boundaries_workspace(
      npixels, nsegments, npixels * 8));
    std::cout << "Merging smooth boundaries\n";
    split::device::ccl::merge_smooth_boundaries(
      d_rgb_image, nsegments, d_segment_labels, d_temp.get(), 0.005f);
    nsegments = split::device::ccl::compress_labels(d_segment_labels.values,
                                                    d_temp.get());
  }

  //----------------------------------------------------------------------------
  // Build the linear system of intrinsic separation equations
  //----------------------------------------------------------------------------
  {
    printf("Generating luminance continuity equations.\n");
    cusp::array2d<real, cusp::device_memory> A;
    cusp::array1d<real, cusp::device_memory> b;
    // Calculate the luminance continuity equations
    std::tie(A, b) = split::device::separation::luminance_continuity(
      d_segment_labels, d_luminance, d_temp.get());
  }
  {
    printf("Generating similar reflectance equations.\n");
    cusp::array2d<real, cusp::device_memory> A;
    cusp::array1d<real, cusp::device_memory> b;
    // Calculate the luminance continuity equations
    std::tie(A, b) =
      split::device::separation::similar_reflectance(d_cluster_labels.values,
                                                     d_segment_labels.values,
                                                     d_rgb_image,
                                                     d_luminance,
                                                     nclusters,
                                                     nsegments,
                                                     d_temp.get());
    //cusp::print(b);
  }

  // Re-calculate the centroids using the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_segment_labels.values, d_rgb_image, d_seg_centroids, d_temp.get());
  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_segment_labels.values, d_seg_centroids, d_rgb_image);

  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/merged.png", h_image);

  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_cluster_labels.values, d_centroids, d_rgb_image);

  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/clusters.png", h_image);
  //------------------------------------------------------------------
#else

  sandbox();
#endif

  return 0;
}

