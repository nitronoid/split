#include "split/device/kmeans/kmeans.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/ccl/connected_components.cuh"
#include "split/device/ccl/compress_labels.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/ccl/merge_small_segments.cuh"
#include "split/device/ccl/merge_smooth_boundaries.cuh"
#include "split/device/color/conversion.cuh"
#include "split/device/color/beta_feature.cuh"
#include "split/device/detail/view_util.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include "split/host/stbi/stbi_raii.hpp"
#include "split/device/detail/cu_raii.cuh"
#include "split/device/morph/erode.cuh"

#include <cusp/print.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <chrono>

#define real split::real

template <typename T>
void BAD_call_destructor(T& io_obj)
{
  io_obj.~T();
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
  switch (error)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

void sandbox()
{
  printf("\n\n\n\n\n SANDBOX\n\n");
  const int A = 0;
  const int B = 1;
  const int C = 2;
  const int D = 3;
  std::vector<int> labels = {
    A,A,A,A,A,A,A,
    A,A,A,A,B,B,B,
    A,A,A,A,B,B,B,
    A,A,B,B,B,B,B,
    C,C,B,B,B,B,B,
    C,C,C,C,C,D,D,
    C,C,C,C,C,D,D
  };
  const int npixels = 42;

  cusp::array2d<int, cusp::device_memory> d_labels(7, 7);
  thrust::copy_n(labels.begin(), npixels, d_labels.values.begin());

  split::device::morph::erode(d_labels, 2);

  cusp::print(d_labels);

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
#if 0
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
  const int nclusters = 5;
  const int npixels = h_image.n_pixels();

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

  // Re-calculate the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_cluster_labels.values, d_rgb_image, d_centroids, d_temp.get());
  //---------------------------------------------------------------------------
  auto d_chrominance = split::device::detail::make_const_array2d_view(
    cusp::make_array2d_view(2,
                            npixels,
                            npixels,
                            d_lab_image.values.subarray(npixels, npixels * 2),
                            cusp::row_major{}));

  for (int i = 0; i < 10 && (nsegments > 5000 || i < 1); ++i)
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

  // Sanity check
  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/sanity.png", h_image);

  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_cluster_labels.values, d_centroids, d_rgb_image);
  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/clusters.png", h_image);

  // Re-calculate the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_segment_labels.values, d_rgb_image, d_seg_centroids, d_temp.get());
  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_segment_labels.values, d_seg_centroids, d_rgb_image);

  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/components.png", h_image);

  //------------------------------------------------------------------
#else

  sandbox();
#endif

  return 0;
}

