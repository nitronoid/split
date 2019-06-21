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
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/shrink_segments.cuh"
#include "split/device/morph/erode.cuh"
#include "split/device/intrinsic/estimate_albedo_intensity.cuh"
#include "split/device/probability/remove_set_outliers.cuh"
#include "split/device/probability/set_probability.cuh"
#include "split/device/probability/set_selection.cuh"

#include <cusp/print.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/count.h>

#include <chrono>

#define real split::real

template <typename T>
void BAD_call_destructor(T& io_obj)
{
  io_obj.~T();
}

static const char* _cudaGetErrorEnum(cublasStatus_t error)
{
  switch (error)
  {
  case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";

  case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";

  case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";

  case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";

  case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";

  case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";

  case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";

  case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

void sandbox()
{
  printf("\n\n\n\n\n SANDBOX\n\n");
  std::vector<int> labels = {0,0,0,0,0,0, 1,1,1, 2,2, 3,3,3,3, 4, 5,5,5,5};
  const int nseg = 6;
  const int ndata = 20;

  cusp::array1d<int, cusp::device_memory> d_dummy(ndata);
  cusp::array1d<int, cusp::device_memory> d_labels(ndata);
  thrust::copy_n(labels.begin(), ndata, d_labels.begin());

  const int nleft = split::device::detail::shrink_segments(
    d_labels.begin(), d_labels.end(), d_dummy.begin(), nseg, 3);

  cusp::print(d_labels.subarray(0, nleft));
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
  const int npixels = h_image.n_pixels();

  //-----------------------------------------------------------------
  cusp::array2d<real, cusp::device_memory> d_intensity_chroma(
    3, h_image.n_pixels());
  split::device::color::convert_color_space(
    d_rgb_image, d_intensity_chroma, split::device::color::rgb_to_ic<>());
  printf("Lab conversion done\n");
  // Alias the intensity and the chroma values
  auto d_intensity = split::device::detail::make_const_array2d_view(
    cusp::make_array2d_view(h_image.height(),
                            h_image.width(),
                            h_image.width(),
                            d_intensity_chroma.row(0),
                            cusp::row_major{}));
  auto d_chroma =
    split::device::detail::make_const_array2d_view(cusp::make_array2d_view(
      2,
      npixels,
      npixels,
      d_intensity_chroma.values.subarray(npixels, npixels * 2),
      cusp::row_major{}));

  cusp::array1d<real, cusp::device_memory> d_albedo_intensity(
    h_image.n_pixels());
  thrust::copy_n(
    d_intensity.values.begin(), h_image.n_pixels(), d_albedo_intensity.begin());
  split::device::intrinsic::estimate_albedo_intensity(
    d_intensity, d_chroma, d_albedo_intensity);

  cusp::array2d<real, cusp::device_memory> d_albedo(3, h_image.n_pixels());
  // Now calc the albedo map
  auto albedo_intensity_begin = split::device::detail::make_cycle_iterator(
    d_albedo_intensity.begin(), npixels);
  thrust::transform(albedo_intensity_begin,
                    albedo_intensity_begin + npixels * 3,
                    d_chroma.values.begin(),
                    d_albedo.values.begin(),
                    thrust::multiplies<float>());
  printf("Intrinsic done\n");

  //-----------------------------------------------------------------
  // Convert the input linear RGB image into L*a*b color space
  split::device::color::convert_color_space(
    d_rgb_image, d_lab_image, split::device::color::rgb_to_lab<>());
  printf("Lab conversion done\n");

  // Make a copy of the image luminance before writing the beta feature
  cusp::array1d<real, cusp::device_memory> d_luminance(h_image.n_pixels());
  thrust::copy(
    d_lab_image.row(0).begin(), d_lab_image.row(0).end(), d_luminance.begin());
  // Compute the beta feature from our lab image, and write it in place of L
  split::device::color::beta_feature(d_lab_image, d_lab_image.row(0));

  // Create initial means
  const int nclusters = 5;

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
  auto d_chrominance =
    split::device::detail::make_const_array2d_view(cusp::make_array2d_view(
      2,
      h_image.n_pixels(),
      h_image.n_pixels(),
      d_lab_image.values.subarray(h_image.n_pixels(), h_image.n_pixels() * 2),
      cusp::row_major{}));

  for (int i = 0; i < 10; ++i)
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
  split::host::stbi::writef("../assets/images/sanity.png", h_image);

  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_cluster_labels.values, d_centroids, d_rgb_image);
  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("../assets/images/clusters.png", h_image);

  // Re-calculate the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_segment_labels.values, d_rgb_image, d_seg_centroids, d_temp.get());
  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_segment_labels.values, d_seg_centroids, d_rgb_image);

  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("../assets/images/components.png", h_image);

  split::device::morph::erode(d_segment_labels, 15);
  nsegments =
    split::device::ccl::compress_labels(d_segment_labels.values, d_temp.get());
  // Re-calculate the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_segment_labels.values, d_rgb_image, d_seg_centroids, d_temp.get());
  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_segment_labels.values, d_seg_centroids, d_rgb_image);
  thrust::transform(d_segment_labels.values.begin(),d_segment_labels.values.end(),d_segment_labels.values.begin(), split::device::detail::unary_minus<int>(1));

  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("../assets/images/eroded.png", h_image);

  printf("Probability begin\n");

  const int nsets = nsegments - 1;  // TODO;
  printf("NSets: %d\n", nsets);
  // Probability mapping
  // Filter out only the pixels left, post-erosion
  int n_set_ids = npixels - thrust::count(d_segment_labels.values.begin(),
                                          d_segment_labels.values.end(),
                                          -1);
  printf("NSet IDs: %d\n", n_set_ids);
  cusp::array1d<int, cusp::device_memory> d_set_ids(n_set_ids);
  const auto count = thrust::make_counting_iterator(0);
  const auto labell = d_segment_labels.values.begin();
  thrust::copy_if(count,
                  count + npixels,
                  d_set_ids.begin(),
                  [=] __host__ __device__ (int i)
                  {
                    return labell[i] != -1;
                  });

  n_set_ids = split::device::probability::set_selection(
    d_albedo, d_seg_centroids, d_segment_labels.values, d_set_ids, nsets);
  printf("NSet IDs: %d\n", n_set_ids);

  printf("Selection done\n");

  cusp::array2d<real, cusp::device_memory> d_probability(nsets,
                                                         h_image.n_pixels());
  split::device::probability::set_probability(
    d_albedo,
    d_segment_labels.values,
    split::device::detail::make_const_array1d_view(
      d_set_ids.subarray(0, n_set_ids)),
    nsets,
    d_probability.values);
  printf("Probability done\n");


  const int prob_idx = 2;

  for (int i = 0; i < nsets; ++i)
  {
  auto prob = d_probability.row(i);
  auto prob_begin = split::device::detail::make_cycle_iterator(
    prob.begin(), npixels);
  thrust::copy_n(prob_begin, npixels*3, d_rgb_image.values.begin());
  make_host_image(d_rgb_image, h_image.get());
  std::string path = "../assets/images/prob"+ std::to_string(i)+".png";
  split::host::stbi::writef(path.c_str(), h_image);
  }

  //------------------------------------------------------------------
#else

  sandbox();
#endif

  return 0;
}

