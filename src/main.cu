#include "split/device/kmeans/kmeans.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/ccl/connected_components.cuh"
#include "split/device/ccl/compress_labels.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/ccl/merge_insignificant.cuh"
#include "split/device/color/conversion.cuh"
#include "split/device/color/beta_feature.cuh"
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
  std::vector<int> labels = {
    0, 0, 0, 3, 4,
    0, 0, 1, 3, 4,
    1, 1, 1, 3, 4,
    1, 2, 2, 3, 3,
    2, 2, 2, 5, 5,
    5, 5, 5, 5, 5};
  const int npixels = labels.size();

  cusp::array2d<int, cusp::device_memory> d_labels(6, 5);
  thrust::copy_n(labels.begin(), npixels, d_labels.values.begin());

  cusp::array1d<int, cusp::device_memory> d_connections(16 * npixels);
  const int nedges = 
    split::device::ccl::segment_adjacency_edges(d_labels, d_connections);
  auto d_edges = split::device::detail::make_const_array2d_view(
    cusp::make_array2d_view(2,
                            nedges,
                            nedges,
                            cusp::make_array1d_view(d_connections),
                            cusp::row_major{}));

  std::cout<<"Num edges: "<<nedges<<'\n';

  for (int i = 0; i < nedges; ++i)
  {
    std::cout<<d_edges(0, i)<<' '<<d_edges(1, i)<<'\n';
  }


  cusp::array1d<int, cusp::device_memory> d_adjacency_keys(nedges);
  cusp::array1d<int, cusp::device_memory> d_adjacency(nedges);
  const int nadjacency =
    split::device::ccl::segment_adjacency(d_labels.values,
                                          d_edges,
                                          d_adjacency_keys,
                                          d_adjacency);
  std::cout << "Number of segment adjacencies: " << nadjacency << '\n';

  for (int i = 0; i < nadjacency; ++i)
  {
    std::cout<<d_adjacency_keys[i]<<' '<<d_adjacency[i]<<'\n';
  }


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
  split::device::kmeans::initialize_centroids(d_lab_image, d_centroids);

  cusp::array2d<int, cusp::device_memory> d_cluster_labels(h_image.height(),
                                                           h_image.width());
  cusp::array2d<int, cusp::device_memory> d_segment_labels(h_image.height(),
                                                           h_image.width());

  // Allocate temporary memory
  TempMemory d_temp(h_image.n_pixels() * nclusters * sizeof(real));

  // K-means cluster the image
  split::device::kmeans::cluster(
    d_lab_image, d_centroids, d_cluster_labels.values, d_temp.get(), 100, 1e-1);

  // Obtain isolated segments from our initial clustering
  split::device::ccl::connected_components(
    d_cluster_labels, d_temp.get(), d_segment_labels.values, 100);
  // Compress the segment labels to produce a contiguous sequence
  const int nsegments =
    split::device::ccl::compress_labels(d_segment_labels.values, d_temp.get());
  std::cout << "Segmented into " << nsegments << " connected components.\n";
  // Re-calculate the centroids using the segment labels
  cusp::array2d<real, cusp::device_memory> d_seg_centroids(h_image.n_channels(),
                                                           nsegments);



  // Re-calculate the centroids using the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_cluster_labels.values, d_rgb_image, d_centroids, d_temp.get());

  // Re-calculate the centroids using the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_segment_labels.values, d_rgb_image, d_seg_centroids, d_temp.get());

  //---------------------------------------------------------------------------


  cusp::array1d<int, cusp::device_memory> d_segment_connections(16 * npixels);
  const int nedges = split::device::ccl::segment_adjacency_edges(
    d_segment_labels, d_segment_connections);
  std::cout << "Number of segment edges: " << nedges << '\n';

  auto d_segment_edges = split::device::detail::make_const_array2d_view(
    cusp::make_array2d_view(2,
                            nedges,
                            nedges,
                            cusp::make_array1d_view(d_segment_connections),
                            cusp::row_major{}));

  cusp::array1d<int, cusp::device_memory> d_segment_adjacency_keys(nedges);
  cusp::array1d<int, cusp::device_memory> d_segment_adjacency(nedges);
  const int nadjacency =
    split::device::ccl::segment_adjacency(d_segment_labels.values,
                                          d_segment_edges,
                                          d_segment_adjacency_keys,
                                          d_segment_adjacency);
  std::cout << "Number of segment adjacencies: " << nadjacency << '\n';

  auto d_chrominance = split::device::detail::make_const_array2d_view(
    cusp::make_array2d_view(2,
                            npixels,
                            npixels,
                            d_lab_image.values.subarray(npixels, npixels * 2),
                            cusp::row_major{}));

  const auto d_sak = 
    d_segment_adjacency_keys.subarray(0, nadjacency);
  const auto d_sa = 
    d_segment_adjacency.subarray(0, nadjacency);

  std::cout<<"MIN: "<<*thrust::min_element(d_segment_labels.values.begin(), d_segment_labels.values.end())<<'\n';
  std::cout << "Merging small clusters\n";
  split::device::ccl::merge_insignificant(
    d_chrominance,
    d_sak,
    d_sa,
    d_segment_labels.values);

  int nseg =
    split::device::ccl::compress_labels(d_segment_labels.values, d_temp.get());
  cusp::array2d<real, cusp::device_memory> d_merged_centroids(h_image.n_channels(),
                                                           nseg);

  // Re-calculate the centroids using the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_segment_labels.values, d_rgb_image, d_merged_centroids, d_temp.get());
  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_segment_labels.values, d_merged_centroids, d_rgb_image);

  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/merged.png", h_image);
  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_segment_labels.values, d_seg_centroids, d_rgb_image);

  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/segments.png", h_image);

  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_cluster_labels.values, d_centroids, d_rgb_image);

  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/clusters.png", h_image);
  //------------------------------------------------------------------


  //sandbox();


  return 0;
}

