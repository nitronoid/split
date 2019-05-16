#include "split/device/separation/luminance_continuity.cuh"
#include "split/device/ccl/links.cuh"
#include "split/device/detail/sorted_segment_adjacency_edges.cuh"
#include "split/device/detail/segment_length.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/average_by_key.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include "split/device/detail/view_util.cuh"
#include <thrust/iterator/transform_output_iterator.h>
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

#define CUDA_ERR_CHECK                                                       \
  cudaDeviceSynchronize();                                                   \
  error = cudaGetLastError();                                                \
  printf("Checking for error on line %d\n", __LINE__);                       \
  if (error != cudaSuccess)                                                  \
  {                                                                          \
    /* print the CUDA error message and exit*/                               \
    printf(                                                                  \
      "CUDA error: %s , on line %d\n", cudaGetErrorString(error), __LINE__); \
    exit(-1);                                                                \
  }

namespace separation
{
namespace
{
struct TupleSum
{
  using vec = thrust::tuple<int, int>;
  __host__ __device__ int operator()(const vec& v) const
  {
    return v.get<0>() + v.get<1>();
  }
};
struct PairProduct
{
  using vec = thrust::tuple<real, real>;
  __host__ __device__ real operator()(const vec& v) const
  {
    return v.get<0>() * v.get<1>();
  }
};

struct LuminanceThreshold
{
  LuminanceThreshold(const real* L_av, real threshold)
    : L_av(L_av), threshold(threshold)
  {
  }

  const real* __restrict__ L_av;
  const real __restrict__ threshold;

  using ivec2 = thrust::tuple<int, int>;
  __host__ __device__ bool operator()(const ivec2& v) const
  {
    return abs(L_av[v.get<0>()] - L_av[v.get<1>()]) < threshold;
  }
};

}  // namespace

SPLIT_API std::tuple<cusp::array2d<real, cusp::device_memory>,
                     cusp::array1d<real, cusp::device_memory>>
similar_reflectance(
  cusp::array1d<int, cusp::device_memory>::const_view di_cluster_labels,
  cusp::array1d<int, cusp::device_memory>::const_view di_segment_labels,
  cusp::array2d<real, cusp::device_memory>::const_view di_rgb,
  cusp::array1d<real, cusp::device_memory>::const_view di_luminance,
  const int i_nclusters,
  const int i_nsegments,
  thrust::device_ptr<void> do_temp)
{
  cudaError_t error;
  const int npoints = di_segment_labels.size();

  cusp::array1d<int, cusp::device_memory> d_indices(npoints);
  cusp::array1d<int, cusp::device_memory> d_cluster_label_cpy(npoints);
  cusp::array1d<int, cusp::device_memory> d_segment_label_cpy(npoints);
  cusp::array1d<int, cusp::device_memory> d_segment_sizes(i_nsegments);
  cusp::array2d<real, cusp::device_memory> d_average_rgb(3, i_nsegments);
  cusp::array1d<real, cusp::device_memory> d_average_lumninance(i_nsegments);
  CUDA_ERR_CHECK

  // Initialize the indices to a default ascending sequence
  thrust::sequence(d_indices.begin(), d_indices.end());
  // Copy the labels across
  thrust::copy_n(
    di_cluster_labels.begin(), npoints, d_cluster_label_cpy.begin());
  thrust::copy_n(
    di_segment_labels.begin(), npoints, d_segment_label_cpy.begin());
  CUDA_ERR_CHECK

  // Useful to reduce verbosity later
  const auto discard_it = thrust::make_discard_iterator();

  // Sort indices by their segment labels
  thrust::sort_by_key(
    d_segment_label_cpy.begin(), d_segment_label_cpy.end(), d_indices.begin());
  CUDA_ERR_CHECK

  // Find the average RGB values per segment
  {
    // Calculate the channel we want to read from
    const auto channel_offset = detail::make_row_iterator(npoints);
    // Read the sorted index that we want to read from
    const auto index_cycle =
      detail::make_cycle_iterator(d_indices.begin(), npoints);
    // Add the channel offset to our index look-up
    const auto read_rgb = thrust::make_transform_iterator(
      detail::zip_it(channel_offset, index_cycle), TupleSum());
    // Loop over the labels per channel
    auto segment_label_cycle_begin =
      detail::make_cycle_iterator(d_segment_label_cpy.begin(), npoints);
    // Average the rgb pixel values for each segment
    detail::average_by_key(
      segment_label_cycle_begin,
      segment_label_cycle_begin + npoints * 3,
      thrust::make_permutation_iterator(di_rgb.values.begin(), read_rgb),
      detail::make_cycle_iterator(d_segment_sizes.begin(), i_nsegments),
      discard_it,
      d_average_rgb.values.begin(),
      i_nsegments);
  }
  CUDA_ERR_CHECK

  // Average the luminance per segment. Don't need to use the cycle iterators
  // for 1D data.
  detail::average_by_key(
    d_segment_label_cpy.begin(),
    d_segment_label_cpy.end(),
    thrust::make_permutation_iterator(di_luminance.begin(), d_indices.begin()),
    d_segment_sizes.begin(),
    discard_it,
    d_average_lumninance.begin(),
    i_nsegments);
  CUDA_ERR_CHECK

  // Copy the segment labels again, to get the original unsorted sequence
  thrust::copy_n(
    di_segment_labels.begin(), npoints, d_segment_label_cpy.begin());
  CUDA_ERR_CHECK
  // Calculate the inter cluster links, as there will be one equation per link
  auto d_links = ccl::inter_cluster_links(d_cluster_label_cpy,
                                          d_segment_label_cpy,
                                          i_nclusters,
                                          i_nsegments,
                                          do_temp);
  int nlinks = d_links.size() / 2;
  CUDA_ERR_CHECK
  cusp::print(d_links.subarray(0, 25));
  cusp::print(d_links.subarray(nlinks, 25));
  printf("NLINKS: %d\n", nlinks);

  // Iterator to access the average luminance per segment
  const auto L_av = d_average_lumninance.begin();
  // Calculate the threshold as 5% of the maximum luminance
  const real L_thresh =
    0.05f * (*thrust::max_element(L_av, L_av + i_nsegments));
  printf("THRESH: %f\n", L_thresh);
  // Iterate over the start and end of the links simultaneously
  auto link_begin = detail::zip_it(d_links.begin(), d_links.begin() + nlinks);
  // Remove links which don't meet the luminance threshold
  nlinks = thrust::remove_if(link_begin,
                             link_begin + nlinks,
                             LuminanceThreshold(L_av.base().get(), L_thresh)) -
           link_begin;
  CUDA_ERR_CHECK

  printf("NLINKS: %d\n", nlinks);

  // We can allocate for the equations now we know how many there are
  cusp::array2d<real, cusp::device_memory> do_A(nlinks * 3, i_nsegments);
  cusp::array1d<real, cusp::device_memory> do_b(nlinks * 3);
  CUDA_ERR_CHECK

  // Cycle over the source indices in the links
  const auto cycle_r_links =
    detail::make_cycle_iterator(d_links.begin(), nlinks);
  // Cycle over the target indices in the links
  const auto cycle_s_links =
    detail::make_cycle_iterator(d_links.begin() + nlinks, nlinks);
  // We need to loop over the link indices, but also move to the next channel
  // when passed and index > nlinks.
  const auto channel_offset =
    thrust::make_transform_iterator(detail::make_row_iterator(nlinks),
                                    detail::unary_multiplies<int>(i_nsegments));
  // Calculate the index that we need to read average rgb values from, this is
  // essentially a matrix lookup where we skip one half (source or target).
  const auto read_Icr = thrust::make_transform_iterator(
    detail::zip_it(channel_offset, cycle_r_links), TupleSum());
  const auto read_Ics = thrust::make_transform_iterator(
    detail::zip_it(channel_offset, cycle_s_links), TupleSum());
  // Iterator to the start of the average rgb
  const auto I_c = d_average_rgb.values.begin();
  // Access our Average rgb per link, per rgb channel
  const auto I_cr = thrust::make_permutation_iterator(I_c, read_Icr);
  const auto I_cs = thrust::make_permutation_iterator(I_c, read_Ics);
  // Get a handle to the average luminance for each link, cycle back for each
  // channel of the rgb
  const auto L_avr = thrust::make_permutation_iterator(L_av, cycle_r_links);
  const auto L_avs = thrust::make_permutation_iterator(L_av, cycle_s_links);
  // Iterate over the opposing pairs for the final equation, getting their
  // product on read
  const auto numer =
    thrust::make_transform_iterator(detail::zip_it(I_cs, L_avr), PairProduct());
  const auto denom =
    thrust::make_transform_iterator(detail::zip_it(I_cr, L_avs), PairProduct());
  CUDA_ERR_CHECK
  cusp::print(do_b.subarray(0, 5));
  CUDA_ERR_CHECK
  // Calculate the b vector entries
  thrust::transform(
    numer, numer + nlinks, denom, do_b.begin(), thrust::divides<real>());
  CUDA_ERR_CHECK
  cusp::print(do_b.subarray(0, 5));

  // Now we write out the coefficient matrix A
  // The equation is ln(fs) - ln(fr), so our coefficients are always -1 and 1
  const auto coefficient_pair =
    thrust::make_constant_iterator(thrust::make_tuple(-1.f, 1.f));
  // Get the row of the dense matrix we'll be writing to
  const auto row_offset =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                    detail::unary_multiplies<int>(i_nsegments));
  // Get the two indices we'll need to write our coefficient pair into
  const auto write_Ar = thrust::make_transform_iterator(
    detail::zip_it(row_offset, cycle_r_links), TupleSum());
  const auto write_As = thrust::make_transform_iterator(
    detail::zip_it(row_offset, cycle_s_links), TupleSum());
  // Package the two indices into one
  auto write_A = detail::zip_it(
    thrust::make_permutation_iterator(do_A.values.begin(), write_Ar),
    thrust::make_permutation_iterator(do_A.values.begin(), write_As));
  // Finally copy the coefficients across, three times as we have one equation
  // per channel.
  thrust::copy_n(coefficient_pair, nlinks * 3, write_A);
  CUDA_ERR_CHECK

  return std::make_tuple(do_A, do_b);
}

}  // namespace separation

SPLIT_DEVICE_NAMESPACE_END

