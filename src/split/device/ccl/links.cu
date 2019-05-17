#include "split/device/ccl/links.cuh"
#include "split/device/detail/segment_length.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/view_util.cuh"
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

namespace ccl
{
namespace
{
struct Combination
{
  using vec2 = thrust::tuple<int, int>;
  using vec3 = thrust::tuple<int, int, int>;
  __host__ __device__ vec2 operator()(const vec3& v) const
  {
    // Extract the ordinal sequence index
    const int i = v.get<0>();
    // Extract the cumulative size prior to this cluster
    const int o = v.get<2>();
    // Extract the cumulative size of the current cluster, subtract the prior
    // size to get the size of just this cluster then -1 to skip diagonals
    const int n = v.get<1>() - o - 1;

    // Small matrix row calculation
    const int r = i / n;
    // Small matrix column calculation
    int c = i % n;
    // If we're on or over the diagonal, we increment to ensure we skip it
    c += (c >= r);
    return thrust::make_tuple(r + o, c + o);
  }
};

struct n_combinations
{
  __host__ __device__ int operator()(int x) const
  {
    return x * (x - 1);
  }
};
}  // namespace

int cluster_segment_map(
  cusp::array1d<int, cusp::device_memory>::view dio_cluster_labels,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_labels)
{
  const int npoints = dio_cluster_labels.size();
  // Range iterators
  auto cluster_begin = dio_cluster_labels.begin();
  auto cluster_end = dio_cluster_labels.end();
  auto segment_begin = dio_segment_labels.begin();
  auto segment_end = dio_segment_labels.end();
  // Sort by segment, then by cluster to obtain neighboring parent child ties
  thrust::sort_by_key(segment_begin, segment_end, cluster_begin);
  thrust::stable_sort_by_key(cluster_begin, cluster_end, segment_begin);
  // Iterate over the cluster and segment labels simultaneously, as "ties"
  auto tie_begin = detail::zip_it(cluster_begin, segment_begin);
  // Get unique ties
  const int nmap = thrust::unique(tie_begin, tie_begin + npoints) - tie_begin;
  // Return the number of mappings
  return nmap;
}

int n_inter_cluster_combinations(
  cusp::array1d<int, cusp::device_memory>::const_view di_cluster_map,
  cusp::array1d<int, cusp::device_memory>::view do_lengths)
{
  // Range iterators
  auto cluster_begin = di_cluster_map.begin();
  auto cluster_end = di_cluster_map.end();
  auto length_begin = do_lengths.begin();
  auto length_end = do_lengths.end();
  // Calculate the number of segments per cluster
  detail::segment_length(cluster_begin, cluster_end, length_begin);
  // Calculate the total number of combinations
  const int ncombinations = thrust::transform_reduce(
    length_begin, length_end, n_combinations(), 0, thrust::plus<int>());
  // Return the number of combinations
  return ncombinations;
}

cusp::array1d<int, cusp::device_memory> inter_cluster_links(
  cusp::array1d<int, cusp::device_memory>::view dio_cluster_labels,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_labels,
  const int i_nclusters,
  const int i_nsegments,
  thrust::device_ptr<void> do_temp)
{
  cudaError_t error;
  cusp::array1d<int, cusp::device_memory> d_cluster_sizes(i_nclusters + 1);
  CUDA_ERR_CHECK
  // Convert to iterators up front
  auto cluster_begin = dio_cluster_labels.begin();
  auto cluster_end = dio_cluster_labels.end();
  auto segment_begin = dio_segment_labels.begin();
  auto segment_end = dio_segment_labels.end();
  // These occupy the same memory
  auto cumulative_length_begin = d_cluster_sizes.begin();
  auto length_begin = d_cluster_sizes.begin();
  auto length_end = d_cluster_sizes.end();

  const int npoints = dio_cluster_labels.size();

  // Compress the connections from clusters to segments to get a mapping
  const int nmapping =
    cluster_segment_map(dio_cluster_labels, dio_segment_labels);
  CUDA_ERR_CHECK
  // Calculate the number of segment combinations in each cluster and get their
  // sum.
  const int ncombinations = n_inter_cluster_combinations(
    detail::make_const_array1d_view(dio_cluster_labels.subarray(0, nmapping)),
    d_cluster_sizes);
  CUDA_ERR_CHECK
  // Store the new end of the cluster range
  cluster_end = cluster_begin + nmapping;

  cusp::array1d<int, cusp::device_memory> do_links(ncombinations * 2);
  CUDA_ERR_CHECK

  // Iterate over the cluster and segment labels simultaneously, as "ties"
  auto tie_begin = detail::zip_it(cluster_begin, segment_begin);
  using ivec2 = thrust::tuple<int, int>;
  // Remove any clusters with only one segment, and get the new end of range
  cluster_end = thrust::remove_if(tie_begin,
                                  tie_begin + npoints,
                                  [=] __host__ __device__(const ivec2& v) {
                                    return length_begin[v.get<0>()] == 1;
                                  })
                  .get_iterator_tuple()
                  .get<0>();
  CUDA_ERR_CHECK
  // Recompute the number of segments per cluster post removal
  detail::segment_length(cluster_begin, cluster_end, length_begin);
  CUDA_ERR_CHECK
  length_end = length_begin + i_nclusters;
  // Prefix-scan the lengths to get cumulative lengths
  thrust::exclusive_scan(length_begin, length_end, cumulative_length_begin);
  CUDA_ERR_CHECK

  cusp::array1d<int, cusp::device_memory> d_ordinal_sequence(ncombinations);
  auto ordinal_begin = d_ordinal_sequence.begin();
  // Obtain the end iterator using the number of combinations
  auto ordinal_end = ordinal_begin + ncombinations;
  // Scatter ones to the positions dictated by cumulative segment lengths
  const auto one = thrust::make_constant_iterator(0);
  thrust::gather(
      cumulative_length_begin, cumulative_length_begin + i_nclusters, one,
      ordinal_begin);
  // Inclusive scan to get which cluster each element of the sequence is in
  thrust::inclusive_scan(ordinal_begin, ordinal_end, ordinal_begin);
  // Subtract the previous cumulative length from the sequence to get a sequence
  // which resets at the head of a cluster
  CUDA_ERR_CHECK
  // Subtract the cumulative length from the sequence, using the scanned keys
  // TODO: combine this with a zip_iterator
  const auto o_length = 
    thrust::make_permutation_iterator(cumulative_length_begin, ordinal_begin);
  thrust::transform(
    ordinal_begin, ordinal_end, o_length, ordinal_begin, thrust::minus<int>());
  CUDA_ERR_CHECK

  // Access the cumulative length of the previous cluster (starts with 0)
  const auto p_length =
    thrust::make_permutation_iterator(cumulative_length_begin, cluster_begin);
  // Iterate over the ordinal sequence, current and previous cumulative cluster
  // lengths
  const auto ordinal_size =
    detail::zip_it(ordinal_begin,
                   thrust::make_permutation_iterator(
                     cumulative_length_begin + 1, cluster_begin),
                   p_length);
  // Iterator for the segment to segment links
  auto link_begin =
    detail::zip_it(do_links.begin(), do_links.begin() + ncombinations);
  // Transform the ordinal sequence and sizes into indices into the mappings
  thrust::transform(
    ordinal_size, ordinal_size + nmapping, link_begin, Combination());
  CUDA_ERR_CHECK
  // Finally, gather the correct mappings using the indices
  thrust::gather(
    do_links.begin(), do_links.end(), segment_begin, do_links.begin());
  CUDA_ERR_CHECK
  // Return the links
  return do_links;
}

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END
