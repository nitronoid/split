#include "split/device/ccl/merge_smooth_boundaries.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/compress_segments.cuh"
#include "split/device/detail/segment_length.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
namespace
{
__host__ __device__ real sqr(real x)
{
  return x * x;
}

struct VecNorm
{
  using vec3 = thrust::tuple<real, real, real>;
  __host__ __device__ real
  operator()(const thrust::tuple<const vec3, const vec3&>& pair) const
  {
    return sqr(pair.get<0>().get<0>() - pair.get<1>().get<0>()) +
           sqr(pair.get<0>().get<1>() - pair.get<1>().get<1>()) +
           sqr(pair.get<0>().get<2>() - pair.get<1>().get<2>());
  }
};
}  // namespace

SPLIT_API void merge_smooth_boundaries(
  cusp::array2d<int, cusp::device_memory>::const_view di_segment_edges,
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array1d<int, cusp::device_memory>::view dio_segment_labels,
  const real i_threshold)
{
  const int npoints = di_points.num_cols;
  const int nedges = di_segment_edges.num_cols;
  // Sort the edges based on their segment labels, so that we can reduce
  // correctly later.
  cusp::array2d<int, cusp::device_memory> edge_copy = di_segment_edges;
  cusp::array2d<int, cusp::device_memory> adjacency(2, nedges);
  //---------------------------------------------------------------------------
  // Make a new list from the edges, containing each edge pairs segment labels
  //---------------------------------------------------------------------------
  // Copy the edge labels into our adjacency lists
  thrust::gather(edge_copy.values.begin(),
                 edge_copy.values.end(),
                 dio_segment_labels.begin(),
                 adjacency.values.begin());
  // Sort by adjacency target and then stable sort by adjacency source so our
  // edges same source and target are neighboring
  thrust::sort_by_key(adjacency.row(1).begin(),
                      adjacency.row(1).end(),
                      detail::zip_it(adjacency.row(0).begin(),
                                     edge_copy.row(0).begin(),
                                     edge_copy.row(1).begin()));
  thrust::stable_sort_by_key(adjacency.row(0).begin(),
                             adjacency.row(0).end(),
                             detail::zip_it(adjacency.row(1).begin(),
                                            edge_copy.row(0).begin(),
                                            edge_copy.row(1).begin()));

  // Iterate over pairs of adjacency keys and values
  auto adj_begin =
    detail::zip_it(adjacency.row(0).begin(), adjacency.row(1).begin());
  auto adj_end = adj_begin + nedges;
  // We can produce a unique single key from two
  auto shift_key_begin = thrust::make_transform_iterator(
    adj_begin, [] __host__ __device__(const thrust::tuple<int, int>& pair) {
      const int64_t long_copy = pair.get<0>();
      return (long_copy << 31) + pair.get<1>();
    });
  auto shift_key_end = shift_key_begin + nedges;
  // Compress the adjacency key pairs into a normal sequence, to find the length
  cusp::array1d<int, cusp::device_memory> segments(nedges);
  detail::compress_segments(shift_key_begin, shift_key_end, segments.begin());
  // Store this as a floating point value for division later
  cusp::array1d<real, cusp::device_memory> segments_length(segments.back());
  detail::segment_length(
    segments.begin(), segments.end(), segments.back(), segments_length.begin());
  //----------------------------------------------------------------------------
  // Reduce the neighboring edge differences
  //----------------------------------------------------------------------------
  cusp::array1d<real, cusp::device_memory> d_d(nedges);
  // Iterate over the pairs of RGB values and transform them into a scalar
  // difference
  auto point_value_it =
    thrust::make_permutation_iterator(detail::zip_it(di_points.row(0).begin(),
                                                     di_points.row(1).begin(),
                                                     di_points.row(2).begin()),
                                      edge_copy.values.begin());
  auto edge_difference_begin = thrust::make_transform_iterator(
    detail::zip_it(point_value_it, point_value_it + nedges), VecNorm());
  // Reduce the differences to a single scalar, per adjacency
  thrust::reduce_by_key(
    adj_begin, adj_end, edge_difference_begin, adj_begin, d_d.begin());
  auto new_end = thrust::unique(adj_begin, adj_end);
  // Divide through to get the averages
  thrust::transform(
    d_d.begin(), d_d.end(), segments_length.begin(), d_d.begin(), thrust::divides<real>());
  //----------------------------------------------------------------------------
  // Finalize targets
  //----------------------------------------------------------------------------
  // Calculate the squared threshold
  const real D =
    i_threshold *
    sqr(*thrust::max_element(di_points.values.begin(), di_points.values.end()));
  // Remove any adjacency mappings that don't meet the threshold
  auto matrix_begin = detail::zip_it(
    adjacency.row(0).begin(), adjacency.row(1).begin(), d_d.begin());
  auto matrix_end = matrix_begin + (new_end - adj_begin);
  auto new_matrix_end = thrust::remove_if(
    matrix_begin,
    matrix_end,
    [=] __host__ __device__(const thrust::tuple<int, int, real>& tup) {
      return tup.get<2>() >= D;
    });
  // Initialize the targets as original labels
  cusp::array1d<int, cusp::device_memory> targets(npoints);
  cusp::array2d<int, cusp::device_memory> d_final_mapping(2, npoints);
  auto discard_it = thrust::make_discard_iterator();
  auto final_map_it = detail::zip_it(
    d_final_mapping.row(0).begin(), d_final_mapping.row(1).begin(), discard_it);
  thrust::copy(
    dio_segment_labels.begin(), dio_segment_labels.end(), targets.begin());
  // If we are faced with multiple options, we take the smallest scalar
  // TODO: We can do better than to load the adjacency keys twice
  auto final_map_end = thrust::reduce_by_key(
    adj_begin,
    adj_end + (new_matrix_end - matrix_begin),
    matrix_begin,
    discard_it,
    final_map_it,
    thrust::equal_to<thrust::tuple<int, int>>(),
    [=] __host__ __device__(const thrust::tuple<int, int, real>& lhs,
                            const thrust::tuple<int, int, real>& rhs) {
      return lhs.get<2>() < rhs.get<2>() ? lhs : rhs;
    });
  const int nmappings = final_map_end.second - final_map_it;

  // Output the final mapping
  thrust::scatter(d_final_mapping.row(1).begin(),
                  d_final_mapping.row(1).begin() + nmappings,
                  d_final_mapping.row(0).begin(),
                  targets.begin());

  cusp::array1d<int, cusp::device_memory> labels_copy(npoints, 0);
  auto old_labels = labels_copy.begin();
  auto new_labels = dio_segment_labels.begin();
  auto has_converged = [=] {
    return thrust::equal(new_labels, new_labels + npoints, old_labels);
  };
  // Map until we've converged
  while (!has_converged())
  {
    thrust::copy_n(new_labels, npoints, old_labels);
    thrust::gather(old_labels, labels_copy.end(), targets.begin(), new_labels);
  }
}

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

