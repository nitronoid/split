#include "split/device/ccl/merge_smooth_boundaries.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/detail/sorted_segment_adjacency_edges.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/compress_segments.cuh"
#include "split/device/detail/segment_length.cuh"
#include "split/device/detail/average_by_key.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include "split/device/detail/fix_map_cycles.cuh"
#include "split/device/detail/map_until_converged.cuh"
#include "split/device/detail/hash_pair.cuh"
#include "split/device/detail/view_util.cuh"

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
    return sqrt(sqr(pair.get<0>().get<0>() - pair.get<1>().get<0>()) +
                sqr(pair.get<0>().get<1>() - pair.get<1>().get<1>()) +
                sqr(pair.get<0>().get<2>() - pair.get<1>().get<2>()));
  }
};

template <typename ForwardIterator>
real max_colwise_norm(ForwardIterator&& points_begin,
                      ForwardIterator&& points_end)
{
  const int npoints = (points_end - points_begin) / 3;
  // Type def a vector
  using vec3 = thrust::tuple<real, real, real>;
  // Create an iterator that produces the norm per column
  const auto colwise_norm_begin = thrust::make_transform_iterator(
    detail::zip_it(
      points_begin, points_begin + npoints, points_begin + npoints * 2),
    [] __host__ __device__(const vec3& v) -> real {
      return sqr(v.get<0>()) + sqr(v.get<1>()) + sqr(v.get<2>());
    });
  const auto colwise_norm_end = colwise_norm_begin + npoints;
  // Reduce to find the maximum, then square root
  return sqrt(thrust::reduce(
    colwise_norm_begin, colwise_norm_end, 0.f, thrust::maximum<real>()));
}

// Struct that defines the temporary memory partition types
struct Workspace
{
  using RealArrayT = cusp::array1d<real, cusp::device_memory>::view;
  using IntArrayT = cusp::array1d<int, cusp::device_memory>::view;
  using IntMatrixT =
    cusp::array2d_view<cusp::array1d_view<thrust::device_ptr<int>>>;
  using TupleT = std::tuple<IntMatrixT,
                            IntArrayT,
                            IntArrayT,
                            IntArrayT,
                            IntMatrixT,
                            IntArrayT,
                            RealArrayT>;
};

Workspace::TupleT partition_workspace(const int i_npoints,
                                      const int i_nsegments,
                                      const int i_nedges,
                                      thrust::device_ptr<void> di_temp)
{
  // Read the temp memory as integer storage
  auto itemp =
    thrust::device_pointer_cast(reinterpret_cast<int*>(di_temp.get()));
  // Partition all of our integer arrays
  auto d_adjacency = cusp::make_array2d_view(
    2,
    i_nedges,
    i_nedges,
    cusp::make_array1d_view(itemp, itemp + i_nedges * 2),
    cusp::row_major{});
  auto d_mask = cusp::make_array1d_view(d_adjacency.values.end(),
                                        d_adjacency.values.end() + i_nedges);
  auto d_segment_sizes =
    cusp::make_array1d_view(d_mask.end(), d_mask.end() + i_nedges);
  auto d_targets = cusp::make_array1d_view(d_segment_sizes.end(),
                                           d_segment_sizes.end() + i_nsegments);
  auto d_mapping = cusp::make_array2d_view(
    2,
    i_nsegments,
    i_nsegments,
    cusp::make_array1d_view(d_targets.end(), d_targets.end() + i_nsegments * 2),
    cusp::row_major{});
  auto d_label_buffer = cusp::make_array1d_view(
    d_mapping.values.end(), d_mapping.values.end() + i_npoints);
  // Read the temp memory as real storage
  auto rtemp = thrust::device_pointer_cast(
    reinterpret_cast<real*>(d_mapping.values.end().get()));
  // Partition the temp memory as real storage
  auto d_d = cusp::make_array1d_view(rtemp, rtemp + i_nedges);
  // Return the struct
  return Workspace::TupleT{d_adjacency,
                           d_mask,
                           d_segment_sizes,
                           d_targets,
                           d_mapping,
                           d_label_buffer,
                           d_d};
}

}  // namespace

SPLIT_API std::size_t merge_smooth_boundaries_workspace(const int i_npoints,
                                                        const int i_nsegments,
                                                        const int i_nedges)
{
  // Calculate the size of all the arrays we require
  const std::size_t adjacency = i_nedges * 2 * sizeof(int);
  const std::size_t segment_sizes = (i_nedges) * sizeof(int);
  const std::size_t targets = i_nsegments * sizeof(int);
  const std::size_t mapping = i_nsegments * 2 * sizeof(int);
  const std::size_t label_buffer = i_npoints * sizeof(int);
  const std::size_t scalars = i_nedges * sizeof(real);
  // Return the sum
  return adjacency + segment_sizes + targets + mapping + label_buffer + scalars;
}

SPLIT_API void merge_smooth_boundaries(
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  const int i_nsegments,
  cusp::array2d<int, cusp::device_memory>::view dio_labels,
  thrust::device_ptr<void> do_temp,
  const real i_threshold)
{
  const int npoints = di_points.num_cols;
  cusp::array1d<int, cusp::device_memory> d_edges(16 * npoints);
  // Calculate the edges
  const int nedges = segment_adjacency_edges(
    detail::make_const_array2d_view(dio_labels), d_edges);
  // Partition the temporary memory into all the regions we need
  Workspace::IntArrayT d_mask, d_segment_sizes, d_targets, d_label_buffer;
  Workspace::IntMatrixT d_adjacency, d_mapping;
  Workspace::RealArrayT d_d;
  std::tie(d_adjacency,
           d_mask,
           d_segment_sizes,
           d_targets,
           d_mapping,
           d_label_buffer,
           d_d) = partition_workspace(npoints, i_nsegments, nedges, do_temp);
  //---------------------------------------------------------------------------
  // Sort the edges by their labels, and obtain those labels in a new list
  detail::sorted_segment_adjacency_edges(d_edges.begin(),
                                         d_edges.begin() + nedges * 2,
                                         dio_labels.values.begin(),
                                         d_adjacency.values.begin());
  // Iterate over pairs of adjacency keys and values
  auto adj_begin =
    detail::zip_it(d_adjacency.row(0).begin(), d_adjacency.row(1).begin());
  auto adj_end = adj_begin + nedges;
  // We can produce a unique single key from two
  auto shift_key_it =
    thrust::make_transform_iterator(adj_begin, detail::hash_pair{});
  // Compress the adjacency key pairs into a normal sequence, to find the length
  detail::compress_segments(
    shift_key_it, shift_key_it + nedges, d_mask.begin());
  // Make this once
  const auto discard_it = thrust::make_discard_iterator();
  // Iterate over segment boundary differences
  auto point_value_it =
    thrust::make_permutation_iterator(detail::zip_it(di_points.row(0).begin(),
                                                     di_points.row(1).begin(),
                                                     di_points.row(2).begin()),
                                      d_edges.begin());
  auto edge_difference_begin = thrust::make_transform_iterator(
    detail::zip_it(point_value_it, point_value_it + nedges), VecNorm());
  // Calculate the average difference over all segment boundaries
  detail::average_by_key(d_mask.begin(),
                         d_mask.end(),
                         edge_difference_begin,
                         d_segment_sizes.begin(),
                         discard_it,
                         d_d.begin(),
                         d_mask.back() + 1,
                         0.f);
  // Compress the adjacency pairs and get the number of entries
  int nadjacency = thrust::unique(adj_begin, adj_end) - adj_begin;
  //----------------------------------------------------------------------------
  // Finalize targets
  //----------------------------------------------------------------------------
  // Calculate the threshold
  const real D = i_threshold * max_colwise_norm(di_points.values.begin(),
                                                di_points.values.end());
  // Iterate over the remaining adjacency pairs, with their scalar difference
  auto matrix_begin = detail::zip_it(d_adjacency.row(0).begin(),
                                     d_adjacency.row(1).begin(),
                                     d_segment_sizes.begin(),
                                     d_d.begin());
  // Remove any adjacency mappings that don't meet the threshold, or have
  // borders which are too small
  nadjacency =
    thrust::remove_if(
      matrix_begin,
      matrix_begin + nadjacency,
      [=] __host__ __device__(const thrust::tuple<int, int, int, real>& tup) {
        return tup.get<3>() >= D || tup.get<2>() <= 15;
      }) -
    matrix_begin;
  // Initialize the targets as original labels
  thrust::sequence(d_targets.begin(), d_targets.end());
  // If we are faced with multiple options, we take the smallest scalar
  const int nmappings =
    thrust::reduce_by_key(
      d_adjacency.row(0).begin(),
      d_adjacency.row(0).begin() + nadjacency,
      detail::zip_it(d_d.begin(), d_adjacency.row(1).begin()),
      d_mapping.row(0).begin(),
      detail::zip_it(discard_it, d_mapping.row(1).begin()),
      thrust::equal_to<int>(),
      thrust::less<thrust::tuple<real, int>>())
      .first -
    d_mapping.row(0).begin();
  // Output the final mapping
  thrust::scatter(d_mapping.row(1).begin(),
                  d_mapping.row(1).begin() + nmappings,
                  d_mapping.row(0).begin(),
                  d_targets.begin());
  // Remove any cyclic mappings, as they would cause oscillations
  detail::fix_map_cycles(d_targets.begin(), d_targets.end());
  // Start with sentinel values in the buffer
  thrust::fill_n(d_label_buffer.begin(), npoints, -1);
  // Map everything to it's target
  detail::map_until_converged(dio_labels.values.begin(),
                              dio_labels.values.end(),
                              d_targets.begin(),
                              d_label_buffer.begin());
}

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

