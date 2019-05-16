#include "split/device/separation/luminance_continuity.cuh"
#include "split/device/detail/sorted_segment_adjacency_edges.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/view_util.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include <thrust/iterator/transform_output_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

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
}  // namespace

SPLIT_API std::tuple<cusp::array2d<real, cusp::device_memory>,
                     cusp::array1d<real, cusp::device_memory>>
luminance_continuity(
  cusp::array2d<int, cusp::device_memory>::const_view di_labels,
  cusp::array1d<real, cusp::device_memory>::const_view di_luminance,
  thrust::device_ptr<void> dio_temp)
{
  const int npoints = di_luminance.size();
  cusp::array1d<int, cusp::device_memory> d_label_copy(npoints);
  // Calculate the edges
  cusp::array1d<int, cusp::device_memory> d_connections(16 *
                                                        di_labels.num_entries);
  // Calculate the edges
  const int nedges = ccl::segment_adjacency_edges(di_labels, d_connections);
  auto d_edges = cusp::make_array2d_view(2,
                                         nedges,
                                         nedges,
                                         d_connections.subarray(0, nedges * 2),
                                         cusp::row_major{});
  cusp::array2d<int, cusp::device_memory> d_adjacency(2, nedges);
  thrust::copy_n(di_labels.values.begin(), npoints, d_label_copy.begin());
  // Sort the edges by their labels, and obtain those labels in a new list
  detail::sorted_segment_adjacency_edges(d_edges.values.begin(),
                                         d_edges.values.end(),
                                         d_label_copy.begin(),
                                         d_adjacency.values.begin());
  // Iterate over pairs of adjacency keys and values
  auto adj_begin =
    detail::zip_it(d_adjacency.row(0).begin(), d_adjacency.row(1).begin());
  auto adj_end = adj_begin + nedges;
  // After sorting, the largest label should be here, so we can easily find the
  // number of segments
  const int nsegments = d_adjacency.row(0).back();

  // Index the luminance using the sorted edges
  auto boundary_luminance_pairs =
    detail::zip_it(thrust::make_permutation_iterator(di_luminance.begin(),
                                                     d_edges.row(0).begin()),
                   thrust::make_permutation_iterator(di_luminance.begin(),
                                                     d_edges.row(1).begin()));

  cusp::array2d<real, cusp::device_memory> d_luminance_totals(2, nedges);
  auto total_it = detail::zip_it(d_luminance_totals.row(0).begin(),
                                 d_luminance_totals.row(1).begin());
  // Get the total luminance at each boundary
  using vec2 = thrust::tuple<real, real>;
  thrust::reduce_by_key(
    adj_begin,
    adj_end,
    boundary_luminance_pairs,
    thrust::make_discard_iterator(),
    total_it,
    thrust::equal_to<thrust::tuple<int, int>>(),
    [] __host__ __device__(const vec2& lhs, const vec2& rhs) {
      return thrust::make_tuple(lhs.get<0>() + rhs.get<0>(),
                                lhs.get<1>() + rhs.get<1>());
    });
  // Compress the adjacency pairs and get the number of entries
  const int nadjacency = thrust::unique(adj_begin, adj_end) - adj_begin;

  // Now that we know the number of equations, allocate for the system
  cusp::array2d<real, cusp::device_memory> do_A(nadjacency, nsegments);
  cusp::array1d<real, cusp::device_memory> do_b(nadjacency);

  // Divide our boundary edge target luminance totals, by the source ones, and
  // write the result into the b vector. On write we take the natural logarithm
  // of the division result.
  const auto lbnd_r = d_luminance_totals.row(0).begin();
  const auto lbnd_s = d_luminance_totals.row(1).begin();
  auto log_b = thrust::make_transform_output_iterator(
    do_b.begin(), detail::unary_log<real>());
  thrust::transform(
    lbnd_s, lbnd_s + nadjacency, lbnd_r, log_b, thrust::divides<real>());

  // Now we write out the coefficient matrix A
  // The equation is ln(fr) - ln(fs), so our coefficients are always 1 and -1
  const auto coefficient_pair =
    thrust::make_constant_iterator(thrust::make_tuple(1.f, -1.f));
  // Get the row of the dense matrix we'll be writing to
  const auto row_offset =
    thrust::make_transform_iterator(thrust::make_counting_iterator(0),
                                    detail::unary_multiplies<int>(nsegments));
  // Get the two indices we'll need to write our coefficient pair into
  const auto write_Ar = thrust::make_transform_iterator(
    detail::zip_it(row_offset, d_adjacency.row(0).begin()), TupleSum());
  const auto write_As = thrust::make_transform_iterator(
    detail::zip_it(row_offset, d_adjacency.row(1).begin()), TupleSum());
  // Package the two indices into one
  auto write_A = detail::zip_it(
    thrust::make_permutation_iterator(do_A.values.begin(), write_Ar),
    thrust::make_permutation_iterator(do_A.values.begin(), write_As));
  // Finally copy the coefficients across
  thrust::copy_n(coefficient_pair, nadjacency, write_A);

  // Return the equations
  return std::make_tuple(do_A, do_b);
}

}  // namespace separation

SPLIT_DEVICE_NAMESPACE_END

