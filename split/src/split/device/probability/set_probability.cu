#include "split/device/probability/set_probability.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/segment_length.cuh"
#include "split/device/detail/shrink_segments.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include <thrust/iterator/transform_output_iterator.h>
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace probability
{
namespace
{
struct Norm2
{
  using vec3 = thrust::tuple<real, real, real>;
  __host__ __device__ real sqr(real x)
  {
    return x * x;
  }

  __host__ __device__ real operator()(const vec3& a, const vec3& b)
  {
    return sqr(a.get<0>() + b.get<0>()) + sqr(a.get<1>() + b.get<1>()) +
           sqr(a.get<2>() + b.get<2>());
  }
};
// rolling my own multiply as CUSP was lazy and uses a serial impl
template <typename SourceBeginIt,
          typename SourceEndIt,
          typename TargetBeginIt,
          typename TargetEndIt,
          typename DistanceIt>
void compute_distances(SourceBeginIt&& source_begin,
                       SourceEndIt&& source_end,
                       TargetBeginIt&& target_begin,
                       TargetEndIt&& target_end,
                       DistanceIt&& distance_begin)
{
  const int32_t N = source_end - source_begin;
  const int32_t M = target_end - target_begin;
  auto A_begin = thrust::make_permutation_iterator(
    source_begin, detail::make_row_iterator(M));
  auto B_begin = thrust::make_permutation_iterator(
    target_begin, detail::make_column_iterator(M));
  thrust::transform(A_begin, A_begin + N * M, B_begin, distance_begin, Norm2());
}  // namespace
}  // namespace

SPLIT_API void set_probability(
  cusp::array2d<real, cusp::device_memory>::const_view di_albedo,
  cusp::array1d<int, cusp::device_memory>::const_view di_set_labels,
  cusp::array1d<int, cusp::device_memory>::const_view di_set_ids,
  const int i_nsets,
  cusp::array1d<real, cusp::device_memory>::view do_probability)
{
  // Get the total number of points, sets and points currently in sets
  const int npoints = di_albedo.num_cols;
  const int nids = di_set_ids.size();

  // Iterate over point albedos
  const auto albedo_begin = detail::zip_it(di_albedo.row(0).begin(),
                                           di_albedo.row(1).begin(),
                                           di_albedo.row(2).begin());
  const auto albedo_end = albedo_begin + npoints;

  // Copy the set labels and albedos for sorting
  // Iterate over the labels of the points in sets
  cusp::array1d<int, cusp::device_memory> d_label_copy(nids);
  auto set_labels_begin = d_label_copy.begin();
  auto set_labels_end = d_label_copy.end();
  thrust::copy_n(thrust::make_permutation_iterator(di_set_labels.begin(),
                                                   di_set_ids.begin()),
                 nids,
                 set_labels_begin);

  // Iterate over point albedos in the provided sets
  cusp::array2d<real, cusp::device_memory> d_albedo_copy(3, nids);
  const auto set_albedo_begin = detail::zip_it(d_albedo_copy.row(0).begin(),
                                               d_albedo_copy.row(1).begin(),
                                               d_albedo_copy.row(2).begin());
  const auto set_albedo_end = set_albedo_begin + nids;
  thrust::copy_n(
    thrust::make_permutation_iterator(albedo_begin, di_set_ids.begin()),
    nids,
    set_albedo_begin);

  // Make a copy of the set ids
  cusp::array1d<int, cusp::device_memory> d_set_id_copy(nids);
  const auto set_id_begin = d_set_id_copy.begin();
  const auto set_id_end = d_set_id_copy.end();
  thrust::copy_n(di_set_ids.begin(), nids, set_id_begin);

  // Now sort the albedos and point id's by the labels to get contiguous
  // segments
  thrust::sort_by_key(set_labels_begin,
                      set_labels_end,
                      detail::zip_it(set_albedo_begin, set_id_begin));
  cusp::print(d_label_copy.subarray(0, 20));

  // Compute the distances from each point, to each point in a set
  // Allocate an MxN matrix where M=num points and N=num points in sets
  cusp::array1d<real, cusp::device_memory> d_distances(npoints * nids);
  compute_distances(albedo_begin,
                    albedo_end,
                    set_albedo_begin,
                    set_albedo_end,
                    d_distances.begin());

  // Now reduce by segment to get the per point-segment combination totals
  const auto set_seq_begin =
    detail::make_cycle_iterator(set_labels_begin, nids);
  const auto set_seq_end = set_seq_begin + nids * npoints;
  const auto discard_it = thrust::make_discard_iterator();
  cusp::array1d<real, cusp::device_memory> d_averages(npoints * i_nsets);
  thrust::reduce_by_key(set_seq_begin,
                        set_seq_end,
                        d_distances.begin(),
                        discard_it,
                        d_averages.begin());
  // Divide through by the segment lengths to get average distances from each
  // point to each set
  cusp::array1d<int, cusp::device_memory> d_segment_lengths(i_nsets);
  detail::segment_length(
    set_labels_begin, set_labels_end, d_segment_lengths.begin());
  const auto seg_length =
    detail::make_cycle_iterator(d_segment_lengths.begin(), i_nsets);
  // Calculate the reciprocal on write
  const auto inv_out = d_averages.begin();// thrust::make_transform_output_iterator(
    //d_averages.begin(), detail::reciprocal<real>());
  thrust::transform(d_averages.begin(),
                    d_averages.end(),
                    seg_length,
                    inv_out,
                    thrust::divides<real>());

  const auto avg_begin = thrust::make_transform_iterator(
    d_averages.begin(), detail::unary_pow<real>(15.f));
  const auto avg_end = avg_begin + npoints * i_nsets;
  // Reduce by row to get the total distance from each point to all sets
  cusp::array1d<real, cusp::device_memory> d_total_distance(npoints);
  const auto seg_begin = detail::make_row_iterator(i_nsets);
  const auto seg_end = seg_begin + npoints * i_nsets;
  thrust::reduce_by_key(
    seg_begin, seg_end, avg_begin, discard_it, d_total_distance.begin());

  // Calculate the probabilities as each total divided by all individual
  // distances. Transpose on write to produce final probability maps.
  auto probabilities_out = thrust::make_permutation_iterator(
    do_probability.begin(), detail::make_transpose_iterator(i_nsets, npoints));
  const auto row_total = thrust::make_permutation_iterator(
    d_total_distance.begin(), detail::make_row_iterator(i_nsets));
  thrust::transform(
    avg_begin, avg_end, row_total, probabilities_out, thrust::divides<real>());
}

}  // namespace probability

SPLIT_DEVICE_NAMESPACE_END

