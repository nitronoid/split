#include "split/device/probability/set_probability.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/segment_length.cuh"
#include <thrust/iterator/transform_output_iterator.h>
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace probability
{
namespace
{
struct SquaredDistance
{
  using vec3 = thrust::tuple<real, real, real>;
  __host__ __device__ real operator()(const vec3& a, const vec3& b)
  {
    return (a.get<0>() * a.get<0>()) + (a.get<1>() * a.get<1>()) +
           (a.get<2>() * a.get<2>());
  }
};

template <typename KeyBegin, typename KeyEnd, typename ValueBegin>
int shrink_segments(KeyBegin&& key_begin,
                    KeyEnd&& key_end,
                    ValueBegin&& value_begin,
                    const int max_length)
{
  const int n_data = key_end - key_begin;
  const int n_lengths = key_end[-1] + 1;
  // Find the length of each segment
  cusp::array1d<int, cusp::device_memory> segment_lengths(n_lengths + 1);
  detail::segment_length(key_begin, key_end, segment_lengths.begin());
  // Calculate the number of elements to be removed per segment
  cusp::array1d<int, cusp::device_memory> segment_removal(n_lengths);
  thrust::transform(segment_lengths.begin(),
                    segment_lengths.end(),
                    segment_removal.begin(),
                    detail::unary_minus<int>(max_length));
  // Scan the lengths to get indices
  thrust::exclusive_scan(segment_lengths.begin(),
                         segment_lengths.end() - 1,
                         segment_lengths.begin());
  // Reset the keys
  thrust::fill(key_begin, key_end, 0);
  // Write ones in the scanned positions
  const auto one = thrust::make_constant_iterator(1);
  const auto scatter_keys =
    thrust::make_permutation_iterator(key_begin, segment_lengths.begin());
  thrust::copy_n(one, n_lengths, scatter_keys);
  // Write ones in the new end positions
  using ivec2 = thrust::tuple<int, int>;
  const auto scatter_ends = thrust::make_permutation_iterator(
    key_begin,
    thrust::make_transform_iterator(
      detail::zip_it(segment_removal.begin(), segment_lengths.begin() + 1),
      [] __host__ __device__(ivec2 seg_pair) {
        return seg_pair.get<1>() - max(0, seg_pair.get<0>());
      }));
  thrust::transform(
    one, one + n_lengths, scatter_keys, scatter_ends, thrust::plus<int>());
  // Scan the markers to get even or odd keys
  thrust::inclusive_scan(key_begin, key_end, key_begin);
  // Remove all even keys
  auto key_value_begin = detail::zip_it(key_begin, value_begin);
  auto key_value_end = key_value_begin + n_data;
  auto new_end =
    thrust::remove_if(key_value_begin,
                      key_value_end,
                      [] __host__ __device__(ivec2 key_value_pair) {
                        return key_value_pair.get<0>() & 1;
                      });

  return new_end - key_value_begin;
}

}  // namespace

SPLIT_API void set_probability(
  cusp::array2d<real, cusp::device_memory>::const_view di_albedo,
  cusp::array1d<int, cusp::device_memory>::const_view di_set_labels,
  cusp::array1d<int, cusp::device_memory>::const_view dio_set_ids,
  cusp::array1d<real, cusp::device_memory>::view do_probability)
{
  // Get the total number of points, sets and points currently in sets
  const int npoints = di_albedo.num_cols;
  const int ninsets = dio_set_ids.size();
  auto set_labels_begin = thrust::make_permutation_iterator(
    di_set_labels.begin(), dio_set_ids.begin());
  auto set_labels_end = set_labels_begin + ninsets;
  const int nsets = *thrust::max_element(set_labels_begin, set_labels_end);

  // Iterate over point albedos in the provided sets
  const auto albedo_begin = detail::zip_it(di_albedo.row(0).begin(),
                                           di_albedo.row(1).begin(),
                                           di_albedo.row(2).begin());
  const auto set_albedo_begin =
    thrust::make_permutation_iterator(albedo_begin, dio_set_ids.begin());
  const auto set_albedo_end = set_albedo_begin + ninsets;

  // A buffer to store the distance from a points albedo to all set albedos
  cusp::array1d<real, cusp::device_memory> d_distances(ninsets);
  auto distances_begin = d_distances.begin();
  auto distances_end = d_distances.end();

  // A buffer to store the average distance from a point to a set
  cusp::array1d<real, cusp::device_memory> avg_distance(npoints * nsets);

  // A buffer to store a copy of the set labels, useful for sorting
  cusp::array1d<int, cusp::device_memory> d_set_label_copy(ninsets);
  auto label_buff_begin = d_set_label_copy.begin();
  auto label_buff_end = d_set_label_copy.end();

  // Functors for the loop
  const SquaredDistance vec_distance;

  const auto discard_it = thrust::make_discard_iterator();
  const int group_size = 10;
  for (int i = 0; i < npoints; ++i)
  {
    // Re-initialize the set labels
    thrust::copy_n(set_labels_begin, ninsets, label_buff_begin);
    // Get a fixed iterator that points to the current albedo
    const auto current_albedo = thrust::make_permutation_iterator(
      albedo_begin, thrust::make_constant_iterator(i));
    // Get albedo distance from the current albedo to all other albedos in all
    // material sets
    thrust::transform(set_albedo_begin,
                      set_albedo_end,
                      current_albedo,
                      distances_begin,
                      vec_distance);
    // Sort the copied labels by distance to our current albedo
    thrust::sort_by_key(distances_begin, distances_end, label_buff_begin);
    // Stable sort distances to get sorted segments of distances to each set
    thrust::stable_sort_by_key(
      label_buff_begin, label_buff_end, distances_begin);
    // Now keep only the 10 smallest distances per set
    const int new_len = shrink_segments(
      label_buff_begin, label_buff_end, distances_begin, group_size);
    // Get the average distance to each set using the remaining 10
    auto avg_distance_out = thrust::make_transform_output_iterator(
      avg_distance.begin() + i * nsets,
      detail::unary_divides<real>(1.f / group_size));

    thrust::reduce_by_key(label_buff_begin,
                          label_buff_begin + new_len,
                          distances_begin,
                          discard_it,
                          avg_distance_out);
  }

  // Reduce by row to get the total distance from each point to each set
  cusp::array1d<real, cusp::device_memory> d_total_distance(npoints);
  const auto seg_begin = detail::make_row_iterator(nsets);
  const auto seg_end = seg_begin + npoints * nsets;
  thrust::reduce_by_key(
    seg_begin, seg_end, distances_begin, discard_it, d_total_distance.begin());

  // Calculate the probabilities as each total divided by all individual
  // distances - 1. Transpose on write to produce final probability maps.
  auto probabilities_out = thrust::make_permutation_iterator(
    thrust::make_transform_output_iterator(do_probability.begin(),
                                           detail::unary_minus<real>(1.f)),
    detail::make_transpose_iterator(npoints, nsets));
  const auto row_total = thrust::make_permutation_iterator(
    d_total_distance.begin(), detail::make_row_iterator(nsets));
  thrust::transform(row_total,
                    row_total + npoints * nsets,
                    avg_distance.begin(),
                    probabilities_out,
                    thrust::divides<real>());
}

}  // namespace probability

SPLIT_DEVICE_NAMESPACE_END

