#include "split/device/probability/remove_set_outliers.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/matrix_functional.cuh"
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
struct CullSet
{
  int min_size;
  __host__ __device__ bool operator()(int c)
  {
    // Mark this point for removal if more than half it's nearest albedos
    // come from outside its own material set
    return c < min_size;
  }
};

}  // namespace

SPLIT_API int remove_set_outliers(
  cusp::array2d<real, cusp::device_memory>::const_view di_albedo,
  cusp::array1d<int, cusp::device_memory>::const_view di_set_labels,
  cusp::array1d<int, cusp::device_memory>::view dio_set_ids,
  thrust::device_ptr<void> dio_temp)
{
  // Get the total number of points in all sets
  const int ninsets = dio_set_ids.size();
  auto set_labels_begin = thrust::make_permutation_iterator(
    di_set_labels.begin(), dio_set_ids.begin());
  // Iterate over point albedos in the provided sets
  const auto albedo_begin =
    thrust::make_permutation_iterator(detail::zip_it(di_albedo.row(0).begin(),
                                                     di_albedo.row(1).begin(),
                                                     di_albedo.row(2).begin()),
                                      dio_set_ids.begin());
  const auto albedo_end = albedo_begin + ninsets;

  // A buffer to store the distance from an albedo to all other albedos
  cusp::array1d<real, cusp::device_memory> d_distances(ninsets);
  auto distances_begin = d_distances.begin();
  auto distances_end = d_distances.end();

  // A buffer to store a copy of the set labels, useful for sorting
  cusp::array1d<int, cusp::device_memory> d_set_label_copy(ninsets);
  auto label_buff_begin = d_set_label_copy.begin();

  // Store the 10 nearest albedo labels for each point
  const int group_size = 10;
  cusp::array1d<int, cusp::device_memory> d_nearest(ninsets * group_size);

  // Functors for the loop
  const SquaredDistance vec_distance;

  for (int i = 0; i < ninsets; ++i)
  {
    // Re-initialize the set labels
    thrust::copy_n(set_labels_begin, ninsets, label_buff_begin);
    // Get a fixed iterator that points to the current albedo
    const auto current_albedo = thrust::make_permutation_iterator(
      albedo_begin, thrust::make_constant_iterator(i));
    // Get albedo distance from the current albedo to all other albedos in all
    // material sets
    thrust::transform(
      albedo_begin, albedo_end, current_albedo, distances_begin, vec_distance);
    // Sort the copied labels by distance to our current albedo
    thrust::sort_by_key(distances_begin, distances_end, label_buff_begin);
    // First 10, but skip the distance to self which is obviously 0
    auto label_eq_begin = thrust::make_transform_iterator(
      label_buff_begin, detail::unary_equal<int>(set_labels_begin[i]));
    // Copy a 1 for each label in the group which matches our label
    thrust::copy_n(
      label_eq_begin + 1, group_size, d_nearest.begin() + group_size * i);
  }

  // Mark boundaries for the reduce
  const auto seg_begin = detail::make_row_iterator(group_size);
  const auto seg_end = seg_begin + d_nearest.size();
  // Will store a 1 at indices of points that should be removed
  cusp::array1d<short, cusp::device_memory> d_removal_markers(ninsets);
  auto removal_out = thrust::make_transform_output_iterator(
    d_removal_markers.begin(), CullSet{group_size / 2});
  // Calculate how many were in the same set, and whether we need to remove
  const auto discard_it = thrust::make_discard_iterator();
  thrust::reduce_by_key(
    seg_begin, seg_end, d_nearest.begin(), discard_it, removal_out);

  // Remove the marked points and get an iterator to the new end of the list
  const auto marker_begin = d_removal_markers.begin();
  const auto new_end = thrust::remove_if(dio_set_ids.begin(),
                                         dio_set_ids.end(),
                                         marker_begin,
                                         detail::constant<bool>(true));
  // Return the number of remaining points
  return dio_set_ids.begin() - new_end;
}

}  // namespace probability

SPLIT_DEVICE_NAMESPACE_END

