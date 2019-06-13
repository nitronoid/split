#include "split/device/probability/remove_set_outliers.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"
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
    return (a.get<0>() * a.get<0>()) + 
           (a.get<1>() * a.get<1>()) + 
           (a.get<2>() * a.get<2>()); 
  }
};
}


SPLIT_API int remove_set_outliers(
  cusp::array2d<int, cusp::device_memory>::const_view di_albedo,
  cusp::array1d<int, cusp::device_memory>::view dio_set_ids,
  cusp::array1d<int, cusp::device_memory>::view dio_set_labels,
  thrust::device_ptr<void> dio_temp)
{
  // Get the total number of points in all sets
  const int ninsets = dio_set_ids.size();
  // Iterate over point albedos in the provided sets
  const auto albedo_begin = thrust::make_permutation_iterator(
    detail::zip_it(di_albedo.row(0).begin(),
                   di_albedo.row(1).begin(),
                   di_albedo.row(2).begin()), dio_set_ids.begin());
  const auto albedo_end = albedo_begin + ninsets;

  // A buffer to store the distance from an albedo to all other albedos
  cusp::array1d<real, cusp::device_memory> d_distances(ninsets);
  auto distances_begin = d_distances.begin();
  auto distances_end = d_distances.end();

  // A buffer to store a copy of the set labels, useful for sorting
  cusp::array1d<int, cusp::device_memory> d_set_label_copy(ninsets);
  auto label_buff_begin = d_set_label_copy.begin();

  // Removal marker
  std::vector<short> removal_markers(ninsets, 0);

  // Functors for the loop
  const SquaredDistance vec_distance;

  const int group_size = 10;
  for (int i = 0; i < ninsets; ++i)
  {
    // Re-initialize the set labels
    thrust::copy_n(dio_set_labels.begin(), ninsets, label_buff_begin);
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
      label_buff_begin, detail::unary_equal<int>(dio_set_labels[i]));
    const int nsimilar = thrust::reduce(
      label_eq_begin + 1, label_eq_begin + 1 + group_size);
    // Mark this point for removal if more than half it's nearest albedos come 
    // from outside its own material set
    removal_markers[i] = nsimilar < group_size / 2;
  }

  // Copy the removal markers to the device
  cusp::array1d<short, cusp::device_memory> d_removal_markers = removal_markers;
  const auto marker_begin = d_removal_markers.begin();
  // Iterate over pairs of the set labels and point id's
  auto set_begin = detail::zip_it(dio_set_ids.begin(), dio_set_labels.begin());
  auto set_end = set_begin + ninsets;
  // Remove the marked points and get an iterator to the new end of the list
  const auto new_end = thrust::remove_if(
    set_begin, set_end, marker_begin, detail::constant<bool>(true));
  // Return the number of remaining points
  return set_begin - new_end;
}

}  // namespace probability

SPLIT_DEVICE_NAMESPACE_END

