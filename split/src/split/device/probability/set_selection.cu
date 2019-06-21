#include "split/device/probability/set_selection.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/segment_length.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include "split/device/detail/shrink_segments.cuh"
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
    return sqr(a.get<0>() + b.get<0>()) + 
           sqr(a.get<1>() + b.get<1>()) + 
           sqr(a.get<2>() + b.get<2>());
  }
};

struct VecAdd
{
  using vec3 = thrust::tuple<real, real, real>;
  __host__ __device__ vec3 operator()(const vec3& a, const vec3& b)
  {
    return thrust::make_tuple(
      a.get<0>() + b.get<0>(), a.get<1>() + b.get<1>(), a.get<2>() + b.get<2>());
  }
};

}  // namespace

SPLIT_API int set_selection(
  cusp::array2d<real, cusp::device_memory>::const_view di_albedo,
  cusp::array2d<real, cusp::device_memory>::const_view di_centroids,
  cusp::array1d<int, cusp::device_memory>::const_view di_set_labels,
  cusp::array1d<int, cusp::device_memory>::view dio_set_ids,
  const int i_nsets)
{
  // Get the total number of points in all sets
  const int ninsets = dio_set_ids.size();
  const auto discard_it = thrust::make_discard_iterator();

  // Iterate over the labels assigned to each id
  auto set_labels_begin = thrust::make_permutation_iterator(
    di_set_labels.begin(), dio_set_ids.begin());

  // Copy the labels from each set id
  cusp::array1d<int, cusp::device_memory> d_label_copy(ninsets);
  thrust::copy_n(set_labels_begin, ninsets, d_label_copy.begin());

  // Sort the ids and their labels to get contiguous segments
  thrust::sort_by_key(d_label_copy.begin(), d_label_copy.end(), dio_set_ids.begin());
  auto sorted_labels_begin = d_label_copy.begin();
  auto sorted_labels_end = d_label_copy.end();

  // Iterate over point albedos in the provided sets
  const auto albedo_begin =
    thrust::make_permutation_iterator(detail::zip_it(di_albedo.row(0).begin(),
                                                     di_albedo.row(1).begin(),
                                                     di_albedo.row(2).begin()),
                                      dio_set_ids.begin());
  const auto albedo_end = albedo_begin + ninsets;

  // Get each points distance to it's set mean
  const auto albedo_average = detail::zip_it(di_centroids.row(0).begin(),
                                             di_centroids.row(1).begin(),
                                             di_centroids.row(2).begin());
  const auto set_average = thrust::make_permutation_iterator(
    albedo_average, sorted_labels_begin);
  cusp::array1d<real, cusp::device_memory> d_distances(ninsets);
  thrust::transform(
    albedo_begin, albedo_end, set_average, d_distances.begin(), Norm2{});

  // Sort the labels and ids by set mean
  auto id_label_pair = 
    detail::zip_it(dio_set_ids.begin(), sorted_labels_begin);
  thrust::sort_by_key(d_distances.begin(), d_distances.end(), id_label_pair);
  // Sort again by set label to yield sorted segments
  thrust::stable_sort_by_key(
    sorted_labels_begin, sorted_labels_end, dio_set_ids.begin());

  cusp::array1d<int, cusp::device_memory> check(ninsets);
  auto e =thrust::unique_copy(sorted_labels_begin, sorted_labels_end, check.begin());
  int n = e - check.begin();
  printf("N:%d\n", n);
  cusp::print(check.subarray(0, n));

  // Pick the 10 id's from each set nearest to the set's mean 
  return detail::shrink_segments(
    sorted_labels_begin, sorted_labels_end, dio_set_ids.begin(), i_nsets, 10);
}

}  // namespace probability

SPLIT_DEVICE_NAMESPACE_END

