#include "split/device/kmeans/label.cuh"
#include "split/device/detail/matrix_functional.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/zip_it.cuh"
#include <cusp/functional.h>
#include <cusp/multiply.h>
#include <cusp/print.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
namespace
{
__host__ __device__ real sqr(real x)
{
  return x * x;
}

struct Norm2
{
  using vec3 = thrust::tuple<real, real, real>;
  __host__ __device__ real operator()(const vec3& source, const vec3& target)
  {
    return sqr(source.get<0>() - target.get<0>()) +
           sqr(source.get<1>() - target.get<1>()) +
           sqr(source.get<2>() - target.get<2>());
  }
};

// rolling my own multiply as CUSP was lazy and uses a serial impl
void compute_distances(
  cusp::array2d<real, cusp::device_memory>::const_view di_sources,
  cusp::array2d<real, cusp::device_memory>::const_view di_targets,
  cusp::array1d<real, cusp::device_memory>::view do_distances)
{
  assert(di_sources.num_rows == 3);
  assert(di_targets.num_rows == 3);

  const int32_t N = di_sources.num_cols;
  const int32_t M = di_targets.num_cols;

  assert(do_distances.size() == N * M);

  auto source_begin =
    thrust::make_permutation_iterator(detail::zip_it(di_sources.row(0).begin(),
                                                     di_sources.row(1).begin(),
                                                     di_sources.row(2).begin()),
                                      detail::make_row_iterator(M));

  auto target_begin =
    thrust::make_permutation_iterator(detail::zip_it(di_targets.row(0).begin(),
                                                     di_targets.row(1).begin(),
                                                     di_targets.row(2).begin()),
                                      detail::make_column_iterator(M));

  thrust::transform(source_begin,
                    source_begin + N * M,
                    target_begin,
                    do_distances.begin(),
                    Norm2());
}  // namespace
}  // namespace

SPLIT_API std::size_t label_points_workspace(const int i_npoints,
                                             const int i_nclusters)
{
  return i_npoints * i_nclusters * sizeof(real);
}

SPLIT_API void
label_points(cusp::array2d<real, cusp::device_memory>::const_view di_centroids,
             cusp::array2d<real, cusp::device_memory>::const_view di_points,
             cusp::array1d<int, cusp::device_memory>::view do_cluster_labels,
             thrust::device_ptr<void> do_temp)
{
  const int npoints = di_points.num_cols;
  const int nclusters = di_centroids.num_cols;
  const int ndistances = npoints * nclusters;

  // Offset the real part pointer by the size of the integer part
  auto d_rtemp = thrust::device_pointer_cast(static_cast<real*>(do_temp.get()));
  // Create a view over the real part
  auto d_distances = cusp::make_array1d_view(d_rtemp, d_rtemp + ndistances);

  compute_distances(di_points, di_centroids, d_distances);

  // Converts a 1D index into a row index
  auto row_indices = detail::make_row_iterator(nclusters);
  // Converts a 1D index into a column index
  auto col_indices = detail::make_column_iterator(nclusters);

  // Reduce each row, by finding the smallest distance contained, and writing
  // it's column index as the label
  thrust::reduce_by_key(
    row_indices,
    row_indices + ndistances,
    detail::zip_it(d_distances.begin(), col_indices),
    thrust::make_discard_iterator(),
    detail::zip_it(thrust::make_discard_iterator(), do_cluster_labels.begin()),
    thrust::equal_to<int>(),
    thrust::minimum<thrust::tuple<real, int>>());
}

}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

