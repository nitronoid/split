#include "split/device/kmeans/centroids.cuh"
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <future>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace kmeans
{
SPLIT_API void calculate_centroids(
  cusp::array1d<int, cusp::device_memory>::const_view di_cluster_labels,
  cusp::array2d<real, cusp::device_memory>::const_view di_points,
  cusp::array2d<real, cusp::device_memory, cusp::column_major>::view
    do_centroids,
  cusp::array1d<int, cusp::device_memory>::view do_temp)
{
  // Grab these sizes upfront with more readable names
  const int nlabels = di_cluster_labels.size();
  const int ndimensions = di_points.num_rows;
  // Reduce the verbosity of async calls
  const auto async = std::launch::async;

  // Iterator for input labels
  auto labels_in = di_cluster_labels.begin();
  // The first chunk of memory will be used to store labels
  auto labels_copy = do_temp.subarray(0, nlabels);
  auto labels_begin = labels_copy.begin(), labels_end = labels_copy.end();
  // Make a copy of the labels for sorting
  auto cpy_handle = std::async(
    async, [=] { thrust::copy(labels_in, labels_in + nlabels, labels_begin); });

  // The next chunk of temp storage will be used to store indices
  auto indices = do_temp.subarray(nlabels, nlabels);
  auto indices_begin = indices.begin(), indices_end = indices.end();
  // Initialize the indices as a sequence
  auto seq_handle =
    std::async(async, [=] { thrust::sequence(indices_begin, indices_end); });

  // Sort the indices by label
  auto sort_handle = std::async(async, [=, &cpy_handle, &seq_handle] {
    // Ensure our other tasks have completed
    cpy_handle.wait();
    seq_handle.wait();
    thrust::sort_by_key(labels_begin, labels_end, indices_begin);
  });

  // Leave the valence uninitialized for now
  auto valence = do_temp.subarray(nlabels * 2, do_centroids.num_rows);
  auto valence_begin = valence.begin(), valence_end = valence.end();
  // Create a counting iter to output the index values from the upper_bound
  thrust::counting_iterator<int> search_begin(0), search_end(valence.size());
  // Calculate a dense histogram to find the cumulative valence
  auto hist_handle = std::async(async, [=, &sort_handle] {
    // The sort must have completed before we can produce a histogram
    sort_handle.wait();
    // Find the indices where a segment boundary occurs
    thrust::upper_bound(
      labels_begin, labels_end, search_begin, search_end, valence_begin);
    // Calculate the non-cumulative valence by subtracting neighboring elements
    thrust::adjacent_difference(valence_begin, valence_end, valence_begin);
  });

  // Create the iterators we'll use, so the capture doesn't capture the arrays
  auto coeff_it = thrust::make_transform_iterator(
    valence_begin, [] __host__ __device__(int x) -> real { return 1.f / x; });
  auto discard_it = thrust::make_discard_iterator();
  auto multiply = thrust::multiplies<real>();

  // Wait for our valence to be ready
  hist_handle.wait();
  // FIXME: VLA's are not allowed in C++
  std::future<void> handles[ndimensions];
  // Launch a reduction and a transform for each dimension of our data
  // asynchronously. This divides the workload and is flexible for N dimensions.
  for (int i = 0; i < ndimensions; ++i)
  {
    // Get the reordered point iterator for this dimension
    auto point_it = thrust::make_permutation_iterator(di_points.row(i).begin(),
                                                      indices_begin);
    // Get the centroid iterators for this dimension
    auto centroid_begin = do_centroids.column(i).begin();
    auto centroid_end = do_centroids.column(i).end();

    // Grab a handle to this dimensions task
    handles[i] = std::async(async, [=] {
      // Reduce by label to find the totals of all points for each cluster
      thrust::reduce_by_key(
        labels_begin, labels_end, point_it, discard_it, centroid_it);
      // Divide through by the number of contributions to each new centroid,
      // but actually multiply by the reciprocal through a transform iterator.
      thrust::transform(
        centroid_end, centroid_begin, coeff_it, centroid_it, multiply);
    });
  }
  // Wait for all of our dimension tasks to complete
  std::for_each_n(handles, ndimensions, [](std::future<void>& h) { h.wait(); });
}
}  // namespace kmeans

SPLIT_DEVICE_NAMESPACE_END

