#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/detail/zip_it.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
namespace
{
struct alignas(32) int8
{
  int8() = default;
  int8(int32_t i) : x(i), y(i), z(i), w(i), a(i), b(i), c(i), d(i)
  {
  }

  int8(int x, int y, int z, int w, int a, int b, int c, int d)
    : x(std::move(x))
    , y(std::move(y))
    , z(std::move(z))
    , w(std::move(w))
    , a(std::move(a))
    , b(std::move(b))
    , c(std::move(c))
    , d(std::move(d))
  {
  }

  int x, y, z, w, a, b, c, d;
};

struct NeighbourKeys
{
  NeighbourKeys() = default;
  NeighbourKeys(int h, int w) : height(h), width(w)
  {
  }
  int32_t height;
  int32_t width;

  __host__ __device__ thrust::tuple<int8, int8> operator()(int i) const
  {
    const int32_t x = i % width;
    const int32_t y = i / width;

    const bool is_left = x == 0;
    const bool is_right = x == width - 1;
    const bool is_upper = y == 0;
    const bool is_lower = y == height - 1;

    const int8 a(i);
    const int8 b((!is_left && !is_upper) ? i - width - 1 : -1,
                 (!is_upper) ? i - width + 0 : -1,
                 (!is_right && !is_upper) ? i - width + 1 : -1,
                 (!is_left) ? i - 1 : -1,
                 (!is_right) ? i + 1 : -1,
                 (!is_left && !is_lower) ? i + width - 1 : -1,
                 (!is_lower) ? i + width + 0 : -1,
                 (!is_right && !is_lower) ? i + width + 1 : -1);
    return thrust::make_tuple(a, b);
  }
};

struct UniqueConnections
{
  UniqueConnections() = default;
  UniqueConnections(const int* labels) : labels(labels)
  {
  }
  const int* labels;

  using Tup2 = thrust::tuple<int, int>;
  __host__ __device__ bool operator()(const Tup2& lhs, const Tup2& rhs) const
  {
    return lhs.get<0>() == rhs.get<0>() &&
           labels[lhs.get<1>()] == labels[rhs.get<1>()];
  }
};

}  // namespace

SPLIT_API int segment_adjacency_edges(
  cusp::array2d<int, cusp::device_memory>::const_view di_labels,
  cusp::array1d<int, cusp::device_memory>::view do_edges)
{
  const int npoints = di_labels.num_entries;
  {
    // Read the edges as packed 8 tuples
    auto edge8_ptr = thrust::device_pointer_cast(
      reinterpret_cast<int8*>(do_edges.begin().base().get()));

    auto edge8_begin = detail::zip_it(edge8_ptr, edge8_ptr + npoints);
    auto edge8_end = edge8_begin + npoints;

    // Fill the first half of every edge with a simple sequence,
    // and the second half with the 8 neighborhood
    thrust::tabulate(edge8_begin,
                     edge8_end,
                     NeighbourKeys(di_labels.num_rows, di_labels.num_cols));
  }

  // Iterate only over the cluster boundary pixels
  auto edge_begin =
    detail::zip_it(do_edges.begin(), do_edges.begin() + do_edges.size() / 2);
  auto edge_end = edge_begin + npoints * 8;
  auto labels = di_labels.values.begin().base().get();
  // Remove any edges accessing out of bounds, or edges internal to a cluster
  auto new_end = thrust::remove_if(
    edge_begin, edge_end, [=] __device__(const thrust::tuple<int, int>& pair) {
      return pair.get<1>() < 0 ||
             labels[pair.get<0>()] == labels[pair.get<1>()];
    });

  // Remove any edges stemming from the same pixel, that end in a common segment
  //new_end = thrust::unique(edge_begin, new_end, UniqueConnections(labels));
  // Copy the targets of each edge, to the memory immediately after the source
  // of each edge.
  thrust::copy(edge_begin.get_iterator_tuple().get<1>(),
               new_end.get_iterator_tuple().get<1>(),
               new_end.get_iterator_tuple().get<0>());

  return new_end - edge_begin;
}

SPLIT_API int segment_adjacency(
  cusp::array1d<int, cusp::device_memory>::const_view di_labels,
  cusp::array2d<int, cusp::device_memory>::const_view di_edges,
  cusp::array1d<int, cusp::device_memory>::view do_segment_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view do_segment_adjacency)
{
  // Look-up the segment labels for each end of an edge
  auto edge_label_it = thrust::make_permutation_iterator(
    di_labels.begin(), di_edges.values.begin());
  // Iterate over the edge label pairs
  auto edge_begin =
    detail::zip_it(edge_label_it, edge_label_it + di_edges.num_cols);
  auto edge_end = edge_begin + di_edges.num_cols;
  // Iterate over pairs of adjacency keys and values
  auto adj_begin = detail::zip_it(do_segment_adjacency_keys.begin(),
                                  do_segment_adjacency.begin());
  auto adj_end = adj_begin + di_edges.num_cols;
  // Copy the edge labels into our adjacency lists
  thrust::copy(edge_begin, edge_end, adj_begin);
  // Sort by value and then stable sort by key so our duplicate edges are
  // neighboring
  thrust::sort_by_key(do_segment_adjacency.begin(),
                      do_segment_adjacency.end(),
                      do_segment_adjacency_keys.begin());
  thrust::stable_sort_by_key(do_segment_adjacency_keys.begin(),
                             do_segment_adjacency_keys.end(),
                             do_segment_adjacency.begin());
  // The result of unique edge labels, is the segment adjacency list
  auto new_end = thrust::unique(adj_begin, adj_end);

  return new_end - adj_begin;
}
}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

