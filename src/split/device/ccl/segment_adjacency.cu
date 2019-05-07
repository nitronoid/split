#include "split/device/ccl/point_point_adjacency.cuh"
#include <cusp/graph/connected_components.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
namespace
{
struct alignas(32) int8
{
  int8() = default;
  int8(int32_t i)
    : x(i)
    , y(i)
    , z(i)
    , w(i)
    , a(i)
    , b(i)
    , c(i)
    , d(i)
  {}

  int8(int32_t x, int32_t y, int32_t z, int32_t w, int32_t a, int32_t b, int32_t c, int32_t d)
    : x(x)
    , y(y)
    , z(z)
    , w(w)
    , a(a)
    , b(b)
    , c(c)
    , d(d)
  {}

  int32_t x, y, z, w, a, b, c, d;
};
}


SPLIT_API int segment_adjacency_edges(
  cusp::array1d<int, cusp::device_memory>::const_view di_labels,
  cusp::array2d<int, cusp::device_memory>::const_view do_edges)
{

  {
  // Read the edges as packed 4 tuples
  auto edge8_ptr = thrust::device_pointer_cast(
      reinterpret_cast<int8*>(do_edges.values.begin().base().get()));

  auto edge8_begin = utils::zip_it(edge8_ptr,
                                   edge8_ptr + di_labels.size());
  auto edge8_end = edge_begin + di_labels.size() * 8;

  // Fill the first half of every edge with a simple sequence, 
  // and the second half with the 8 nieghbour hood
  thrust::tabulate(edge8_begin, edge8_end, [] __device__ (int i)
      {
        const int32_t x = i % width;
        const int32_t y = i / width;

        const bool is_left = x == 0;
        const bool is_right = x == width - 1;
        const bool is_upper = y == 0;
        const bool is_lower = y == height - 1;


        const int8 a(i >> 3);
        const int8 b(
          (!is_left && !is_upper) ? i - width - 1 : -1,
          (!is_upper) ? i - width + 0 : -1,
          (!is_right && !is_upper) ? i - width + 1 : -1,
          (!is_left) ? i - 1 : -1,
          (!is_right) ? i + 1 : -1,
          (!is_left && !is_lower) ? i + width - 1 : -1,
          (!is_lower) ? i + width + 0 : -1,
          (!is_right && !is_lower) ? i + width + 1 : -1,
          );
        return thrust::make_tuple(a, b);
      });
  }

  // Iterate only over the cluster boundary pixels
  auto edge_begin = utils::zip_it(di_edges.row(0).begin(),
                                  di_edges.row(1).begin());
  auto edge_end = edge_begin + di_labels.size() * 8;
  auto labels = di_labels.begin();
  // Remove any edges accessing out of bounds, or edges internal to a cluster
  auto new_end = thrust::remove_if(edge_begin, edge_end,
      [=] __device__ (const thrust::tuple<int, int>& pair)
      {
        return pair.get<1>() < 0 || labels[pair.get<0>()] == labels[pair.get<1>()];
      });

  return new_end - edge_begin;
}

SPLIT_API int segment_adjacency(
  cusp::array1d<int, cusp::device_memory>::const_view di_labels,
  cusp::array2d<int, cusp::device_memory>::const_view di_edges,
  cusp::array1d<int, cusp::device_memory>::view do_segment_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view do_segment_adjacency,
  cusp::array1d<int, cusp::device_memory>::view do_segment_valence,
  cusp::array1d<int, cusp::device_memory>::view do_segment_cumulative_valence)
{
  // Iterate only over the cluster boundary pixels
  auto edge_it = utils::zip_it(di_edges.row(0).begin(),
                               di_edges.row(1).begin());

  

}
}  // namespace ccl


SPLIT_DEVICE_NAMESPACE_END

