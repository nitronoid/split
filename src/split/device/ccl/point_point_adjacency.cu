#include "split/device/ccl/point_point_adjacency.cuh"
#include <cusp/graph/connected_components.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
namespace
{
struct EightNeighborhood
{
  // 8*4=32, this could enable writing the 8-neighborhood in one instruction
  struct alignas(32) int8
  {
    int32_t x, y, z, w, a, b, c, d;
  };

  EightNeighborhood(int w, int h, const int* __restrict__ l)
    : width(w), height(h), label(l)
  {
  }

  int width;
  int height;
  const int* __restrict__ label;

  __device__ int8 operator()(int x) const
  {
    // Row and column coordinates as guards against out of bounds memory
    // reads
    const int R = x / width;
    const int C = x % width;
    // Get our label
    const int O = label[x];
    // Direction labels
    const int NW = x - width - 1;
    const int N = x - width;
    const int NE = x - width + 1;
    const int W = x - 1;
    const int E = x + 1;
    const int SW = x + width - 1;
    const int S = x + width;
    const int SE = x + width + 1;

    // Get neighbor label in all directions, vec packed
    // If we would read out of bounds, we instead write a -1 as an invalid
    const int8 NW_N_NE_W_E_SW_S_SE = {
      // NW corner must guard against reading up and left, out of bounds
      R && C && label[NW] == O ? NW : -1,
      // N must guard against reading up out of bounds
      R && label[N] == O ? N : -1,
      // NE corner must guard against reading up and right, out of bounds
      R && (C != width - 1) && label[NE] == O ? NE : -1,
      // W must guard against reading left, out of bounds
      C && label[W] == O ? W : -1,
      // E must guard against reading right, out of bounds
      (C != width - 1) && label[E] == O ? E : -1,
      // SW must guard against reading down and left, out of bounds
      (R != height - 1) && C && label[SW] == O ? SW : -1,
      // S must guard against reading down, out of bounds
      (R != height - 1) && label[S] == O ? S : -1,
      // SE must guard against reading down and right, out of bounds
      (R != height - 1) && (C != width - 1) && label[SE] == O ? SE : -1};
    // Write all through a single struct
    return NW_N_NE_W_E_SW_S_SE;
  }
};
}  // namespace

SPLIT_API int point_point_adjacency(
  cusp::array2d<int, cusp::device_memory>::const_view di_labels,
  cusp::array1d<int, cusp::device_memory>::view do_point_adjacency_keys,
  cusp::array1d<int, cusp::device_memory>::view do_point_adjacency)
{
  // Width and height dimensions
  const int width = di_labels.num_cols;
  const int height = di_labels.num_rows;
  const int npoints = di_labels.num_entries;

  using int8 = EightNeighborhood::int8;
  // Fill with 8-neighborhood grid indices:
  // 0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2...
  {
    thrust::device_ptr<int8> key_vec_begin{
      reinterpret_cast<int8*>(do_point_adjacency.begin().base().get())};
    auto key_vec_end = key_vec_begin + npoints;
    thrust::tabulate(key_vec_begin, key_vec_begin, [] __device__(int x) {
      const int v = x >> 3;
      return int8{v, v, v, v, v, v, v, v};
    });
  }

  // Write the adjacent pixels, from the 8-neighborhood. Two pixels are adjacent
  // if they share the same cluster label
  {
    // Interpret our data as packed int8s
    thrust::device_ptr<int8> adj_vec_begin{
      reinterpret_cast<int8*>(do_point_adjacency.begin().base().get())};
    auto adj_vec_end = adj_vec_begin + npoints;
    // pointer to the labels
    auto label_ptr = di_labels.values.begin().base().get();
    // Produce the 8-neighborhood
    thrust::tabulate(
      adj_vec_begin, adj_vec_end, EightNeighborhood(width, height, label_ptr));
  }

  auto adj_begin = thrust::make_zip_iterator(thrust::make_tuple(
    do_point_adjacency_keys.begin(), do_point_adjacency.begin()));
  auto adj_end = adj_begin + npoints * 8;
  // Get the new end
  adj_end = thrust::remove_if(
    adj_begin, adj_end, [] __device__(const thrust::tuple<int, int>& entry) {
      return entry.get<1>() < 0;
    });
  // Return the new size
  return adj_end - adj_begin;
}

}  // namespace ccl


SPLIT_DEVICE_NAMESPACE_END

