#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/detail/zip_it.cuh"
#include "split/device/detail/unary_functional.cuh"
#include <cusp/print.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
__global__ void d_segment_edges(const int* __restrict__ di_labels,
                                const int32_t i_width,
                                const int32_t i_height,
                                int* __restrict__ do_segment_connections)
{
  // Shared memory for storing blocks of labels from global memory
  extern __shared__ int s_labels[];

  // We subtract one from the block dimensions, to allow the blocks to overlap
  const int32_t g_x = threadIdx.x + blockIdx.x * (blockDim.x - 1);
  const int32_t g_y = threadIdx.y + blockIdx.y * (blockDim.y - 1);
  // Calculate the label in the 2D grid we need to access
  const int32_t g_tid = g_x + g_y * i_width;
  // Calculate the address in shared memory that we need to write to
  const int16_t l_tid = threadIdx.x + threadIdx.y * blockDim.x;

  // Guard against accessing memory outside of the provided global memory
  if ((g_x < i_width) && (g_y < i_height))
  {
    // Gather the labels from global memory, and store them in our shared block
    s_labels[l_tid] = di_labels[g_tid];
  }
  __syncthreads();

  // A thread will be actively calculating adjacency, if it is not a part of the
  // East/South border of the block, and it is not past the East/South border of
  // the provided global memory. These will be taken care of by neighboring
  // blocks due to the overlap.
  if ((threadIdx.x == blockDim.x - 1) || (threadIdx.y == blockDim.y - 1) ||
      (g_x > i_width - 2) && (g_y > i_height - 2))
    return;

  // Put our label in a register as we'll need it a lot
  const int l_label = s_labels[l_tid];
  // Useful enum for more readable directional memory access
  enum DIRECTION { NW, N, NE, W, E, SW, S, SE };

  // We check for a boundary in 4 directions: NE, E, S, SE, and write the point
  // index back to global memory at the same time. Simultaneously we write out
  // the inverse of the segment edge.
  // The write pattern is:
  /***
     NW  N  NE    0 1 2
     W      E  -> 3 - 4
     SW  S  SE    5 6 7
     ***/

  // First we check the North East direction, we skip this if we're on the
  // North boundary of the block
  if (threadIdx.y && (l_label != s_labels[l_tid - blockDim.x + 1]))
  {
    const int32_t boundary = g_tid - i_width + 1;
    do_segment_connections[g_tid * 8 + NE] = boundary;
    do_segment_connections[boundary * 8 + SW] = g_tid;
  }
  // Check the East direction
  if (l_label != s_labels[l_tid + 1])
  {
    const int32_t boundary = g_tid + 1;
    do_segment_connections[g_tid * 8 + E] = boundary;
    do_segment_connections[boundary * 8 + W] = g_tid;
  }
  // Check the South direction
  if (l_label != s_labels[l_tid + blockDim.x])
  {
    const int32_t boundary = g_tid + i_width;
    do_segment_connections[g_tid * 8 + S] = boundary;
    do_segment_connections[boundary * 8 + N] = g_tid;
  }
  // Check the South East direction
  if (l_label != s_labels[l_tid + blockDim.x + 1])
  {
    const int32_t boundary = g_tid + i_width + 1;
    do_segment_connections[g_tid * 8 + SE] = boundary;
    do_segment_connections[boundary * 8 + NW] = g_tid;
  }
}

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

int overlapping_blocks(const int2 grid_dim, const int2 block_dim)
{
  // Add one to effectively apply a ceiling rather than floor to the calculation
  const int x_count = ((grid_dim.x - 1) / (block_dim.x - 1)) + 1;
  const int y_count = ((grid_dim.y - 1) / (block_dim.y - 1)) + 1;
  return x_count * y_count;
}

}  // namespace

SPLIT_API int segment_adjacency_edges(
  cusp::array2d<int, cusp::device_memory>::const_view di_labels,
  cusp::array1d<int, cusp::device_memory>::view do_edges)
{
#if 0
  const int npoints = di_labels.num_entries;
  // Handy iterators
  auto source_begin = do_edges.begin();
  auto target_begin = do_edges.begin() + npoints * 8;
  auto target_end = target_begin + npoints * 8;
  // Initialize the source labels for the segment edges
  thrust::tabulate(source_begin, target_begin, detail::unary_divides<int>(8));
  // Initialize the target labels with a sentinel value
  thrust::fill(target_begin, target_end, -1);
  printf("debug %d\n", __LINE__);

  // Raw pointer to all labels
  const int* label_ptr = di_labels.values.begin().base().get();
  // Raw pointer to the edge target labels
  int* connection_ptr = target_begin.base().get();

  const dim3 block_dim(32, 32);
  const int nblock_threads = block_dim.x * block_dim.y * block_dim.z;
  const int nblocks = overlapping_blocks(
    {di_labels.num_cols, di_labels.num_rows}, {block_dim.x, block_dim.y});
  const std::size_t nshared_mem = nblock_threads * sizeof(int);

  // Launch the kernel
  d_segment_edges<<<nblocks, block_dim, nshared_mem>>>(
    label_ptr, di_labels.num_cols, di_labels.num_rows, connection_ptr);
  // Wait for the kernel to complete
  cudaDeviceSynchronize();
  printf("debug %d\n", __LINE__);

  // Iterate only over the segment edge pairs
  auto edge_begin = detail::zip_it(source_begin, target_begin);
  auto edge_end = edge_begin + npoints * 8;
  // Remove any sentinel edges
  auto new_end = thrust::remove_if(
    edge_begin, edge_end, [] __device__(const thrust::tuple<int, int>& pair) {
      return pair.get<1>() == -1;
    });
  printf("debug %d\n", __LINE__);
  cusp::print(do_edges.subarray(npoints * 8, 25));

  // Remove any edges stemming from the same pixel, that end in a common segment
  // new_end = thrust::unique(edge_begin, new_end, UniqueConnections(labels));
  // Copy the targets of each edge, to the memory immediately after the source
  // of each edge.
  thrust::copy(edge_begin.get_iterator_tuple().get<1>(),
               new_end.get_iterator_tuple().get<1>(),
               new_end.get_iterator_tuple().get<0>());
  printf("debug %d\n", __LINE__);

  return new_end - edge_begin;
#else

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

  // Copy the targets of each edge to the memory immediately after the source of
  // each edge.
  thrust::copy(edge_begin.get_iterator_tuple().get<1>(),
               new_end.get_iterator_tuple().get<1>(),
               new_end.get_iterator_tuple().get<0>());

  return new_end - edge_begin;
#endif
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

