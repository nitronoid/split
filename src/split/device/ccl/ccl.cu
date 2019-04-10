#include "split/device/ccl/ccl.cuh"
#include <cusp/graph/connected_components.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{
namespace
{
// Read the 8-neighborhood into shared memory
// FIXME: Clamp global reads so we don't go out of bounds, mostly done, need to
// fix cases where the image doesn't align with block sizes
__device__ void d_read_boundaries(const int* __restrict__ di_labels,
                                  int* __restrict__ do_shared_memory,
                                  const int i_mem_width,
                                  const int i_global_width)
{
  // Global thread index
  const int32_t tid = (threadIdx.x + threadIdx.y * blockDim.x) +
                      (blockIdx.x + blockIdx.y * gridDim.x);
  // Local thread index is offset by the 8-neighborhood
  const int16_t mem_idx = (threadIdx.x + 1) + (threadIdx.y + 1) * i_mem_width;

  // Conditionals
  const int8_t is_top = threadIdx.y == 0;
  const int8_t is_botm = threadIdx.y == blockDim.y - 1;
  const int8_t is_left = threadIdx.x == 0;
  const int8_t is_right = threadIdx.x == blockDim.x - 1;

  // Check if we would be accessing out of bounds memory, store these as guards
  const int8_t b_top = blockIdx.y != 0;
  const int8_t b_botm = blockIdx.y != gridDim.y - 1;
  const int8_t b_left = blockIdx.x != 0;
  const int8_t b_right = blockIdx.x != gridDim.x - 1;

  // Left = look behind by one
  // Right = look ahead by one
  // Up = look behind by width
  // Down = look ahead by width

  // Read upper neighbors
  if (is_top)
  {
    do_shared_memory[mem_idx - i_mem_width] =
      di_labels[tid - i_global_width * b_top];
  }
  // Read lower neighbors
  if (is_botm)
  {
    do_shared_memory[mem_idx + i_mem_width] =
      di_labels[tid + i_global_width * b_botm];
  }
  // Read left neighbors
  if (is_left)
  {
    do_shared_memory[mem_idx - 1] = di_labels[tid - b_left];
  }
  // Read right neighbors
  if (is_right)
  {
    do_shared_memory[mem_idx + 1] = di_labels[tid + b_right];
  }
  // Read corners
  if (is_top && is_left)
  {
    do_shared_memory[mem_idx - i_mem_width - 1] =
      di_labels[tid - i_global_width * b_top - b_left];
  }
  // Read corners
  if (is_top && is_right)
  {
    do_shared_memory[mem_idx - i_mem_width + 1] =
      di_labels[tid - i_global_width * b_top + b_left];
  }
  // Read corners
  if (is_botm && is_left)
  {
    do_shared_memory[mem_idx - i_mem_width - 1] =
      di_labels[tid + i_global_width * b_top - b_right];
  }
  // Read corners
  if (is_botm && is_right)
  {
    do_shared_memory[mem_idx - i_mem_width + 1] =
      di_labels[tid + i_global_width * b_top + b_right];
  }
}

__global__ void d_cluster_adjacency(const int* __restrict__ di_cluster_labels,
                                    int8_t* __restrict__ do_connected,
                                    const int i_width,
                                    const int i_height)
{
  // Global thread index
  const int32_t tid = (threadIdx.x + threadIdx.y * blockDim.x) +
                      (blockIdx.x + blockIdx.y * gridDim.x);

  // Guard against out of bounds memory reads
  if (tid >= i_width * i_height)
    return;

  // Add padding to the block dimensions to get our shared memory dimensions
  const int16_t mem_width = blockDim.x + 2;

  // The shared memory block
  extern __shared__ uint8_t shared_memory[];
  // Two ints for every thread in the block, plus the surrounding
  // 8-neighborhood, storing their original cluster labels
  int* __restrict__ cluster_label = reinterpret_cast<int*>(shared_memory);

  // Local thread index is offset by the 8-neighborhood
  const int16_t mem_idx = (threadIdx.x + 1) + (threadIdx.y + 1) * mem_width;
  // Read the cluster labels into shared memory
  cluster_label[mem_idx] = di_cluster_labels[tid];
  // Read the boundary data, to access to the 8-neighborhood from this block
  d_read_boundaries(di_cluster_labels, cluster_label, mem_width, i_width);
  // Ensure all shared memory writes have been completed
  __syncthreads();

  /***
    Stored in order NW,N,NE,W,E,SW,S,SE
    NW - N - NE
    |    |    |
    W  - O -  E
    |    |    |
    SW - S - SE
    ***/
  // Output the results using two vectorized writes
  *reinterpret_cast<char4*>(do_connected + tid * 8 + 0) = make_char4(
    /*NW*/ (cluster_label[mem_idx] == cluster_label[mem_idx - mem_width - 1]),
    /*N */ (cluster_label[mem_idx] == cluster_label[mem_idx - mem_width]),
    /*NE*/ (cluster_label[mem_idx] == cluster_label[mem_idx - mem_width + 1]),
    /*W */ (cluster_label[mem_idx] == cluster_label[mem_idx - 1]));
  *reinterpret_cast<char4*>(do_connected + tid * 8 + 4) = make_char4(
    /*E */ (cluster_label[mem_idx] == cluster_label[mem_idx + 1]),
    /*SW*/ (cluster_label[mem_idx] == cluster_label[mem_idx + mem_width - 1]),
    /*S */ (cluster_label[mem_idx] == cluster_label[mem_idx + mem_width]),
    /*SE*/ (cluster_label[mem_idx] == cluster_label[mem_idx + mem_width + 1]));
}
}  // namespace

SPLIT_API void connected_components(
  cusp::array2d<int, cusp::device_memory>::const_view di_cluster_labels,
  cusp::array1d<int, cusp::device_memory>::view do_labels)
{
  const int width = di_cluster_labels.num_cols;
  const int height = di_cluster_labels.num_rows;
  // Number of labels
  const int nlabels = width * height;

  // Matrix has max of 8 entries per label
  cusp::csr_matrix<int, int8_t, cusp::device_memory> d_adjacency(
    nlabels, nlabels, di_cluster_labels.num_entries * 8);

  dim3 grid_dim, block_dim;
  block_dim.x = 32;
  block_dim.y = 32;
  grid_dim.x = width / block_dim.x + 1;
  grid_dim.y = height / block_dim.y + 1;

  const int shared_memory_size =
    (block_dim.x + 2) * (block_dim.y + 2) * sizeof(int32_t);

  // Get adjacency information, 8-neighborhood minus non equal cluster labels
  d_cluster_adjacency<<<grid_dim, block_dim, shared_memory_size>>>(
    di_cluster_labels.values.begin().base().get(),
    d_adjacency.values.begin().base().get(),
    width,
    height);

  // Get our components
  cusp::graph::connected_components(d_adjacency, do_labels);
}

}  // namespace ccl

SPLIT_DEVICE_NAMESPACE_END

