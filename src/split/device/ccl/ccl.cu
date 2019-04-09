#include "split/device/ccl/ccl.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace ccl
{

namespace
{
__device__ __forceinline__ segvalue()
{
}

__global__ d_connected_components(const real* __restrict__ di_cluster_labels, int* __restrict__ do_segement_labels)
{
  // The shared memory block
  extern __shared__ uint8_t shared_memory[];
  // One int for every thread in the block, plus the surrounding 8-neighbourhood,
  // storing their original cluster labels
  int* __restrict__ cluster_label = reinterpret_cast<int*>(shared_memory);
  // One int for every thread in the block, representing it's new segment
  //int* __restrict__ segment_label =
  //  reinterpret_cast<int*>(cluster_label + (blockDim.x + 2) * (blockDim.y + 2));

  // Global thread index
  const int tid = 
    (threadIdx.x + threadIdx.y * blockDim.x) + (blockIdx.x + blockIdx.y * gridDim.x);

  // Guard against out of bounds memory reads
  if (tid >= i_width * i_height) return;

  // Add padding to the block dimensions to get our shared memory dimensions
  const int16_t mem_width = blockDim.x + 2;
  const int16_t mem_height = blockDim.y + 2;
  // Local thread index is offset by the 8-neighbourhood
  const int16_t mem_idx = (threadIdx.x + 1) + (threadIdx.y + 1) * mem_width;

  // Read the cluster labels into shared memory
  cluster_label[mem_idx] = di_cluster_labels[tid];
  // Initial values for the segment labels, are just indices
  segement_label[mem_idx] = di_cluster_labels[tid] + 1;

  const auto read_boundaries = [=]{
  // Read the 8-neighbourhood into shared memory
  if (threadIdx.y <= 1)
  {
    const int16_t local_tid = threadIdx.x + threadIdx.y * blockDim.x;
    // If we're in the second row of threads,
    // we need to read from the opposite side of the block
    const int8_t is_second = threadIdx.y;

    // First we read the top and bottom rows
    // If we're in the first row, we need to subtract the width of the 
    // block from our index. If we're in the second, we'll 
    // add the width times the height of our block, minus one row.
    const int16_t mem_row_idx =
      mem_idx + mem_width * (-1 + blockDim.y * is_second);
    const int32_t global_row_idx = 
      tid + i_width * (-1 + blockDim.y * is_second);

    // Copy across
    cluster_label[mem_row_idx] = di_cluster_labels[global_row_idx];

    // Next we read the left and right columns
    // We take the current thread index and add the width of the block times
    // our local thread index. We subtract our memory x coordinate from that.
    // Then if we're in the second row, we need to offset by the width of the 
    // thread block.
    const int16_t mem_col_idx =
      mem_idx + threadIdx.x * mem_width - mem_x + is_second * (mem_width - 1);
    const int32_t global_col_idx = 
      tid + threadIdx.x * i_width - mem_x + is_second * (mem_width - 1);

    // Copy across
    cluster_label[mem_col_idx] = di_cluster_labels[global_col_idx];

    // Copy the corners across
    if (threadIdx.x <= 1)
    {
      const int16_t mem_corner_idx = 
        (mem_width - 1) * threadIdx.x + (mem_height - 1) * is_second * i_width;
      const int32_t global_corner_idx = 
        tid + i_width * (-1 + blockDim.y * is_second) + (-1 + blockDim.x * threadIdx.x);

      cluster_label[mem_corner_idx] = di_cluster_labels[global_corner_idx];
    }
  }};

  const auto write_boundaries = [=]{
  };

  read_boundaries();

  // each thread can access it's 8 neighbourhood in shared memory 
  const auto segmax = [a] (int16_t a, int16_t b)
  {
    return max(segment_label[a], segment_label[b] * (cluster_label[a] == cluster_label[b]));
  };

  const int16_t up = (threadIdx.x + 1) + threadIdx.y * blockDim.x;
  const int16_t down = (threadIdx.x + 1) + (threadIdx.y + 2) * blockDim.x;
  while (!converged)
  {
    // Check row neighbours
    segment_label[mem_idx] = segmax(segmax(mem_idx, mem_idx - 1), mem_idx + 1);
    // Check col neighbours, no need to check diags, as the two passes cover it
    segment_label[mem_idx] = segmax(segmax(mem_idx, up), down);

    // At the end of each iter, we must update the boundaries, so that we can
    // propagate our results across blocks
    write_boundaries();
    read_boundaries();
  }
  
  


}
}

SPLIT_API void
connected_components(cusp::array2d<real, cusp::device_memory>::const_view di_points,
                     cusp::array1d<real, cusp::device_memory>::view do_labels)
{

}

}

SPLIT_DEVICE_NAMESPACE_END

