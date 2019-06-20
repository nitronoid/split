#include "split/device/intrinsic/estimate_albedo_intensity.cuh"
#include <cub/cub.cuh>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace
{
__constant__ real c_max_chroma[2];

template <typename T>
constexpr T vmin(T&& t)
{
  return std::forward<T>(t);
}

template <typename T0, typename T1, typename... Ts>
constexpr typename std::common_type<T0, T1, Ts...>::type
vmin(T0&& val1, T1&& val2, Ts&&... vs)
{
  return (val2 < val1) ? vmin(val2, std::forward<Ts>(vs)...)
                       : vmin(val1, std::forward<Ts>(vs)...);
}

__host__ __device__ int
n_contributions(int2 coord, int2 block_dim, int2 image_dim)
{
  const int x = vmin(coord.x + 1, block_dim.x, image_dim.x - coord.x);
  const int y = vmin(coord.y + 1, block_dim.y, image_dim.y - coord.y);
  return x * y;
}

__host__ __device__ int hash_chroma(real a, real b, float2 max, int nslots)
{
  const int x = (a / max.x) * (nslots - 1);
  const int y = (b / max.y) * (nslots - 1);
  return y * (nslots - 1) + x;
}

template <int BLOCK_DIM>
__global__ void
d_estimate_albedo_intensity(const real* __restrict__ di_intensity,
                            const real* __restrict__ di_albedo_intensity,
                            const real* __restrict__ di_chroma_a,
                            const real* __restrict__ di_chroma_b,
                            const int i_width,
                            const int i_height,
                            const int i_nslots,
                            real* __restrict__ do_albedo_intensity)
{
  // Allocate shared memory for the block reduce
  using CubReduce =
    cub::BlockReduce<real, BLOCK_DIM, cub::BLOCK_REDUCE_RAKING, BLOCK_DIM>;
  __shared__ typename CubReduce::TempStorage s_reduce_mem;
  // Shared memory for storing the intensity estimates
  __shared__ real s_shading_intensity_avg;
  extern __shared__ int s_mem[];
  int* s_contributions = s_mem;
  real* s_estimates = (real*)(s_contributions + i_nslots * i_nslots);
  // Global x and y, the blocks overlap and only offset threads by one
  // Calculate the pixel in the 2D grid we need to access
  const int32_t g_tid =
    (threadIdx.x + blockIdx.x) + (threadIdx.y + blockIdx.y) * i_width;

  // Init the estimates
  for (int i = threadIdx.x; i < i_nslots * i_nslots; i += blockDim.x)
  {
    s_estimates[i] = 0.f;
    s_contributions[i] = 0;
  }
  __syncthreads();
  // Hash our chroma values from global memory
  const int16_t chroma_hash = hash_chroma(di_chroma_a[g_tid],
                                          di_chroma_b[g_tid],
                                          {c_max_chroma[0], c_max_chroma[1]},
                                          i_nslots);
  // Load the intensity from global memory
  const real intensity = di_intensity[g_tid];
  // Add our intensity to the correct slot and count the contribution
  atomicAdd(s_estimates + chroma_hash, intensity);
  atomicAdd(s_contributions + chroma_hash, 1);
  // Divide the intensity by albedo intensity to get the shading intensity
  real shading_intensity = intensity / di_albedo_intensity[g_tid];
  // Reduce the shading intensities, to find the average across the block
  const real sum = CubReduce(s_reduce_mem).Sum(shading_intensity);
  if (threadIdx.x == 0 && threadIdx.y == 0)
    s_shading_intensity_avg = sum / (BLOCK_DIM * BLOCK_DIM);
  __syncthreads();
  // The resulting value is the average estimate times the average shading
  // intensity
  const real result = s_estimates[chroma_hash] /
                      (s_contributions[chroma_hash] * s_shading_intensity_avg);
  // Add it to the pixels value
  atomicAdd(do_albedo_intensity + g_tid, result);
}
}  // namespace

namespace intrinsic
{
/***
 ***/
SPLIT_API void estimate_albedo_intensity(
  cusp::array2d<real, cusp::device_memory>::const_view di_intensity,
  cusp::array2d<real, cusp::device_memory>::const_view di_chroma,
  cusp::array1d<real, cusp::device_memory>::view dio_albedo_intensity,
  const int i_nslots,
  const int i_niterations)
{
  const int npixels = di_intensity.num_entries;
  const int width = di_intensity.num_cols;
  const int height = di_intensity.num_rows;

  // Store the maximum chroma in constant memory
  const real max_c[2] = {
    *thrust::max_element(di_chroma.row(0).begin(), di_chroma.row(0).end()),
    *thrust::max_element(di_chroma.row(1).begin(), di_chroma.row(1).end())};
  cudaMemcpyToSymbol(c_max_chroma, max_c, sizeof(real) * 2);

  const int n_chroma = i_nslots * i_nslots;
  constexpr int scale = 16;
  const dim3 block_dim(scale, scale, 1);
  const dim3 nblocks{width - scale + 1, height - scale + 1, 1};
  const std::size_t nshared_mem =
    n_chroma * sizeof(real) + n_chroma * sizeof(int);

  cusp::array1d<real, cusp::device_memory> d_estimated_albedo_intensity(
    npixels);

  for (int i = 0; i < i_niterations; ++i)
  {
    // Reset the estimates
    thrust::fill_n(d_estimated_albedo_intensity.begin(), npixels, 0.f);

    // Estimate the albedo intensity per region, and copy back to pixels
    d_estimate_albedo_intensity<scale><<<nblocks, block_dim, nshared_mem>>>(
      di_intensity.values.begin().base().get(),
      dio_albedo_intensity.begin().base().get(),
      di_chroma.row(0).begin().base().get(),
      di_chroma.row(1).begin().base().get(),
      width,
      height,
      i_nslots,
      d_estimated_albedo_intensity.begin().base().get());
    cudaDeviceSynchronize();

    const auto ncontributions_begin = thrust::make_transform_iterator(
      thrust::make_counting_iterator(0), [=] __host__ __device__(int i) {
        const int2 coord = {i % width, i / width};
        return n_contributions(coord, {scale, scale}, {width, height});
      });
    // Average the estimates for each pixel
    thrust::transform(d_estimated_albedo_intensity.begin(),
                      d_estimated_albedo_intensity.end(),
                      ncontributions_begin,
                      dio_albedo_intensity.begin(),
                      thrust::divides<real>());
  }
}

}  // namespace intrinsic

SPLIT_DEVICE_NAMESPACE_END
