#include "split/device/kmeans/kmeans.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/ccl/connected_components.cuh"
#include "split/device/ccl/compress_labels.cuh"
#include "split/device/ccl/segment_adjacency.cuh"
#include "split/device/ccl/merge_small_segments.cuh"
#include "split/device/ccl/merge_smooth_boundaries.cuh"
#include "split/device/color/conversion.cuh"
#include "split/device/color/beta_feature.cuh"
#include "split/device/separation/luminance_continuity.cuh"
#include "split/device/separation/similar_reflectance.cuh"
#include "split/device/detail/view_util.cuh"
#include "split/device/detail/unary_functional.cuh"
#include "split/device/detail/cycle_iterator.cuh"
#include "split/host/stbi/stbi_raii.hpp"
#include "split/device/detail/cu_raii.cuh"

#include <cusp/print.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/array2d.h>
#include <cusp/monitor.h>
#include <cusp/krylov/gmres.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <chrono>

#define real split::real

template <typename T>
void BAD_call_destructor(T& io_obj)
{
  io_obj.~T();
}

static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
  switch (error)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

void sandbox()
{
  printf("\n\n\n\n\n SANDBOX\n\n");
  std::vector<int> labels = {0, 0, 0, 1, 1, 4, 5, 0, 0, 0, 1, 1, 4, 5, 0, 0, 0,
                             0, 1, 4, 4, 2, 2, 1, 1, 1, 4, 4, 2, 2, 1, 1, 1, 3,
                             3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3};

  const int npixels = labels.size();
  int nsegments = 5;

  cusp::array2d<int, cusp::device_memory> d_labels(7, 7);
  thrust::copy_n(labels.begin(), npixels, d_labels.values.begin());
  cusp::array1d<int, cusp::device_memory> d_connections(16 * npixels);

  cusp::array2d<real, cusp::device_memory> d_chroma(3, npixels);
  thrust::tabulate(d_chroma.values.begin(),
                   d_chroma.values.end(),
                   [=] __device__(int) { return 0.5f; });
  cusp::array2d<real, cusp::device_memory> d_rgb_image(3, npixels);
  thrust::tabulate(d_rgb_image.values.begin(),
                   d_rgb_image.values.end(),
                   [=] __device__(int i) {
                     int x = i % npixels;
                     int y = i / npixels;

                     return x * (0.5f / npixels) + y * (0.5f / 3.f);
                   });

  thrust::device_ptr<void> d_temp = thrust::device_malloc(
    split::device::ccl::merge_small_segments_workspace(npixels, 5));

  for (int i = 0; i < 1; ++i)
  {
    std::cout << "Merging small clusters\n";
    split::device::ccl::merge_small_segments(d_chroma, d_labels, d_temp);
    nsegments = split::device::ccl::compress_labels(d_labels.values, d_temp);
  }
  std::cout << "Post small merging\n";
  for (int i = 0; i < d_labels.num_rows; ++i)
  {
    for (int j = 0; j < d_labels.num_cols; ++j)
    {
      std::cout << d_labels(i, j) << ' ';
    }
    std::cout << '\n';
  }

  for (int i = 0; i < 1; ++i)
  {
    std::cout << "Merging smooth boundaries\n";
    split::device::ccl::merge_smooth_boundaries(
      d_rgb_image, nsegments, d_labels, d_temp);
    nsegments = split::device::ccl::compress_labels(d_labels.values, d_temp);
  }
  std::cout << "Post smooth merging\n";
  for (int i = 0; i < d_labels.num_rows; ++i)
  {
    for (int j = 0; j < d_labels.num_cols; ++j)
    {
      std::cout << d_labels(i, j) << ' ';
    }
    std::cout << '\n';
  }

  thrust::device_free(d_temp);
}

template <typename T>
void strided_copy(const T* i_src,
                  T* i_dest,
                  int src_stride,
                  int dest_stride,
                  int n,
                  cudaMemcpyKind i_kind)
{
  cudaMemcpy2D(i_dest,
               sizeof(T) * dest_stride,
               i_src,
               sizeof(T) * src_stride,
               sizeof(T),
               n,
               i_kind);
}

void make_device_image(gsl::not_null<const real*> h_image,
                       cusp::array2d<real, cusp::device_memory>::view d_image)
{
  const int npixels = d_image.num_cols;
  const int nchannels = d_image.num_rows;
  for (int c = 0; c < nchannels; ++c)
  {
    auto d_image_channel = d_image.values.begin().base().get() + npixels * c;
    const auto h_image_channel = h_image.get() + c;
    strided_copy(h_image_channel,
                 d_image_channel,
                 nchannels,
                 1,
                 npixels,
                 cudaMemcpyHostToDevice);
  }
}

void make_host_image(cusp::array2d<real, cusp::device_memory>::view d_image,
                     gsl::not_null<real*> h_image)
{
  const int npixels = d_image.num_cols;
  const int nchannels = d_image.num_rows;
  for (int c = 0; c < nchannels; ++c)
  {
    auto d_image_channel = d_image.values.begin().base().get() + npixels * c;
    const auto h_image_channel = h_image.get() + c;
    strided_copy(d_image_channel,
                 h_image_channel,
                 1,
                 nchannels,
                 npixels,
                 cudaMemcpyDeviceToHost);
  }
}

struct TempMemory
{
  TempMemory(std::size_t N) : m_ptr(thrust::device_malloc(N))
  {
  }

  ~TempMemory()
  {
    thrust::device_free(m_ptr);
  }

  thrust::device_ptr<void> get() const
  {
    return m_ptr;
  }

private:
  thrust::device_ptr<void> m_ptr;
};

/// @breif Performs a matrix multiplication with the self transpose A^t * A
cusp::array2d<real, cusp::device_memory>
cublas_AtA(cusp::array2d<real, cusp::device_memory, cusp::column_major>::const_view di_At)
{
  // Scalar coefficients for the matrix multiply
  const float alpha = 1.0f;
  const float beta = 0.0f;

  split::device::detail::cu_raii::blas::Handle blas_handle;
  std::cout<< _cudaGetErrorEnum(blas_handle.status) << '\n';

  // cublas expects our matrix to be column major, however we have it in row major.
  // This is essentially the transpose, so we swap the matrix dimensions. 
  const int At_m = di_At.num_rows;
  const int At_n = di_At.num_cols;

  printf("A^t is mxn: %dx%d.\n", At_m, At_n);

  // A^t * A, where A^t is MxN will yeild a square matrix of MxM
  cusp::array2d<real, cusp::device_memory> do_AtA(At_m, At_m);

  const real* At_ptr = di_At.values.begin().base().get();
  real* AtA_ptr = do_AtA.values.begin().base().get();

  // We don't transpose the first matrix as it's row major, and cublas expects
  // column major. This means it's reading it as if it was already transposed.
  // Performs: alpha * op(A) * op(B) + beta * C
  cublasSgemm(blas_handle,
              CUBLAS_OP_N, // Don't transpose
              CUBLAS_OP_T, // Do transpose
              At_m,        // Num rows of op(A)
              At_m,        // Num cols of op(B)
              At_n,        // Num cols of op(A) == rows of op(B)
              &alpha,      // Coefficient for op(A)*op(B)
              At_ptr,      // Pointer to matrix A
              At_m,        // Pass num rows of A, as the op doesn't transpose
              At_ptr,      // Pointer to matrix B
              At_m,        // Pass num cols of B, as the op does transpose
              &beta,       // Coefficient for C
              AtA_ptr,     // Pointer to matrix C
              At_m);       // Pass the num cols of A

  std::cout<< _cudaGetErrorEnum(blas_handle.status) << '\n';

  return do_AtA;
}

cusp::array1d<real, cusp::device_memory>
cublas_Atb(cusp::array2d<real, cusp::device_memory, cusp::column_major>::const_view di_At, cusp::array1d<real, cusp::device_memory>::const_view di_b)
{
  // Scalar coefficients for the matrix multiply
  const float alpha = 1.0f;
  const float beta = 0.0f;

  split::device::detail::cu_raii::blas::Handle blas_handle;
  std::cout<< _cudaGetErrorEnum(blas_handle.status) << '\n';

  // cublas expects our matrix to be column major, however we have it in row major.
  // This is essentially the transpose, so we swap the matrix dimensions. 
  const int At_m = di_At.num_rows;
  const int At_n = di_At.num_cols;

  printf("A^t is mxn: %dx%d.\n", At_m, At_n);

  // A^t * b, where A^t is MxN will yeild a square matrix of MxM
  cusp::array1d<real, cusp::device_memory> do_Atb(At_m);

  const real* At_ptr = di_At.values.begin().base().get();
  const real* b_ptr = di_b.begin().base().get();
  real* Atb_ptr = do_Atb.begin().base().get();

  // Performs: y = alpha * op(A) * x + beta * y
  cublasSgemv(blas_handle,
              CUBLAS_OP_N,
              At_m,
              At_n,
              &alpha,
              At_ptr,
              At_m,
              b_ptr,
              1,
              &beta,
              Atb_ptr,
              1);

  std::cout<< _cudaGetErrorEnum(blas_handle.status) << '\n';

  return do_Atb;
}

void solve_intrinsic_separation(
  cusp::array2d<real, cusp::device_memory>::const_view di_A,
  cusp::array1d<real, cusp::device_memory>::const_view di_b,
  cusp::array1d<real, cusp::device_memory>::view do_x
  )
{
  split::device::detail::cu_raii::solver::SolverDn cusolver_handle;
  cublasFillMode_t fill = CUBLAS_FILL_MODE_LOWER;

  // Copy A to L
  cusp::array2d<real, cusp::device_memory> L = di_A;
  const int A_m = di_A.num_rows;
  real* L_ptr = L.values.begin().base().get();
  int buffer_size;

  cusolverDnSpotrf_bufferSize(cusolver_handle, fill, A_m, L_ptr, A_m, &buffer_size);
  // Allocate the buffer
  thrust::device_vector<real> buffer(buffer_size);
  real* buffer_ptr = buffer.data().get();

  thrust::device_vector<int> v_info(1);
  int* info = v_info.data().get();
  // Cholesky factorization of A, into L^T * L
  cusolverDnSpotrf(
    cusolver_handle, fill, A_m, L_ptr, A_m, buffer_ptr, buffer_size, info);

  // Copy from b to x, potrs will compute x inplace
  thrust::copy(di_b.begin(), di_b.end(), do_x.begin());
  real* x_ptr = do_x.begin().base().get();

  // Solve the factored linear system (L^T * L) * X = b
  cusolverDnSpotrs(cusolver_handle, fill, A_m, 1, L_ptr, A_m, x_ptr, A_m, info); 

}

int main(int argc, char* argv[])
{
#if 1
  assert(argc == 2);
  auto h_image = split::host::stbi::loadf(argv[1], 3);
  printf("Loaded image with dim: %dx%dx%d\n",
         h_image.width(),
         h_image.height(),
         h_image.n_channels());

  cusp::array2d<real, cusp::device_memory> d_rgb_image(h_image.n_channels(),
                                                       h_image.n_pixels());
  cusp::array2d<real, cusp::device_memory> d_lab_image(h_image.n_channels(),
                                                       h_image.n_pixels());
  make_device_image(h_image.get(), d_rgb_image);

  // Convert the input linear RGB image into L*a*b color space
  split::device::color::convert_color_space(
    d_rgb_image, d_lab_image, split::device::color::rgb_to_lab());
  // Make a copy of the image luminance before writing the beta feature
  cusp::array1d<real, cusp::device_memory> d_luminance(h_image.n_pixels());
  thrust::copy(
    d_lab_image.row(0).begin(), d_lab_image.row(0).end(), d_luminance.begin());
  // Compute the beta feature from our lab image, and write it in place of L
  split::device::color::beta_feature(d_lab_image, d_lab_image.row(0));

  // Create initial means
  const int nclusters = 16;
  const int npixels = h_image.n_pixels();

  cusp::array2d<real, cusp::device_memory> d_centroids(h_image.n_channels(),
                                                       nclusters);
  split::device::kmeans::uniform_random_initialize(d_lab_image, d_centroids);

  cusp::array2d<int, cusp::device_memory> d_cluster_labels(h_image.height(),
                                                           h_image.width());
  cusp::array2d<int, cusp::device_memory> d_segment_labels(h_image.height(),
                                                           h_image.width());

  // Allocate temporary memory
  TempMemory d_temp(split::device::kmeans::cluster_workspace(
    npixels, nclusters, h_image.n_channels()));

  // K-means cluster the image
  split::device::kmeans::cluster(
    d_lab_image, d_centroids, d_cluster_labels.values, d_temp.get(), 100, 1e-3);

  // Obtain isolated segments from our initial clustering
  split::device::ccl::connected_components(
    d_cluster_labels, d_temp.get(), d_segment_labels.values, 100);
  // Compress the segment labels to produce a contiguous sequence
  int nsegments =
    split::device::ccl::compress_labels(d_segment_labels.values, d_temp.get());
  std::cout << "Segmented into " << nsegments << " connected components.\n";
  // Re-calculate the centroids using the segment labels
  cusp::array2d<real, cusp::device_memory> d_seg_centroids(h_image.n_channels(),
                                                           nsegments);

  // Re-calculate the centroids using the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_cluster_labels.values, d_rgb_image, d_centroids, d_temp.get());
  //---------------------------------------------------------------------------
  auto d_chrominance = split::device::detail::make_const_array2d_view(
    cusp::make_array2d_view(2,
                            npixels,
                            npixels,
                            d_lab_image.values.subarray(npixels, npixels * 2),
                            cusp::row_major{}));

  for (int i = 0; i < 10 && (nsegments > 5000 || i < 1); ++i)
  {
    TempMemory d_temp(
      split::device::ccl::merge_small_segments_workspace(npixels, nsegments));
    std::cout << "Merging small clusters\n";
    split::device::ccl::merge_small_segments(
      d_chrominance, d_segment_labels, d_temp.get(), 10 * (i + 1));
    nsegments = split::device::ccl::compress_labels(d_segment_labels.values,
                                                    d_temp.get());
  }
  printf("Number of segments post merge: %d\n", nsegments);

  for (int i = 0; i < 0; ++i)
  {
    TempMemory d_temp(split::device::ccl::merge_smooth_boundaries_workspace(
      npixels, nsegments, npixels * 8));
    std::cout << "Merging smooth boundaries\n";
    split::device::ccl::merge_smooth_boundaries(
      d_rgb_image, nsegments, d_segment_labels, d_temp.get(), 0.005f);
    nsegments = split::device::ccl::compress_labels(d_segment_labels.values,
                                                    d_temp.get());
  }

  //----------------------------------------------------------------------------
  // Build the linear system of intrinsic separation equations
  //----------------------------------------------------------------------------
  printf("Generating luminance continuity equations.\n");
  cusp::array2d<real, cusp::device_memory> lc_A;
  cusp::array1d<real, cusp::device_memory> lc_b;
  // Calculate the luminance continuity equations
  std::tie(lc_A, lc_b) = split::device::separation::luminance_continuity(
    d_segment_labels, d_luminance, d_temp.get());

  printf("Generating similar reflectance equations.\n");
  cusp::array2d<real, cusp::device_memory> sr_A;
  cusp::array1d<real, cusp::device_memory> sr_b;
  // Calculate the luminance continuity equations
  cusp::print(d_cluster_labels.values.subarray(0, 10));
  std::tie(sr_A, sr_b) =
    split::device::separation::similar_reflectance(d_cluster_labels.values,
                                                   d_segment_labels.values,
                                                   d_rgb_image,
                                                   d_luminance,
                                                   nclusters,
                                                   nsegments,
                                                   d_temp.get());
  cusp::print(d_cluster_labels.values.subarray(0, 10));

  // Combine the equations into one matrix
  printf("Combining equations.\n");
  const int nequations = lc_A.num_rows + sr_A.num_rows + 1;
  cusp::array2d<real, cusp::device_memory> A(nequations, nsegments);
  cusp::array1d<real, cusp::device_memory> b(nequations);
  // Copy across the two A matrices, and add the regularization equation
  thrust::copy_n(lc_A.values.begin(), lc_A.num_entries, A.values.begin());
  thrust::copy_n(sr_A.values.begin(), sr_A.num_entries, A.values.begin() + lc_A.num_entries);
  thrust::fill_n(A.values.begin() + nequations - 1, nsegments, 1.f);
  // Copy across the two b vectors, and add the regularization equation
  thrust::copy(lc_b.begin(), lc_b.end(), b.begin());
  thrust::copy(sr_b.begin(), sr_b.end(), b.begin() + lc_b.size());
  b.back() = 0;

  //BAD_call_destructor(d_temp);
  //BAD_call_destructor(lc_A);
  //BAD_call_destructor(sr_A);
  //BAD_call_destructor(lc_b);
  //BAD_call_destructor(sr_b);

  auto At_col_major = split::device::detail::make_const_array2d_view(
      cusp::make_array2d_view(
      A.num_cols,
      A.num_rows,
      A.num_rows,
      cusp::make_array1d_view(A.values),
      cusp::column_major{}));
  auto AtA = cublas_AtA(At_col_major);
  auto Atb = cublas_Atb(At_col_major, b);

  cusp::array1d<real, cusp::device_memory> x(nequations);
  solve_intrinsic_separation(AtA, Atb, x);

  cusp::print(x.subarray(0, 10));
  // Transform out of log space
  thrust::transform(
    x.begin(), x.end(), x.begin(), split::device::detail::unary_exp<real>());
  printf("X:\n");
  cusp::print(x.subarray(0, 10));

  // Convert to 3 dimensional
  cusp::array2d<real, cusp::device_memory> d_shading(3, nsegments);
  auto x_cycle = 
    split::device::detail::make_cycle_iterator(x.begin(), nsegments);
  thrust::copy_n(x_cycle, nsegments * 3, 
      thrust::make_transform_output_iterator(d_shading.values.begin(),
        split::device::detail::unary_multiplies<real>(0.01f)));

  // Sanity check
  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/sanity.png", h_image);

  split::device::kmeans::propagate_centroids(
    d_segment_labels.values, d_shading, d_rgb_image);

  auto c_lum =
    split::device::detail::make_cycle_iterator(d_luminance.begin(), npixels);
  thrust::transform(
      d_rgb_image.values.begin(), d_rgb_image.values.end(), c_lum, d_rgb_image.values.begin(),
      thrust::multiplies<real>());

  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/shading.png", h_image);

  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_cluster_labels.values, d_centroids, d_rgb_image);
  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/clusters.png", h_image);

  // Re-calculate the centroids using the centroids using the segment labels
  split::device::kmeans::calculate_centroids(
    d_segment_labels.values, d_rgb_image, d_seg_centroids, d_temp.get());
  // Copy the segment means to their member pixels
  split::device::kmeans::propagate_centroids(
    d_segment_labels.values, d_seg_centroids, d_rgb_image);

  make_host_image(d_rgb_image, h_image.get());
  split::host::stbi::writef("assets/images/merged.png", h_image);

  //------------------------------------------------------------------
#else

  sandbox();
#endif

  return 0;
}

