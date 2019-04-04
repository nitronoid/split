#if !defined(SPLIT_INCLUDED)

// clang-format off
#define SPLIT_VERSION_MAJOR      0
#define SPLIT_VERSION_MINOR      1
#define SPLIT_VERSION_PATCH      0
#define SPLIT_VERSION           10
#define SPLIT_VERSION_MESSAGE  "SPLIT: version 0.1.0"
#define SPLIT_INCLUDED SPLIT_VERSION

#define SPLIT_NAMESPACE split
#define SPLIT_NAMESPACE_BEGIN namespace SPLIT_NAMESPACE { 
#define SPLIT_NAMESPACE_END }

#define SPLIT_HOST_NAMESPACE_BEGIN SPLIT_NAMESPACE_BEGIN namespace host {
#define SPLIT_HOST_NAMESPACE_END }}

#define SPLIT_DEVICE_NAMESPACE_BEGIN SPLIT_NAMESPACE_BEGIN namespace device {
#define SPLIT_DEVICE_NAMESPACE_END }}

#if !defined(__host__)
#define __host__ 
#endif
#if !defined(__device__)
#define __device__ 
#endif

#define SPLIT_API __host__
#define SPLIT_DEVICE_ONLY_API __device__
#define SPLIT_SHARED_API SPLIT_API SPLIT_DEVICE_ONLY_API
// clang-format on

SPLIT_NAMESPACE_BEGIN

#if defined(SPLIT_USE_DOUBLE_PRECISION)
typedef double real;
#else
typedef float real;
#endif

SPLIT_NAMESPACE_END

#endif  // SPLIT_INCLUDED

