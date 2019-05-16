#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_FIX_MAP_CYCLES)
#define SPLIT_DEVICE_INCLUDED_DETAIL_FIX_MAP_CYCLES

#include "split/detail/internal.h"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
/// @brief A function for removing cycles within a mapping
template <typename MapIterator>
void fix_map_cycles(MapIterator&& map_begin,
                    MapIterator&& map_end)
{
  const int nsegments = map_end - map_begin;
  // Make a counting iterator
  const auto count = thrust::make_counting_iterator(0);
  // Iterate over the source to target mapping, with sizes
  auto map_pair_begin = detail::zip_it(count, map_begin);
  auto map_pair_end = map_pair_begin + nsegments;
  // Iterate over the targets of the targets
  auto target_target_begin =
    thrust::make_permutation_iterator(map_begin, map_begin);
  // Remove any loops in the map that would cause oscillations by selecting the
  // largest region of the loop
  thrust::transform(
    map_pair_begin,
    map_pair_end,
    target_target_begin,
    map_begin,
    [] __device__(const thrust::tuple<int, int>& map, int target_target) {
      // If our target is targeting us, then we have a cycle that needs to be
      // fixed, other wise the two will oscillate
      const bool is_cycle = map.get<0>() == target_target;
      // We select one of the two, based on whichever is currently the largest
      if (is_cycle && map.get<0>() > map.get<1>())
      {
        return map.get<0>();
      }
      return map.get<1>();
    });
}

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_FIX_MAP_CYCLES

