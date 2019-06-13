#if !defined(SPLIT_DEVICE_INCLUDED_DETAIL_SORTED_SEGMENT_ADJACENCY_EDGES)
#define SPLIT_DEVICE_INCLUDED_DETAIL_SORTED_SEGMENT_ADJACENCY_EDGES

#include "split/detail/internal.h"
#include "split/device/detail/zip_it.cuh"

SPLIT_DEVICE_NAMESPACE_BEGIN

namespace detail
{
/***
   @brief Builds sorts the given segment edges, using their labels, so that
   edges with the same source and target segment are neighboring.
   ***/
template <typename EdgeIterator,
          typename LabelIterator,
          typename AdjacencyIterator>
void sorted_segment_adjacency_edges(EdgeIterator&& edge_begin,
                                    EdgeIterator&& edge_end,
                                    LabelIterator&& label_begin,
                                    AdjacencyIterator&& adjacency_begin)
{
  const int nedges = (edge_end - edge_begin) / 2;
  // Copy the edge labels into our adjacency lists
  thrust::gather(edge_begin, edge_end, label_begin, adjacency_begin);
  // Sort by adjacency target and then stable sort by adjacency source so our
  // edges same source and target are neighboring
  thrust::sort_by_key(
    adjacency_begin + nedges,
    adjacency_begin + nedges * 2,
    detail::zip_it(adjacency_begin, edge_begin, edge_begin + nedges));
  thrust::stable_sort_by_key(
    adjacency_begin,
    adjacency_begin + nedges,
    detail::zip_it(adjacency_begin + nedges, edge_begin, edge_begin + nedges));
}

}  // namespace detail

SPLIT_DEVICE_NAMESPACE_END

#endif  // SPLIT_DEVICE_INCLUDED_DETAIL_SORTED_SEGMENT_ADJACENCY_EDGES

