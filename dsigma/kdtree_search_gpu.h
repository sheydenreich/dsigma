#ifndef DSIGMA_KDTREE_SEARCH_GPU_H
#define DSIGMA_KDTREE_SEARCH_GPU_H

#include <cuda_runtime.h>
#include <vector_types.h> // For float3

/**
 * @brief Callback function pointer type for processing neighbors found in radius search.
 * @param point_kdtree_idx Index of the found neighbor point in the original array
 *                         that the KD-tree was built upon (after reordering by cudaKDTree).
 * @param user_data Arbitrary user data passed to the callback.
 */
typedef void (*process_neighbor_callback_t)(int point_kdtree_idx, void* user_data);

/**
 * @brief Performs a radius search on a KD-tree represented by a reordered point array.
 *
 * This function is a conceptual device-side recursive implementation.
 * Production KD-tree searches on GPU are typically iterative to manage stack usage.
 *
 * @param query_point The point around which to search.
 * @param search_radius_sq The squared radius of the search sphere.
 * @param tree_points Pointer to the array of 3D points, reordered by cudaKDTree
 *                    to represent the implicit KD-tree structure.
 * @param num_tree_points The total number of points in the tree_points array.
 * @param current_node_idx The index of the current node in the tree_points array
 *                         to visit (starts at 0 for the root).
 * @param depth The current depth in the KD-tree (starts at 0 for the root),
 *              used to determine the splitting axis.
 * @param callback A function pointer to be called for each point found within the radius.
 * @param callback_user_data User-defined data to be passed to the callback function.
 */
__device__ void cukd_radius_search_with_callback(
    const float3& query_point,
    float search_radius_sq,
    const float3* tree_points,
    int num_tree_points,
    int current_node_idx,
    int depth,
    process_neighbor_callback_t callback,
    void* callback_user_data);

#endif // DSIGMA_KDTREE_SEARCH_GPU_H
