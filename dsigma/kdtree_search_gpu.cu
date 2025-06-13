#include "kdtree_search_gpu.h"
#include "dsigma/healpix_gpu.h" // For dot_gpu (though not strictly needed for this version)
#include <cmath> // For fabsf

// Helper function to calculate squared distance between two float3 points
__device__ inline float distance_sq(const float3& p1, const float3& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    float dz = p1.z - p2.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ void cukd_radius_search_with_callback(
    const float3& query_point,
    float search_radius_sq,
    const float3* tree_points,
    int num_tree_points,
    int current_node_idx,
    int depth,
    process_neighbor_callback_t callback,
    void* callback_user_data) {

    // Base Case: If current_node_idx is out of bounds, return.
    if (current_node_idx < 0 || current_node_idx >= num_tree_points) {
        return;
    }

    // Get Current Point
    const float3& node_point = tree_points[current_node_idx];

    // Distance Check: Calculate squared distance between query_point and node_point.
    // If less than or equal to search_radius_sq, call the callback.
    if (distance_sq(query_point, node_point) <= search_radius_sq) {
        callback(current_node_idx, callback_user_data);
    }

    // Determine Splitting Dimension
    int axis = depth % 3; // 0 for x, 1 for y, 2 for z

    // Calculate Difference Along Axis
    float delta_axis;
    float node_coord_axis;
    float query_coord_axis;

    if (axis == 0) { // X-axis
        node_coord_axis = node_point.x;
        query_coord_axis = query_point.x;
    } else if (axis == 1) { // Y-axis
        node_coord_axis = node_point.y;
        query_coord_axis = query_point.y;
    } else { // Z-axis (axis == 2)
        node_coord_axis = node_point.z;
        query_coord_axis = query_point.z;
    }
    delta_axis = query_coord_axis - node_coord_axis;

    // Determine child indices (assuming complete binary tree layout)
    // Left child: 2 * i + 1
    // Right child: 2 * i + 2
    int near_child_idx;
    int far_child_idx;

    if (delta_axis < 0) {
        // Query point is on the "left" side of the splitting plane
        near_child_idx = 2 * current_node_idx + 1; // Left child
        far_child_idx = 2 * current_node_idx + 2;  // Right child
    } else {
        // Query point is on the "right" side of the splitting plane (or on the plane)
        near_child_idx = 2 * current_node_idx + 2; // Right child
        far_child_idx = 2 * current_node_idx + 1;  // Left child
    }

    // Recursively call for the "near" child
    cukd_radius_search_with_callback(
        query_point,
        search_radius_sq,
        tree_points,
        num_tree_points,
        near_child_idx,
        depth + 1,
        callback,
        callback_user_data);

    // Pruning: If the splitting plane is within the search sphere's projection on the axis,
    // then also recursively call for the "far" child.
    // The squared distance from the query point to the splitting plane is delta_axis * delta_axis.
    // If this distance is less than or equal to search_radius_sq, the sphere intersects the plane,
    // so the other subtree ("far" child) might contain neighbors.
    if (delta_axis * delta_axis <= search_radius_sq) {
        cukd_radius_search_with_callback(
            query_point,
            search_radius_sq,
            tree_points,
            num_tree_points,
            far_child_idx,
            depth + 1,
            callback,
            callback_user_data);
    }
}
