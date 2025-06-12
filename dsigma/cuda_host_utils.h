#ifndef CUDA_HOST_UTILS_H
#define CUDA_HOST_UTILS_H

#include <vector>
#include <string> // For std::string, if needed for Healpix_cxx or table interactions

// Forward declarations if necessary, or include minimal headers.
// For Healpix_cxx, we might need something like:
// #include <healpix_map.h> // Or specific headers for pixel operations
// #include <arr.h> // For Healpix_cxx array types if used directly

// For nanoflann:
#include "nanoflann.hpp" // Assume this is in the include path

// Placeholder for a structure to represent points for KD-tree
template <typename T>
struct PointCloud {
    struct Point {
        T x, y, z;
    };
    std::vector<Point> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else return pts[idx].z;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

// Function declarations for C++ implementations
std::vector<long> convert_to_healpix_ids_cpp(
    const std::vector<double>& ra_deg,
    const std::vector<double>& dec_deg,
    int nside,
    const std::string& order);

PointCloud<double> build_healpix_xyz_celestial_cpp(
    const std::vector<long>& unique_pix_ids,
    int nside,
    const std::string& order);

std::vector<size_t> find_neighboring_pixel_indices_nanoflann_cpp(
    const PointCloud<double>& source_pixel_cloud,
    const double query_pt[3], // Single query point
    double search_radius_sq);

#endif // CUDA_HOST_UTILS_H
