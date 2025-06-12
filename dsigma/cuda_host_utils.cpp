#include "cuda_host_utils.h"
#include <cmath> // For M_PI, sin, cos, acos, etc.
#include <vector>
#include <stdexcept> // For std::runtime_error
#include <iostream>  // For warnings or errors

// Define DEG2RAD here if it's not accessible from a common header for host code
#ifndef DEG2RAD_Host // Avoid redefinition if it were in a common host header
#define DEG2RAD_Host 0.017453292519943295
#endif

// Conditional include for Healpix_cxx and setup
#if __has_include(<healpix_cxx/healpix_base.h>) && __has_include(<healpix_cxx/pointing.h>)
    #include <healpix_cxx/healpix_base.h>
    #include <healpix_cxx/pointing.h>
    // Potentially needed for Healpix_Ordering enum if not in healpix_base.h
    #if __has_include(<healpix_cxx/healpix_orderings.h>)
      #include <healpix_cxx/healpix_orderings.h>
    #endif
    #ifndef HEALPIX_ORDERING_DEFINED
        // Older versions of Healpix_cxx might have RING/NESTED in Healpix_Base
        // If Healpix_Ordering is a specific type, this might need adjustment
        // For typical usage, RING and NESTED are enum values.
        #define HEALPIX_ORDERING_DEFINED
    #endif
    #define HEALPIX_FOUND 1
#elif __has_include(<chealpix.h>)
    // Basic C API might be available.
    // This example will focus on C++ API. If C API is the only one,
    // these functions would need a different implementation.
    #warning "Found <chealpix.h>, but C++ Healpix_cxx API is preferred. Stubs will be used if C++ API is not found."
    #include <chealpix.h>
    // Define HEALPIX_FOUND only if we intend to use C API as fallback.
    // For now, let's assume we need the C++ API for Healpix_Base and pointing.
    #ifndef HEALPIX_FOUND // Only if C++ API not found above
        #define HEALPIX_FOUND 0 // Or a specific flag for C API, e.g., HEALPIX_C_FOUND
        #warning "Healpix_cxx C++ headers not found. Healpix related functions will be dummies."
    #endif
#else
    #warning "Healpix_cxx headers not found. Healpix related functions will be dummies."
    #define HEALPIX_FOUND 0
#endif

// nanoflann is header only, already included in cuda_host_utils.h

std::vector<long> convert_to_healpix_ids_cpp(
    const std::vector<double>& ra_deg,
    const std::vector<double>& dec_deg,
    int nside,
    const std::string& order_str) {

#if HEALPIX_FOUND
    if (ra_deg.size() != dec_deg.size()) {
        throw std::runtime_error("RA and Dec vectors must have the same size.");
    }
    if (nside <= 0) {
        throw std::runtime_error("NSIDE must be positive.");
    }

    Healpix_Ordering_Scheme hpix_order_scheme;
    if (order_str == "ring") {
        hpix_order_scheme = RING;
    } else if (order_str == "nested") {
        hpix_order_scheme = NESTED;
    } else {
        throw std::runtime_error("Invalid HEALPix order: must be 'ring' or 'nested'.");
    }

    Healpix_Base hp_base(nside, hpix_order_scheme);
    std::vector<long> ids(ra_deg.size());

    for (size_t i = 0; i < ra_deg.size(); ++i) {
        double theta = (90.0 - dec_deg[i]) * DEG2RAD_Host; // colatitude
        double phi = ra_deg[i] * DEG2RAD_Host;
        pointing pt(theta, phi);
        ids[i] = hp_base.ang2pix(pt);
    }
    return ids;
#else
    // Dummy implementation or error if Healpix is not found
    std::cerr << "Warning: convert_to_healpix_ids_cpp called but HEALPIX_FOUND is false. Returning empty vector." << std::endl;
    return std::vector<long>();
#endif
}

PointCloud<double> build_healpix_xyz_celestial_cpp(
    const std::vector<long>& unique_pix_ids,
    int nside,
    const std::string& order_str) {

    PointCloud<double> cloud;
#if HEALPIX_FOUND
    if (nside <= 0) {
        throw std::runtime_error("NSIDE must be positive.");
    }
    Healpix_Ordering_Scheme hpix_order_scheme;
    if (order_str == "ring") {
        hpix_order_scheme = RING;
    } else if (order_str == "nested") {
        hpix_order_scheme = NESTED;
    } else {
        throw std::runtime_error("Invalid HEALPix order: must be 'ring' or 'nested'.");
    }

    Healpix_Base hp_base(nside, hpix_order_scheme);
    cloud.pts.resize(unique_pix_ids.size());

    for (size_t i = 0; i < unique_pix_ids.size(); ++i) {
        pointing pt = hp_base.pix2ang(unique_pix_ids[i]); // pt.theta, pt.phi are colatitude and longitude
        // Convert spherical (theta, phi) to Cartesian (x,y,z)
        // x = sin(theta)cos(phi)
        // y = sin(theta)sin(phi)
        // z = cos(theta)
        // Assuming a unit sphere for pixel centers. If a specific distance is needed, multiply by radius.
        // For celestial coordinates, this is fine for angular queries.
        cloud.pts[i].x = std::sin(pt.theta) * std::cos(pt.phi);
        cloud.pts[i].y = std::sin(pt.theta) * std::sin(pt.phi);
        cloud.pts[i].z = std::cos(pt.theta);
    }
#else
    std::cerr << "Warning: build_healpix_xyz_celestial_cpp called but HEALPIX_FOUND is false. Returning empty PointCloud." << std::endl;
#endif
    return cloud;
}

std::vector<size_t> find_neighboring_pixel_indices_nanoflann_cpp(
    const PointCloud<double>& source_pixel_cloud,
    const double query_pt[3],
    double search_radius_sq) {

    std::vector<size_t> result_indices;
    if (source_pixel_cloud.kdtree_get_point_count() == 0) {
        return result_indices; // No points to search in
    }

    // KD-tree construction
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>,
        PointCloud<double>,
        3 /* dimensionality */
        > kd_tree_t;

    kd_tree_t index(3, source_pixel_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();

    std::vector<nanoflann::ResultItem<size_t, double>> ret_matches;
    nanoflann::SearchParams params;
    // params.sorted = true; // If sorted results are needed

    const size_t n_matches = index.radiusSearch(&query_pt[0], search_radius_sq, ret_matches, params);

    result_indices.resize(n_matches);
    for (size_t i = 0; i < n_matches; ++i) {
        result_indices[i] = ret_matches[i].first;
    }
    return result_indices;
}
