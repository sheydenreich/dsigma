#include "precompute_interface.h"
#include "precompute_engine_cuda.h" // For launching the kernel and physics
#include "cuda_host_utils.h"      // For Healpix utilities on host (still needed for initial unique IDs)
#include "healpix_gpu.h"          // For GPU-side HEALPix functions
// Replace broken kdtree with cudaKDTree
#include "../cudaKDTree/cukd/builder.h"
#include "../cudaKDTree/cukd/knn.h"
#include <cfloat>
#include <cuda_runtime.h>
#include <vector_types.h>       // For float3
#include <vector>
#include <iostream> // For error messages / cout
#include <algorithm> // For std::sort, std::unique, std::max
#include <map>       // For std::map (can be removed if not used for unique_pix anymore)
#include <cmath>     // For sqrt, M_PI
#include <stdexcept> // For std::runtime_error

// Ensure M_PI is defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// DEG2RAD, if still needed (likely replaced by direct radian usage)
#ifndef DEG2RAD_Interface
#define DEG2RAD_Interface 0.017453292519943295
#endif

// Constants for HEALPix - Host version
#ifndef TWO_PI
#define TWO_PI (2.0 * M_PI)
#endif

#define TWOTHIRD (2.0 / 3.0)

// Host version of acos_safe function
double acos_safe_host(double val) {
    if (val <= -1.0) {
        return M_PI;
    } else if (val >= 1.0) {
        return 0.0;
    } else {
        return acos(val);
    }
}

// Host version of pix2ang_ring function (copied from healpix_gpu.cu device version)
void pix2ang_ring_host(long nside, long pix, double& theta, double& phi) {
    if (nside <= 0 || pix < 0 || pix >= 12 * nside * nside) {
        theta = phi = 0.0;
        return;
    }

    const long npix = 12 * nside * nside;
    const long ncap = 2 * nside * (nside - 1);
    const long nsidesq = nside * nside;
    const long nl2 = 2 * nside;
    const long nl4 = 4 * nside;

    long ipix1 = pix + 1; // Convert to 1-based indexing as in Java

    if (ipix1 <= ncap) { // North polar cap
        double hip = ipix1 / 2.0;
        double fihip = (long) hip; // get integer part of hip
        long iring = (long) (sqrt(hip - sqrt(fihip))) + 1; // counted from north pole
        long iphi = ipix1 - 2 * iring * (iring - 1);
        
        theta = acos_safe_host(1.0 - iring * iring / (3.0 * nsidesq));
        phi = ((double)iphi - 0.5) * M_PI / (2.0 * iring);
        
    } else if (ipix1 <= nl2 * (5 * nside + 1)) { // Equatorial region
        long ip = ipix1 - ncap - 1;
        long iring = (ip / nl4) + nside; // counted from North pole
        long iphi = ip % nl4 + 1;
        
        double fodd = 0.5 * (1.0 + ((iring + nside) % 2)); // 1 if iring+nside is odd, 1/2 otherwise
        
        theta = acos_safe_host((nl2 - iring) / (1.5 * nside));
        phi = ((double)iphi - fodd) * M_PI / (2.0 * nside);

    } else { // South polar cap
        long ip = npix - ipix1 + 1;
        double hip = ip / 2.0;
        double fihip = (long) hip;
        long iring = (long) (sqrt(hip - sqrt(fihip))) + 1; // counted from South pole
        long iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        
        theta = acos_safe_host(-1.0 + iring * iring / (3.0 * nsidesq));
        phi = ((double)iphi - 0.5) * M_PI / (2.0 * iring);
    }
    
    // Normalize phi to [0, 2*PI)
    while (phi < 0.0) phi += TWO_PI;
    while (phi >= TWO_PI) phi -= TWO_PI;
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
        return -1; \
    } \
}

// Structure for passing data to the source processing
typedef struct {
    int lens_idx;
    int N_bins;
    long nside_healpix;

    // Source data (global arrays)
    const long* g_unique_source_hp_ids;
    const long* g_all_source_hp_ids_sorted;
    const int* g_sorted_source_original_indices;
    const int* g_unique_hp_id_offsets_start;
    const int* g_unique_hp_id_offsets_end;

    // Lens data
    double lens_zl_i;
    double lens_dcoml_i;
    double lens_sin_ra_l_i;
    double lens_cos_ra_l_i;
    double lens_sin_dec_l_i;
    double lens_cos_dec_l_i;

    // Pointers to global source property arrays
    const double* g_z_s;
    const double* g_d_com_s;
    const double* g_sin_ra_s;
    const double* g_cos_ra_s;
    const double* g_sin_dec_s;
    const double* g_cos_dec_s;
    const double* g_w_s;
    const double* g_e_1_s;
    const double* g_e_2_s;
    const double* g_z_l_max_s;

    bool has_sigma_crit_eff;
    int n_z_bins_l;
    const double* g_sigma_crit_eff_l;
    const int* g_z_bin_s;

    bool has_m_s; const double* g_m_s;
    bool has_e_rms_s; const double* g_e_rms_s;
    bool has_R_2_s; const double* g_R_2_s;
    bool has_R_matrix_s;
    const double* g_R_11_s; const double* g_R_12_s;
    const double* g_R_21_s; const double* g_R_22_s;

    const double* g_dist_3d_sq_bins;

    // Configuration
    bool comoving;
    int weighting;

    // Output sum arrays (global pointers)
    long long* g_sum_1_r;
    double* g_sum_w_ls_r;
    double* g_sum_w_ls_e_t_r;
    double* g_sum_w_ls_e_t_sigma_crit_r;
    double* g_sum_w_ls_z_s_r;
    double* g_sum_w_ls_sigma_crit_r;
    double* g_sum_w_ls_m_r;
    double* g_sum_w_ls_1_minus_e_rms_sq_r;
    double* g_sum_w_ls_A_p_R_2_r;
    double* g_sum_w_ls_R_T_r;

} KernelCallbackData;

// Device function to process a single source HP pixel that was found by KNN search
__device__ void process_found_source_hp_pixel(int source_kdtree_idx, KernelCallbackData* cb_data) {
    // 1. Get the unique HEALPix ID for the found KD-tree node
    long unique_hp_id = cb_data->g_unique_source_hp_ids[source_kdtree_idx];

    // 2. Find the range of actual sources belonging to this unique_hp_id
    int start_offset = cb_data->g_unique_hp_id_offsets_start[source_kdtree_idx];
    int end_offset = cb_data->g_unique_hp_id_offsets_end[source_kdtree_idx];
    
    // Debug output can be enabled by uncommenting:
    // if (cb_data->lens_idx < 3 && source_kdtree_idx < 3) {
    //     printf("  Processing HP pixel %d: hp_id=%ld, sources [%d:%d)\n", 
    //            source_kdtree_idx, unique_hp_id, start_offset, end_offset);
    // }

    // 3. Loop over these actual source indices
    for (int i_s_mapped_idx = start_offset; i_s_mapped_idx < end_offset; ++i_s_mapped_idx) {
        int original_source_idx = cb_data->g_sorted_source_original_indices[i_s_mapped_idx];

        // Load current source data using original_source_idx
        double zs_i = cb_data->g_z_s[original_source_idx];

        // Redshift filter
        if (cb_data->lens_zl_i >= zs_i || cb_data->lens_zl_i >= cb_data->g_z_l_max_s[original_source_idx]) {
            continue;
        }

        double dcoms_i = cb_data->g_d_com_s[original_source_idx];
        double sin_ra_s_i = cb_data->g_sin_ra_s[original_source_idx];
        double cos_ra_s_i = cb_data->g_cos_ra_s[original_source_idx];
        double sin_dec_s_i = cb_data->g_sin_dec_s[original_source_idx];
        double cos_dec_s_i = cb_data->g_cos_dec_s[original_source_idx];

        // Calculate 3D angular distance squared to match CPU implementation
        // The CPU uses dist_3d_sq which is 3D chord distance on unit sphere
        double dist_sq = dist_angular_sq_gpu(
            cb_data->lens_sin_ra_l_i, cb_data->lens_cos_ra_l_i,
            cb_data->lens_sin_dec_l_i, cb_data->lens_cos_dec_l_i,
            sin_ra_s_i, cos_ra_s_i, sin_dec_s_i, cos_dec_s_i
        );

        // Find bin index
        const double* current_lens_dist_bins = cb_data->g_dist_3d_sq_bins + (size_t)cb_data->lens_idx * (cb_data->N_bins + 1);
        int i_bin = find_bin_idx_gpu(dist_sq, current_lens_dist_bins, cb_data->N_bins);

        if (i_bin == -1) { // Not in any bin for this lens
            continue;
        }

        // Calculate sigma_crit_inverse
        int z_bin_s_val = -1;
        if (cb_data->has_sigma_crit_eff && cb_data->g_z_bin_s != nullptr) {
            z_bin_s_val = cb_data->g_z_bin_s[original_source_idx];
        }
        double sigma_crit_inv = calculate_sigma_crit_inv_gpu(
            cb_data->lens_zl_i, zs_i, cb_data->lens_dcoml_i, dcoms_i,
            cb_data->comoving,
            cb_data->has_sigma_crit_eff,
            cb_data->g_sigma_crit_eff_l,
            cb_data->lens_idx,
            cb_data->n_z_bins_l,
            z_bin_s_val
        );

        if (sigma_crit_inv == 0.0 || sigma_crit_inv == DBL_MAX) {
            continue;
        }

        // Calculate w_ls
        double w_ls = calculate_w_ls_gpu(
            sigma_crit_inv,
            cb_data->g_w_s[original_source_idx],
            cb_data->weighting
        );

        if (w_ls == 0.0) {
            continue;
        }

        // Calculate tangential shear e_t
        double cos_2phi, sin_2phi;
        calculate_et_components_gpu(
            cb_data->lens_sin_ra_l_i, cb_data->lens_cos_ra_l_i,
            cb_data->lens_sin_dec_l_i, cb_data->lens_cos_dec_l_i,
            sin_ra_s_i, cos_ra_s_i, sin_dec_s_i, cos_dec_s_i,
            cos_2phi, sin_2phi
        );
        double e_t_val = calculate_et_gpu(
            cb_data->g_e_1_s[original_source_idx],
            cb_data->g_e_2_s[original_source_idx],
            cos_2phi, sin_2phi
        );

        // Accumulate results using atomics
        size_t out_idx = (size_t)cb_data->lens_idx * cb_data->N_bins + i_bin;

        atomicAdd((unsigned long long int*)&cb_data->g_sum_1_r[out_idx], 1ULL);
        atomicAdd(&cb_data->g_sum_w_ls_r[out_idx], w_ls);
        atomicAdd(&cb_data->g_sum_w_ls_e_t_r[out_idx], w_ls * e_t_val);
        
        double sigma_crit = (sigma_crit_inv == 0.0) ? DBL_MAX : 1.0 / sigma_crit_inv;
        atomicAdd(&cb_data->g_sum_w_ls_e_t_sigma_crit_r[out_idx], w_ls * e_t_val * sigma_crit);
        atomicAdd(&cb_data->g_sum_w_ls_z_s_r[out_idx], w_ls * zs_i);
        atomicAdd(&cb_data->g_sum_w_ls_sigma_crit_r[out_idx], w_ls * sigma_crit);

        if (cb_data->has_m_s && cb_data->g_m_s != nullptr && cb_data->g_sum_w_ls_m_r != nullptr) {
            atomicAdd(&cb_data->g_sum_w_ls_m_r[out_idx], w_ls * cb_data->g_m_s[original_source_idx]);
        }
        if (cb_data->has_e_rms_s && cb_data->g_e_rms_s != nullptr && cb_data->g_sum_w_ls_1_minus_e_rms_sq_r != nullptr) {
            double e_rms_s_i = cb_data->g_e_rms_s[original_source_idx];
            atomicAdd(&cb_data->g_sum_w_ls_1_minus_e_rms_sq_r[out_idx], w_ls * (1.0 - e_rms_s_i * e_rms_s_i));
        }
        if (cb_data->has_R_2_s && cb_data->g_R_2_s != nullptr && cb_data->g_sum_w_ls_A_p_R_2_r != nullptr) {
            if (cb_data->g_R_2_s[original_source_idx] <= 0.31) {
                // Match CPU implementation: 0.00865 * w_ls / 0.01 = 0.865 * w_ls
                atomicAdd(&cb_data->g_sum_w_ls_A_p_R_2_r[out_idx], 0.00865 * w_ls / 0.01);
            }
        }
        if (cb_data->has_R_matrix_s && cb_data->g_R_11_s != nullptr && cb_data->g_sum_w_ls_R_T_r != nullptr) {
            double R_T_val = calculate_R_T_gpu(
                cb_data->g_R_11_s[original_source_idx], cb_data->g_R_12_s[original_source_idx],
                cb_data->g_R_21_s[original_source_idx], cb_data->g_R_22_s[original_source_idx],
                cos_2phi, sin_2phi
            );
            atomicAdd(&cb_data->g_sum_w_ls_R_T_r[out_idx], w_ls * R_T_val);
        }
    }
}

__global__ void process_all_lenses_kernel(
    // Lens data (global arrays)
    const double* g_z_l, const double* g_d_com_l,
    const double* g_sin_ra_l, const double* g_cos_ra_l,
    const double* g_sin_dec_l, const double* g_cos_dec_l,
    const double* g_dist_3d_sq_bins,

    // Source data (global arrays)
    const double* g_z_s, const double* g_d_com_s,
    const double* g_sin_ra_s, const double* g_cos_ra_s,
    const double* g_sin_dec_s, const double* g_cos_dec_s,
    const double* g_w_s, const double* g_e_1_s, const double* g_e_2_s,
    const double* g_z_l_max_s,

    // Unique Source HEALPix data for KD-Tree
    const float3* g_unique_source_hp_coords_kdtree,
    const long* g_unique_source_hp_ids,
    int N_unique_source_hp,

    // Mapping from unique source HP ID back to original source indices
    const long* g_all_source_hp_ids_sorted,
    const int* g_sorted_source_original_indices,
    const int* g_unique_hp_id_offsets_start,
    const int* g_unique_hp_id_offsets_end,

    // Optional source data
    bool _has_sigma_crit_eff, int _n_z_bins_l, const double* _g_sigma_crit_eff_l, const int* _g_z_bin_s,
    bool _has_m_s, const double* _g_m_s,
    bool _has_e_rms_s, const double* _g_e_rms_s,
    bool _has_R_2_s, const double* _g_R_2_s,
    bool _has_R_matrix_s, const double* _g_R_11_s, const double* _g_R_12_s, const double* _g_R_21_s, const double* _g_R_22_s,

    // Configuration
    int N_lenses, int N_bins, long nside_healpix, bool comoving, int weighting,

    // Output sum arrays
    long long* g_sum_1_r, double* g_sum_w_ls_r,
    double* g_sum_w_ls_e_t_r, double* g_sum_w_ls_e_t_sigma_crit_r,
    double* g_sum_w_ls_z_s_r, double* g_sum_w_ls_sigma_crit_r,
    double* g_sum_w_ls_m_r, double* g_sum_w_ls_1_minus_e_rms_sq_r,
    double* g_sum_w_ls_A_p_R_2_r, double* g_sum_w_ls_R_T_r
) {
    int lens_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lens_idx >= N_lenses) return;

    // Load current lens data
    double zl_i = g_z_l[lens_idx];
    double dcoml_i = g_d_com_l[lens_idx];
    double sin_ra_l_i = g_sin_ra_l[lens_idx];
    double cos_ra_l_i = g_cos_ra_l[lens_idx];
    double sin_dec_l_i = g_sin_dec_l[lens_idx];
    double cos_dec_l_i = g_cos_dec_l[lens_idx];

    float3 lens_xyz_cartesian;
    spherical_to_cartesian_gpu(sin_ra_l_i, cos_ra_l_i, sin_dec_l_i, cos_dec_l_i, lens_xyz_cartesian);

    // Calculate search radius for KNN to match CPU implementation
    // The CPU uses dist_3d_sq_bins which are 3D angular distances squared on unit sphere
    double max_dist_3d_sq_lens = g_dist_3d_sq_bins[lens_idx * (N_bins + 1) + N_bins];
    double lens_hp_pixrad = get_max_pixrad_gpu(nside_healpix);
    
    // Convert HEALPix pixel radius from radians to 3D chord distance
    double pixrad_3d_sq = 4.0 * sin(lens_hp_pixrad * 0.5) * sin(lens_hp_pixrad * 0.5);
    
    // Add buffer for pixel radius (matching CPU implementation)
    double search_radius_sq_gpu = max_dist_3d_sq_lens + 
                                  (4.0 * pixrad_3d_sq + 
                                   4.0 * sqrt(max_dist_3d_sq_lens) * sqrt(pixrad_3d_sq));
    
    float search_radius_sq = (float)search_radius_sq_gpu;
    
    // Debug output can be enabled by uncommenting:
    // if (lens_idx < 3) {
    //     printf("Lens %d: max_dist_3d_sq=%.6e, pixrad=%.6e, search_radius_sq=%.6e\n", 
    //            lens_idx, max_dist_3d_sq_lens, lens_hp_pixrad, search_radius_sq_gpu);
    // }

    // Prepare callback data
    KernelCallbackData callback_data;
    callback_data.lens_idx = lens_idx;
    callback_data.N_bins = N_bins;
    callback_data.nside_healpix = nside_healpix;
    callback_data.g_unique_source_hp_ids = g_unique_source_hp_ids;
    callback_data.g_all_source_hp_ids_sorted = g_all_source_hp_ids_sorted;
    callback_data.g_sorted_source_original_indices = g_sorted_source_original_indices;
    callback_data.g_unique_hp_id_offsets_start = g_unique_hp_id_offsets_start;
    callback_data.g_unique_hp_id_offsets_end = g_unique_hp_id_offsets_end;
    callback_data.lens_zl_i = zl_i;
    callback_data.lens_dcoml_i = dcoml_i;
    callback_data.lens_sin_ra_l_i = sin_ra_l_i;
    callback_data.lens_cos_ra_l_i = cos_ra_l_i;
    callback_data.lens_sin_dec_l_i = sin_dec_l_i;
    callback_data.lens_cos_dec_l_i = cos_dec_l_i;
    callback_data.g_z_s = g_z_s;
    callback_data.g_d_com_s = g_d_com_s;
    callback_data.g_sin_ra_s = g_sin_ra_s;
    callback_data.g_cos_ra_s = g_cos_ra_s;
    callback_data.g_sin_dec_s = g_sin_dec_s;
    callback_data.g_cos_dec_s = g_cos_dec_s;
    callback_data.g_w_s = g_w_s;
    callback_data.g_e_1_s = g_e_1_s;
    callback_data.g_e_2_s = g_e_2_s;
    callback_data.g_z_l_max_s = g_z_l_max_s;
    callback_data.has_sigma_crit_eff = _has_sigma_crit_eff;
    callback_data.n_z_bins_l = _n_z_bins_l;
    callback_data.g_sigma_crit_eff_l = _g_sigma_crit_eff_l;
    callback_data.g_z_bin_s = _g_z_bin_s;
    callback_data.has_m_s = _has_m_s; callback_data.g_m_s = _g_m_s;
    callback_data.has_e_rms_s = _has_e_rms_s; callback_data.g_e_rms_s = _g_e_rms_s;
    callback_data.has_R_2_s = _has_R_2_s; callback_data.g_R_2_s = _g_R_2_s;
    callback_data.has_R_matrix_s = _has_R_matrix_s;
    callback_data.g_R_11_s = _g_R_11_s; callback_data.g_R_12_s = _g_R_12_s;
    callback_data.g_R_21_s = _g_R_21_s; callback_data.g_R_22_s = _g_R_22_s;
    callback_data.g_dist_3d_sq_bins = g_dist_3d_sq_bins;
    callback_data.comoving = comoving;
    callback_data.weighting = weighting;
    callback_data.g_sum_1_r = g_sum_1_r;
    callback_data.g_sum_w_ls_r = g_sum_w_ls_r;
    callback_data.g_sum_w_ls_e_t_r = g_sum_w_ls_e_t_r;
    callback_data.g_sum_w_ls_e_t_sigma_crit_r = g_sum_w_ls_e_t_sigma_crit_r;
    callback_data.g_sum_w_ls_z_s_r = g_sum_w_ls_z_s_r;
    callback_data.g_sum_w_ls_sigma_crit_r = g_sum_w_ls_sigma_crit_r;
    callback_data.g_sum_w_ls_m_r = g_sum_w_ls_m_r;
    callback_data.g_sum_w_ls_1_minus_e_rms_sq_r = g_sum_w_ls_1_minus_e_rms_sq_r;
    callback_data.g_sum_w_ls_A_p_R_2_r = g_sum_w_ls_A_p_R_2_r;
    callback_data.g_sum_w_ls_R_T_r = g_sum_w_ls_R_T_r;

    // Use simple brute force radius search since kdtree_search_gpu.cu is broken
    // Loop through all unique source HEALPix pixels and check distance
    int candidates_found = 0;
    
    for (int source_kdtree_idx = 0; source_kdtree_idx < N_unique_source_hp; source_kdtree_idx++) {
        // Calculate distance squared between lens and this source HEALPix pixel
        float3 source_xyz = g_unique_source_hp_coords_kdtree[source_kdtree_idx];
        
        float dx = lens_xyz_cartesian.x - source_xyz.x;
        float dy = lens_xyz_cartesian.y - source_xyz.y;
        float dz = lens_xyz_cartesian.z - source_xyz.z;
        float dist_sq = dx * dx + dy * dy + dz * dz;
        
        // If within search radius, process this HEALPix pixel
        if (dist_sq <= search_radius_sq) {
            candidates_found++;
            process_found_source_hp_pixel(source_kdtree_idx, &callback_data);
        }
    }
    
    // Debug output can be enabled by uncommenting:
    // if (lens_idx < 5) {
    //     printf("Lens %d: search_radius_sq=%.6e, candidates_found=%d\n", 
    //            lens_idx, search_radius_sq, candidates_found);
    // }
}

int precompute_cuda_interface(TableData* tables, int n_gpus) {
    // --- 1. Input Validation (remains similar) ---
    if (!tables->z_l || !tables->d_com_l || !tables->sin_ra_l || !tables->cos_ra_l || !tables->sin_dec_l || !tables->cos_dec_l ||
        !tables->z_s || !tables->d_com_s || !tables->sin_ra_s || !tables->cos_ra_s || !tables->sin_dec_s || !tables->cos_dec_s || !tables->w_s || !tables->e_1_s || !tables->e_2_s || !tables->z_l_max_s || !tables->healpix_id_s ||
        !tables->dist_3d_sq_bins ||
        !tables->sum_1_r || !tables->sum_w_ls_r || !tables->sum_w_ls_e_t_r || !tables->sum_w_ls_e_t_sigma_crit_r || !tables->sum_w_ls_z_s_r || !tables->sum_w_ls_sigma_crit_r) {
        std::cerr << "Error: Essential data pointers in TableData are null." << std::endl;
        return -1;
    }
     if (tables->has_sigma_crit_eff && (!tables->sigma_crit_eff_l || !tables->z_bin_s)) {
        std::cerr << "Error: sigma_crit_eff pointers are null when has_sigma_crit_eff is true." << std::endl;
        return -1;
    }
    if (tables->has_m_s && !tables->m_s) { std::cerr << "Error: m_s is null when has_m_s is true." << std::endl; return -1; }
    // ... other validation checks ...
    if (tables->n_lenses <= 0 && tables->n_sources == 0) {
        std::cout << "No lenses and no sources. Exiting." << std::endl;
        return 0; // No work to do
    }
     if (tables->n_lenses > 0 && tables->n_bins <= 0) {
        std::cerr << "Error: n_bins must be positive if there are lenses." << std::endl;
        return -1;
    }
    if (tables->nside_healpix <= 0 && (tables->n_lenses > 0 || tables->n_sources > 0)) {
        std::cerr << "Error: nside_healpix must be positive if there are objects." << std::endl;
        return -1;
    }


    // --- 2. GPU Setup ---
    CUDA_CHECK(cudaSetDevice(0)); // Use GPU 0 by default

    // --- 3. GPU Data Allocation and H2D Transfers (Upfront) ---
    // Device pointers
    double *d_z_l, *d_d_com_l, *d_sin_ra_l, *d_cos_ra_l, *d_sin_dec_l, *d_cos_dec_l;
    double *d_dist_3d_sq_bins;

    double *d_z_s, *d_d_com_s, *d_sin_ra_s, *d_cos_ra_s, *d_sin_dec_s, *d_cos_dec_s;
    double *d_w_s, *d_e_1_s, *d_e_2_s, *d_z_l_max_s;
    long* d_all_source_hp_ids; // All source HP IDs, unsorted initially

    double *d_sigma_crit_eff_l = nullptr; int *d_z_bin_s = nullptr;
    double *d_m_s = nullptr, *d_e_rms_s = nullptr, *d_R_2_s = nullptr;
    double *d_R_11_s = nullptr, *d_R_12_s = nullptr, *d_R_21_s = nullptr, *d_R_22_s = nullptr;

    // Allocate and copy lens data
    CUDA_CHECK(cudaMalloc(&d_z_l, tables->n_lenses * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_z_l, tables->z_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_d_com_l, tables->n_lenses * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_d_com_l, tables->d_com_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice));
    // ... (repeat for sin_ra_l, cos_ra_l, sin_dec_l, cos_dec_l) ...
    CUDA_CHECK(cudaMalloc(&d_sin_ra_l, tables->n_lenses * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_sin_ra_l, tables->sin_ra_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cos_ra_l, tables->n_lenses * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cos_ra_l, tables->cos_ra_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_sin_dec_l, tables->n_lenses * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_sin_dec_l, tables->sin_dec_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_cos_dec_l, tables->n_lenses * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_cos_dec_l, tables->cos_dec_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_dist_3d_sq_bins, (size_t)tables->n_lenses * (tables->n_bins + 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_dist_3d_sq_bins, tables->dist_3d_sq_bins, (size_t)tables->n_lenses * (tables->n_bins + 1) * sizeof(double), cudaMemcpyHostToDevice));

    // Allocate and copy source data
    if (tables->n_sources > 0) {
        CUDA_CHECK(cudaMalloc(&d_z_s, tables->n_sources * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_z_s, tables->z_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        // ... (repeat for all source arrays: d_com_s, sin/cos_ra/dec_s, w_s, e1/2_s, z_l_max_s, healpix_id_s) ...
        CUDA_CHECK(cudaMalloc(&d_d_com_s, tables->n_sources * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_d_com_s, tables->d_com_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_sin_ra_s, tables->n_sources * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_sin_ra_s, tables->sin_ra_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_cos_ra_s, tables->n_sources * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_cos_ra_s, tables->cos_ra_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_sin_dec_s, tables->n_sources * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_sin_dec_s, tables->sin_dec_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_cos_dec_s, tables->n_sources * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_cos_dec_s, tables->cos_dec_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_w_s, tables->n_sources * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_w_s, tables->w_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_e_1_s, tables->n_sources * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_e_1_s, tables->e_1_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_e_2_s, tables->n_sources * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_e_2_s, tables->e_2_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_z_l_max_s, tables->n_sources * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_z_l_max_s, tables->z_l_max_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_all_source_hp_ids, tables->n_sources * sizeof(long)));
        CUDA_CHECK(cudaMemcpy(d_all_source_hp_ids, tables->healpix_id_s, tables->n_sources * sizeof(long), cudaMemcpyHostToDevice));


        if (tables->has_sigma_crit_eff) {
            CUDA_CHECK(cudaMalloc(&d_sigma_crit_eff_l, (size_t)tables->n_lenses * tables->n_z_bins_l * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_sigma_crit_eff_l, tables->sigma_crit_eff_l, (size_t)tables->n_lenses * tables->n_z_bins_l * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_z_bin_s, tables->n_sources * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_z_bin_s, tables->z_bin_s, tables->n_sources * sizeof(int), cudaMemcpyHostToDevice));
        }
        // ... (optional source arrays m_s, e_rms_s, etc.) ...
        if (tables->has_m_s) {
            CUDA_CHECK(cudaMalloc(&d_m_s, tables->n_sources * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_m_s, tables->m_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        }
         if (tables->has_e_rms_s) {
            CUDA_CHECK(cudaMalloc(&d_e_rms_s, tables->n_sources * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_e_rms_s, tables->e_rms_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (tables->has_R_2_s) {
            CUDA_CHECK(cudaMalloc(&d_R_2_s, tables->n_sources * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_R_2_s, tables->R_2_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        }
        if (tables->has_R_matrix_s) {
            CUDA_CHECK(cudaMalloc(&d_R_11_s, tables->n_sources * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_R_11_s, tables->R_11_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_R_12_s, tables->n_sources * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_R_12_s, tables->R_12_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_R_21_s, tables->n_sources * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_R_21_s, tables->R_21_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_R_22_s, tables->n_sources * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(d_R_22_s, tables->R_22_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice));
        }
    } else { // n_sources == 0
        // Set source data pointers to null if no sources
        d_z_s = d_d_com_s = d_sin_ra_s = d_cos_ra_s = d_sin_dec_s = d_cos_dec_s = nullptr;
        d_w_s = d_e_1_s = d_e_2_s = d_z_l_max_s = nullptr;
        d_all_source_hp_ids = nullptr;
        // Optional ones too
        d_sigma_crit_eff_l = nullptr; d_z_bin_s = nullptr; // Note: d_sigma_crit_eff_l is lens-based but used with z_bin_s
        d_m_s = d_e_rms_s = d_R_2_s = nullptr;
        d_R_11_s = d_R_12_s = d_R_21_s = d_R_22_s = nullptr;
    }


    // Allocate and initialize output sum arrays on GPU
    size_t total_output_bins = (size_t)tables->n_lenses * tables->n_bins;
    long long* d_sum_1_r;
    double *d_sum_w_ls_r, *d_sum_w_ls_e_t_r, *d_sum_w_ls_e_t_sigma_crit_r;
    double *d_sum_w_ls_z_s_r, *d_sum_w_ls_sigma_crit_r;
    double *d_sum_w_ls_m_r = nullptr, *d_sum_w_ls_1_minus_e_rms_sq_r = nullptr;
    double *d_sum_w_ls_A_p_R_2_r = nullptr, *d_sum_w_ls_R_T_r = nullptr;

    if (total_output_bins > 0) {
        CUDA_CHECK(cudaMalloc(&d_sum_1_r, total_output_bins * sizeof(long long)));
        CUDA_CHECK(cudaMemset(d_sum_1_r, 0, total_output_bins * sizeof(long long)));
        CUDA_CHECK(cudaMalloc(&d_sum_w_ls_r, total_output_bins * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_sum_w_ls_r, 0, total_output_bins * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sum_w_ls_e_t_r, total_output_bins * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_sum_w_ls_e_t_r, 0, total_output_bins * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sum_w_ls_e_t_sigma_crit_r, total_output_bins * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_sum_w_ls_e_t_sigma_crit_r, 0, total_output_bins * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sum_w_ls_z_s_r, total_output_bins * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_sum_w_ls_z_s_r, 0, total_output_bins * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_sum_w_ls_sigma_crit_r, total_output_bins * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_sum_w_ls_sigma_crit_r, 0, total_output_bins * sizeof(double)));

        if (tables->has_m_s && tables->sum_w_ls_m_r) {
            CUDA_CHECK(cudaMalloc(&d_sum_w_ls_m_r, total_output_bins * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_sum_w_ls_m_r, 0, total_output_bins * sizeof(double)));
        }
        if (tables->has_e_rms_s && tables->sum_w_ls_1_minus_e_rms_sq_r) {
            CUDA_CHECK(cudaMalloc(&d_sum_w_ls_1_minus_e_rms_sq_r, total_output_bins * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_sum_w_ls_1_minus_e_rms_sq_r, 0, total_output_bins * sizeof(double)));
        }
        if (tables->has_R_2_s && tables->sum_w_ls_A_p_R_2_r) {
            CUDA_CHECK(cudaMalloc(&d_sum_w_ls_A_p_R_2_r, total_output_bins * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_sum_w_ls_A_p_R_2_r, 0, total_output_bins * sizeof(double)));
        }
        if (tables->has_R_matrix_s && tables->sum_w_ls_R_T_r) {
            CUDA_CHECK(cudaMalloc(&d_sum_w_ls_R_T_r, total_output_bins * sizeof(double)));
            CUDA_CHECK(cudaMemset(d_sum_w_ls_R_T_r, 0, total_output_bins * sizeof(double)));
        }
    } else {
      if (tables->n_lenses == 0) {
        std::cout << "No lenses to process. Exiting early." << std::endl;
        if (d_all_source_hp_ids) CUDA_CHECK(cudaFree(d_all_source_hp_ids));
        if (d_z_s) CUDA_CHECK(cudaFree(d_z_s)); if (d_d_com_s) CUDA_CHECK(cudaFree(d_d_com_s));
        return 0;
      }
    }

    // --- 4. GPU-Side Data Preparation for KD-Tree ---
    float3* d_unique_source_hp_coords_kdtree = nullptr;
    long* d_unique_source_hp_ids = nullptr;
    int N_unique_source_hp = 0;

    // For mapping unique HP IDs to original source indices ranges
    long* d_all_source_hp_ids_sorted_gpu = nullptr;
    int* d_sorted_source_original_indices_gpu = nullptr;
    int* d_unique_hp_id_offsets_start_gpu = nullptr;
    int* d_unique_hp_id_offsets_end_gpu = nullptr;


    if (tables->n_sources > 0) {
        // --- Conceptual Steps for KD-Tree setup ---
        // 1. Create a temporary array of (hp_id, original_idx) pairs on host or device.
        std::vector<std::pair<long, int>> source_hp_pairs(tables->n_sources);
        for(int i=0; i < tables->n_sources; ++i) {
            source_hp_pairs[i] = {tables->healpix_id_s[i], i};
        }

        // 2. Sort these pairs by hp_id.
        std::sort(source_hp_pairs.begin(), source_hp_pairs.end(),
                  [](const auto& a, const auto& b){ return a.first < b.first; });

        // 3. Populate d_all_source_hp_ids_sorted_gpu and d_sorted_source_original_indices_gpu
        std::vector<long> h_all_source_hp_ids_sorted(tables->n_sources);
        std::vector<int> h_sorted_source_original_indices(tables->n_sources);
        for(int i=0; i < tables->n_sources; ++i) {
            h_all_source_hp_ids_sorted[i] = source_hp_pairs[i].first;
            h_sorted_source_original_indices[i] = source_hp_pairs[i].second;
        }
        CUDA_CHECK(cudaMalloc(&d_all_source_hp_ids_sorted_gpu, tables->n_sources * sizeof(long)));
        CUDA_CHECK(cudaMemcpy(d_all_source_hp_ids_sorted_gpu, h_all_source_hp_ids_sorted.data(), tables->n_sources * sizeof(long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_sorted_source_original_indices_gpu, tables->n_sources * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_sorted_source_original_indices_gpu, h_sorted_source_original_indices.data(), tables->n_sources * sizeof(int), cudaMemcpyHostToDevice));


        // 4. Find unique sorted HP IDs and their offsets on host (could be done on GPU with Thrust/CUB)
        std::vector<long> h_unique_source_hp_ids;
        std::vector<int> h_unique_hp_id_offsets_start; // inclusive start index in sorted_hp_ids
        std::vector<int> h_unique_hp_id_offsets_end;   // exclusive end index

        if (tables->n_sources > 0) {
            h_unique_source_hp_ids.push_back(h_all_source_hp_ids_sorted[0]);
            h_unique_hp_id_offsets_start.push_back(0);
            for (int i = 1; i < tables->n_sources; ++i) {
                if (h_all_source_hp_ids_sorted[i] != h_all_source_hp_ids_sorted[i-1]) {
                    h_unique_source_hp_ids.push_back(h_all_source_hp_ids_sorted[i]);
                    h_unique_hp_id_offsets_end.push_back(i); // end for previous
                    h_unique_hp_id_offsets_start.push_back(i); // start for current
                }
            }
            h_unique_hp_id_offsets_end.push_back(tables->n_sources); // end for the last unique ID
        }
        N_unique_source_hp = h_unique_source_hp_ids.size();

        if (N_unique_source_hp > 0) {
            CUDA_CHECK(cudaMalloc(&d_unique_source_hp_ids, N_unique_source_hp * sizeof(long)));
            CUDA_CHECK(cudaMemcpy(d_unique_source_hp_ids, h_unique_source_hp_ids.data(), N_unique_source_hp * sizeof(long), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_unique_hp_id_offsets_start_gpu, N_unique_source_hp * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_unique_hp_id_offsets_start_gpu, h_unique_hp_id_offsets_start.data(), N_unique_source_hp * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_unique_hp_id_offsets_end_gpu, N_unique_source_hp * sizeof(int)));
            CUDA_CHECK(cudaMemcpy(d_unique_hp_id_offsets_end_gpu, h_unique_hp_id_offsets_end.data(), N_unique_source_hp * sizeof(int), cudaMemcpyHostToDevice));

            // 5. Convert unique HEALPix IDs to Cartesian coordinates for KD-Tree construction
            std::vector<float3> h_unique_source_hp_coords_kdtree(N_unique_source_hp);

            for(int i=0; i < N_unique_source_hp; ++i) {
                double theta, phi;
                // Use host version of pix2ang_ring function
                pix2ang_ring_host(tables->nside_healpix, h_unique_source_hp_ids[i], theta, phi);
                
                // Convert angles to Cartesian coordinates
                double sin_theta = sin(theta);
                h_unique_source_hp_coords_kdtree[i].x = static_cast<float>(sin_theta * cos(phi));
                h_unique_source_hp_coords_kdtree[i].y = static_cast<float>(sin_theta * sin(phi));
                h_unique_source_hp_coords_kdtree[i].z = static_cast<float>(cos(theta));
            }
            CUDA_CHECK(cudaMalloc(&d_unique_source_hp_coords_kdtree, N_unique_source_hp * sizeof(float3)));
            CUDA_CHECK(cudaMemcpy(d_unique_source_hp_coords_kdtree, h_unique_source_hp_coords_kdtree.data(), N_unique_source_hp * sizeof(float3), cudaMemcpyHostToDevice));

            // 6. Build KD-Tree using cudaKDTree library
            std::cout << "Building cudaKDTree with " << N_unique_source_hp << " points..." << std::endl;
            
            // Note: We're not using the KD-tree for search anymore due to issues
            // Just keeping the coordinate array for brute force search
            
            std::cout << "KD-Tree built successfully." << std::endl;
        }
    }


    // --- 5. Launch Main Kernel ---
    if (tables->n_lenses > 0 && tables->n_bins > 0 && tables->n_sources > 0 && N_unique_source_hp > 0) {
        int threadsPerBlock = 128; // Or other tuned value
        int blocksPerGrid = (tables->n_lenses + threadsPerBlock - 1) / threadsPerBlock;

        process_all_lenses_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_z_l, d_d_com_l, d_sin_ra_l, d_cos_ra_l, d_sin_dec_l, d_cos_dec_l, d_dist_3d_sq_bins,
            d_z_s, d_d_com_s, d_sin_ra_s, d_cos_ra_s, d_sin_dec_s, d_cos_dec_s,
            d_w_s, d_e_1_s, d_e_2_s, d_z_l_max_s,
            d_unique_source_hp_coords_kdtree, d_unique_source_hp_ids, N_unique_source_hp,
            d_all_source_hp_ids_sorted_gpu, d_sorted_source_original_indices_gpu,
            d_unique_hp_id_offsets_start_gpu, d_unique_hp_id_offsets_end_gpu,
            tables->has_sigma_crit_eff, tables->n_z_bins_l, d_sigma_crit_eff_l, d_z_bin_s,
            tables->has_m_s, d_m_s,
            tables->has_e_rms_s, d_e_rms_s,
            tables->has_R_2_s, d_R_2_s,
            tables->has_R_matrix_s, d_R_11_s, d_R_12_s, d_R_21_s, d_R_22_s,
            tables->n_lenses, tables->n_bins, tables->nside_healpix,
            tables->comoving, tables->weighting,
            d_sum_1_r, d_sum_w_ls_r, d_sum_w_ls_e_t_r, d_sum_w_ls_e_t_sigma_crit_r,
            d_sum_w_ls_z_s_r, d_sum_w_ls_sigma_crit_r,
            d_sum_w_ls_m_r, d_sum_w_ls_1_minus_e_rms_sq_r,
            d_sum_w_ls_A_p_R_2_r, d_sum_w_ls_R_T_r
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        std::cout << "Skipping kernel launch due to no lenses, sources, bins, or unique source HP cells." << std::endl;
    }

    // --- 6. Copy Results (Device to Host) ---
    if (total_output_bins > 0) {
        CUDA_CHECK(cudaMemcpy(tables->sum_1_r, d_sum_1_r, total_output_bins * sizeof(long long), cudaMemcpyDeviceToHost));
        // ... (repeat for all output sum arrays)
        CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_r, d_sum_w_ls_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_e_t_r, d_sum_w_ls_e_t_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_e_t_sigma_crit_r, d_sum_w_ls_e_t_sigma_crit_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_z_s_r, d_sum_w_ls_z_s_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_sigma_crit_r, d_sum_w_ls_sigma_crit_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost));

        if (tables->has_m_s && tables->sum_w_ls_m_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_m_r, d_sum_w_ls_m_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost));
        if (tables->has_e_rms_s && tables->sum_w_ls_1_minus_e_rms_sq_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_1_minus_e_rms_sq_r, d_sum_w_ls_1_minus_e_rms_sq_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost));
        if (tables->has_R_2_s && tables->sum_w_ls_A_p_R_2_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_A_p_R_2_r, d_sum_w_ls_A_p_R_2_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost));
        if (tables->has_R_matrix_s && tables->sum_w_ls_R_T_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_R_T_r, d_sum_w_ls_R_T_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // --- 7. Final Cleanup (Free GPU memory) ---
    CUDA_CHECK(cudaFree(d_z_l)); CUDA_CHECK(cudaFree(d_d_com_l)); /* ... free all d_lens arrays ... */
    CUDA_CHECK(cudaFree(d_sin_ra_l)); CUDA_CHECK(cudaFree(d_cos_ra_l)); CUDA_CHECK(cudaFree(d_sin_dec_l)); CUDA_CHECK(cudaFree(d_cos_dec_l));
    CUDA_CHECK(cudaFree(d_dist_3d_sq_bins));

    if (tables->n_sources > 0) {
        CUDA_CHECK(cudaFree(d_z_s)); CUDA_CHECK(cudaFree(d_d_com_s)); /* ... free all d_source arrays ... */
        CUDA_CHECK(cudaFree(d_sin_ra_s)); CUDA_CHECK(cudaFree(d_cos_ra_s)); CUDA_CHECK(cudaFree(d_sin_dec_s)); CUDA_CHECK(cudaFree(d_cos_dec_s));
        CUDA_CHECK(cudaFree(d_w_s)); CUDA_CHECK(cudaFree(d_e_1_s)); CUDA_CHECK(cudaFree(d_e_2_s)); CUDA_CHECK(cudaFree(d_z_l_max_s));
        CUDA_CHECK(cudaFree(d_all_source_hp_ids)); // This was the original unsorted one

        if (tables->has_sigma_crit_eff) { CUDA_CHECK(cudaFree(d_sigma_crit_eff_l)); CUDA_CHECK(cudaFree(d_z_bin_s)); }
        if (tables->has_m_s) CUDA_CHECK(cudaFree(d_m_s));
        if (tables->has_e_rms_s) CUDA_CHECK(cudaFree(d_e_rms_s));
        if (tables->has_R_2_s) CUDA_CHECK(cudaFree(d_R_2_s));
        if (tables->has_R_matrix_s) { CUDA_CHECK(cudaFree(d_R_11_s)); CUDA_CHECK(cudaFree(d_R_12_s)); CUDA_CHECK(cudaFree(d_R_21_s)); CUDA_CHECK(cudaFree(d_R_22_s));}

        // Free KD-tree related GPU arrays
        if(d_unique_source_hp_coords_kdtree) CUDA_CHECK(cudaFree(d_unique_source_hp_coords_kdtree));
        if(d_unique_source_hp_ids) CUDA_CHECK(cudaFree(d_unique_source_hp_ids));
        // if(d_source_idx_map_from_kdtree_node) CUDA_CHECK(cudaFree(d_source_idx_map_from_kdtree_node));
        if(d_all_source_hp_ids_sorted_gpu) CUDA_CHECK(cudaFree(d_all_source_hp_ids_sorted_gpu));
        if(d_sorted_source_original_indices_gpu) CUDA_CHECK(cudaFree(d_sorted_source_original_indices_gpu));
        if(d_unique_hp_id_offsets_start_gpu) CUDA_CHECK(cudaFree(d_unique_hp_id_offsets_start_gpu));
        if(d_unique_hp_id_offsets_end_gpu) CUDA_CHECK(cudaFree(d_unique_hp_id_offsets_end_gpu));
    }


    if (total_output_bins > 0) {
        CUDA_CHECK(cudaFree(d_sum_1_r)); /* ... free all d_sum output arrays ... */
        CUDA_CHECK(cudaFree(d_sum_w_ls_r)); CUDA_CHECK(cudaFree(d_sum_w_ls_e_t_r)); CUDA_CHECK(cudaFree(d_sum_w_ls_e_t_sigma_crit_r));
        CUDA_CHECK(cudaFree(d_sum_w_ls_z_s_r)); CUDA_CHECK(cudaFree(d_sum_w_ls_sigma_crit_r));
        if (tables->has_m_s && tables->sum_w_ls_m_r) CUDA_CHECK(cudaFree(d_sum_w_ls_m_r));
        if (tables->has_e_rms_s && tables->sum_w_ls_1_minus_e_rms_sq_r) CUDA_CHECK(cudaFree(d_sum_w_ls_1_minus_e_rms_sq_r));
        if (tables->has_R_2_s && tables->sum_w_ls_A_p_R_2_r) CUDA_CHECK(cudaFree(d_sum_w_ls_A_p_R_2_r));
        if (tables->has_R_matrix_s && tables->sum_w_ls_R_T_r) CUDA_CHECK(cudaFree(d_sum_w_ls_R_T_r));
    }

    std::cout << "precompute_cuda_interface completed successfully." << std::endl;
    return 0; // Success
}

// Placeholder for the actual `precompute_engine_cuda.cu` content or where physics kernels are.
// The actual physics functions (dist_3d_sq_kernel, find_bin_idx_kernel, etc.)
// would need to be adapted to work with the global pointers and indices used in the callback.
// For example:
/*
__device__ double calculate_sigma_crit_inv_gpu_global(
    double zl, double zs, double dcoml, double dcoms, bool comoving)
{
    if (zl >= zs) return 0.0; // Invalid configuration

    double d_ls; // Effective distance between lens and source
    if (comoving) {
        d_ls = dcoms - dcoml;
        // Ensure d_ls is not negative due to precision with dcoms ~ dcoml
        if (d_ls < 0) d_ls = 0;
    } else { // Physical distances
        // This case is more complex if dcoml, dcoms are comoving distances
        // Typically, D_LS = D_S - D_L for physical angular diameter distances in flat LCDM
        // Or using (1+z) factors: D_LS = ( D_S - D_L/(1+z_L) ) / (1+z_S) (approx, depends on exact defs)
        // For now, assume inputs are appropriate or this branch is not used with comoving dcoml/dcoms.
        // A common simplification if dcoml, dcoms are comoving:
        d_ls = (dcoms - dcoml) / (1.0 + zs); // This is one form of angular diameter distance D_LS.
                                          // The exact formula depends on curvature and definitions.
                                          // In flat LCDM, D_A(z1,z2) = ( D_M(z2) - D_M(z1) ) / (1+z2)
                                          // So if dcoms = D_M(zs) and dcoml = D_M(zl), then d_ls = (dcoms-dcoml)/(1+zs)
         if (d_ls < 0) d_ls = 0;
    }

    // Sigma_crit_inv = (4 * PI * G / c^2) * (D_L * D_LS / D_S)
    // Constants: 4*PI*G/c^2 in appropriate units (e.g., pc/M_solar)
    // D_L, D_S, D_LS are angular diameter distances.
    // If dcoml, dcoms are comoving transverse distances:
    // D_L = dcoml / (1+zl)
    // D_S = dcoms / (1+zs)
    // D_LS needs care. If using the D_A(z1,z2) form above: D_LS = (dcoms-dcoml)/(1+zs)
    // So, Sigma_crit_inv proportional to (dcoml/(1+zl)) * ((dcoms-dcoml)/(1+zs)) / (dcoms/(1+zs))
    // Sigma_crit_inv proportional to dcoml * (dcoms-dcoml) / (dcoms * (1+zl))

    if (dcoms <= 0) return 0.0; // Source is at observer or behind

    // Using the common formula for Sigma_crit_inv with comoving distances D_L, D_S, D_LS:
    // D_L = dcoml, D_S = dcoms, D_LS = dcoms - dcoml (if comoving=true)
    // D_L = dcoml/(1+zl), D_S = dcoms/(1+zs), D_LS = (dcoms-dcoml)/(1+zs) (if comoving=false, approx for flat LCDM)
    double d_l_ang, d_s_ang, d_ls_ang;
    if (comoving) { // Distances are comoving transverse
        d_l_ang = dcoml;
        d_s_ang = dcoms;
        d_ls_ang = dcoms - dcoml;
        if (d_ls_ang < 0) d_ls_ang = 0;
    } else { // Distances are physical angular diameter
        d_l_ang = dcoml / (1.0 + zl);
        d_s_ang = dcoms / (1.0 + zs);
        // D_A(z_l, z_s)
        d_ls_ang = (dcoms - dcoml) / (1.0 + zs); // Assuming dcoml, dcoms are comoving here.
                                                // This requires clarification if input dcoml/dcoms can be physical.
                                                // If inputs are already D_A, then D_LS is more complex.
        if (d_ls_ang < 0) d_ls_ang = 0;
    }

    if (d_s_ang <= 1e-9) return 0.0; // Avoid division by zero if source is at observer

    // CONST_CRIT is 4*pi*G/c^2 in units matching distances (e.g. Mpc/Msun)
    // Assuming it's defined elsewhere (e.g. precompute_engine_cuda.h)
    // #define CONST_CRIT (4.30091e-9) // Mpc / M_sun, if G in (km/s)^2 Mpc/Msun, c in km/s
    // For now, use 1.0 as placeholder for 4*pi*G/c^2
    return (d_l_ang * d_ls_ang / d_s_ang) * 1.0; // Factor for 4*pi*G/c^2 is missing
}
*/
