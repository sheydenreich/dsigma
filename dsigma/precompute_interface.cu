#include "precompute_interface.h"
#include "precompute_engine_cuda.h" // For launching the kernel and physics
#include "cuda_host_utils.h"      // For Healpix utilities on host
#include "healpix_gpu.h"          // For GPU-side HEALPix functions
#include "../cudaKDTree/cukd/common.h"
#include "../cudaKDTree/cukd/data.h"
#include "../cudaKDTree/cukd/builder.h"
#include "../cudaKDTree/cukd/knn.h"
#include "../cudaKDTree/cukd/box.h"
#include <cfloat>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <map>
#include <cmath>
#include <stdexcept>

// Ensure M_PI is defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Constants for HEALPix
#ifndef TWO_PI
#define TWO_PI (2.0 * M_PI)
#endif
#define TWOTHIRD (2.0 / 3.0)

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
        return -1; \
    } \
}

// Custom data structures for KD-tree with index mapping
struct PointWithIndex {
    float3 coord;
    int original_index;
};

// Define data traits for our custom point type
struct PointWithIndexTraits : public cukd::default_data_traits<float3> {
    using point_t = float3;
    using data_t = PointWithIndex;
    static inline __host__ __device__ const float3& get_point(const PointWithIndex& data) { return data.coord; }
    static inline __host__ __device__ float get_coord(const PointWithIndex& data, int dim) { return cukd::get_coord(data.coord, dim); }
};


//-------------------------------------------------------------------------------------------
// SCALABLE CANDIDATE LIST SOLUTION
//-------------------------------------------------------------------------------------------
struct KnnCandidate {
    float distance_sq;
    int   index;
};

class WorkspaceCandidateList {
private:
    KnnCandidate* m_elements;
    int           m_capacity;
    int           m_count;
    float         m_radius_sq;

public:
    __device__ WorkspaceCandidateList(void* workspace_ptr, int capacity, float search_radius) {
        m_elements = (KnnCandidate*)workspace_ptr;
        m_capacity = capacity;
        m_count = 0;
        m_radius_sq = search_radius * search_radius;
    }
    __device__ inline float initialCullDist2() const { return m_radius_sq; }
    __device__ inline float processCandidate(int pointID, float dist2) {
        if (m_count < m_capacity) {
            m_elements[m_count].distance_sq = dist2;
            m_elements[m_count].index = pointID;
            m_count++;
        }
        return m_radius_sq;
    }
    __device__ inline float returnValue() const { return sqrtf(m_radius_sq); }
    __device__ inline int get_pointID(int i) const { return (i < m_count) ? m_elements[i].index : -1; }
    __device__ inline float get_dist2(int i) const { return (i < m_count) ? m_elements[i].distance_sq : CUDART_INF_F; }
    __device__ inline int get_final_count() const { return m_count; }
};
//-------------------------------------------------------------------------------------------

double get_max_pixrad_gpu_host(long nside) {
    double area = 4*M_PI / (12 * nside * nside);
    return sqrt(area);
}

// Host-side HEALPix utility functions
double acos_safe_host(double val) {
    if (val <= -1.0) return M_PI;
    if (val >= 1.0) return 0.0;
    return acos(val);
}

void pix2ang_ring_host(long nside, long pix, double& theta, double& phi) {
    if (nside <= 0 || pix < 0 || pix >= 12 * nside * nside) { theta = phi = 0.0; return; }
    const long npix = 12 * nside * nside;
    const long ncap = 2 * nside * (nside - 1);
    const long nsidesq = nside * nside;
    const long nl2 = 2 * nside;
    const long nl4 = 4 * nside;
    long ipix1 = pix + 1;
    if (ipix1 <= ncap) {
        double hip = ipix1 / 2.0;
        long iring = (long)(sqrt(hip - sqrt((long)hip))) + 1;
        long iphi = ipix1 - 2 * iring * (iring - 1);
        theta = acos_safe_host(1.0 - iring * iring / (3.0 * nsidesq));
        phi = ((double)iphi - 0.5) * M_PI / (2.0 * iring);
    } else if (ipix1 <= nl2 * (5 * nside + 1)) {
        long ip = ipix1 - ncap - 1;
        long iring = (ip / nl4) + nside;
        long iphi = ip % nl4 + 1;
        double fodd = 0.5 * (1.0 + ((iring + nside) % 2));
        theta = acos_safe_host((nl2 - iring) / (1.5 * nside));
        phi = ((double)iphi - fodd) * M_PI / (2.0 * nside);
    } else {
        long ip = npix - ipix1 + 1;
        double hip = ip / 2.0;
        long iring = (long)(sqrt(hip - sqrt((long)hip))) + 1;
        long iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
        theta = acos_safe_host(-1.0 + iring * iring / (3.0 * nsidesq));
        phi = ((double)iphi - 0.5) * M_PI / (2.0 * iring);
    }
    while (phi < 0.0) phi += TWO_PI;
    while (phi >= TWO_PI) phi -= TWO_PI;
}

// NEW: Host-side function to calculate the needed K for a given lens.
// This allows us to pre-calculate the maximum K needed across all lenses.
int calculate_max_k_host(long nside, double search_radius_sq) {
    if (nside <= 0) return 32; // Default minimum
    double pixel_area = (4.0 * M_PI) / (12.0 * (double)nside * nside);
    double search_area_approx = M_PI * search_radius_sq;
    double theoretical_max = search_area_approx / pixel_area;
    int max_k = static_cast<int>(theoretical_max * 2.0) + 10; // 1.2x safety factor + 10
    return (max_k < 32) ? 32 : max_k;
}


typedef struct { /* ... all fields remain identical to original ... */ 
    int lens_idx; int N_bins; long nside_healpix;
    const long* g_unique_source_hp_ids; const long* g_all_source_hp_ids_sorted;
    const int* g_sorted_source_original_indices; const int* g_unique_hp_id_offsets_start;
    const int* g_unique_hp_id_offsets_end; double lens_zl_i; double lens_dcoml_i;
    double lens_sin_ra_l_i; double lens_cos_ra_l_i; double lens_sin_dec_l_i; double lens_cos_dec_l_i;
    const double* g_z_s; const double* g_d_com_s; const double* g_sin_ra_s; const double* g_cos_ra_s;
    const double* g_sin_dec_s; const double* g_cos_dec_s; const double* g_w_s; const double* g_e_1_s;
    const double* g_e_2_s; const double* g_z_l_max_s; bool has_sigma_crit_eff; int n_z_bins_l;
    const double* g_sigma_crit_eff_l; const int* g_z_bin_s; bool has_m_s; const double* g_m_s;
    bool has_e_rms_s; const double* g_e_rms_s; bool has_R_2_s; const double* g_R_2_s;
    bool has_R_matrix_s; const double* g_R_11_s; const double* g_R_12_s;
    const double* g_R_21_s; const double* g_R_22_s; const double* g_dist_3d_sq_bins;
    bool comoving; int weighting; long long* g_sum_1_r; double* g_sum_w_ls_r;
    double* g_sum_w_ls_e_t_r; double* g_sum_w_ls_e_t_sigma_crit_r;
    double* g_sum_w_ls_z_s_r; double* g_sum_w_ls_sigma_crit_r; double* g_sum_w_ls_m_r;
    double* g_sum_w_ls_1_minus_e_rms_sq_r; double* g_sum_w_ls_A_p_R_2_r;
    double* g_sum_w_ls_R_T_r;
} KernelCallbackData;

__device__ void process_found_source_hp_pixel(int source_kdtree_idx, KernelCallbackData* cb_data) {
    int start_offset = cb_data->g_unique_hp_id_offsets_start[source_kdtree_idx];
    int end_offset = cb_data->g_unique_hp_id_offsets_end[source_kdtree_idx];
    for (int i_s_mapped_idx = start_offset; i_s_mapped_idx < end_offset; ++i_s_mapped_idx) {
        int original_source_idx = cb_data->g_sorted_source_original_indices[i_s_mapped_idx];
        double zs_i = cb_data->g_z_s[original_source_idx];
        if (cb_data->lens_zl_i >= zs_i || cb_data->lens_zl_i >= cb_data->g_z_l_max_s[original_source_idx]) continue;
        double dcoms_i = cb_data->g_d_com_s[original_source_idx];
        double sin_ra_s_i = cb_data->g_sin_ra_s[original_source_idx]; double cos_ra_s_i = cb_data->g_cos_ra_s[original_source_idx];
        double sin_dec_s_i = cb_data->g_sin_dec_s[original_source_idx]; double cos_dec_s_i = cb_data->g_cos_dec_s[original_source_idx];
        double dist_sq = dist_angular_sq_gpu(cb_data->lens_sin_ra_l_i, cb_data->lens_cos_ra_l_i, cb_data->lens_sin_dec_l_i, cb_data->lens_cos_dec_l_i, sin_ra_s_i, cos_ra_s_i, sin_dec_s_i, cos_dec_s_i);
        const double* current_lens_dist_bins = cb_data->g_dist_3d_sq_bins + (size_t)cb_data->lens_idx * (cb_data->N_bins + 1);
        int i_bin = find_bin_idx_gpu(dist_sq, current_lens_dist_bins, cb_data->N_bins);
        if (i_bin == -1) continue;
        int z_bin_s_val = (cb_data->has_sigma_crit_eff && cb_data->g_z_bin_s != nullptr) ? cb_data->g_z_bin_s[original_source_idx] : -1;
        double sigma_crit_inv = calculate_sigma_crit_inv_gpu(cb_data->lens_zl_i, zs_i, cb_data->lens_dcoml_i, dcoms_i, cb_data->comoving, cb_data->has_sigma_crit_eff, cb_data->g_sigma_crit_eff_l, cb_data->lens_idx, cb_data->n_z_bins_l, z_bin_s_val);
        if (sigma_crit_inv == 0.0 || sigma_crit_inv == DBL_MAX) continue;
        double w_ls = calculate_w_ls_gpu(sigma_crit_inv, cb_data->g_w_s[original_source_idx], cb_data->weighting);
        if (w_ls == 0.0) continue;
        double cos_2phi, sin_2phi;
        calculate_et_components_gpu(cb_data->lens_sin_ra_l_i, cb_data->lens_cos_ra_l_i, cb_data->lens_sin_dec_l_i, cb_data->lens_cos_dec_l_i, sin_ra_s_i, cos_ra_s_i, sin_dec_s_i, cos_dec_s_i, cos_2phi, sin_2phi);
        double e_t_val = calculate_et_gpu(cb_data->g_e_1_s[original_source_idx], cb_data->g_e_2_s[original_source_idx], cos_2phi, sin_2phi);
        size_t out_idx = (size_t)cb_data->lens_idx * cb_data->N_bins + i_bin;
        atomicAdd((unsigned long long int*)&cb_data->g_sum_1_r[out_idx], 1ULL);
        atomicAdd(&cb_data->g_sum_w_ls_r[out_idx], w_ls);
        atomicAdd(&cb_data->g_sum_w_ls_e_t_r[out_idx], w_ls * e_t_val);
        double sigma_crit = (sigma_crit_inv == 0.0) ? DBL_MAX : 1.0 / sigma_crit_inv;
        atomicAdd(&cb_data->g_sum_w_ls_e_t_sigma_crit_r[out_idx], w_ls * e_t_val * sigma_crit);
        atomicAdd(&cb_data->g_sum_w_ls_z_s_r[out_idx], w_ls * zs_i);
        atomicAdd(&cb_data->g_sum_w_ls_sigma_crit_r[out_idx], w_ls * sigma_crit);
        if (cb_data->has_m_s && cb_data->g_m_s && cb_data->g_sum_w_ls_m_r) atomicAdd(&cb_data->g_sum_w_ls_m_r[out_idx], w_ls * cb_data->g_m_s[original_source_idx]);
        if (cb_data->has_e_rms_s && cb_data->g_e_rms_s && cb_data->g_sum_w_ls_1_minus_e_rms_sq_r) { double e_rms_s_i = cb_data->g_e_rms_s[original_source_idx]; atomicAdd(&cb_data->g_sum_w_ls_1_minus_e_rms_sq_r[out_idx], w_ls * (1.0 - e_rms_s_i * e_rms_s_i)); }
        if (cb_data->has_R_2_s && cb_data->g_R_2_s && cb_data->g_sum_w_ls_A_p_R_2_r) { if (cb_data->g_R_2_s[original_source_idx] <= 0.31) atomicAdd(&cb_data->g_sum_w_ls_A_p_R_2_r[out_idx], 0.00865 * w_ls / 0.01); }
        if (cb_data->has_R_matrix_s && cb_data->g_R_11_s && cb_data->g_sum_w_ls_R_T_r) { double R_T_val = calculate_R_T_gpu(cb_data->g_R_11_s[original_source_idx], cb_data->g_R_12_s[original_source_idx], cb_data->g_R_21_s[original_source_idx], cb_data->g_R_22_s[original_source_idx], cos_2phi, sin_2phi); atomicAdd(&cb_data->g_sum_w_ls_R_T_r[out_idx], w_ls * R_T_val); }
    }
}

__device__ void kdtree_radius_search_and_process(
    float3 query_point, float search_radius_sq,
    const float3* d_kdtree_points, int num_kdtree_points,
    const cukd::box_t<float3>* d_world_bounds, const int* d_kdtree_to_original_mapping,
    KernelCallbackData* callback_data, void* g_knn_workspace, int max_k_per_thread
) {
    void* my_workspace = (char*)g_knn_workspace + ((size_t)callback_data->lens_idx * max_k_per_thread * sizeof(KnnCandidate));
    WorkspaceCandidateList candidates(my_workspace, max_k_per_thread, sqrtf(search_radius_sq));
    cukd::stackBased::knn(candidates, query_point, *d_world_bounds, d_kdtree_points, num_kdtree_points);
    
    int processed_count = candidates.get_final_count();
    for (int i = 0; i < processed_count; i++) {
        int kdtree_index = candidates.get_pointID(i);
        if (kdtree_index >= 0) {
             int original_index = d_kdtree_to_original_mapping[kdtree_index];
             process_found_source_hp_pixel(original_index, callback_data);
        }
    }

    if (processed_count >= max_k_per_thread) {
        printf("FATAL ERROR: Lens %d (zl=%.4f) exhausted its candidate buffer of %d!\n",
               callback_data->lens_idx, callback_data->lens_zl_i, max_k_per_thread);
        printf("SOLUTION: Increase MAX_K_PER_THREAD in precompute_cuda_interface() in the host code.\n");
    }
}

__global__ void process_all_lenses_kernel(
    const double* g_z_l, const double* g_d_com_l, const double* g_sin_ra_l, const double* g_cos_ra_l, const double* g_sin_dec_l, const double* g_cos_dec_l, const double* g_dist_3d_sq_bins,
    const double* g_z_s, const double* g_d_com_s, const double* g_sin_ra_s, const double* g_cos_ra_s, const double* g_sin_dec_s, const double* g_cos_dec_s,
    const double* g_w_s, const double* g_e_1_s, const double* g_e_2_s, const double* g_z_l_max_s,
    const float3* g_unique_source_hp_coords_kdtree, const long* g_unique_source_hp_ids, int N_unique_source_hp, const cukd::box_t<float3>* g_world_bounds, const int* g_kdtree_to_original_mapping,
    const long* g_all_source_hp_ids_sorted, const int* g_sorted_source_original_indices, const int* g_unique_hp_id_offsets_start, const int* g_unique_hp_id_offsets_end,
    bool _has_sigma_crit_eff, int _n_z_bins_l, const double* _g_sigma_crit_eff_l, const int* _g_z_bin_s,
    bool _has_m_s, const double* _g_m_s, bool _has_e_rms_s, const double* _g_e_rms_s, bool _has_R_2_s, const double* _g_R_2_s,
    bool _has_R_matrix_s, const double* _g_R_11_s, const double* _g_R_12_s, const double* _g_R_21_s, const double* _g_R_22_s,
    int N_lenses, int N_bins, long nside_healpix, bool comoving, int weighting,
    void* g_knn_workspace, int max_k_per_thread,
    long long* g_sum_1_r, double* g_sum_w_ls_r, double* g_sum_w_ls_e_t_r, double* g_sum_w_ls_e_t_sigma_crit_r,
    double* g_sum_w_ls_z_s_r, double* g_sum_w_ls_sigma_crit_r, double* g_sum_w_ls_m_r,
    double* g_sum_w_ls_1_minus_e_rms_sq_r, double* g_sum_w_ls_A_p_R_2_r, double* g_sum_w_ls_R_T_r
) {
    int lens_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (lens_idx >= N_lenses) return;

    double zl_i = g_z_l[lens_idx]; double dcoml_i = g_d_com_l[lens_idx];
    double sin_ra_l_i = g_sin_ra_l[lens_idx]; double cos_ra_l_i = g_cos_ra_l[lens_idx];
    double sin_dec_l_i = g_sin_dec_l[lens_idx]; double cos_dec_l_i = g_cos_dec_l[lens_idx];
    float3 lens_xyz_cartesian;
    spherical_to_cartesian_gpu(sin_ra_l_i, cos_ra_l_i, sin_dec_l_i, cos_dec_l_i, lens_xyz_cartesian);
    double max_dist_3d_sq_lens = g_dist_3d_sq_bins[lens_idx * (N_bins + 1) + N_bins];
    double lens_hp_pixrad = get_max_pixrad_gpu(nside_healpix);
    double pixrad_3d_sq = 4.0 * sin(lens_hp_pixrad * 0.5) * sin(lens_hp_pixrad * 0.5);
    double search_radius_sq_gpu = max_dist_3d_sq_lens + (4.0 * pixrad_3d_sq + 4.0 * sqrt(max_dist_3d_sq_lens) * sqrt(pixrad_3d_sq));
    float search_radius_sq = (float)search_radius_sq_gpu;

    KernelCallbackData callback_data;
    callback_data.lens_idx = lens_idx; callback_data.N_bins = N_bins; callback_data.nside_healpix = nside_healpix;
    callback_data.g_unique_source_hp_ids = g_unique_source_hp_ids; callback_data.g_all_source_hp_ids_sorted = g_all_source_hp_ids_sorted;
    callback_data.g_sorted_source_original_indices = g_sorted_source_original_indices; callback_data.g_unique_hp_id_offsets_start = g_unique_hp_id_offsets_start;
    callback_data.g_unique_hp_id_offsets_end = g_unique_hp_id_offsets_end; callback_data.lens_zl_i = zl_i; callback_data.lens_dcoml_i = dcoml_i;
    callback_data.lens_sin_ra_l_i = sin_ra_l_i; callback_data.lens_cos_ra_l_i = cos_ra_l_i; callback_data.lens_sin_dec_l_i = sin_dec_l_i; callback_data.lens_cos_dec_l_i = cos_dec_l_i;
    callback_data.g_z_s = g_z_s; callback_data.g_d_com_s = g_d_com_s; callback_data.g_sin_ra_s = g_sin_ra_s; callback_data.g_cos_ra_s = g_cos_ra_s;
    callback_data.g_sin_dec_s = g_sin_dec_s; callback_data.g_cos_dec_s = g_cos_dec_s; callback_data.g_w_s = g_w_s; callback_data.g_e_1_s = g_e_1_s;
    callback_data.g_e_2_s = g_e_2_s; callback_data.g_z_l_max_s = g_z_l_max_s; callback_data.has_sigma_crit_eff = _has_sigma_crit_eff; callback_data.n_z_bins_l = _n_z_bins_l;
    callback_data.g_sigma_crit_eff_l = _g_sigma_crit_eff_l; callback_data.g_z_bin_s = _g_z_bin_s; callback_data.has_m_s = _has_m_s; callback_data.g_m_s = _g_m_s;
    callback_data.has_e_rms_s = _has_e_rms_s; callback_data.g_e_rms_s = _g_e_rms_s; callback_data.has_R_2_s = _has_R_2_s; callback_data.g_R_2_s = _g_R_2_s;
    callback_data.has_R_matrix_s = _has_R_matrix_s; callback_data.g_R_11_s = _g_R_11_s; callback_data.g_R_12_s = _g_R_12_s;
    callback_data.g_R_21_s = _g_R_21_s; callback_data.g_R_22_s = _g_R_22_s; callback_data.g_dist_3d_sq_bins = g_dist_3d_sq_bins;
    callback_data.comoving = comoving; callback_data.weighting = weighting; callback_data.g_sum_1_r = g_sum_1_r; callback_data.g_sum_w_ls_r = g_sum_w_ls_r;
    callback_data.g_sum_w_ls_e_t_r = g_sum_w_ls_e_t_r; callback_data.g_sum_w_ls_e_t_sigma_crit_r = g_sum_w_ls_e_t_sigma_crit_r;
    callback_data.g_sum_w_ls_z_s_r = g_sum_w_ls_z_s_r; callback_data.g_sum_w_ls_sigma_crit_r = g_sum_w_ls_sigma_crit_r; callback_data.g_sum_w_ls_m_r = g_sum_w_ls_m_r;
    callback_data.g_sum_w_ls_1_minus_e_rms_sq_r = g_sum_w_ls_1_minus_e_rms_sq_r; callback_data.g_sum_w_ls_A_p_R_2_r = g_sum_w_ls_A_p_R_2_r;
    callback_data.g_sum_w_ls_R_T_r = g_sum_w_ls_R_T_r;

    kdtree_radius_search_and_process(
        lens_xyz_cartesian, search_radius_sq,
        g_unique_source_hp_coords_kdtree, N_unique_source_hp, g_world_bounds,
        g_kdtree_to_original_mapping, &callback_data,
        g_knn_workspace, max_k_per_thread
    );
}

int precompute_cuda_interface(TableData* tables, int n_gpus) {
    // --- 1. Input Validation (unchanged) ---
    if (!tables->z_l || !tables->d_com_l || !tables->sin_ra_l || !tables->cos_ra_l || !tables->sin_dec_l || !tables->cos_dec_l || !tables->z_s || !tables->d_com_s || !tables->sin_ra_s || !tables->cos_ra_s || !tables->sin_dec_s || !tables->cos_dec_s || !tables->w_s || !tables->e_1_s || !tables->e_2_s || !tables->z_l_max_s || !tables->healpix_id_s || !tables->dist_3d_sq_bins || !tables->sum_1_r || !tables->sum_w_ls_r || !tables->sum_w_ls_e_t_r || !tables->sum_w_ls_e_t_sigma_crit_r || !tables->sum_w_ls_z_s_r || !tables->sum_w_ls_sigma_crit_r) { std::cerr << "Error: Essential data pointers in TableData are null." << std::endl; return -1; }
    if (tables->has_sigma_crit_eff && (!tables->sigma_crit_eff_l || !tables->z_bin_s)) { std::cerr << "Error: sigma_crit_eff pointers are null." << std::endl; return -1; }
    if (tables->has_m_s && !tables->m_s) { std::cerr << "Error: m_s is null." << std::endl; return -1; }
    if (tables->n_lenses <= 0 && tables->n_sources == 0) { std::cout << "No lenses or sources. Exiting." << std::endl; return 0; }
    if (tables->n_lenses > 0 && tables->n_bins <= 0) { std::cerr << "Error: n_bins must be positive." << std::endl; return -1; }
    if (tables->nside_healpix <= 0 && (tables->n_lenses > 0 || tables->n_sources > 0)) { std::cerr << "Error: nside_healpix must be positive." << std::endl; return -1; }

    CUDA_CHECK(cudaSetDevice(0));

    // --- 3. GPU Data Allocation and H2D Transfers ---
    // Pointers are declared here
    double *d_z_l, *d_d_com_l, *d_sin_ra_l, *d_cos_ra_l, *d_sin_dec_l, *d_cos_dec_l, *d_dist_3d_sq_bins;
    double *d_z_s, *d_d_com_s, *d_sin_ra_s, *d_cos_ra_s, *d_sin_dec_s, *d_cos_dec_s, *d_w_s, *d_e_1_s, *d_e_2_s, *d_z_l_max_s;
    long* d_all_source_hp_ids;
    double *d_sigma_crit_eff_l = nullptr; int *d_z_bin_s = nullptr;
    double *d_m_s = nullptr, *d_e_rms_s = nullptr, *d_R_2_s = nullptr;
    double *d_R_11_s = nullptr, *d_R_12_s = nullptr, *d_R_21_s = nullptr, *d_R_22_s = nullptr;
    long long* d_sum_1_r;
    double *d_sum_w_ls_r, *d_sum_w_ls_e_t_r, *d_sum_w_ls_e_t_sigma_crit_r, *d_sum_w_ls_z_s_r, *d_sum_w_ls_sigma_crit_r;
    double *d_sum_w_ls_m_r = nullptr, *d_sum_w_ls_1_minus_e_rms_sq_r = nullptr, *d_sum_w_ls_A_p_R_2_r = nullptr, *d_sum_w_ls_R_T_r = nullptr;
    
    // ... Bulk of cudaMalloc/cudaMemcpy remains the same ...
    CUDA_CHECK(cudaMalloc(&d_z_l, tables->n_lenses * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_z_l, tables->z_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_d_com_l, tables->n_lenses * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_d_com_l, tables->d_com_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_sin_ra_l, tables->n_lenses * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_sin_ra_l, tables->sin_ra_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_cos_ra_l, tables->n_lenses * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_cos_ra_l, tables->cos_ra_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_sin_dec_l, tables->n_lenses * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_sin_dec_l, tables->sin_dec_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_cos_dec_l, tables->n_lenses * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_cos_dec_l, tables->cos_dec_l, tables->n_lenses * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_dist_3d_sq_bins, (size_t)tables->n_lenses * (tables->n_bins + 1) * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_dist_3d_sq_bins, tables->dist_3d_sq_bins, (size_t)tables->n_lenses * (tables->n_bins + 1) * sizeof(double), cudaMemcpyHostToDevice));
    if (tables->n_sources > 0) { CUDA_CHECK(cudaMalloc(&d_z_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_z_s, tables->z_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_d_com_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_d_com_s, tables->d_com_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_sin_ra_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_sin_ra_s, tables->sin_ra_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_cos_ra_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_cos_ra_s, tables->cos_ra_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_sin_dec_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_sin_dec_s, tables->sin_dec_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_cos_dec_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_cos_dec_s, tables->cos_dec_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_w_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_w_s, tables->w_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_e_1_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_e_1_s, tables->e_1_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_e_2_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_e_2_s, tables->e_2_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_z_l_max_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_z_l_max_s, tables->z_l_max_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_all_source_hp_ids, tables->n_sources * sizeof(long))); CUDA_CHECK(cudaMemcpy(d_all_source_hp_ids, tables->healpix_id_s, tables->n_sources * sizeof(long), cudaMemcpyHostToDevice)); if (tables->has_sigma_crit_eff) { CUDA_CHECK(cudaMalloc(&d_sigma_crit_eff_l, (size_t)tables->n_lenses * tables->n_z_bins_l * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_sigma_crit_eff_l, tables->sigma_crit_eff_l, (size_t)tables->n_lenses * tables->n_z_bins_l * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_z_bin_s, tables->n_sources * sizeof(int))); CUDA_CHECK(cudaMemcpy(d_z_bin_s, tables->z_bin_s, tables->n_sources * sizeof(int), cudaMemcpyHostToDevice)); } if (tables->has_m_s) { CUDA_CHECK(cudaMalloc(&d_m_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_m_s, tables->m_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); } if (tables->has_e_rms_s) { CUDA_CHECK(cudaMalloc(&d_e_rms_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_e_rms_s, tables->e_rms_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); } if (tables->has_R_2_s) { CUDA_CHECK(cudaMalloc(&d_R_2_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_R_2_s, tables->R_2_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); } if (tables->has_R_matrix_s) { CUDA_CHECK(cudaMalloc(&d_R_11_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_R_11_s, tables->R_11_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_R_12_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_R_12_s, tables->R_12_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_R_21_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_R_21_s, tables->R_21_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMalloc(&d_R_22_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpy(d_R_22_s, tables->R_22_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice)); } }
    size_t total_output_bins = (size_t)tables->n_lenses * tables->n_bins; if (total_output_bins > 0) { CUDA_CHECK(cudaMalloc(&d_sum_1_r, total_output_bins * sizeof(long long))); CUDA_CHECK(cudaMemset(d_sum_1_r, 0, total_output_bins * sizeof(long long))); CUDA_CHECK(cudaMalloc(&d_sum_w_ls_r, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMemset(d_sum_w_ls_r, 0, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_sum_w_ls_e_t_r, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMemset(d_sum_w_ls_e_t_r, 0, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_sum_w_ls_e_t_sigma_crit_r, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMemset(d_sum_w_ls_e_t_sigma_crit_r, 0, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_sum_w_ls_z_s_r, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMemset(d_sum_w_ls_z_s_r, 0, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_sum_w_ls_sigma_crit_r, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMemset(d_sum_w_ls_sigma_crit_r, 0, total_output_bins * sizeof(double))); if (tables->has_m_s && tables->sum_w_ls_m_r) { CUDA_CHECK(cudaMalloc(&d_sum_w_ls_m_r, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMemset(d_sum_w_ls_m_r, 0, total_output_bins * sizeof(double))); } if (tables->has_e_rms_s && tables->sum_w_ls_1_minus_e_rms_sq_r) { CUDA_CHECK(cudaMalloc(&d_sum_w_ls_1_minus_e_rms_sq_r, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMemset(d_sum_w_ls_1_minus_e_rms_sq_r, 0, total_output_bins * sizeof(double))); } if (tables->has_R_2_s && tables->sum_w_ls_A_p_R_2_r) { CUDA_CHECK(cudaMalloc(&d_sum_w_ls_A_p_R_2_r, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMemset(d_sum_w_ls_A_p_R_2_r, 0, total_output_bins * sizeof(double))); } if (tables->has_R_matrix_s && tables->sum_w_ls_R_T_r) { CUDA_CHECK(cudaMalloc(&d_sum_w_ls_R_T_r, total_output_bins * sizeof(double))); CUDA_CHECK(cudaMemset(d_sum_w_ls_R_T_r, 0, total_output_bins * sizeof(double))); } } else if (tables->n_lenses == 0) { return 0; }
    
    // --- 4. GPU-Side Data Preparation for KD-Tree ---
    float3* d_unique_source_hp_coords_kdtree = nullptr; long* d_unique_source_hp_ids = nullptr;
    int N_unique_source_hp = 0; cukd::box_t<float3>* d_world_bounds = nullptr;
    int* d_kdtree_to_original_mapping = nullptr; long* d_all_source_hp_ids_sorted_gpu = nullptr;
    int* d_sorted_source_original_indices_gpu = nullptr; int* d_unique_hp_id_offsets_start_gpu = nullptr; int* d_unique_hp_id_offsets_end_gpu = nullptr;
    if (tables->n_sources > 0) {
        std::vector<std::pair<long, int>> source_hp_pairs(tables->n_sources); for(int i=0; i < tables->n_sources; ++i) source_hp_pairs[i] = {tables->healpix_id_s[i], i};
        std::sort(source_hp_pairs.begin(), source_hp_pairs.end());
        std::vector<long> h_all_source_hp_ids_sorted(tables->n_sources); std::vector<int> h_sorted_source_original_indices(tables->n_sources);
        for(int i=0; i < tables->n_sources; ++i) { h_all_source_hp_ids_sorted[i] = source_hp_pairs[i].first; h_sorted_source_original_indices[i] = source_hp_pairs[i].second; }
        CUDA_CHECK(cudaMalloc(&d_all_source_hp_ids_sorted_gpu, tables->n_sources * sizeof(long))); CUDA_CHECK(cudaMemcpy(d_all_source_hp_ids_sorted_gpu, h_all_source_hp_ids_sorted.data(), tables->n_sources * sizeof(long), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_sorted_source_original_indices_gpu, tables->n_sources * sizeof(int))); CUDA_CHECK(cudaMemcpy(d_sorted_source_original_indices_gpu, h_sorted_source_original_indices.data(), tables->n_sources * sizeof(int), cudaMemcpyHostToDevice));
        std::vector<long> h_unique_source_hp_ids; std::vector<int> h_unique_hp_id_offsets_start; std::vector<int> h_unique_hp_id_offsets_end;
        if (tables->n_sources > 0) { h_unique_source_hp_ids.push_back(h_all_source_hp_ids_sorted[0]); h_unique_hp_id_offsets_start.push_back(0); for (int i = 1; i < tables->n_sources; ++i) if (h_all_source_hp_ids_sorted[i] != h_all_source_hp_ids_sorted[i-1]) { h_unique_source_hp_ids.push_back(h_all_source_hp_ids_sorted[i]); h_unique_hp_id_offsets_end.push_back(i); h_unique_hp_id_offsets_start.push_back(i); } h_unique_hp_id_offsets_end.push_back(tables->n_sources); }
        N_unique_source_hp = h_unique_source_hp_ids.size();
        if (N_unique_source_hp > 0) {
            CUDA_CHECK(cudaMalloc(&d_unique_source_hp_ids, N_unique_source_hp * sizeof(long))); CUDA_CHECK(cudaMemcpy(d_unique_source_hp_ids, h_unique_source_hp_ids.data(), N_unique_source_hp * sizeof(long), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_unique_hp_id_offsets_start_gpu, N_unique_source_hp * sizeof(int))); CUDA_CHECK(cudaMemcpy(d_unique_hp_id_offsets_start_gpu, h_unique_hp_id_offsets_start.data(), N_unique_source_hp * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_unique_hp_id_offsets_end_gpu, N_unique_source_hp * sizeof(int))); CUDA_CHECK(cudaMemcpy(d_unique_hp_id_offsets_end_gpu, h_unique_hp_id_offsets_end.data(), N_unique_source_hp * sizeof(int), cudaMemcpyHostToDevice));
            std::vector<PointWithIndex> h_points_with_indices(N_unique_source_hp);
            for(int i=0; i < N_unique_source_hp; ++i) { double theta, phi; pix2ang_ring_host(tables->nside_healpix, h_unique_source_hp_ids[i], theta, phi); double sin_theta = sin(theta); h_points_with_indices[i].coord.x = static_cast<float>(sin_theta * cos(phi)); h_points_with_indices[i].coord.y = static_cast<float>(sin_theta * sin(phi)); h_points_with_indices[i].coord.z = static_cast<float>(cos(theta)); h_points_with_indices[i].original_index = i; }
            PointWithIndex* d_points_with_indices; CUDA_CHECK(cudaMalloc(&d_points_with_indices, N_unique_source_hp * sizeof(PointWithIndex))); CUDA_CHECK(cudaMemcpy(d_points_with_indices, h_points_with_indices.data(), N_unique_source_hp * sizeof(PointWithIndex), cudaMemcpyHostToDevice));
            std::cout << "Building cudaKDTree..." << std::endl;
            CUDA_CHECK(cudaMalloc(&d_world_bounds, sizeof(cukd::box_t<float3>))); cukd::buildTree<PointWithIndex, PointWithIndexTraits>(d_points_with_indices, N_unique_source_hp, d_world_bounds);
            std::vector<float3> h_unique_source_hp_coords_kdtree(N_unique_source_hp); std::vector<int> h_kdtree_to_original_mapping(N_unique_source_hp);
            std::vector<PointWithIndex> h_reordered_points(N_unique_source_hp);
            CUDA_CHECK(cudaMemcpy(h_reordered_points.data(), d_points_with_indices, N_unique_source_hp * sizeof(PointWithIndex), cudaMemcpyDeviceToHost));
            for(int i=0; i < N_unique_source_hp; ++i) { h_unique_source_hp_coords_kdtree[i] = h_reordered_points[i].coord; h_kdtree_to_original_mapping[i] = h_reordered_points[i].original_index; }
            CUDA_CHECK(cudaMalloc(&d_unique_source_hp_coords_kdtree, N_unique_source_hp * sizeof(float3))); CUDA_CHECK(cudaMemcpy(d_unique_source_hp_coords_kdtree, h_unique_source_hp_coords_kdtree.data(), N_unique_source_hp * sizeof(float3), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMalloc(&d_kdtree_to_original_mapping, N_unique_source_hp * sizeof(int))); CUDA_CHECK(cudaMemcpy(d_kdtree_to_original_mapping, h_kdtree_to_original_mapping.data(), N_unique_source_hp * sizeof(int), cudaMemcpyHostToDevice));
            cudaFree(d_points_with_indices); std::cout << "KD-Tree built successfully." << std::endl;
        }
    }
    
    // >>> NEW: Calculate max K needed and allocate the workspace <<<
    int max_k_for_workspace = 32; // Default minimum
    if (tables->n_lenses > 0 && tables->n_sources > 0) {
        double lens_hp_pixrad = get_max_pixrad_gpu_host(tables->nside_healpix);
        double pixrad_3d_sq = 4.0 * sin(lens_hp_pixrad * 0.5) * sin(lens_hp_pixrad * 0.5);
        for (int i = 0; i < tables->n_lenses; ++i) {
            double max_dist_3d_sq_lens = tables->dist_3d_sq_bins[i * (tables->n_bins + 1) + tables->n_bins];
            double search_radius_sq_gpu = max_dist_3d_sq_lens + (4.0 * pixrad_3d_sq + 4.0 * sqrt(max_dist_3d_sq_lens) * sqrt(pixrad_3d_sq));
            int required_k = calculate_max_k_host(tables->nside_healpix, search_radius_sq_gpu);
            if (required_k > max_k_for_workspace) {
                max_k_for_workspace = required_k;
            }
        }
    }
    
    void* d_knn_workspace = nullptr;
    if (tables->n_lenses > 0 && tables->n_sources > 0) {
        size_t workspace_size = (size_t)tables->n_lenses * max_k_for_workspace * sizeof(KnnCandidate);
        std::cout << "Max K estimated to be " << max_k_for_workspace << " per lens." << std::endl;
        std::cout << "Allocating KNN workspace for " << tables->n_lenses << " lenses (" << workspace_size / (1024.0 * 1024.0 * 1024.0) << " GB)..." << std::endl;
        CUDA_CHECK(cudaMalloc(&d_knn_workspace, workspace_size));
    }

    // --- 5. Launch Main Kernel ---
    if (tables->n_lenses > 0 && tables->n_bins > 0 && tables->n_sources > 0 && N_unique_source_hp > 0) {
        int threadsPerBlock = 128;
        int blocksPerGrid = (tables->n_lenses + threadsPerBlock - 1) / threadsPerBlock;

        process_all_lenses_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_z_l, d_d_com_l, d_sin_ra_l, d_cos_ra_l, d_sin_dec_l, d_cos_dec_l, d_dist_3d_sq_bins,
            d_z_s, d_d_com_s, d_sin_ra_s, d_cos_ra_s, d_sin_dec_s, d_cos_dec_s,
            d_w_s, d_e_1_s, d_e_2_s, d_z_l_max_s,
            d_unique_source_hp_coords_kdtree, d_unique_source_hp_ids, N_unique_source_hp, d_world_bounds, d_kdtree_to_original_mapping,
            d_all_source_hp_ids_sorted_gpu, d_sorted_source_original_indices_gpu,
            d_unique_hp_id_offsets_start_gpu, d_unique_hp_id_offsets_end_gpu,
            tables->has_sigma_crit_eff, tables->n_z_bins_l, d_sigma_crit_eff_l, d_z_bin_s,
            tables->has_m_s, d_m_s, tables->has_e_rms_s, d_e_rms_s, tables->has_R_2_s, d_R_2_s,
            tables->has_R_matrix_s, d_R_11_s, d_R_12_s, d_R_21_s, d_R_22_s,
            tables->n_lenses, tables->n_bins, tables->nside_healpix, tables->comoving, tables->weighting,
            d_knn_workspace, max_k_for_workspace, // Pass the dynamically calculated max K
            d_sum_1_r, d_sum_w_ls_r, d_sum_w_ls_e_t_r, d_sum_w_ls_e_t_sigma_crit_r,
            d_sum_w_ls_z_s_r, d_sum_w_ls_sigma_crit_r, d_sum_w_ls_m_r,
            d_sum_w_ls_1_minus_e_rms_sq_r, d_sum_w_ls_A_p_R_2_r, d_sum_w_ls_R_T_r
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        std::cout << "Skipping kernel launch..." << std::endl;
    }

    // --- 6. Copy Results (Device to Host) ---
    if (total_output_bins > 0) {
        // ... D2H copies remain the same ...
        CUDA_CHECK(cudaMemcpy(tables->sum_1_r, d_sum_1_r, total_output_bins * sizeof(long long), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_r, d_sum_w_ls_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_e_t_r, d_sum_w_ls_e_t_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_e_t_sigma_crit_r, d_sum_w_ls_e_t_sigma_crit_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_z_s_r, d_sum_w_ls_z_s_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_sigma_crit_r, d_sum_w_ls_sigma_crit_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost)); if (tables->has_m_s && tables->sum_w_ls_m_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_m_r, d_sum_w_ls_m_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost)); if (tables->has_e_rms_s && tables->sum_w_ls_1_minus_e_rms_sq_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_1_minus_e_rms_sq_r, d_sum_w_ls_1_minus_e_rms_sq_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost)); if (tables->has_R_2_s && tables->sum_w_ls_A_p_R_2_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_A_p_R_2_r, d_sum_w_ls_A_p_R_2_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost)); if (tables->has_R_matrix_s && tables->sum_w_ls_R_T_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_R_T_r, d_sum_w_ls_R_T_r, total_output_bins * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // --- 7. Final Cleanup ---
    if(d_knn_workspace) CUDA_CHECK(cudaFree(d_knn_workspace));
    // ... all other cudaFrees remain the same ...
    CUDA_CHECK(cudaFree(d_z_l)); CUDA_CHECK(cudaFree(d_d_com_l)); CUDA_CHECK(cudaFree(d_sin_ra_l)); CUDA_CHECK(cudaFree(d_cos_ra_l)); CUDA_CHECK(cudaFree(d_sin_dec_l)); CUDA_CHECK(cudaFree(d_cos_dec_l)); CUDA_CHECK(cudaFree(d_dist_3d_sq_bins));
    if (tables->n_sources > 0) {
        CUDA_CHECK(cudaFree(d_z_s)); CUDA_CHECK(cudaFree(d_d_com_s)); CUDA_CHECK(cudaFree(d_sin_ra_s)); CUDA_CHECK(cudaFree(d_cos_ra_s)); CUDA_CHECK(cudaFree(d_sin_dec_s)); CUDA_CHECK(cudaFree(d_cos_dec_s)); CUDA_CHECK(cudaFree(d_w_s)); CUDA_CHECK(cudaFree(d_e_1_s)); CUDA_CHECK(cudaFree(d_e_2_s)); CUDA_CHECK(cudaFree(d_z_l_max_s)); CUDA_CHECK(cudaFree(d_all_source_hp_ids));
        if (tables->has_sigma_crit_eff) { CUDA_CHECK(cudaFree(d_sigma_crit_eff_l)); CUDA_CHECK(cudaFree(d_z_bin_s)); }
        if (tables->has_m_s) CUDA_CHECK(cudaFree(d_m_s)); if (tables->has_e_rms_s) CUDA_CHECK(cudaFree(d_e_rms_s)); if (tables->has_R_2_s) CUDA_CHECK(cudaFree(d_R_2_s));
        if (tables->has_R_matrix_s) { CUDA_CHECK(cudaFree(d_R_11_s)); CUDA_CHECK(cudaFree(d_R_12_s)); CUDA_CHECK(cudaFree(d_R_21_s)); CUDA_CHECK(cudaFree(d_R_22_s)); }
        if(d_unique_source_hp_coords_kdtree) CUDA_CHECK(cudaFree(d_unique_source_hp_coords_kdtree)); if(d_unique_source_hp_ids) CUDA_CHECK(cudaFree(d_unique_source_hp_ids));
        if(d_world_bounds) CUDA_CHECK(cudaFree(d_world_bounds)); if(d_kdtree_to_original_mapping) CUDA_CHECK(cudaFree(d_kdtree_to_original_mapping));
        if(d_all_source_hp_ids_sorted_gpu) CUDA_CHECK(cudaFree(d_all_source_hp_ids_sorted_gpu)); if(d_sorted_source_original_indices_gpu) CUDA_CHECK(cudaFree(d_sorted_source_original_indices_gpu));
        if(d_unique_hp_id_offsets_start_gpu) CUDA_CHECK(cudaFree(d_unique_hp_id_offsets_start_gpu)); if(d_unique_hp_id_offsets_end_gpu) CUDA_CHECK(cudaFree(d_unique_hp_id_offsets_end_gpu));
    }
    if (total_output_bins > 0) {
        CUDA_CHECK(cudaFree(d_sum_1_r)); CUDA_CHECK(cudaFree(d_sum_w_ls_r)); CUDA_CHECK(cudaFree(d_sum_w_ls_e_t_r)); CUDA_CHECK(cudaFree(d_sum_w_ls_e_t_sigma_crit_r)); CUDA_CHECK(cudaFree(d_sum_w_ls_z_s_r)); CUDA_CHECK(cudaFree(d_sum_w_ls_sigma_crit_r));
        if (tables->has_m_s && tables->sum_w_ls_m_r) CUDA_CHECK(cudaFree(d_sum_w_ls_m_r));
        if (tables->has_e_rms_s && tables->sum_w_ls_1_minus_e_rms_sq_r) CUDA_CHECK(cudaFree(d_sum_w_ls_1_minus_e_rms_sq_r));
        if (tables->has_R_2_s && tables->sum_w_ls_A_p_R_2_r) CUDA_CHECK(cudaFree(d_sum_w_ls_A_p_R_2_r));
        if (tables->has_R_matrix_s && tables->sum_w_ls_R_T_r) CUDA_CHECK(cudaFree(d_sum_w_ls_R_T_r));
    }

    std::cout << "precompute_cuda_interface completed successfully." << std::endl;
    return 0; // Success
}
