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
#include <omp.h> // NEW: Include OpenMP for host-side multi-threading

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
        fprintf(stderr, "CUDA Error: %s in file %s at line %d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
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

double get_max_pixrad_gpu_host(long nside) {
    double area = 4*M_PI / (12 * nside * nside);
    return sqrt(area);
}


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
        if (dist2 < m_radius_sq && m_count < m_capacity) {
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

int calculate_max_k_host(long nside, double search_radius_sq) {
    if (nside <= 0) return 32;
    double pixel_area = (4.0 * M_PI) / (12.0 * (double)nside * nside);
    double search_area_approx = M_PI * search_radius_sq;
    double theoretical_max = search_area_approx / pixel_area;
    int max_k = static_cast<int>(theoretical_max * 1.5) + 32; // Increased safety margin
    return (max_k < 32) ? 32 : max_k;
}


typedef struct { 
    int lens_idx_global;
    int lens_idx_batch;
    int N_bins; long nside_healpix;
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
    const double* g_R_21_s; const double* g_R_22_s; const double* g_dist_3d_sq_bins_batch;
    bool comoving; int weighting; long long* g_sum_1_r_batch; double* g_sum_w_ls_r_batch;
    double* g_sum_w_ls_e_t_r_batch; double* g_sum_w_ls_e_t_sigma_crit_r_batch;
    double* g_sum_w_ls_z_s_r_batch; double* g_sum_w_ls_sigma_crit_r_batch; double* g_sum_w_ls_m_r_batch;
    double* g_sum_w_ls_1_minus_e_rms_sq_r_batch; double* g_sum_w_ls_A_p_R_2_r_batch;
    double* g_sum_w_ls_R_T_r_batch;
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
        const double* current_lens_dist_bins = cb_data->g_dist_3d_sq_bins_batch + (size_t)cb_data->lens_idx_batch * (cb_data->N_bins + 1);
        int i_bin = find_bin_idx_gpu(dist_sq, current_lens_dist_bins, cb_data->N_bins);
        if (i_bin == -1) continue;
        int z_bin_s_val = (cb_data->has_sigma_crit_eff && cb_data->g_z_bin_s != nullptr) ? cb_data->g_z_bin_s[original_source_idx] : -1;
        double sigma_crit_inv = calculate_sigma_crit_inv_gpu(cb_data->lens_zl_i, zs_i, cb_data->lens_dcoml_i, dcoms_i, cb_data->comoving, cb_data->has_sigma_crit_eff, cb_data->g_sigma_crit_eff_l, cb_data->lens_idx_global, cb_data->n_z_bins_l, z_bin_s_val);
        if (sigma_crit_inv == 0.0 || sigma_crit_inv == DBL_MAX) continue;
        double w_ls = calculate_w_ls_gpu(sigma_crit_inv, cb_data->g_w_s[original_source_idx], cb_data->weighting);
        if (w_ls == 0.0) continue;
        double cos_2phi, sin_2phi;
        calculate_et_components_gpu(cb_data->lens_sin_ra_l_i, cb_data->lens_cos_ra_l_i, cb_data->lens_sin_dec_l_i, cb_data->lens_cos_dec_l_i, sin_ra_s_i, cos_ra_s_i, sin_dec_s_i, cos_dec_s_i, cos_2phi, sin_2phi);
        double e_t_val = calculate_et_gpu(cb_data->g_e_1_s[original_source_idx], cb_data->g_e_2_s[original_source_idx], cos_2phi, sin_2phi);
        size_t out_idx = (size_t)cb_data->lens_idx_batch * cb_data->N_bins + i_bin;
        atomicAdd((unsigned long long int*)&cb_data->g_sum_1_r_batch[out_idx], 1ULL);
        atomicAdd(&cb_data->g_sum_w_ls_r_batch[out_idx], w_ls);
        atomicAdd(&cb_data->g_sum_w_ls_e_t_r_batch[out_idx], w_ls * e_t_val);
        double sigma_crit = (sigma_crit_inv == 0.0) ? DBL_MAX : 1.0 / sigma_crit_inv;
        atomicAdd(&cb_data->g_sum_w_ls_e_t_sigma_crit_r_batch[out_idx], w_ls * e_t_val * sigma_crit);
        atomicAdd(&cb_data->g_sum_w_ls_z_s_r_batch[out_idx], w_ls * zs_i);
        atomicAdd(&cb_data->g_sum_w_ls_sigma_crit_r_batch[out_idx], w_ls * sigma_crit);
        if (cb_data->has_m_s && cb_data->g_m_s && cb_data->g_sum_w_ls_m_r_batch) atomicAdd(&cb_data->g_sum_w_ls_m_r_batch[out_idx], w_ls * cb_data->g_m_s[original_source_idx]);
        if (cb_data->has_e_rms_s && cb_data->g_e_rms_s && cb_data->g_sum_w_ls_1_minus_e_rms_sq_r_batch) { double e_rms_s_i = cb_data->g_e_rms_s[original_source_idx]; atomicAdd(&cb_data->g_sum_w_ls_1_minus_e_rms_sq_r_batch[out_idx], w_ls * (1.0 - e_rms_s_i * e_rms_s_i)); }
        if (cb_data->has_R_2_s && cb_data->g_R_2_s && cb_data->g_sum_w_ls_A_p_R_2_r_batch) { if (cb_data->g_R_2_s[original_source_idx] <= 0.31) atomicAdd(&cb_data->g_sum_w_ls_A_p_R_2_r_batch[out_idx], 0.00865 * w_ls / 0.01); }
        if (cb_data->has_R_matrix_s && cb_data->g_R_11_s && cb_data->g_sum_w_ls_R_T_r_batch) { double R_T_val = calculate_R_T_gpu(cb_data->g_R_11_s[original_source_idx], cb_data->g_R_12_s[original_source_idx], cb_data->g_R_21_s[original_source_idx], cb_data->g_R_22_s[original_source_idx], cos_2phi, sin_2phi); atomicAdd(&cb_data->g_sum_w_ls_R_T_r_batch[out_idx], w_ls * R_T_val); }
    }
}

// CORRECTED: Added gpu_id parameter
__device__ void kdtree_radius_search_and_process(
    float3 query_point, float search_radius_sq,
    const float3* d_kdtree_points, int num_kdtree_points,
    const cukd::box_t<float3>* d_world_bounds, const int* d_kdtree_to_original_mapping,
    KernelCallbackData* callback_data, void* g_knn_workspace, int max_k_per_thread,
    int gpu_id // NEW
) {
    void* my_workspace = (char*)g_knn_workspace + ((size_t)callback_data->lens_idx_batch * max_k_per_thread * sizeof(KnnCandidate));
    WorkspaceCandidateList candidates(my_workspace, max_k_per_thread, sqrtf(search_radius_sq));
    cukd::stackBased::knn(candidates, query_point, *d_world_bounds, d_kdtree_points, num_kdtree_points);
    
    int processed_count = 0;
    for (int i = 0; i < max_k_per_thread; i++) {
        int kdtree_index = candidates.get_pointID(i);
        if (kdtree_index < 0) {
            // printf("GPU %d, Lens %d (zl=%.4f) exhausted its candidate buffer of %d!\n",
            //        gpu_id, callback_data->lens_idx_global, callback_data->lens_zl_i, max_k_per_thread);
            break;
        }
        int original_index = d_kdtree_to_original_mapping[kdtree_index];
        process_found_source_hp_pixel(original_index, callback_data);
        processed_count++;
    }

    if (processed_count >= max_k_per_thread) {
        // CORRECTED: Replaced omp_get_thread_num() with the passed-in gpu_id
        printf("FATAL ERROR: GPU %d, Lens %d (zl=%.4f) exhausted its candidate buffer of %d!\n",
               gpu_id, callback_data->lens_idx_global, callback_data->lens_zl_i, max_k_per_thread);
    }
}

// CORRECTED: Added gpu_id parameter
__global__ void process_lens_batch_kernel(
    const double* g_z_l_batch, const double* g_d_com_l_batch, const double* g_sin_ra_l_batch, const double* g_cos_ra_l_batch, const double* g_sin_dec_l_batch, const double* g_cos_dec_l_batch, const double* g_dist_3d_sq_bins_batch,
    const double* g_z_s, const double* g_d_com_s, const double* g_sin_ra_s, const double* g_cos_ra_s, const double* g_sin_dec_s, const double* g_cos_dec_s,
    const double* g_w_s, const double* g_e_1_s, const double* g_e_2_s, const double* g_z_l_max_s,
    const float3* g_unique_source_hp_coords_kdtree, const long* g_unique_source_hp_ids, int N_unique_source_hp, const cukd::box_t<float3>* g_world_bounds, const int* g_kdtree_to_original_mapping,
    const long* g_all_source_hp_ids_sorted, const int* g_sorted_source_original_indices, const int* g_unique_hp_id_offsets_start, const int* g_unique_hp_id_offsets_end,
    bool _has_sigma_crit_eff, int _n_z_bins_l, const double* g_sigma_crit_eff_l, const int* _g_z_bin_s,
    bool _has_m_s, const double* g_m_s, bool _has_e_rms_s, const double* g_e_rms_s, bool _has_R_2_s, const double* g_R_2_s,
    bool _has_R_matrix_s, const double* g_R_11_s, const double* g_R_12_s, const double* g_R_21_s, const double* g_R_22_s,
    int N_lenses_in_batch, int N_bins, long nside_healpix, bool comoving, int weighting, int global_lens_offset,
    void* g_knn_workspace, int max_k_per_thread,
    int gpu_id, // NEW
    long long* g_sum_1_r_batch, double* g_sum_w_ls_r_batch, double* g_sum_w_ls_e_t_r_batch, double* g_sum_w_ls_e_t_sigma_crit_r_batch,
    double* g_sum_w_ls_z_s_r_batch, double* g_sum_w_ls_sigma_crit_r_batch, double* g_sum_w_ls_m_r_batch,
    double* g_sum_w_ls_1_minus_e_rms_sq_r_batch, double* g_sum_w_ls_A_p_R_2_r_batch, double* g_sum_w_ls_R_T_r_batch
) {
    int lens_idx_batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (lens_idx_batch >= N_lenses_in_batch) return;

    int lens_idx_global = global_lens_offset + lens_idx_batch;

    double zl_i = g_z_l_batch[lens_idx_batch]; double dcoml_i = g_d_com_l_batch[lens_idx_batch];
    double sin_ra_l_i = g_sin_ra_l_batch[lens_idx_batch]; double cos_ra_l_i = g_cos_ra_l_batch[lens_idx_batch];
    double sin_dec_l_i = g_sin_dec_l_batch[lens_idx_batch]; double cos_dec_l_i = g_cos_dec_l_batch[lens_idx_batch];
    float3 lens_xyz_cartesian;
    spherical_to_cartesian_gpu(sin_ra_l_i, cos_ra_l_i, sin_dec_l_i, cos_dec_l_i, lens_xyz_cartesian);
    double max_dist_3d_sq_lens = g_dist_3d_sq_bins_batch[lens_idx_batch * (N_bins + 1) + N_bins];
    double lens_hp_pixrad = get_max_pixrad_gpu(nside_healpix);
    double pixrad_3d_sq = 4.0 * sin(lens_hp_pixrad * 0.5) * sin(lens_hp_pixrad * 0.5);
    double search_radius_sq_gpu = max_dist_3d_sq_lens + (4.0 * pixrad_3d_sq + 4.0 * sqrt(max_dist_3d_sq_lens) * sqrt(pixrad_3d_sq));
    float search_radius_sq = (float)search_radius_sq_gpu;

    KernelCallbackData callback_data;
    callback_data.lens_idx_global = lens_idx_global; callback_data.lens_idx_batch = lens_idx_batch;
    callback_data.N_bins = N_bins; callback_data.nside_healpix = nside_healpix;
    callback_data.g_unique_source_hp_ids = g_unique_source_hp_ids; callback_data.g_all_source_hp_ids_sorted = g_all_source_hp_ids_sorted;
    callback_data.g_sorted_source_original_indices = g_sorted_source_original_indices; callback_data.g_unique_hp_id_offsets_start = g_unique_hp_id_offsets_start;
    callback_data.g_unique_hp_id_offsets_end = g_unique_hp_id_offsets_end; callback_data.lens_zl_i = zl_i; callback_data.lens_dcoml_i = dcoml_i;
    callback_data.lens_sin_ra_l_i = sin_ra_l_i; callback_data.lens_cos_ra_l_i = cos_ra_l_i; callback_data.lens_sin_dec_l_i = sin_dec_l_i; callback_data.lens_cos_dec_l_i = cos_dec_l_i;
    callback_data.g_z_s = g_z_s; callback_data.g_d_com_s = g_d_com_s; callback_data.g_sin_ra_s = g_sin_ra_s; callback_data.g_cos_ra_s = g_cos_ra_s;
    callback_data.g_sin_dec_s = g_sin_dec_s; callback_data.g_cos_dec_s = g_cos_dec_s; callback_data.g_w_s = g_w_s; callback_data.g_e_1_s = g_e_1_s;
    callback_data.g_e_2_s = g_e_2_s; callback_data.g_z_l_max_s = g_z_l_max_s; callback_data.has_sigma_crit_eff = _has_sigma_crit_eff; callback_data.n_z_bins_l = _n_z_bins_l;
    callback_data.g_sigma_crit_eff_l = g_sigma_crit_eff_l; callback_data.g_z_bin_s = _g_z_bin_s; callback_data.has_m_s = _has_m_s; callback_data.g_m_s = g_m_s;
    callback_data.has_e_rms_s = _has_e_rms_s; callback_data.g_e_rms_s = g_e_rms_s; callback_data.has_R_2_s = _has_R_2_s; callback_data.g_R_2_s = g_R_2_s;
    callback_data.has_R_matrix_s = _has_R_matrix_s; callback_data.g_R_11_s = g_R_11_s; callback_data.g_R_12_s = g_R_12_s;
    callback_data.g_R_21_s = g_R_21_s; callback_data.g_R_22_s = g_R_22_s; callback_data.g_dist_3d_sq_bins_batch = g_dist_3d_sq_bins_batch;
    callback_data.comoving = comoving; callback_data.weighting = weighting; callback_data.g_sum_1_r_batch = g_sum_1_r_batch; callback_data.g_sum_w_ls_r_batch = g_sum_w_ls_r_batch;
    callback_data.g_sum_w_ls_e_t_r_batch = g_sum_w_ls_e_t_r_batch; callback_data.g_sum_w_ls_e_t_sigma_crit_r_batch = g_sum_w_ls_e_t_sigma_crit_r_batch;
    callback_data.g_sum_w_ls_z_s_r_batch = g_sum_w_ls_z_s_r_batch; callback_data.g_sum_w_ls_sigma_crit_r_batch = g_sum_w_ls_sigma_crit_r_batch;
    callback_data.g_sum_w_ls_m_r_batch = g_sum_w_ls_m_r_batch;
    callback_data.g_sum_w_ls_1_minus_e_rms_sq_r_batch = g_sum_w_ls_1_minus_e_rms_sq_r_batch; callback_data.g_sum_w_ls_A_p_R_2_r_batch = g_sum_w_ls_A_p_R_2_r_batch;
    callback_data.g_sum_w_ls_R_T_r_batch = g_sum_w_ls_R_T_r_batch;

    kdtree_radius_search_and_process(
        lens_xyz_cartesian, search_radius_sq,
        g_unique_source_hp_coords_kdtree, N_unique_source_hp, g_world_bounds,
        g_kdtree_to_original_mapping, &callback_data,
        g_knn_workspace, max_k_per_thread,
        gpu_id // NEW
    );
}

int precompute_cuda_interface(TableData* tables, int n_gpus_to_use) {
    // --- 1. Input Validation (unchanged) ---
    if (!tables->z_l || !tables->d_com_l || !tables->sin_ra_l || !tables->cos_ra_l || !tables->sin_dec_l || !tables->cos_dec_l || !tables->z_s || !tables->d_com_s || !tables->sin_ra_s || !tables->cos_ra_s || !tables->sin_dec_s || !tables->cos_dec_s || !tables->w_s || !tables->e_1_s || !tables->e_2_s || !tables->z_l_max_s || !tables->healpix_id_s || !tables->dist_3d_sq_bins || !tables->sum_1_r || !tables->sum_w_ls_r || !tables->sum_w_ls_e_t_r || !tables->sum_w_ls_e_t_sigma_crit_r || !tables->sum_w_ls_z_s_r || !tables->sum_w_ls_sigma_crit_r) { std::cerr << "Error: Essential data pointers in TableData are null." << std::endl; return -1; }
    if (tables->has_sigma_crit_eff && (!tables->sigma_crit_eff_l || !tables->z_bin_s)) { std::cerr << "Error: sigma_crit_eff pointers are null." << std::endl; return -1; }
    if (tables->has_m_s && !tables->m_s) { std::cerr << "Error: m_s is null." << std::endl; return -1; }
    if (tables->n_lenses <= 0 && tables->n_sources == 0) { std::cout << "No lenses or sources. Exiting." << std::endl; return 0; }
    if (tables->n_lenses > 0 && tables->n_bins <= 0) { std::cerr << "Error: n_bins must be positive." << std::endl; return -1; }
    if (tables->nside_healpix <= 0 && (tables->n_lenses > 0 || tables->n_sources > 0)) { std::cerr << "Error: nside_healpix must be positive." << std::endl; return -1; }
    
    // --- 2. GLOBAL SETUP for KD-Tree (Host-side) ---
    std::vector<long> h_unique_source_hp_ids;
    std::vector<int> h_kdtree_to_original_mapping;
    std::vector<float3> h_unique_source_hp_coords_kdtree;
    int N_unique_source_hp = 0;
    
    std::vector<std::pair<long, int>> source_hp_pairs(tables->n_sources);
    for(int i=0; i < tables->n_sources; ++i) source_hp_pairs[i] = {tables->healpix_id_s[i], i};
    std::sort(source_hp_pairs.begin(), source_hp_pairs.end());

    std::vector<long> h_all_source_hp_ids_sorted(tables->n_sources);
    std::vector<int> h_sorted_source_original_indices(tables->n_sources);
    for(int i=0; i < tables->n_sources; ++i) { h_all_source_hp_ids_sorted[i] = source_hp_pairs[i].first; h_sorted_source_original_indices[i] = source_hp_pairs[i].second; }

    std::vector<int> h_unique_hp_id_offsets_start, h_unique_hp_id_offsets_end;
    if (tables->n_sources > 0) {
        h_unique_source_hp_ids.push_back(h_all_source_hp_ids_sorted[0]); h_unique_hp_id_offsets_start.push_back(0);
        for (int i = 1; i < tables->n_sources; ++i) if (h_all_source_hp_ids_sorted[i] != h_all_source_hp_ids_sorted[i-1]) { h_unique_source_hp_ids.push_back(h_all_source_hp_ids_sorted[i]); h_unique_hp_id_offsets_end.push_back(i); h_unique_hp_id_offsets_start.push_back(i); }
        h_unique_hp_id_offsets_end.push_back(tables->n_sources);
    }
    N_unique_source_hp = h_unique_source_hp_ids.size();
    
    if (N_unique_source_hp > 0) {
        std::vector<PointWithIndex> h_points_with_indices(N_unique_source_hp);
        for(int i=0; i < N_unique_source_hp; ++i) {
            double theta, phi; pix2ang_ring_host(tables->nside_healpix, h_unique_source_hp_ids[i], theta, phi);
            double sin_theta = sin(theta); h_points_with_indices[i].coord.x = static_cast<float>(sin_theta * cos(phi));
            h_points_with_indices[i].coord.y = static_cast<float>(sin_theta * sin(phi)); h_points_with_indices[i].coord.z = static_cast<float>(cos(theta));
            h_points_with_indices[i].original_index = i;
        }
        
        PointWithIndex* d_temp_points;
        cukd::box_t<float3>* d_temp_bounds;
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaMalloc(&d_temp_points, N_unique_source_hp * sizeof(PointWithIndex)));
        CUDA_CHECK(cudaMalloc(&d_temp_bounds, sizeof(cukd::box_t<float3>)));
        CUDA_CHECK(cudaMemcpy(d_temp_points, h_points_with_indices.data(), N_unique_source_hp * sizeof(PointWithIndex), cudaMemcpyHostToDevice));
        
        std::cout << "Building cudaKDTree on GPU 0..." << std::endl;
        cukd::buildTree<PointWithIndex, PointWithIndexTraits>(d_temp_points, N_unique_source_hp, d_temp_bounds);
        
        std::vector<PointWithIndex> h_reordered_points(N_unique_source_hp);
        CUDA_CHECK(cudaMemcpy(h_reordered_points.data(), d_temp_points, N_unique_source_hp * sizeof(PointWithIndex), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_temp_points));
        CUDA_CHECK(cudaFree(d_temp_bounds));
        
        h_unique_source_hp_coords_kdtree.resize(N_unique_source_hp);
        h_kdtree_to_original_mapping.resize(N_unique_source_hp);
        for(int i=0; i < N_unique_source_hp; ++i) { 
            h_unique_source_hp_coords_kdtree[i] = h_reordered_points[i].coord; 
            h_kdtree_to_original_mapping[i] = h_reordered_points[i].original_index;
        }
        std::cout << "KD-Tree pre-build complete." << std::endl;
    }
    
    // --- 3. MULTI-GPU EXECUTION WITH DYNAMIC BATCHING ---
    std::cout << "Starting computation on " << n_gpus_to_use << " GPU(s)..." << std::endl;

    #pragma omp parallel for num_threads(n_gpus_to_use)
    for (int gpu_id = 0; gpu_id < n_gpus_to_use; ++gpu_id) {
        // --- A. PER-GPU SETUP ---
        CUDA_CHECK(cudaSetDevice(gpu_id));
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        printf("GPU %d: Thread %d checking in.\n", gpu_id, omp_get_thread_num());

        // --- B. REPLICATE SHARED DATA ON THIS GPU ---
        double *d_z_s, *d_d_com_s, *d_sin_ra_s, *d_cos_ra_s, *d_sin_dec_s, *d_cos_dec_s, *d_w_s, *d_e_1_s, *d_e_2_s, *d_z_l_max_s;
        double *d_sigma_crit_eff_l = nullptr; int *d_z_bin_s = nullptr;
        double *d_m_s = nullptr, *d_e_rms_s = nullptr, *d_R_2_s = nullptr;
        double *d_R_11_s = nullptr, *d_R_12_s = nullptr, *d_R_21_s = nullptr, *d_R_22_s = nullptr;
        float3* d_unique_source_hp_coords_kdtree = nullptr; long* d_unique_source_hp_ids = nullptr;
        cukd::box_t<float3>* d_world_bounds = nullptr; int* d_kdtree_to_original_mapping = nullptr;
        long* d_all_source_hp_ids_sorted_gpu = nullptr; int* d_sorted_source_original_indices_gpu = nullptr;
        int* d_unique_hp_id_offsets_start_gpu = nullptr; int* d_unique_hp_id_offsets_end_gpu = nullptr;
        
        if (tables->n_sources > 0) {
            CUDA_CHECK(cudaMalloc(&d_z_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_z_s, tables->z_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMalloc(&d_d_com_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_d_com_s, tables->d_com_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_sin_ra_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_sin_ra_s, tables->sin_ra_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_cos_ra_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_cos_ra_s, tables->cos_ra_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_sin_dec_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_sin_dec_s, tables->sin_dec_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_cos_dec_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_cos_dec_s, tables->cos_dec_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_w_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_w_s, tables->w_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_e_1_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_e_1_s, tables->e_1_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_e_2_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_e_2_s, tables->e_2_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_z_l_max_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_z_l_max_s, tables->z_l_max_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream));
            if (tables->has_sigma_crit_eff) { CUDA_CHECK(cudaMalloc(&d_sigma_crit_eff_l, (size_t)tables->n_lenses * tables->n_z_bins_l * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_sigma_crit_eff_l, tables->sigma_crit_eff_l, (size_t)tables->n_lenses * tables->n_z_bins_l * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_z_bin_s, tables->n_sources * sizeof(int))); CUDA_CHECK(cudaMemcpyAsync(d_z_bin_s, tables->z_bin_s, tables->n_sources * sizeof(int), cudaMemcpyHostToDevice, stream)); }
            if (tables->has_m_s) { CUDA_CHECK(cudaMalloc(&d_m_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_m_s, tables->m_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); } if (tables->has_e_rms_s) { CUDA_CHECK(cudaMalloc(&d_e_rms_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_e_rms_s, tables->e_rms_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); } if (tables->has_R_2_s) { CUDA_CHECK(cudaMalloc(&d_R_2_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_R_2_s, tables->R_2_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); } if (tables->has_R_matrix_s) { CUDA_CHECK(cudaMalloc(&d_R_11_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_R_11_s, tables->R_11_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_R_12_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_R_12_s, tables->R_12_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_R_21_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_R_21_s, tables->R_21_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMalloc(&d_R_22_s, tables->n_sources * sizeof(double))); CUDA_CHECK(cudaMemcpyAsync(d_R_22_s, tables->R_22_s, tables->n_sources * sizeof(double), cudaMemcpyHostToDevice, stream)); }
            CUDA_CHECK(cudaMalloc(&d_unique_source_hp_coords_kdtree, N_unique_source_hp * sizeof(float3))); CUDA_CHECK(cudaMemcpyAsync(d_unique_source_hp_coords_kdtree, h_unique_source_hp_coords_kdtree.data(), N_unique_source_hp * sizeof(float3), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMalloc(&d_unique_source_hp_ids, N_unique_source_hp * sizeof(long))); CUDA_CHECK(cudaMemcpyAsync(d_unique_source_hp_ids, h_unique_source_hp_ids.data(), N_unique_source_hp * sizeof(long), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMalloc(&d_kdtree_to_original_mapping, N_unique_source_hp * sizeof(int))); CUDA_CHECK(cudaMemcpyAsync(d_kdtree_to_original_mapping, h_kdtree_to_original_mapping.data(), N_unique_source_hp * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMalloc(&d_all_source_hp_ids_sorted_gpu, tables->n_sources * sizeof(long))); CUDA_CHECK(cudaMemcpyAsync(d_all_source_hp_ids_sorted_gpu, h_all_source_hp_ids_sorted.data(), tables->n_sources * sizeof(long), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMalloc(&d_sorted_source_original_indices_gpu, tables->n_sources * sizeof(int))); CUDA_CHECK(cudaMemcpyAsync(d_sorted_source_original_indices_gpu, h_sorted_source_original_indices.data(), tables->n_sources * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMalloc(&d_unique_hp_id_offsets_start_gpu, N_unique_source_hp * sizeof(int))); CUDA_CHECK(cudaMemcpyAsync(d_unique_hp_id_offsets_start_gpu, h_unique_hp_id_offsets_start.data(), N_unique_source_hp * sizeof(int), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMalloc(&d_unique_hp_id_offsets_end_gpu, N_unique_source_hp * sizeof(int))); CUDA_CHECK(cudaMemcpyAsync(d_unique_hp_id_offsets_end_gpu, h_unique_hp_id_offsets_end.data(), N_unique_source_hp * sizeof(int), cudaMemcpyHostToDevice, stream));
            cukd::box_t<float3> h_world_bounds; for(int i=0; i < h_unique_source_hp_coords_kdtree.size(); ++i) h_world_bounds.grow(h_unique_source_hp_coords_kdtree[i]);
            CUDA_CHECK(cudaMalloc(&d_world_bounds, sizeof(cukd::box_t<float3>))); CUDA_CHECK(cudaMemcpyAsync(d_world_bounds, &h_world_bounds, sizeof(cukd::box_t<float3>), cudaMemcpyHostToDevice, stream));
        }
        
        // --- C. DYNAMIC BATCHING LOGIC FOR THIS GPU ---
        int total_lenses = tables->n_lenses;
        int lenses_per_gpu = (total_lenses + n_gpus_to_use - 1) / n_gpus_to_use;
        int start_lens_idx = gpu_id * lenses_per_gpu;
        int end_lens_idx = std::min(start_lens_idx + lenses_per_gpu, total_lenses);
        int num_lenses_for_this_gpu = end_lens_idx - start_lens_idx;

        if (num_lenses_for_this_gpu <= 0) {
            printf("GPU %d has no lenses to process.\n", gpu_id);
        } else {
            int max_k_for_this_gpu = 32;
            double lens_hp_pixrad = get_max_pixrad_gpu_host(tables->nside_healpix);
            double pixrad_3d_sq = 4.0 * sin(lens_hp_pixrad * 0.5) * sin(lens_hp_pixrad * 0.5);
            for (int i = start_lens_idx; i < end_lens_idx; ++i) {
                double max_dist_3d_sq_lens = tables->dist_3d_sq_bins[i * (tables->n_bins + 1) + tables->n_bins];
                double search_radius_sq_gpu = max_dist_3d_sq_lens + (4.0 * pixrad_3d_sq + 4.0 * sqrt(max_dist_3d_sq_lens) * sqrt(pixrad_3d_sq));
                int required_k = calculate_max_k_host(tables->nside_healpix, search_radius_sq_gpu);
                if (required_k > max_k_for_this_gpu) max_k_for_this_gpu = required_k;
            }

            size_t free_mem, total_mem; CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
            size_t memory_per_lens = sizeof(double) * 6 + (tables->n_bins + 1) * sizeof(double) + tables->n_bins * (sizeof(long long) + 8 * sizeof(double));
            size_t workspace_per_lens = max_k_for_this_gpu * sizeof(KnnCandidate);
            size_t total_per_lens = memory_per_lens + workspace_per_lens;
            size_t usable_mem = free_mem > (500 * 1024 * 1024) ? free_mem - (500 * 1024 * 1024) : free_mem / 2;
            int batch_size = (total_per_lens > 0) ? (usable_mem / total_per_lens) : num_lenses_for_this_gpu;
            if (batch_size == 0) batch_size = 1;
            if (batch_size > num_lenses_for_this_gpu) batch_size = num_lenses_for_this_gpu;

            printf("GPU %d processing lenses [%d, %d). Max K: %d. Available VRAM: %.2f GB. Decided batch size: %d lenses.\n", gpu_id, start_lens_idx, end_lens_idx, max_k_for_this_gpu, free_mem / (1024.0*1024.0*1024.0), batch_size);

            int lenses_processed = 0;
            while(lenses_processed < num_lenses_for_this_gpu) {
                int current_batch_size = std::min(batch_size, num_lenses_for_this_gpu - lenses_processed);
                int global_lens_offset = start_lens_idx + lenses_processed;
                
                double *d_z_l_batch, *d_d_com_l_batch, *d_sin_ra_l_batch, *d_cos_ra_l_batch, *d_sin_dec_l_batch, *d_cos_dec_l_batch, *d_dist_3d_sq_bins_batch;
                void* d_knn_workspace_batch;
                long long* d_sum_1_r_batch; double* d_sum_w_ls_r_batch, *d_sum_w_ls_e_t_r_batch, *d_sum_w_ls_e_t_sigma_crit_r_batch, *d_sum_w_ls_z_s_r_batch, *d_sum_w_ls_sigma_crit_r_batch, *d_sum_w_ls_m_r_batch, *d_sum_w_ls_1_minus_e_rms_sq_r_batch, *d_sum_w_ls_A_p_R_2_r_batch, *d_sum_w_ls_R_T_r_batch;
                CUDA_CHECK(cudaMalloc(&d_z_l_batch, current_batch_size * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_d_com_l_batch, current_batch_size * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_sin_ra_l_batch, current_batch_size * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_cos_ra_l_batch, current_batch_size * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_sin_dec_l_batch, current_batch_size * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_cos_dec_l_batch, current_batch_size * sizeof(double))); CUDA_CHECK(cudaMalloc(&d_dist_3d_sq_bins_batch, (size_t)current_batch_size * (tables->n_bins + 1) * sizeof(double)));
                CUDA_CHECK(cudaMalloc(&d_knn_workspace_batch, (size_t)current_batch_size * max_k_for_this_gpu * sizeof(KnnCandidate)));
                size_t batch_output_bins = (size_t)current_batch_size * tables->n_bins;
                CUDA_CHECK(cudaMalloc(&d_sum_1_r_batch, batch_output_bins * sizeof(long long))); CUDA_CHECK(cudaMemsetAsync(d_sum_1_r_batch, 0, batch_output_bins * sizeof(long long), stream));
                CUDA_CHECK(cudaMalloc(&d_sum_w_ls_r_batch, batch_output_bins * sizeof(double))); CUDA_CHECK(cudaMemsetAsync(d_sum_w_ls_r_batch, 0, batch_output_bins * sizeof(double), stream)); CUDA_CHECK(cudaMalloc(&d_sum_w_ls_e_t_r_batch, batch_output_bins * sizeof(double))); CUDA_CHECK(cudaMemsetAsync(d_sum_w_ls_e_t_r_batch, 0, batch_output_bins * sizeof(double), stream)); CUDA_CHECK(cudaMalloc(&d_sum_w_ls_e_t_sigma_crit_r_batch, batch_output_bins * sizeof(double))); CUDA_CHECK(cudaMemsetAsync(d_sum_w_ls_e_t_sigma_crit_r_batch, 0, batch_output_bins * sizeof(double), stream)); CUDA_CHECK(cudaMalloc(&d_sum_w_ls_z_s_r_batch, batch_output_bins * sizeof(double))); CUDA_CHECK(cudaMemsetAsync(d_sum_w_ls_z_s_r_batch, 0, batch_output_bins * sizeof(double), stream)); CUDA_CHECK(cudaMalloc(&d_sum_w_ls_sigma_crit_r_batch, batch_output_bins * sizeof(double))); CUDA_CHECK(cudaMemsetAsync(d_sum_w_ls_sigma_crit_r_batch, 0, batch_output_bins * sizeof(double), stream));
                if (tables->has_m_s) { CUDA_CHECK(cudaMalloc(&d_sum_w_ls_m_r_batch, batch_output_bins * sizeof(double))); CUDA_CHECK(cudaMemsetAsync(d_sum_w_ls_m_r_batch, 0, batch_output_bins * sizeof(double), stream)); } else { d_sum_w_ls_m_r_batch = nullptr; }
                if (tables->has_e_rms_s) { CUDA_CHECK(cudaMalloc(&d_sum_w_ls_1_minus_e_rms_sq_r_batch, batch_output_bins * sizeof(double))); CUDA_CHECK(cudaMemsetAsync(d_sum_w_ls_1_minus_e_rms_sq_r_batch, 0, batch_output_bins * sizeof(double), stream)); } else { d_sum_w_ls_1_minus_e_rms_sq_r_batch = nullptr; }
                if (tables->has_R_2_s) { CUDA_CHECK(cudaMalloc(&d_sum_w_ls_A_p_R_2_r_batch, batch_output_bins * sizeof(double))); CUDA_CHECK(cudaMemsetAsync(d_sum_w_ls_A_p_R_2_r_batch, 0, batch_output_bins * sizeof(double), stream)); } else { d_sum_w_ls_A_p_R_2_r_batch = nullptr; }
                if (tables->has_R_matrix_s) { CUDA_CHECK(cudaMalloc(&d_sum_w_ls_R_T_r_batch, batch_output_bins * sizeof(double))); CUDA_CHECK(cudaMemsetAsync(d_sum_w_ls_R_T_r_batch, 0, batch_output_bins * sizeof(double), stream)); } else { d_sum_w_ls_R_T_r_batch = nullptr; }
                
                CUDA_CHECK(cudaMemcpyAsync(d_z_l_batch, tables->z_l + global_lens_offset, current_batch_size * sizeof(double), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_d_com_l_batch, tables->d_com_l + global_lens_offset, current_batch_size * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMemcpyAsync(d_sin_ra_l_batch, tables->sin_ra_l + global_lens_offset, current_batch_size * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMemcpyAsync(d_cos_ra_l_batch, tables->cos_ra_l + global_lens_offset, current_batch_size * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMemcpyAsync(d_sin_dec_l_batch, tables->sin_dec_l + global_lens_offset, current_batch_size * sizeof(double), cudaMemcpyHostToDevice, stream)); CUDA_CHECK(cudaMemcpyAsync(d_cos_dec_l_batch, tables->cos_dec_l + global_lens_offset, current_batch_size * sizeof(double), cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaMemcpyAsync(d_dist_3d_sq_bins_batch, tables->dist_3d_sq_bins + (size_t)global_lens_offset * (tables->n_bins + 1), (size_t)current_batch_size * (tables->n_bins + 1) * sizeof(double), cudaMemcpyHostToDevice, stream));
                
                int threadsPerBlock = 256;
                int blocksPerGrid = (current_batch_size + threadsPerBlock - 1) / threadsPerBlock;
                
                process_lens_batch_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
                    d_z_l_batch, d_d_com_l_batch, d_sin_ra_l_batch, d_cos_ra_l_batch, d_sin_dec_l_batch, d_cos_dec_l_batch, d_dist_3d_sq_bins_batch,
                    d_z_s, d_d_com_s, d_sin_ra_s, d_cos_ra_s, d_sin_dec_s, d_cos_dec_s,
                    d_w_s, d_e_1_s, d_e_2_s, d_z_l_max_s,
                    d_unique_source_hp_coords_kdtree, d_unique_source_hp_ids, N_unique_source_hp, d_world_bounds, d_kdtree_to_original_mapping,
                    d_all_source_hp_ids_sorted_gpu, d_sorted_source_original_indices_gpu,
                    d_unique_hp_id_offsets_start_gpu, d_unique_hp_id_offsets_end_gpu,
                    tables->has_sigma_crit_eff, tables->n_z_bins_l, d_sigma_crit_eff_l, d_z_bin_s,
                    tables->has_m_s, d_m_s, tables->has_e_rms_s, d_e_rms_s, tables->has_R_2_s, d_R_2_s,
                    tables->has_R_matrix_s, d_R_11_s, d_R_12_s, d_R_21_s, d_R_22_s,
                    current_batch_size, tables->n_bins, tables->nside_healpix, tables->comoving, tables->weighting, global_lens_offset,
                    d_knn_workspace_batch, max_k_for_this_gpu, gpu_id,
                    d_sum_1_r_batch, d_sum_w_ls_r_batch, d_sum_w_ls_e_t_r_batch, d_sum_w_ls_e_t_sigma_crit_r_batch,
                    d_sum_w_ls_z_s_r_batch, d_sum_w_ls_sigma_crit_r_batch, d_sum_w_ls_m_r_batch,
                    d_sum_w_ls_1_minus_e_rms_sq_r_batch, d_sum_w_ls_A_p_R_2_r_batch, d_sum_w_ls_R_T_r_batch
                );
                
                size_t host_offset = (size_t)global_lens_offset * tables->n_bins;
                CUDA_CHECK(cudaMemcpyAsync(tables->sum_1_r + host_offset, d_sum_1_r_batch, batch_output_bins * sizeof(long long), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaMemcpyAsync(tables->sum_w_ls_r + host_offset, d_sum_w_ls_r_batch, batch_output_bins * sizeof(double), cudaMemcpyDeviceToHost, stream)); CUDA_CHECK(cudaMemcpyAsync(tables->sum_w_ls_e_t_r + host_offset, d_sum_w_ls_e_t_r_batch, batch_output_bins * sizeof(double), cudaMemcpyDeviceToHost, stream)); CUDA_CHECK(cudaMemcpyAsync(tables->sum_w_ls_e_t_sigma_crit_r + host_offset, d_sum_w_ls_e_t_sigma_crit_r_batch, batch_output_bins * sizeof(double), cudaMemcpyDeviceToHost, stream)); CUDA_CHECK(cudaMemcpyAsync(tables->sum_w_ls_z_s_r + host_offset, d_sum_w_ls_z_s_r_batch, batch_output_bins * sizeof(double), cudaMemcpyDeviceToHost, stream)); CUDA_CHECK(cudaMemcpyAsync(tables->sum_w_ls_sigma_crit_r + host_offset, d_sum_w_ls_sigma_crit_r_batch, batch_output_bins * sizeof(double), cudaMemcpyDeviceToHost, stream));
                if (tables->has_m_s) CUDA_CHECK(cudaMemcpyAsync(tables->sum_w_ls_m_r + host_offset, d_sum_w_ls_m_r_batch, batch_output_bins * sizeof(double), cudaMemcpyDeviceToHost, stream));
                if (tables->has_e_rms_s) CUDA_CHECK(cudaMemcpyAsync(tables->sum_w_ls_1_minus_e_rms_sq_r + host_offset, d_sum_w_ls_1_minus_e_rms_sq_r_batch, batch_output_bins * sizeof(double), cudaMemcpyDeviceToHost, stream));
                if (tables->has_R_2_s) CUDA_CHECK(cudaMemcpyAsync(tables->sum_w_ls_A_p_R_2_r + host_offset, d_sum_w_ls_A_p_R_2_r_batch, batch_output_bins * sizeof(double), cudaMemcpyDeviceToHost, stream));
                if (tables->has_R_matrix_s) CUDA_CHECK(cudaMemcpyAsync(tables->sum_w_ls_R_T_r + host_offset, d_sum_w_ls_R_T_r_batch, batch_output_bins * sizeof(double), cudaMemcpyDeviceToHost, stream));
                
                CUDA_CHECK(cudaStreamSynchronize(stream));
                
                CUDA_CHECK(cudaFree(d_z_l_batch)); CUDA_CHECK(cudaFree(d_d_com_l_batch)); CUDA_CHECK(cudaFree(d_sin_ra_l_batch)); CUDA_CHECK(cudaFree(d_cos_ra_l_batch)); CUDA_CHECK(cudaFree(d_sin_dec_l_batch)); CUDA_CHECK(cudaFree(d_cos_dec_l_batch)); CUDA_CHECK(cudaFree(d_dist_3d_sq_bins_batch)); CUDA_CHECK(cudaFree(d_knn_workspace_batch));
                CUDA_CHECK(cudaFree(d_sum_1_r_batch)); CUDA_CHECK(cudaFree(d_sum_w_ls_r_batch)); CUDA_CHECK(cudaFree(d_sum_w_ls_e_t_r_batch)); CUDA_CHECK(cudaFree(d_sum_w_ls_e_t_sigma_crit_r_batch)); CUDA_CHECK(cudaFree(d_sum_w_ls_z_s_r_batch)); CUDA_CHECK(cudaFree(d_sum_w_ls_sigma_crit_r_batch));
                if (tables->has_m_s) CUDA_CHECK(cudaFree(d_sum_w_ls_m_r_batch)); if (tables->has_e_rms_s) CUDA_CHECK(cudaFree(d_sum_w_ls_1_minus_e_rms_sq_r_batch)); if (tables->has_R_2_s) CUDA_CHECK(cudaFree(d_sum_w_ls_A_p_R_2_r_batch)); if (tables->has_R_matrix_s) CUDA_CHECK(cudaFree(d_sum_w_ls_R_T_r_batch));
                
                lenses_processed += current_batch_size;
                printf("GPU %d: Completed batch. Total lenses processed on this GPU: %d / %d\n", gpu_id, lenses_processed, num_lenses_for_this_gpu);
            }
        }
        
        // --- D. PER-GPU CLEANUP ---
        if (tables->n_sources > 0) {
            CUDA_CHECK(cudaFree(d_z_s)); CUDA_CHECK(cudaFree(d_d_com_s)); CUDA_CHECK(cudaFree(d_sin_ra_s)); CUDA_CHECK(cudaFree(d_cos_ra_s)); CUDA_CHECK(cudaFree(d_sin_dec_s)); CUDA_CHECK(cudaFree(d_cos_dec_s)); CUDA_CHECK(cudaFree(d_w_s)); CUDA_CHECK(cudaFree(d_e_1_s)); CUDA_CHECK(cudaFree(d_e_2_s)); CUDA_CHECK(cudaFree(d_z_l_max_s));
            if (d_sigma_crit_eff_l) CUDA_CHECK(cudaFree(d_sigma_crit_eff_l)); if (d_z_bin_s) CUDA_CHECK(cudaFree(d_z_bin_s));
            if (d_m_s) CUDA_CHECK(cudaFree(d_m_s)); if (d_e_rms_s) CUDA_CHECK(cudaFree(d_e_rms_s)); if (d_R_2_s) CUDA_CHECK(cudaFree(d_R_2_s));
            if (d_R_11_s) { CUDA_CHECK(cudaFree(d_R_11_s)); CUDA_CHECK(cudaFree(d_R_12_s)); CUDA_CHECK(cudaFree(d_R_21_s)); CUDA_CHECK(cudaFree(d_R_22_s)); }
            if(d_unique_source_hp_coords_kdtree) CUDA_CHECK(cudaFree(d_unique_source_hp_coords_kdtree)); if(d_unique_source_hp_ids) CUDA_CHECK(cudaFree(d_unique_source_hp_ids));
            if(d_world_bounds) CUDA_CHECK(cudaFree(d_world_bounds)); if(d_kdtree_to_original_mapping) CUDA_CHECK(cudaFree(d_kdtree_to_original_mapping));
            if(d_all_source_hp_ids_sorted_gpu) CUDA_CHECK(cudaFree(d_all_source_hp_ids_sorted_gpu)); if(d_sorted_source_original_indices_gpu) CUDA_CHECK(cudaFree(d_sorted_source_original_indices_gpu));
            if(d_unique_hp_id_offsets_start_gpu) CUDA_CHECK(cudaFree(d_unique_hp_id_offsets_start_gpu)); if(d_unique_hp_id_offsets_end_gpu) CUDA_CHECK(cudaFree(d_unique_hp_id_offsets_end_gpu));
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        printf("GPU %d: Work complete. Cleaning up.\n", gpu_id);
    } // End of omp parallel for loop

    std::cout << "precompute_cuda_interface completed successfully for all GPUs." << std::endl;
    return 0; // Success
}
