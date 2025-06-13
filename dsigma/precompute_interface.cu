#include "precompute_interface.h"
#include "precompute_engine_cuda.h" // For launching the kernel
#include "cuda_host_utils.h"      // For Healpix/KDTree utilities on host
#include "healpix_base.h"
#include <cuda_runtime.h>
#include <vector>
#include <iostream> // For error messages / cout
#include <algorithm> // For std::sort, std::unique, std::lower_bound, std::upper_bound, std::max
#include <map>       // For std::map to get unique pixels and offsets
#include <cmath>     // For sqrt, M_PI (though M_PI might not be in cmath for MSVC, cmath should have PI if C++20)
#include <stdexcept> // For std::runtime_error

// Ensure M_PI is defined (it's not standard C++ before C++20, but common in cmath extensions)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// DEG2RAD for this file, consistent with cuda_host_utils.cpp
#ifndef DEG2RAD_Interface
#define DEG2RAD_Interface 0.017453292519943295
#endif

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err_) << " in file " << __FILE__ << " at line " << __LINE__ << std::endl; \
        return -1; \
    } \
}

// Helper to get HEALPix scheme from string
#if defined(HEALPIX_FOUND) && HEALPIX_FOUND == 1
// This check is to ensure Healpix_Ordering_Scheme and related enums are available
// It's already guarded in cuda_host_utils.cpp, but good for clarity here too
Healpix_Ordering_Scheme get_healpix_scheme(const std::string& order_str) {
    if (order_str == "ring") return RING;
    if (order_str == "nested") return NEST;
    throw std::runtime_error("Invalid HEALPix order string: " + order_str);
}
#endif


// Helper function to get unique pixel IDs and their offsets
void get_unique_pix_and_offsets(
    long* healpix_ids_global, int n_global,
    std::vector<long>& unique_pix_ids,
    std::vector<int>& pix_offsets) {

    unique_pix_ids.clear();
    pix_offsets.clear();

    if (n_global == 0) {
        pix_offsets.push_back(0);
        return;
    }

    // Assumes healpix_ids_global is sorted
    for (int i = 0; i < n_global; ++i) {
        if (unique_pix_ids.empty() || unique_pix_ids.back() != healpix_ids_global[i]) {
            unique_pix_ids.push_back(healpix_ids_global[i]);
            pix_offsets.push_back(i);
        }
    }
    pix_offsets.push_back(n_global); // Add offset for the end of the last group
}


extern "C" int precompute_cuda_interface(
    TableData* tables,
    int n_gpus
) {
    // --- 1. Input Validation ---
    if (!tables->z_l || !tables->d_com_l || !tables->sin_ra_l || !tables->cos_ra_l || !tables->sin_dec_l || !tables->cos_dec_l || !tables->healpix_id_l ||
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
    // Add checks for other optional arrays if their boolean flag is true
    if (tables->has_m_s && !tables->m_s) { std::cerr << "Error: m_s is null when has_m_s is true." << std::endl; return -1; }
    if (tables->has_e_rms_s && !tables->e_rms_s) { std::cerr << "Error: e_rms_s is null when has_e_rms_s is true." << std::endl; return -1; }
    if (tables->has_R_2_s && !tables->R_2_s) { std::cerr << "Error: R_2_s is null when has_R_2_s is true." << std::endl; return -1; }
    if (tables->has_R_matrix_s && (!tables->R_11_s || !tables->R_12_s || !tables->R_21_s || !tables->R_22_s)) {
         std::cerr << "Error: R_matrix_s pointers are null when has_R_matrix_s is true." << std::endl; return -1;
    }


    if (tables->n_lenses <= 0 && tables->n_sources > 0) { // Allow n_lenses = 0 if n_sources = 0 (vacuously true)
         // If there are sources but no lenses, it's probably an issue or just no work to do.
         // If n_lenses is 0, the main loop won't run, so it's safe.
    }
    if (tables->n_bins <= 0 && tables->n_lenses > 0) { // Only an issue if there are lenses
        std::cerr << "Error: n_bins must be positive if there are lenses." << std::endl;
        return -1;
    }
    if (tables->nside_healpix <= 0 && (tables->n_lenses > 0 || tables->n_sources > 0)) {
        std::cerr << "Error: nside_healpix must be positive if there are objects." << std::endl;
        return -1;
    }


    // --- 2. GPU Setup ---
    if (n_gpus <= 0) {
        std::cerr << "No GPUs specified or invalid number of GPUs." << std::endl;
        return -1;
    }
    CUDA_CHECK(cudaSetDevice(0)); // Use GPU 0
    if (n_gpus > 1) {
        std::cout << "Warning: Multi-GPU support (n_gpus > 1) is not yet implemented. Using GPU 0." << std::endl;
    }

    // --- 3. Host Data Analysis (CPU side) ---
    std::vector<long> u_pix_l, u_pix_s;
    std::vector<int> lens_pix_offsets, source_pix_offsets;

    get_unique_pix_and_offsets(tables->healpix_id_l, tables->n_lenses, u_pix_l, lens_pix_offsets);
    get_unique_pix_and_offsets(tables->healpix_id_s, tables->n_sources, u_pix_s, source_pix_offsets);

    if (u_pix_l.empty() || u_pix_s.empty()) {
        std::cout << "No lens or source HEALPix pixels to process. Exiting." << std::endl;
        // Initialize all output arrays to 0, as kernel won't run
        size_t total_output_bins = (size_t)tables->n_lenses * tables->n_bins;
        if (total_output_bins > 0) { // only if n_lenses > 0 and n_bins > 0
            std::fill(tables->sum_1_r, tables->sum_1_r + total_output_bins, 0);
            std::fill(tables->sum_w_ls_r, tables->sum_w_ls_r + total_output_bins, 0.0);
            // ... and so on for all other output arrays
        }
        return 0; // No work to do
    }

#if !(defined(HEALPIX_FOUND) && HEALPIX_FOUND == 1)
    std::cerr << "Error: Healpix_cxx library not found, cannot proceed with HEALPix dependent operations." << std::endl;
    return -1;
#endif

    PointCloud<double> source_hp_xyz = build_healpix_xyz_celestial_cpp(u_pix_s, tables->nside_healpix, tables->order_healpix);
    if (source_hp_xyz.pts.empty() && !u_pix_s.empty()) { // Check if build failed
         std::cerr << "Error: Building source HEALPix XYZ coordinates failed." << std::endl;
         return -1;
    }


    // --- 4. Allocate fixed GPU arrays (if any, none in this design yet beyond batch) ---
    // Example: double* fixed_array_gpu; CUDA_CHECK(cudaMalloc(&fixed_array_gpu, size)); ...

    // --- 5. Main Loop over unique lens HEALPix pixels ---
    Healpix_Base hp_base_for_res(tables->nside_healpix, get_healpix_scheme(tables->order_healpix), SET_NSIDE);
    double max_pixrad_rad = hp_base_for_res.max_pixrad(); // Returns radians

    for (size_t idx_u_pix_l = 0; idx_u_pix_l < u_pix_l.size(); ++idx_u_pix_l) {
        long current_pix_l = u_pix_l[idx_u_pix_l];
        int i_l_min = lens_pix_offsets[idx_u_pix_l];
        int i_l_max = lens_pix_offsets[idx_u_pix_l + 1];
        int n_lenses_batch = i_l_max - i_l_min;

        if (n_lenses_batch == 0) continue;

        double dist_3d_sq_max_batch = 0.0;
        for (int i_l_abs = i_l_min; i_l_abs < i_l_max; ++i_l_abs) {
            dist_3d_sq_max_batch = std::max(dist_3d_sq_max_batch, tables->dist_3d_sq_bins[i_l_abs * (tables->n_bins + 1) + tables->n_bins]);
        }
        // Add margin based on lens pixel size and source pixel sizes (more complex, simplified here)
        // Original Cython: dist_3d_sq_max += (4 * DEG2RAD * DEG2RAD * max_pixrad * max_pixrad + 4 * sqrt(dist_3d_sq_max) * DEG2RAD * max_pixrad);
        // Using max_pixrad_rad (already in radians)
        double margin = 2.0 * max_pixrad_rad; // Simplified margin based on approx pixel diameter
        dist_3d_sq_max_batch += margin * margin + 2.0 * std::sqrt(dist_3d_sq_max_batch) * margin;


        std::vector<long> current_lens_pix_vec = {current_pix_l};
        PointCloud<double> xyz_current_pix_l_cloud = build_healpix_xyz_celestial_cpp(current_lens_pix_vec, tables->nside_healpix, tables->order_healpix);
        if (xyz_current_pix_l_cloud.pts.empty()) {
            std::cerr << "Warning: Could not get coordinates for lens pixel " << current_pix_l << ". Skipping." << std::endl;
            continue;
        }
        const double query_pt[3] = {xyz_current_pix_l_cloud.pts[0].x, xyz_current_pix_l_cloud.pts[0].y, xyz_current_pix_l_cloud.pts[0].z};

        std::vector<size_t> neighboring_u_source_pix_indices = find_neighboring_pixel_indices_nanoflann_cpp(source_hp_xyz, query_pt, dist_3d_sq_max_batch);

        std::vector<int> relevant_s_indices_original;
        for (size_t u_s_idx_in_list : neighboring_u_source_pix_indices) {
            // u_s_idx_in_list is an index into u_pix_s and source_hp_xyz
            int i_s_min = source_pix_offsets[u_s_idx_in_list];
            int i_s_max = source_pix_offsets[u_s_idx_in_list + 1];
            for (int i_s_orig = i_s_min; i_s_orig < i_s_max; ++i_s_orig) {
                relevant_s_indices_original.push_back(i_s_orig);
            }
        }

        int n_sources_batch = relevant_s_indices_original.size();
        if (n_sources_batch == 0) continue;

        // --- Gather Batch Data (Host) ---
        // Lens data for the batch is contiguous in the input `tables` arrays from i_l_min to i_l_max-1.
        // Source data needs to be gathered.
        std::vector<double> z_s_batch(n_sources_batch), d_com_s_batch(n_sources_batch);
        std::vector<double> sin_ra_s_batch(n_sources_batch), cos_ra_s_batch(n_sources_batch);
        std::vector<double> sin_dec_s_batch(n_sources_batch), cos_dec_s_batch(n_sources_batch);
        std::vector<double> w_s_batch(n_sources_batch), e_1_s_batch(n_sources_batch), e_2_s_batch(n_sources_batch);
        std::vector<double> z_l_max_s_batch(n_sources_batch);
        std::vector<int> z_bin_s_batch_host; // Only if has_sigma_crit_eff
        if (tables->has_sigma_crit_eff) z_bin_s_batch_host.resize(n_sources_batch);

        std::vector<double> m_s_batch_host, e_rms_s_batch_host, R_2_s_batch_host;
        std::vector<double> R_11_s_batch_host, R_12_s_batch_host, R_21_s_batch_host, R_22_s_batch_host;
        if (tables->has_m_s) m_s_batch_host.resize(n_sources_batch);
        if (tables->has_e_rms_s) e_rms_s_batch_host.resize(n_sources_batch);
        if (tables->has_R_2_s) R_2_s_batch_host.resize(n_sources_batch);
        if (tables->has_R_matrix_s) {
            R_11_s_batch_host.resize(n_sources_batch); R_12_s_batch_host.resize(n_sources_batch);
            R_21_s_batch_host.resize(n_sources_batch); R_22_s_batch_host.resize(n_sources_batch);
        }

        for (int i = 0; i < n_sources_batch; ++i) {
            int original_idx = relevant_s_indices_original[i];
            z_s_batch[i] = tables->z_s[original_idx];
            d_com_s_batch[i] = tables->d_com_s[original_idx];
            sin_ra_s_batch[i] = tables->sin_ra_s[original_idx];
            cos_ra_s_batch[i] = tables->cos_ra_s[original_idx];
            sin_dec_s_batch[i] = tables->sin_dec_s[original_idx];
            cos_dec_s_batch[i] = tables->cos_dec_s[original_idx];
            w_s_batch[i] = tables->w_s[original_idx];
            e_1_s_batch[i] = tables->e_1_s[original_idx];
            e_2_s_batch[i] = tables->e_2_s[original_idx];
            z_l_max_s_batch[i] = tables->z_l_max_s[original_idx];
            if (tables->has_sigma_crit_eff) z_bin_s_batch_host[i] = tables->z_bin_s[original_idx];
            if (tables->has_m_s) m_s_batch_host[i] = tables->m_s[original_idx];
            if (tables->has_e_rms_s) e_rms_s_batch_host[i] = tables->e_rms_s[original_idx];
            if (tables->has_R_2_s) R_2_s_batch_host[i] = tables->R_2_s[original_idx];
            if (tables->has_R_matrix_s) {
                R_11_s_batch_host[i] = tables->R_11_s[original_idx]; R_12_s_batch_host[i] = tables->R_12_s[original_idx];
                R_21_s_batch_host[i] = tables->R_21_s[original_idx]; R_22_s_batch_host[i] = tables->R_22_s[original_idx];
            }
        }

        // --- GPU Memory Allocation for Batch ---
        // Lens data
        double *z_l_batch_gpu, *d_com_l_batch_gpu, *sin_ra_l_batch_gpu, *cos_ra_l_batch_gpu, *sin_dec_l_batch_gpu, *cos_dec_l_batch_gpu;
        CUDA_CHECK(cudaMalloc(&z_l_batch_gpu, n_lenses_batch * sizeof(double)));
        // ... (cudaMalloc for other lens arrays: d_com_l, sin_ra_l, etc.)
        CUDA_CHECK(cudaMalloc(&d_com_l_batch_gpu, n_lenses_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&sin_ra_l_batch_gpu, n_lenses_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&cos_ra_l_batch_gpu, n_lenses_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&sin_dec_l_batch_gpu, n_lenses_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&cos_dec_l_batch_gpu, n_lenses_batch * sizeof(double)));


        // Source data (already gathered into host std::vectors)
        double *z_s_batch_gpu, *d_com_s_batch_gpu, *sin_ra_s_batch_gpu, *cos_ra_s_batch_gpu, *sin_dec_s_batch_gpu, *cos_dec_s_batch_gpu;
        double *w_s_batch_gpu, *e_1_s_batch_gpu, *e_2_s_batch_gpu, *z_l_max_s_batch_gpu;
        int* z_bin_s_batch_gpu = nullptr;
        // ... (cudaMalloc for all source arrays)
        CUDA_CHECK(cudaMalloc(&z_s_batch_gpu, n_sources_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_com_s_batch_gpu, n_sources_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&sin_ra_s_batch_gpu, n_sources_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&cos_ra_s_batch_gpu, n_sources_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&sin_dec_s_batch_gpu, n_sources_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&cos_dec_s_batch_gpu, n_sources_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&w_s_batch_gpu, n_sources_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&e_1_s_batch_gpu, n_sources_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&e_2_s_batch_gpu, n_sources_batch * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&z_l_max_s_batch_gpu, n_sources_batch * sizeof(double)));


        double* sigma_crit_eff_l_batch_gpu = nullptr;
        if (tables->has_sigma_crit_eff) {
            CUDA_CHECK(cudaMalloc(&sigma_crit_eff_l_batch_gpu, (size_t)n_lenses_batch * tables->n_z_bins_l * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&z_bin_s_batch_gpu, n_sources_batch * sizeof(int)));
        }

        double *m_s_batch_gpu = nullptr, *e_rms_s_batch_gpu = nullptr, *R_2_s_batch_gpu = nullptr;
        double *R_11_s_batch_gpu = nullptr, *R_12_s_batch_gpu = nullptr, *R_21_s_batch_gpu = nullptr, *R_22_s_batch_gpu = nullptr;
        if (tables->has_m_s) CUDA_CHECK(cudaMalloc(&m_s_batch_gpu, n_sources_batch * sizeof(double)));
        if (tables->has_e_rms_s) CUDA_CHECK(cudaMalloc(&e_rms_s_batch_gpu, n_sources_batch * sizeof(double)));
        if (tables->has_R_2_s) CUDA_CHECK(cudaMalloc(&R_2_s_batch_gpu, n_sources_batch * sizeof(double)));
        if (tables->has_R_matrix_s) {
            CUDA_CHECK(cudaMalloc(&R_11_s_batch_gpu, n_sources_batch * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&R_12_s_batch_gpu, n_sources_batch * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&R_21_s_batch_gpu, n_sources_batch * sizeof(double)));
            CUDA_CHECK(cudaMalloc(&R_22_s_batch_gpu, n_sources_batch * sizeof(double)));
        }


        double* dist_3d_sq_bins_batch_gpu;
        CUDA_CHECK(cudaMalloc(&dist_3d_sq_bins_batch_gpu, (size_t)n_lenses_batch * (tables->n_bins + 1) * sizeof(double)));

        // Output sums for the batch
        long long* sum_1_r_batch_gpu;
        double *sum_w_ls_r_batch_gpu, *sum_w_ls_e_t_r_batch_gpu, *sum_w_ls_e_t_sigma_crit_r_batch_gpu;
        double *sum_w_ls_z_s_r_batch_gpu, *sum_w_ls_sigma_crit_r_batch_gpu;
        size_t output_batch_size = (size_t)n_lenses_batch * tables->n_bins;

        CUDA_CHECK(cudaMalloc(&sum_1_r_batch_gpu, output_batch_size * sizeof(long long)));
        // ... (cudaMalloc for other output sum arrays)
        CUDA_CHECK(cudaMalloc(&sum_w_ls_r_batch_gpu, output_batch_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&sum_w_ls_e_t_r_batch_gpu, output_batch_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&sum_w_ls_e_t_sigma_crit_r_batch_gpu, output_batch_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&sum_w_ls_z_s_r_batch_gpu, output_batch_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&sum_w_ls_sigma_crit_r_batch_gpu, output_batch_size * sizeof(double)));

        CUDA_CHECK(cudaMemset(sum_1_r_batch_gpu, 0, output_batch_size * sizeof(long long)));
        CUDA_CHECK(cudaMemset(sum_w_ls_r_batch_gpu, 0, output_batch_size * sizeof(double)));
        // ... (cudaMemset for other output sum arrays)
        CUDA_CHECK(cudaMemset(sum_w_ls_e_t_r_batch_gpu, 0, output_batch_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(sum_w_ls_e_t_sigma_crit_r_batch_gpu, 0, output_batch_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(sum_w_ls_z_s_r_batch_gpu, 0, output_batch_size * sizeof(double)));
        CUDA_CHECK(cudaMemset(sum_w_ls_sigma_crit_r_batch_gpu, 0, output_batch_size * sizeof(double)));


        double *sum_w_ls_m_r_batch_gpu = nullptr, *sum_w_ls_1_minus_e_rms_sq_r_batch_gpu = nullptr;
        double *sum_w_ls_A_p_R_2_r_batch_gpu = nullptr, *sum_w_ls_R_T_r_batch_gpu = nullptr;

        if (tables->has_m_s && tables->sum_w_ls_m_r) {
            CUDA_CHECK(cudaMalloc(&sum_w_ls_m_r_batch_gpu, output_batch_size * sizeof(double)));
            CUDA_CHECK(cudaMemset(sum_w_ls_m_r_batch_gpu, 0, output_batch_size * sizeof(double)));
        }
        // ... (Malloc and Memset for other optional output sums)
        if (tables->has_e_rms_s && tables->sum_w_ls_1_minus_e_rms_sq_r) {
             CUDA_CHECK(cudaMalloc(&sum_w_ls_1_minus_e_rms_sq_r_batch_gpu, output_batch_size * sizeof(double)));
             CUDA_CHECK(cudaMemset(sum_w_ls_1_minus_e_rms_sq_r_batch_gpu, 0, output_batch_size * sizeof(double)));
        }
        if (tables->has_R_2_s && tables->sum_w_ls_A_p_R_2_r) {
             CUDA_CHECK(cudaMalloc(&sum_w_ls_A_p_R_2_r_batch_gpu, output_batch_size * sizeof(double)));
             CUDA_CHECK(cudaMemset(sum_w_ls_A_p_R_2_r_batch_gpu, 0, output_batch_size * sizeof(double)));
        }
        if (tables->has_R_matrix_s && tables->sum_w_ls_R_T_r) {
             CUDA_CHECK(cudaMalloc(&sum_w_ls_R_T_r_batch_gpu, output_batch_size * sizeof(double)));
             CUDA_CHECK(cudaMemset(sum_w_ls_R_T_r_batch_gpu, 0, output_batch_size * sizeof(double)));
        }


        // --- Copy Batch Data (Host to Device) ---
        CUDA_CHECK(cudaMemcpy(z_l_batch_gpu, tables->z_l + i_l_min, n_lenses_batch * sizeof(double), cudaMemcpyHostToDevice));
        // ... (cudaMemcpy for other lens arrays: d_com_l, sin_ra_l, etc. using pointer arithmetic: tables->array_name + i_l_min)
        CUDA_CHECK(cudaMemcpy(d_com_l_batch_gpu, tables->d_com_l + i_l_min, n_lenses_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sin_ra_l_batch_gpu, tables->sin_ra_l + i_l_min, n_lenses_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cos_ra_l_batch_gpu, tables->cos_ra_l + i_l_min, n_lenses_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sin_dec_l_batch_gpu, tables->sin_dec_l + i_l_min, n_lenses_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cos_dec_l_batch_gpu, tables->cos_dec_l + i_l_min, n_lenses_batch * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemcpy(z_s_batch_gpu, z_s_batch.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        // ... (cudaMemcpy for other source arrays using .data() method of std::vector)
        CUDA_CHECK(cudaMemcpy(d_com_s_batch_gpu, d_com_s_batch.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sin_ra_s_batch_gpu, sin_ra_s_batch.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cos_ra_s_batch_gpu, cos_ra_s_batch.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(sin_dec_s_batch_gpu, sin_dec_s_batch.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(cos_dec_s_batch_gpu, cos_dec_s_batch.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w_s_batch_gpu, w_s_batch.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(e_1_s_batch_gpu, e_1_s_batch.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(e_2_s_batch_gpu, e_2_s_batch.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(z_l_max_s_batch_gpu, z_l_max_s_batch.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));


        if (tables->has_sigma_crit_eff) {
            CUDA_CHECK(cudaMemcpy(sigma_crit_eff_l_batch_gpu, tables->sigma_crit_eff_l + (size_t)i_l_min * tables->n_z_bins_l, (size_t)n_lenses_batch * tables->n_z_bins_l * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(z_bin_s_batch_gpu, z_bin_s_batch_host.data(), n_sources_batch * sizeof(int), cudaMemcpyHostToDevice));
        }
        if (tables->has_m_s) CUDA_CHECK(cudaMemcpy(m_s_batch_gpu, m_s_batch_host.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        // ... (cudaMemcpy for other optional source arrays) ...
        if (tables->has_e_rms_s) CUDA_CHECK(cudaMemcpy(e_rms_s_batch_gpu, e_rms_s_batch_host.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        if (tables->has_R_2_s) CUDA_CHECK(cudaMemcpy(R_2_s_batch_gpu, R_2_s_batch_host.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        if (tables->has_R_matrix_s) {
            CUDA_CHECK(cudaMemcpy(R_11_s_batch_gpu, R_11_s_batch_host.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(R_12_s_batch_gpu, R_12_s_batch_host.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(R_21_s_batch_gpu, R_21_s_batch_host.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(R_22_s_batch_gpu, R_22_s_batch_host.data(), n_sources_batch * sizeof(double), cudaMemcpyHostToDevice));
        }


        CUDA_CHECK(cudaMemcpy(dist_3d_sq_bins_batch_gpu, tables->dist_3d_sq_bins + (size_t)i_l_min * (tables->n_bins + 1), (size_t)n_lenses_batch * (tables->n_bins + 1) * sizeof(double), cudaMemcpyHostToDevice));

        // --- Kernel Launch ---
        int threadsPerBlock = 128;
        int blocksPerGrid = (n_lenses_batch + threadsPerBlock - 1) / threadsPerBlock;

        precompute_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            z_l_batch_gpu, d_com_l_batch_gpu, sin_ra_l_batch_gpu, cos_ra_l_batch_gpu, sin_dec_l_batch_gpu, cos_dec_l_batch_gpu,
            z_s_batch_gpu, d_com_s_batch_gpu, sin_ra_s_batch_gpu, cos_ra_s_batch_gpu, sin_dec_s_batch_gpu, cos_dec_s_batch_gpu,
            w_s_batch_gpu, e_1_s_batch_gpu, e_2_s_batch_gpu, z_l_max_s_batch_gpu,
            tables->has_sigma_crit_eff, tables->n_z_bins_l, sigma_crit_eff_l_batch_gpu, z_bin_s_batch_gpu,
            tables->has_m_s, m_s_batch_gpu,
            tables->has_e_rms_s, e_rms_s_batch_gpu,
            tables->has_R_2_s, R_2_s_batch_gpu,
            tables->has_R_matrix_s, R_11_s_batch_gpu, R_12_s_batch_gpu, R_21_s_batch_gpu, R_22_s_batch_gpu,
            dist_3d_sq_bins_batch_gpu, tables->n_bins,
            tables->comoving, tables->weighting,
            sum_1_r_batch_gpu, sum_w_ls_r_batch_gpu, sum_w_ls_e_t_r_batch_gpu, sum_w_ls_e_t_sigma_crit_r_batch_gpu,
            sum_w_ls_z_s_r_batch_gpu, sum_w_ls_sigma_crit_r_batch_gpu,
            sum_w_ls_m_r_batch_gpu, sum_w_ls_1_minus_e_rms_sq_r_batch_gpu,
            sum_w_ls_A_p_R_2_r_batch_gpu, sum_w_ls_R_T_r_batch_gpu,
            n_lenses_batch, n_sources_batch
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Result Retrieval (Device to Host) ---
        size_t offset_global_output = (size_t)i_l_min * tables->n_bins;
        CUDA_CHECK(cudaMemcpy(tables->sum_1_r + offset_global_output, sum_1_r_batch_gpu, output_batch_size * sizeof(long long), cudaMemcpyDeviceToHost));
        // ... (cudaMemcpy for other output sum arrays, adding offset_global_output to destination pointer)
        CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_r + offset_global_output, sum_w_ls_r_batch_gpu, output_batch_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_e_t_r + offset_global_output, sum_w_ls_e_t_r_batch_gpu, output_batch_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_e_t_sigma_crit_r + offset_global_output, sum_w_ls_e_t_sigma_crit_r_batch_gpu, output_batch_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_z_s_r + offset_global_output, sum_w_ls_z_s_r_batch_gpu, output_batch_size * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_sigma_crit_r + offset_global_output, sum_w_ls_sigma_crit_r_batch_gpu, output_batch_size * sizeof(double), cudaMemcpyDeviceToHost));

        if (tables->has_m_s && tables->sum_w_ls_m_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_m_r + offset_global_output, sum_w_ls_m_r_batch_gpu, output_batch_size * sizeof(double), cudaMemcpyDeviceToHost));
        // ... (cudaMemcpy for other optional output sums)
        if (tables->has_e_rms_s && tables->sum_w_ls_1_minus_e_rms_sq_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_1_minus_e_rms_sq_r + offset_global_output, sum_w_ls_1_minus_e_rms_sq_r_batch_gpu, output_batch_size * sizeof(double), cudaMemcpyDeviceToHost));
        if (tables->has_R_2_s && tables->sum_w_ls_A_p_R_2_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_A_p_R_2_r + offset_global_output, sum_w_ls_A_p_R_2_r_batch_gpu, output_batch_size * sizeof(double), cudaMemcpyDeviceToHost));
        if (tables->has_R_matrix_s && tables->sum_w_ls_R_T_r) CUDA_CHECK(cudaMemcpy(tables->sum_w_ls_R_T_r + offset_global_output, sum_w_ls_R_T_r_batch_gpu, output_batch_size * sizeof(double), cudaMemcpyDeviceToHost));


        // --- Cleanup GPU Memory for Batch ---
        CUDA_CHECK(cudaFree(z_l_batch_gpu)); /* ... free all lens_batch_gpu ... */
        CUDA_CHECK(cudaFree(d_com_l_batch_gpu)); CUDA_CHECK(cudaFree(sin_ra_l_batch_gpu)); CUDA_CHECK(cudaFree(cos_ra_l_batch_gpu)); CUDA_CHECK(cudaFree(sin_dec_l_batch_gpu)); CUDA_CHECK(cudaFree(cos_dec_l_batch_gpu));
        CUDA_CHECK(cudaFree(z_s_batch_gpu)); /* ... free all source_batch_gpu ... */
        CUDA_CHECK(cudaFree(d_com_s_batch_gpu)); CUDA_CHECK(cudaFree(sin_ra_s_batch_gpu)); CUDA_CHECK(cudaFree(cos_ra_s_batch_gpu)); CUDA_CHECK(cudaFree(sin_dec_s_batch_gpu)); CUDA_CHECK(cudaFree(cos_dec_s_batch_gpu));
        CUDA_CHECK(cudaFree(w_s_batch_gpu)); CUDA_CHECK(cudaFree(e_1_s_batch_gpu)); CUDA_CHECK(cudaFree(e_2_s_batch_gpu)); CUDA_CHECK(cudaFree(z_l_max_s_batch_gpu));
        if (tables->has_sigma_crit_eff) { CUDA_CHECK(cudaFree(sigma_crit_eff_l_batch_gpu)); CUDA_CHECK(cudaFree(z_bin_s_batch_gpu)); }
        if (tables->has_m_s) CUDA_CHECK(cudaFree(m_s_batch_gpu));
        if (tables->has_e_rms_s) CUDA_CHECK(cudaFree(e_rms_s_batch_gpu));
        if (tables->has_R_2_s) CUDA_CHECK(cudaFree(R_2_s_batch_gpu));
        if (tables->has_R_matrix_s) { CUDA_CHECK(cudaFree(R_11_s_batch_gpu)); CUDA_CHECK(cudaFree(R_12_s_batch_gpu)); CUDA_CHECK(cudaFree(R_21_s_batch_gpu)); CUDA_CHECK(cudaFree(R_22_s_batch_gpu));}

        CUDA_CHECK(cudaFree(dist_3d_sq_bins_batch_gpu));
        CUDA_CHECK(cudaFree(sum_1_r_batch_gpu)); /* ... free all sum_..._batch_gpu ... */
        CUDA_CHECK(cudaFree(sum_w_ls_r_batch_gpu)); CUDA_CHECK(cudaFree(sum_w_ls_e_t_r_batch_gpu)); CUDA_CHECK(cudaFree(sum_w_ls_e_t_sigma_crit_r_batch_gpu));
        CUDA_CHECK(cudaFree(sum_w_ls_z_s_r_batch_gpu)); CUDA_CHECK(cudaFree(sum_w_ls_sigma_crit_r_batch_gpu));
        if (tables->has_m_s && tables->sum_w_ls_m_r) CUDA_CHECK(cudaFree(sum_w_ls_m_r_batch_gpu));
        if (tables->has_e_rms_s && tables->sum_w_ls_1_minus_e_rms_sq_r) CUDA_CHECK(cudaFree(sum_w_ls_1_minus_e_rms_sq_r_batch_gpu));
        if (tables->has_R_2_s && tables->sum_w_ls_A_p_R_2_r) CUDA_CHECK(cudaFree(sum_w_ls_A_p_R_2_r_batch_gpu));
        if (tables->has_R_matrix_s && tables->sum_w_ls_R_T_r) CUDA_CHECK(cudaFree(sum_w_ls_R_T_r_batch_gpu));

        // std::cout << "Processed lens pixel " << current_pix_l << " (batch " << idx_u_pix_l+1 << "/" << u_pix_l.size() << ") with " << n_lenses_batch << " lenses and " << n_sources_batch << " sources." << std::endl;
    }

    // --- 6. Final Cleanup ---
    // CUDA_CHECK(cudaFree(fixed_array_gpu)); // If any were allocated

    std::cout << "precompute_cuda_interface completed successfully." << std::endl;
    return 0; // Success
}
