#ifndef PRECOMPUTE_ENGINE_CUDA_H
#define PRECOMPUTE_ENGINE_CUDA_H

#include "precompute_interface.h" // To get the definition of TableData

// Declaration of the host function that launches the CUDA kernel.
// This function should be defined in precompute_engine_cuda.cu.
void launch_cuda_precomputation(TableData* tables, int n_gpus);
__global__ void precompute_kernel(
    // Lens properties
    double* z_l, double* d_com_l,
    double* sin_ra_l, double* cos_ra_l, double* sin_dec_l, double* cos_dec_l,
    // Source properties
    double* z_s, double* d_com_s,
    double* sin_ra_s, double* cos_ra_s, double* sin_dec_s, double* cos_dec_s,
    double* w_s, double* e_1_s, double* e_2_s, double* z_l_max_s,
    // Sigma_crit_eff related
    bool has_sigma_crit_eff, int n_z_bins_l, double* sigma_crit_eff_l, int* z_bin_s,
    // Optional source properties
    bool has_m_s, double* m_s,
    bool has_e_rms_s, double* e_rms_s,
    bool has_R_2_s, double* R_2_s,
    bool has_R_matrix_s, double* R_11_s, double* R_12_s, double* R_21_s, double* R_22_s,
    // Binning
    double* dist_3d_sq_bins, // Flattened: n_lenses * (n_bins + 1)
    int n_bins,
    // Configuration
    bool comoving, float weighting,
    // Output sum arrays (flattened: n_lenses * n_bins)
    long long* sum_1_r,
    double* sum_w_ls_r,
    double* sum_w_ls_e_t_r,
    double* sum_w_ls_e_t_sigma_crit_r,
    double* sum_w_ls_z_s_r,
    double* sum_w_ls_sigma_crit_r,
    // Optional output sum arrays
    double* sum_w_ls_m_r,                 // if has_m_s
    double* sum_w_ls_1_minus_e_rms_sq_r,  // if has_e_rms_s
    double* sum_w_ls_A_p_R_2_r,           // if has_R_2_s
    double* sum_w_ls_R_T_r,               // if has_R_matrix_s
    // Counts
    int n_lenses, int n_sources
    // No n_pix_l, n_pix_s needed as per current design (1 thread per lens)
);
#endif // PRECOMPUTE_ENGINE_CUDA_H