#ifndef PRECOMPUTE_ENGINE_CUDA_H
#define PRECOMPUTE_ENGINE_CUDA_H

#include <cuda_runtime.h> // For __device__

// Constants (can also be defined in .cu if not needed by other host files)
#define SIGMA_CRIT_FACTOR_ENGINE 1.6629165401756011e+06 // Renamed to avoid conflict if precompute_interface also defines it
#define DEG2RAD_ENGINE 0.017453292519943295


// Device function to calculate 3D angular distance squared (on unit sphere)
__device__ double dist_angular_sq_gpu(
    double sin_ra_1, double cos_ra_1, double sin_dec_1, double cos_dec_1,
    double sin_ra_2, double cos_ra_2, double sin_dec_2, double cos_dec_2);

// Device function to calculate projected physical distance squared
__device__ double dist_projected_sq_gpu(
    double d_com_l, // Comoving distance to lens
    double sin_ra_l, double cos_ra_l, double sin_dec_l, double cos_dec_l, // Lens coords
    double sin_ra_s, double cos_ra_s, double sin_dec_s, double cos_dec_s, // Source coords
    bool comoving // if true, d_com_l is used as physical transverse distance for projection
);


// Device function to find the bin index for a given squared distance
__device__ int find_bin_idx_gpu(
    double dist_sq,
    const double* current_lens_dist_bins, // Pointer to this lens's bins: g_dist_3d_sq_bins + lens_idx * (N_bins+1)
    int N_bins);

// Device function to calculate sigma_crit_inverse
__device__ double calculate_sigma_crit_inv_gpu(
    double zl_i, double zs_i,
    double dcoml_i, double dcoms_i,
    bool comoving,
    // For effective sigma_crit
    bool has_sigma_crit_eff,
    const double* g_sigma_crit_eff_l, // Base pointer: lens_idx * n_z_bins_l + z_bin_s_idx
    int current_lens_idx, // The current global lens index
    int n_z_bins_l,       // Total number of redshift bins for lenses for sigma_crit_eff
    int z_bin_s_val       // Source's redshift bin index for sigma_crit_eff
);

// Device function to calculate w_ls
__device__ double calculate_w_ls_gpu(
    double sigma_crit_inv, // Inverse of sigma_crit
    double w_s_i,
    int weighting_type // Changed from float to int for clarity if it's type enum
);

// Device function to calculate tangential shear e_t components (cos2phi, sin2phi)
__device__ void calculate_et_components_gpu(
    double sin_ra_l, double cos_ra_l, double sin_dec_l, double cos_dec_l, // Lens
    double sin_ra_s, double cos_ra_s, double sin_dec_s, double cos_dec_s, // Source
    double& cos_2phi, double& sin_2phi);

// Device function to calculate tangential shear e_t from components and source ellipticities
__device__ double calculate_et_gpu(
    double e1_s_i, double e2_s_i,
    double cos_2phi, double sin_2phi);

// Optional: Device function for R_T calculation if complex
__device__ double calculate_R_T_gpu(
    double R_11_s_i, double R_12_s_i,
    double R_21_s_i, double R_22_s_i,
    double cos_2phi, double sin_2phi);


#endif // PRECOMPUTE_ENGINE_CUDA_H