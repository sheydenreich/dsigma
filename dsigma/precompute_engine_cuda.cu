#include <cuda_runtime.h>
#include <cmath>
#include <cfloat> // For DBL_MAX

#include "precompute_engine_cuda.h"

// Constants
#define SIGMA_CRIT_FACTOR 1.6629165401756011e+06
#define DEG2RAD 0.017453292519943295

// Forward declarations of device functions (if needed, or define before use)

// Device function to calculate 3D distance squared
__device__ double dist_3d_sq_gpu(
    double sin_ra_1, double cos_ra_1, double sin_dec_1, double cos_dec_1,
    double sin_ra_2, double cos_ra_2, double sin_dec_2, double cos_dec_2) {

    // Match CPU implementation exactly: 3D Cartesian distance on unit sphere
    double dx = cos_ra_1 * cos_dec_1 - cos_ra_2 * cos_dec_2;
    double dy = sin_ra_1 * cos_dec_1 - sin_ra_2 * cos_dec_2;
    double dz = sin_dec_1 - sin_dec_2;
    
    return dx * dx + dy * dy + dz * dz;
}

// Device function to calculate sigma_crit
__device__ double calculate_sigma_crit_gpu(
    double d_com_l, double d_com_s, double z_l, double z_s,
    bool has_sigma_crit_eff, int n_z_bins_l, double* sigma_crit_eff_l, int i_l, int z_bin_s_val,
    bool comoving) {

    if (has_sigma_crit_eff) {
        if (sigma_crit_eff_l != nullptr && z_bin_s_val >= 0 && z_bin_s_val < n_z_bins_l) {
             return sigma_crit_eff_l[i_l * n_z_bins_l + z_bin_s_val];
        } else {
            // Fallback or error handling if z_bin_s_val is out of bounds or pointer is null
            // This case should ideally be prevented by input validation or logic upstream
            return DBL_MAX; // Or some indicator of an issue
        }
    } else {
        if (d_com_l >= d_com_s) { // lens behind source or at same distance
            return DBL_MAX; // Effectively infinite, w_ls will be 0
        }
        // Match CPU implementation exactly
        double sigma_crit = SIGMA_CRIT_FACTOR * (1.0 + z_l) * d_com_s / d_com_l / (d_com_s - d_com_l);
        if (comoving) {
            sigma_crit /= (1.0 + z_l) * (1.0 + z_l);
        }
        return sigma_crit;
    }
}

// Device function to calculate w_ls
__device__ double calculate_w_ls_gpu(double sigma_crit, double w_s, float weighting) {
    if (sigma_crit == DBL_MAX || sigma_crit == 0.0) { // Check for invalid sigma_crit
        return 0.0;
    }
    
    if (weighting == 0.0f) {
        return w_s;
    } else if (weighting == -2.0f) {
        return w_s / (sigma_crit * sigma_crit);
    } else {
        return w_s * pow(sigma_crit, weighting);
    }
}


__device__ void calculate_et_components_gpu(
    double sin_ra_l, double cos_ra_l, double sin_dec_l, double cos_dec_l,
    double sin_ra_s, double cos_ra_s, double sin_dec_s, double cos_dec_s,
    double& cos_2phi, double& sin_2phi) {

    // Match CPU implementation exactly
    double sin_ra_l_minus_ra_s = sin_ra_l * cos_ra_s - cos_ra_l * sin_ra_s;
    double cos_ra_l_minus_ra_s = cos_ra_l * cos_ra_s + sin_ra_l * sin_ra_s;
    double tan_phi_num = cos_dec_s * sin_dec_l - sin_dec_s * cos_dec_l * cos_ra_l_minus_ra_s;
    double tan_phi_den = cos_dec_l * sin_ra_l_minus_ra_s;
    
    if (tan_phi_den == 0) {
        cos_2phi = -1.0;
        sin_2phi = 0.0;
    } else {
        double tan_phi = tan_phi_num / tan_phi_den;
        cos_2phi = (2.0 / (1.0 + tan_phi * tan_phi)) - 1.0;
        sin_2phi = 2.0 * tan_phi / (1.0 + tan_phi * tan_phi);
    }
}


// Device function to calculate tangential shear e_t
__device__ double calculate_et_gpu(double e1_s, double e2_s, double cos_2phi, double sin_2phi) {
    return -e1_s * cos_2phi + e2_s * sin_2phi;
}


// Main CUDA kernel
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
) {
    int i_l = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_l >= n_lenses) {
        return;
    }

    // Retrieve lens data for this thread
    double zl_i = z_l[i_l];
    double dcoml_i = d_com_l[i_l];
    double sin_ra_l_i = sin_ra_l[i_l];
    double cos_ra_l_i = cos_ra_l[i_l];
    double sin_dec_l_i = sin_dec_l[i_l];
    double cos_dec_l_i = cos_dec_l[i_l];

    for (int i_s = 0; i_s < n_sources; ++i_s) {
        // Apply redshift filter
        if (zl_i > z_l_max_s[i_s]) {
            continue;
        }

        // Retrieve source data
        double zs_i = z_s[i_s];
        double dcoms_i = d_com_s[i_s];
        double sin_ra_s_i = sin_ra_s[i_s];
        double cos_ra_s_i = cos_ra_s[i_s];
        double sin_dec_s_i = sin_dec_s[i_s];
        double cos_dec_s_i = cos_dec_s[i_s];

        // Calculate 3D distance squared
        double dist_3d_sq_ls = dist_3d_sq_gpu(
            sin_ra_l_i, cos_ra_l_i, sin_dec_l_i, cos_dec_l_i,
            sin_ra_s_i, cos_ra_s_i, sin_dec_s_i, cos_dec_s_i);

        // Binning logic
        long i_bin = n_bins;
        // Pointer to the start of bins for the current lens i_l
        double* current_lens_bins = dist_3d_sq_bins + i_l * (n_bins + 1);
        while (i_bin >= 0) {
            if (dist_3d_sq_ls > current_lens_bins[i_bin]) {
                break;
            }
            i_bin--;
        }

        if (i_bin == n_bins || i_bin < 0) {
            continue;
        }

        // Valid bin found: i_bin is the index for this lens-source pair

        // Calculate sigma_crit
        int z_bin_s_val = -1; // Default if not used
        if (has_sigma_crit_eff && z_bin_s != nullptr) {
            z_bin_s_val = z_bin_s[i_s];
        }
        double sigma_crit = calculate_sigma_crit_gpu(
            dcoml_i, dcoms_i, zl_i, zs_i,
            has_sigma_crit_eff, n_z_bins_l, sigma_crit_eff_l, i_l, z_bin_s_val, comoving);

        if (sigma_crit == DBL_MAX) { // Or check against any other "invalid" value from calculate_sigma_crit_gpu
            continue;
        }

        // Calculate w_ls
        double w_ls = calculate_w_ls_gpu(sigma_crit, w_s[i_s], weighting);

        if (w_ls == 0.0) {
            continue;
        }

        // Calculate tangential shear components
        double cos_2phi, sin_2phi;
        calculate_et_components_gpu(
            sin_ra_l_i, cos_ra_l_i, sin_dec_l_i, cos_dec_l_i,
            sin_ra_s_i, cos_ra_s_i, sin_dec_s_i, cos_dec_s_i,
            cos_2phi, sin_2phi);

        double e_t = calculate_et_gpu(e_1_s[i_s], e_2_s[i_s], cos_2phi, sin_2phi);

        // Update sums for the bin: i_l * n_bins + i_bin
        long long result_idx = (long long)i_l * n_bins + i_bin;

        // Atomcis not strictly needed here if one thread handles one lens and its sources sequentially.
        // If multiple threads could update the same lens's bins (not the current design), atomics would be vital.
        sum_1_r[result_idx]++;
        sum_w_ls_r[result_idx] += w_ls;
        sum_w_ls_e_t_r[result_idx] += w_ls * e_t;
        sum_w_ls_e_t_sigma_crit_r[result_idx] += w_ls * e_t * sigma_crit;
        sum_w_ls_z_s_r[result_idx] += w_ls * zs_i;
        sum_w_ls_sigma_crit_r[result_idx] += w_ls * sigma_crit;

        // Optional sums
        if (has_m_s && m_s != nullptr && sum_w_ls_m_r != nullptr) {
            sum_w_ls_m_r[result_idx] += w_ls * m_s[i_s];
        }

        if (has_e_rms_s && e_rms_s != nullptr && sum_w_ls_1_minus_e_rms_sq_r != nullptr) {
            sum_w_ls_1_minus_e_rms_sq_r[result_idx] += w_ls * (1.0 - e_rms_s[i_s] * e_rms_s[i_s]);
        }

        if (has_R_2_s && R_2_s != nullptr && sum_w_ls_A_p_R_2_r != nullptr) {
            if (R_2_s[i_s] <= 0.31) {
                sum_w_ls_A_p_R_2_r[result_idx] += 0.00865 * w_ls / 0.01;
            }
        }

        if (has_R_matrix_s && R_11_s != nullptr && R_12_s != nullptr &&
            R_21_s != nullptr && R_22_s != nullptr && sum_w_ls_R_T_r != nullptr) {
            double R_T_val = R_11_s[i_s] * cos_2phi * cos_2phi +
                             R_22_s[i_s] * sin_2phi * sin_2phi +
                             (R_12_s[i_s] + R_21_s[i_s]) * sin_2phi * cos_2phi;
            sum_w_ls_R_T_r[result_idx] += w_ls * R_T_val;
        }
    }
}
