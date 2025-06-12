#include <cuda_runtime.h>
#include <cmath>
#include <cfloat> // For DBL_MAX

// Constants
#define SIGMA_CRIT_FACTOR 1.662890013800909e+09
#define DEG2RAD 0.017453292519943295

// Forward declarations of device functions (if needed, or define before use)

// Device function to calculate 3D distance squared
__device__ double dist_3d_sq_gpu(
    double sin_ra_1, double cos_ra_1, double sin_dec_1, double cos_dec_1, double d_com_1,
    double sin_ra_2, double cos_ra_2, double sin_dec_2, double cos_dec_2, double d_com_2,
    bool comoving) {

    double cos_angle = sin_dec_1 * sin_dec_2 + cos_dec_1 * cos_dec_2 * (sin_ra_1 * sin_ra_2 + cos_ra_1 * cos_ra_2);
    // Ensure cos_angle is within [-1, 1] to avoid NaNs from acos due to precision errors
    if (cos_angle > 1.0) cos_angle = 1.0;
    if (cos_angle < -1.0) cos_angle = -1.0;

    if (comoving) {
        // Comoving distance squared
        return d_com_1 * d_com_1 + d_com_2 * d_com_2 - 2.0 * d_com_1 * d_com_2 * cos_angle;
    } else {
        // Projected distance squared
        double d_ang_12_sq = d_com_1 * d_com_1 * (1.0 - cos_angle * cos_angle) / (1.0 + cos_angle); // Approximation for small angles
        if (cos_angle < 0.999999) { // if angle is not very small
             d_ang_12_sq = d_com_1 * d_com_1 * 2.0 * (1.0 - cos_angle); // More stable for larger angles
        }
        // Ensure d_ang_12_sq is not negative due to precision errors
        if (d_ang_12_sq < 0) d_ang_12_sq = 0;
        return d_ang_12_sq;
    }
}

// Device function to calculate sigma_crit
__device__ double calculate_sigma_crit_gpu(
    double d_com_l, double d_com_s, double z_l, double z_s,
    bool has_sigma_crit_eff, int n_z_bins_l, double* sigma_crit_eff_l, int i_l, int z_bin_s_val) {

    if (has_sigma_crit_eff) {
        if (sigma_crit_eff_l != nullptr && z_bin_s_val >= 0 && z_bin_s_val < n_z_bins_l) {
             return sigma_crit_eff_l[i_l * n_z_bins_l + z_bin_s_val];
        } else {
            // Fallback or error handling if z_bin_s_val is out of bounds or pointer is null
            // This case should ideally be prevented by input validation or logic upstream
            return DBL_MAX; // Or some indicator of an issue
        }
    } else {
        if (d_com_s <= d_com_l) { // lens behind source or at same distance
            return DBL_MAX; // Effectively infinite, w_ls will be 0
        }
        double d_ls = (d_com_s - d_com_l) / (1.0 + z_s); // Proper distance
        return SIGMA_CRIT_FACTOR * d_com_s / (d_com_l * d_ls * (1.0 + z_l) * (1.0 + z_l));
    }
}

// Device function to calculate w_ls
__device__ double calculate_w_ls_gpu(double sigma_crit, double w_s, float weighting) {
    if (sigma_crit == DBL_MAX || sigma_crit == 0.0) { // Check for invalid sigma_crit
        return 0.0;
    }
    double sigma_crit_sq = sigma_crit * sigma_crit;
    return w_s / (sigma_crit_sq * pow(sigma_crit_sq, weighting / 2.0f));
}

// Device function to calculate cos(2*phi) and sin(2*phi) for tangential shear
__device__ void calculate_et_components_gpu(
    double sin_ra_l, double cos_ra_l, double sin_dec_l, double cos_dec_l,
    double sin_ra_s, double cos_ra_s, double sin_dec_s, double cos_dec_s,
    double& cos_2phi, double& sin_2phi) {

    // Calculate position angle phi of source relative to lens
    // Similar to astropy.coordinates.position_angle
    // RA/Dec are in radians here
    double delta_ra = ra_s - ra_l; // Assuming ra_s, ra_l are already available or passed
                                   // This needs actual ra_l, dec_l, ra_s, dec_s
                                   // The current inputs are sin/cos components.
                                   // We need to reconstruct angles or use a different formulation.

    // Reconstruct angles (less efficient, but direct from inputs if needed)
    // double ra_l_rad = atan2(sin_ra_l, cos_ra_l);
    // double dec_l_rad = asin(sin_dec_l); // asin range is [-pi/2, pi/2] which is correct for dec
    // double ra_s_rad = atan2(sin_ra_s, cos_ra_s);
    // double dec_s_rad = asin(sin_dec_s);

    // Using spherical trigonometry for position angle
    // cos(theta_s) = sin(dec_l)sin(dec_s) + cos(dec_l)cos(dec_s)cos(ra_s - ra_l)
    // sin_pa = cos(dec_s) sin(ra_s - ra_l) / sin(theta_s)
    // cos_pa = (sin(dec_s)cos(dec_l) - cos(dec_s)sin(dec_l)cos(ra_s-ra_l)) / sin(theta_s)
    // This is for PA. For phi (lens-source orientation for shear), it's more complex.

    // Let's use the delta_x, delta_y approach on the sphere (gnomonic projection approximation for small angles)
    // Or better, use the definition from the existing Cython code's delta_alpha, delta_delta
    // delta_alpha = (ra_s - ra_l) * cos(dec_l)
    // delta_delta = dec_s - dec_l
    // phi = atan2(delta_delta, delta_alpha)
    // This requires actual angles.

    // Alternative using vectors (more robust for large separations, though shear is usually small sep.)
    // Vector L (lens) and S (source) from observer O.
    // We need the angle of the vector L->S in the tangent plane at L, relative to the local North/East.
    // Let's use the same logic as in the Cython `precompute_engine.pyx` for `phi_l`.
    // It uses `np.arctan2(y, x)` where:
    // x = cos_dec_s * sin(ra_s - ra_l)
    // y = sin_dec_s * cos_dec_l - cos_dec_s * sin_dec_l * cos(ra_s - ra_l)
    // This is the position angle of S *from* L.

    double delta_ra_rad = acos(fmax(-1.0, fmin(1.0, cos_ra_s * cos_ra_l + sin_ra_s * sin_ra_l))); // More stable way to get delta_ra if cos(delta_ra) is known
                                                                                           // This is not delta_ra directly.
                                                                                           // It's simpler to use atan2(sin_delta_ra, cos_delta_ra)

    // We need ra_l, dec_l, ra_s, dec_s. The inputs are sin/cos of these.
    // It's more numerically stable to work with sin/cos as much as possible.
    // delta_ra = ra_s - ra_l.
    // sin(delta_ra) = sin_ra_s * cos_ra_l - cos_ra_s * sin_ra_l
    // cos(delta_ra) = cos_ra_s * cos_ra_l + sin_ra_s * sin_ra_l

    double sin_delta_ra = sin_ra_s * cos_ra_l - cos_ra_s * sin_ra_l;
    double cos_delta_ra = cos_ra_s * cos_ra_l + sin_ra_s * sin_ra_l;

    double x = cos_dec_s * sin_delta_ra;
    double y = sin_dec_s * cos_dec_l - cos_dec_s * sin_dec_l * cos_delta_ra;

    // phi is atan2(y,x). We need cos(2*phi) and sin(2*phi).
    // cos(2a) = cos^2(a) - sin^2(a) = (x^2 - y^2) / (x^2 + y^2)
    // sin(2a) = 2sin(a)cos(a) = 2xy / (x^2 + y^2)
    double norm_sq = x * x + y * y;
    if (norm_sq == 0) { // Should not happen if lens and source are not at the exact same RA/Dec
        cos_2phi = 1.0; // Default to avoid division by zero, e.g., e_t = e_1
        sin_2phi = 0.0;
    } else {
        cos_2phi = (x * x - y * y) / norm_sq;
        sin_2phi = (2.0 * x * y) / norm_sq;
    }
}


// Device function to calculate tangential shear e_t
__device__ double calculate_et_gpu(double e1_s, double e2_s, double cos_2phi, double sin_2phi) {
    return -e1_s * cos_2phi - e2_s * sin_2phi;
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
            sin_ra_l_i, cos_ra_l_i, sin_dec_l_i, cos_dec_l_i, dcoml_i,
            sin_ra_s_i, cos_ra_s_i, sin_dec_s_i, cos_dec_s_i, dcoms_i,
            comoving);

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
            has_sigma_crit_eff, n_z_bins_l, sigma_crit_eff_l, i_l, z_bin_s_val);

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
