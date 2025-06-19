#include <cuda_runtime.h>
#include <cmath> // For sqrt, sin, cos, acos, pow
#include <cfloat> // For DBL_MAX

#include "precompute_engine_cuda.h" // Contains new declarations and constants

// Note: SIGMA_CRIT_FACTOR_ENGINE and DEG2RAD_ENGINE are defined in the header.

// Device function to calculate 3D angular distance squared (on unit sphere)
// This is the previous dist_3d_sq_gpu, renamed for clarity.
__device__ double dist_angular_sq_gpu(
    double sin_ra_1, double cos_ra_1, double sin_dec_1, double cos_dec_1,
    double sin_ra_2, double cos_ra_2, double sin_dec_2, double cos_dec_2) {
    // 3D Cartesian distance on unit sphere from angular coordinates
    double dx = cos_ra_1 * cos_dec_1 - cos_ra_2 * cos_dec_2;
    double dy = sin_ra_1 * cos_dec_1 - sin_ra_2 * cos_dec_2;
    double dz = sin_dec_1 - sin_dec_2;
    return dx * dx + dy * dy + dz * dz; // This is chord distance squared.
                                       // For small angles, (chord_dist)^2 approx (angular_dist_rad)^2
}

// Device function to calculate projected physical distance squared
__device__ double dist_projected_sq_gpu(
    double d_com_l, // Comoving distance to lens
    double sin_ra_l, double cos_ra_l, double sin_dec_l, double cos_dec_l, // Lens coords
    double sin_ra_s, double cos_ra_s, double sin_dec_s, double cos_dec_s, // Source coords
    bool comoving // if true, d_com_l is used as physical transverse distance for projection
                  // This 'comoving' flag here refers to how d_com_l relates to physical transverse separation.
                  // If d_com_l is already D_A(lens), then use directly.
                  // If d_com_l is comoving D_M(lens), then D_A(lens) = D_M(lens)/(1+zl).
                  // The problem description's `comoving` flag for Sigma_crit calculation
                  // indicates whether D_L, D_S, D_LS are comoving or physical angular diameter distances.
                  // Let's assume d_com_l is comoving distance D_M.
                  // The projected distance R_proj = D_A(lens) * theta.
                  // theta is the angular separation. theta^2 approx dist_angular_sq_gpu for small angles.
) {
    double ang_dist_sq = dist_angular_sq_gpu(sin_ra_l, cos_ra_l, sin_dec_l, cos_dec_l,
                                             sin_ra_s, cos_ra_s, sin_dec_s, cos_dec_s);
    
    // ang_dist_sq is roughly theta^2 for small theta.
    // More accurately, ang_dist_sq = (2*sin(theta/2))^2. So theta = 2*asin(sqrt(ang_dist_sq)/2).
    // For small theta, theta approx sqrt(ang_dist_sq).
    // R_proj = D_A_lens * theta. So R_proj_sq = D_A_lens^2 * theta^2.
    // If d_com_l is comoving distance D_M(lens), then D_A(lens) = D_M(lens)/(1+z_lens).
    // The `comoving` parameter in TableData seems to refer to the definition of SigmaCrit distances.
    // For projected distance, we usually use Angular Diameter Distance to the lens.
    // Let's assume d_com_l is the comoving distance D_M. The problem needs z_l to get D_A(lens).
    // This function needs z_l to correctly calculate D_A(lens) if d_com_l is D_M(lens).
    // However, the `KernelCallbackData` has `dcoml_i` which is `g_d_com_l[lens_idx]`.
    // The original kernel used `d_com_l` directly in `calculate_sigma_crit_gpu`.
    // Let's assume for now `d_com_l` is the appropriate distance scale for projection (e.g. D_A(lens) or D_M(lens) if used consistently).
    // If `d_com_l` is D_M(lens), then projected separation squared is (D_M(lens) * theta_approx)^2
    // This is a simplification. The actual projected distance depends on cosmological model and definitions.
    // A common approximation for small angles: (d_com_l * sqrt(ang_dist_sq))^2
    return d_com_l * d_com_l * ang_dist_sq; // This is (D_M * theta_approx)^2
                                            // If d_com_l is already D_A, then this is correct.
}


// Device function to find the bin index
__device__ int find_bin_idx_gpu(
    double dist_sq,
    const double* current_lens_dist_bins, // Pointer to this lens's bins
    int N_bins) {
    // Bins are [bin_edges[0], bin_edges[1]), [bin_edges[1], bin_edges[2]), ..., [bin_edges[N-1], bin_edges[N])
    // current_lens_dist_bins has N_bins+1 elements (edges).
    // Perform a binary search.
    if (dist_sq < current_lens_dist_bins[0]) return -1;
    if (dist_sq >= current_lens_dist_bins[N_bins]) return -1;


    int low = 0;
    int high = N_bins;
    int mid;

    while (low < high) {
        // Calculate mid-point to avoid potential overflow
        mid = low + (high - low) / 2;

        if (dist_sq >= current_lens_dist_bins[mid]) {
            // The value is in the current bin or a higher one.
            // Move the lower bound up.
            low = mid + 1;
        } else {
            // The value is in a lower bin.
            // Move the upper bound down.
            high = mid;
        }
    }

    // The loop terminates when low == high. 'low' now points to the first element
    // in bin_edges that is greater than x. The bin index is therefore low - 1.
    return low - 1;
}


// Device function to calculate sigma_crit_inverse
__device__ double calculate_sigma_crit_inv_gpu(
    double zl_i, double zs_i,
    double dcoml_i, double dcoms_i,
    bool comoving, // This flag dictates interpretation of dcoml_i, dcoms_i for SigmaCrit
    bool has_sigma_crit_eff,
    const double* g_sigma_crit_eff_l, // Base pointer for all lenses' sigma_crit_eff tables
    int current_lens_idx,      // Global lens index
    int n_z_bins_l,            // Number of redshift bins for lens in sigma_crit_eff table
    int z_bin_s_val            // Source's redshift bin for sigma_crit_eff table
) {
    if (zl_i >= zs_i) return 0.0; // Lens not in front of source, so SigmaCrit is infinite, inverse is 0.

    double sigma_crit_val;
    if (has_sigma_crit_eff) {
        if (g_sigma_crit_eff_l != nullptr && z_bin_s_val >= 0 && z_bin_s_val < n_z_bins_l) {
            // Access the specific sigma_crit_eff for this lens and this source's redshift bin
            sigma_crit_val = g_sigma_crit_eff_l[(size_t)current_lens_idx * n_z_bins_l + z_bin_s_val];
        } else {
            return 0.0; // Error or invalid input, treat as infinite SigmaCrit
        }
    } else {
        // Standard calculation
        if (dcoml_i >= dcoms_i && dcoms_i > 0) { // dcoms_i > 0 ensures not at observer
             // Lens behind or at same comoving distance as source.
             // (But zl_i < zs_i already checked, so this implies non-standard distance-redshift relation)
             // Or, more likely, dcoml_i, dcoms_i are such that D_LS would be <=0
            return 0.0; // SigmaCrit infinite
        }
        if (dcoml_i <= 0 || dcoms_i <= 0) return 0.0; // Invalid distances

        // Using the formula from the original kernel:
        // sigma_crit = SIGMA_CRIT_FACTOR * (1.0 + z_l) * d_com_s / d_com_l / (d_com_s - d_com_l);
        // if (comoving) { sigma_crit /= (1.0 + z_l) * (1.0 + z_l); }
        // This can be rearranged for SigmaCrit_inv = 1.0 / sigma_crit

        double term_dl_dsl_ds = dcoml_i * (dcoms_i - dcoml_i) / dcoms_i; // D_L * D_LS / D_S part (comoving)
        if (term_dl_dsl_ds <= 0) return 0.0; // Avoid division by zero or negative if dcoml_i > dcoms_i

        double sigma_crit_inv_val = term_dl_dsl_ds / (SIGMA_CRIT_FACTOR_ENGINE * (1.0 + zl_i));
        if (comoving) {
            // Original: sigma_crit_comoving = sigma_crit_phys / (1+zl)^2
            // So, sigma_crit_inv_comoving = sigma_crit_inv_phys * (1+zl)^2
            sigma_crit_inv_val *= (1.0 + zl_i) * (1.0 + zl_i);
        }
        sigma_crit_val = 1.0 / sigma_crit_inv_val; // This function calculates sigma_crit_inv, so return sigma_crit_inv_val
                                                // The original was `calculate_sigma_crit_gpu`
                                                // Let's stick to inv.
        return sigma_crit_inv_val;
    }

    // If sigma_crit_val was from table and is 0 or DBL_MAX, its inverse is problematic
    if (sigma_crit_val == 0.0) return DBL_MAX; // Inverse of 0 is infinite
    if (sigma_crit_val == DBL_MAX) return 0.0;   // Inverse of infinite is 0
    return 1.0 / sigma_crit_val;
}


// Device function to calculate w_ls
__device__ double calculate_w_ls_gpu(
    double sigma_crit_inv, // Inverse of SigmaCrit
    double w_s_i,
    int weighting_type // (0: w_s, -2: w_s * SigmaCrit_inv^2, other: w_s * (1/SigmaCrit_inv)^weighting_type)
) {
    if (sigma_crit_inv == DBL_MAX) { // sigma_crit was 0
        if (weighting_type == 0) return w_s_i;
        else return 0.0; // Any power of 0 is 0, unless 0^0 (undefined) or negative power (inf)
                         // For lensing, if sigma_crit is 0, w_ls is typically 0 or undefined.
    }
    if (sigma_crit_inv == 0.0) { // sigma_crit was infinite
         return 0.0; // w_ls is 0 if sigma_crit is infinite and weighting is applied
    }

    if (weighting_type == 0) { // weight_type 0 means w_s
        return w_s_i;
    } else if (weighting_type == -2) { // weight_type -2 means w_s * SigmaCrit_inv^2
        return w_s_i * sigma_crit_inv * sigma_crit_inv;
    } else {
        // General case: w_s * SigmaCrit^weighting_type = w_s * (1/SigmaCrit_inv)^weighting_type
        return w_s_i * pow(1.0 / sigma_crit_inv, static_cast<double>(weighting_type));
    }
}

// Device function to calculate tangential shear e_t components
__device__ void calculate_et_components_gpu(
    double sin_ra_l, double cos_ra_l, double sin_dec_l, double cos_dec_l,
    double sin_ra_s, double cos_ra_s, double sin_dec_s, double cos_dec_s,
    double& cos_2phi, double& sin_2phi) {
    // This implementation is identical to the original kernel's version.
    double sin_ra_l_minus_ra_s = sin_ra_l * cos_ra_s - cos_ra_l * sin_ra_s;
    double cos_ra_l_minus_ra_s = cos_ra_l * cos_ra_s + sin_ra_l * sin_ra_s;
    double tan_phi_num = cos_dec_s * sin_dec_l - sin_dec_s * cos_dec_l * cos_ra_l_minus_ra_s;
    double tan_phi_den = cos_dec_l * sin_ra_l_minus_ra_s;
    
    // Handle cases where tan_phi_den is zero (phi is 0 or PI)
    // If tan_phi_den is zero, phi is +/- PI/2 or 3PI/2, so 2phi is +/- PI or 3PI. cos(2phi) = -1, sin(2phi) = 0.
    // This logic seems to handle it: if tan_phi_den is 0, tan_phi would be inf.
    // The original code's check was `if (tan_phi_den == 0)`.
    // A more robust check for tan_phi_den being very small might be needed if precision issues arise.
    if (fabs(tan_phi_den) < 1e-12) { // Avoid division by zero, effectively tan_phi is infinite or phi is fixed.
        // If tan_phi_den is 0, then ra_l = ra_s.
        // If additionally cos_dec_l is 0 (lens at pole), this could be an issue.
        // However, typical case: phi is PI/2 or -PI/2.
        // If tan_phi_num > 0, phi ~ PI/2, 2phi ~ PI  => cos(2phi)=-1, sin(2phi)=0
        // If tan_phi_num < 0, phi ~ -PI/2, 2phi ~ -PI => cos(2phi)=-1, sin(2phi)=0
        // If tan_phi_num = 0, then source is at same declination as lens (if also ra_l=ra_s), undefined or special case.
        // The original code simply set cos_2phi = -1, sin_2phi = 0. Let's stick to that for consistency.
        cos_2phi = -1.0;
        sin_2phi = 0.0;
    } else {
        double tan_phi = tan_phi_num / tan_phi_den;
        double tan_phi_sq = tan_phi * tan_phi;
        // cos(2a) = (1 - tan^2(a)) / (1 + tan^2(a))
        // sin(2a) = 2*tan(a) / (1 + tan^2(a))
        // The original CPU code used: cos_2phi = (2.0 / (1.0 + tan_phi * tan_phi)) - 1.0
        // Let's match the CPU implementation exactly
        cos_2phi = (2.0 / (1.0 + tan_phi_sq)) - 1.0;
        sin_2phi = 2.0 * tan_phi / (1.0 + tan_phi_sq);
    }
}

// Device function to calculate tangential shear e_t
__device__ double calculate_et_gpu(
    double e1_s_i, double e2_s_i,
    double cos_2phi, double sin_2phi) {
    // e_t = -(e1*cos(2phi) + e2*sin(2phi)) -- This is a common convention.
    // The original kernel had: -e1_s * cos_2phi + e2_s * sin_2phi
    // This seems to be a difference in definition of e_t vs e_x (cross shear) or phi angle.
    // Let's stick to the original formulation:
    return -e1_s_i * cos_2phi + e2_s_i * sin_2phi;
}

// Optional: Device function for R_T calculation
__device__ double calculate_R_T_gpu(
    double R_11_s_i, double R_12_s_i,
    double R_21_s_i, double R_22_s_i,
    double cos_2phi, double sin_2phi) {
    // R_T = R_11 * cos^2(2phi) + R_22 * sin^2(2phi) + (R_12+R_21)*sin(2phi)cos(2phi)
    // Original kernel: R_11_s[i_s] * cos_2phi * cos_2phi + R_22_s[i_s] * sin_2phi * sin_2phi + (R_12_s[i_s] + R_21_s[i_s]) * sin_2phi * cos_2phi;
    // This seems fine.
    return R_11_s_i * cos_2phi * cos_2phi +
           R_22_s_i * sin_2phi * sin_2phi +
           (R_12_s_i + R_21_s_i) * sin_2phi * cos_2phi;
}

// The old precompute_kernel is removed as it's replaced by process_all_lenses_kernel
// in precompute_interface.cu.
/*
__global__ void precompute_kernel( ... ) {
    // ... old implementation ...
}
*/
