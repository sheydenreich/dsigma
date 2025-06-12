#ifndef PRECOMPUTE_INTERFACE_H
#define PRECOMPUTE_INTERFACE_H

#include <vector> // Used for potential data handling on C++ side, though raw pointers are for kernel

// Struct to hold all data tables (pointers to arrays)
struct TableData {
    // Lens properties
    double* z_l;
    double* d_com_l;
    double* sin_ra_l;
    double* cos_ra_l;
    double* sin_dec_l;
    double* cos_dec_l;
    long* healpix_id_l;      // HEALPix ID for each lens

    // Source properties
    double* z_s;
    double* d_com_s;
    double* sin_ra_s;
    double* cos_ra_s;
    double* sin_dec_s;
    double* cos_dec_s;
    double* w_s;
    double* e_1_s;
    double* e_2_s;
    double* z_l_max_s;       // Max lens redshift for each source
    long* healpix_id_s;      // HEALPix ID for each source

    // HEALPix configuration (for interpreting healpix_id_l and healpix_id_s)
    int nside_healpix;
    std::string order_healpix; // "ring" or "nested"

    // Sigma_crit_eff related (effective sigma crit, precomputed per lens per source z_bin)
    bool has_sigma_crit_eff;
    int n_z_bins_l;          // Number of redshift bins for sigma_crit_eff_l
    double* sigma_crit_eff_l; // Flattened: n_lenses * n_z_bins_l
    int* z_bin_s;            // Redshift bin index for each source

    // Optional source properties
    bool has_m_s;
    double* m_s;             // Multiplicative bias

    bool has_e_rms_s;
    double* e_rms_s;         // RMS ellipticity

    bool has_R_2_s;
    double* R_2_s;           // Response R_2 (scalar)

    bool has_R_matrix_s;
    double* R_11_s;          // R_11 component of response matrix
    double* R_12_s;          // R_12 component of response matrix
    double* R_21_s;          // R_21 component of response matrix
    double* R_22_s;          // R_22 component of response matrix

    // Binning configuration
    double* dist_3d_sq_bins; // Radial bins for each lens, flattened: n_lenses * (n_bins + 1)
    int n_bins;              // Number of radial bins

    // Configuration parameters
    bool comoving;           // Use comoving (true) or projected (false) distances
    float weighting;         // Power for sigma_crit weighting (0, 1, or 2 typically)

    // Output sum arrays (results) - these will be populated by the CUDA kernel
    // These should be pre-allocated by the caller to the correct size (n_lenses * n_bins)
    long long* sum_1_r;
    double* sum_w_ls_r;
    double* sum_w_ls_e_t_r;
    double* sum_w_ls_e_t_sigma_crit_r;
    double* sum_w_ls_z_s_r;
    double* sum_w_ls_sigma_crit_r;

    // Optional output sum arrays
    double* sum_w_ls_m_r;                 // if has_m_s
    double* sum_w_ls_1_minus_e_rms_sq_r;  // if has_e_rms_s
    double* sum_w_ls_A_p_R_2_r;           // if has_R_2_s
    double* sum_w_ls_R_T_r;               // if has_R_matrix_s

    // Counts
    int n_lenses;
    int n_sources;

    // Constructor to initialize pointers to nullptr and basic types to zero/false
    TableData() :
        z_l(nullptr), d_com_l(nullptr), sin_ra_l(nullptr), cos_ra_l(nullptr), sin_dec_l(nullptr), cos_dec_l(nullptr), healpix_id_l(nullptr),
        z_s(nullptr), d_com_s(nullptr), sin_ra_s(nullptr), cos_ra_s(nullptr), sin_dec_s(nullptr), cos_dec_s(nullptr),
        w_s(nullptr), e_1_s(nullptr), e_2_s(nullptr), z_l_max_s(nullptr), healpix_id_s(nullptr),
        nside_healpix(0), order_healpix(""),
        has_sigma_crit_eff(false), n_z_bins_l(0), sigma_crit_eff_l(nullptr), z_bin_s(nullptr),
        has_m_s(false), m_s(nullptr),
        has_e_rms_s(false), e_rms_s(nullptr),
        has_R_2_s(false), R_2_s(nullptr),
        has_R_matrix_s(false), R_11_s(nullptr), R_12_s(nullptr), R_21_s(nullptr), R_22_s(nullptr),
        dist_3d_sq_bins(nullptr), n_bins(0),
        comoving(false), weighting(0.0f),
        sum_1_r(nullptr), sum_w_ls_r(nullptr), sum_w_ls_e_t_r(nullptr), sum_w_ls_e_t_sigma_crit_r(nullptr),
        sum_w_ls_z_s_r(nullptr), sum_w_ls_sigma_crit_r(nullptr),
        sum_w_ls_m_r(nullptr), sum_w_ls_1_minus_e_rms_sq_r(nullptr), sum_w_ls_A_p_R_2_r(nullptr), sum_w_ls_R_T_r(nullptr),
        n_lenses(0), n_sources(0)
    {}
};

// Extern "C" interface function
#ifdef __cplusplus
extern "C" {
#endif

int precompute_cuda_interface(TableData& tables, int n_gpus);

#ifdef __cplusplus
}
#endif

#endif // PRECOMPUTE_INTERFACE_H
