# dsigma/_precompute_cuda.pxd

cdef extern from "precompute_interface.h": # Assumes precompute_interface.h is in include path
    ctypedef struct TableData:
        # Lens data
        double* z_l
        double* d_com_l
        double* sin_ra_l
        double* cos_ra_l
        double* sin_dec_l
        double* cos_dec_l
        int n_lenses # Matches C++ int
        long* healpix_id_l

        # Source data
        double* z_s
        double* d_com_s
        double* sin_ra_s
        double* cos_ra_s
        double* sin_dec_s
        double* cos_dec_s
        double* w_s
        double* e_1_s
        double* e_2_s
        double* z_l_max_s
        int n_sources # Matches C++ int
        long* healpix_id_s

        # HEALPix configuration
        int nside_healpix
        const char* order_healpix # Assuming C++ side uses const char*

        # Optional lens data for sigma_crit_eff
        bint has_sigma_crit_eff
        int n_z_bins_l
        double* sigma_crit_eff_l

        # Optional source data
        int* z_bin_s # In C++, this is int*

        bint has_m_s
        double* m_s
        bint has_e_rms_s
        double* e_rms_s
        bint has_R_2_s
        double* R_2_s
        bint has_R_matrix_s
        double* R_11_s
        double* R_12_s
        double* R_21_s
        double* R_22_s

        # Binning information
        double* dist_3d_sq_bins
        int n_bins

        # Configuration
        bint comoving
        float weighting
        # nside_healpix and order_healpix moved to HEALPix config section

        # Output arrays (pre-allocated by Python, filled by C++)
        long long* sum_1_r
        double* sum_w_ls_r
        double* sum_w_ls_e_t_r
        double* sum_w_ls_e_t_sigma_crit_r
        double* sum_w_ls_sigma_crit_r
        double* sum_w_ls_z_s_r
        double* sum_w_ls_m_r
        double* sum_w_ls_1_minus_e_rms_sq_r
        double* sum_w_ls_A_p_R_2_r
        double* sum_w_ls_R_T_r

    # Structure for max_k check results
    ctypedef struct MaxKCheckResult:
        long adjusted_nside
        int max_k
        bint fits_in_shared_memory
    
    # Changed to pass TableData by pointer as per C++ interface
    int precompute_cuda_interface(TableData* table_data_ptr, int n_gpus, bint force_shared_memory, bint force_global_memory)
    MaxKCheckResult check_max_k_for_precompute(long nside, int n_bins, double max_distance_sq_estimate, int n_lenses, bint force_shared)


# For numpy arrays
cimport numpy as np

# No cpdef needed here as per instructions.
# Public Python function will be in .pyx file.
