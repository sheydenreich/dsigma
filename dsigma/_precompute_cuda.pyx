# dsigma/_precompute_cuda.pyx

# cython: language_level=3, boundscheck=False, wraparound=False
# cython: nonecheck=False, cdivision=True, initializedcheck=False

cimport _precompute_cuda
import numpy as np
cimport numpy as np

# Ensure NumPy is initialized for C API usage
np.import_array()

def precompute_gpu_wrapper(
    # Lens data (NumPy arrays)
    np.ndarray[np.double_t, ndim=1, mode="c"] z_l_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] d_com_l_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sin_ra_l_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] cos_ra_l_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sin_dec_l_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] cos_dec_l_np,
    np.ndarray[np.int64_t, ndim=1, mode="c"] healpix_id_l_np,

    # Source data (NumPy arrays)
    np.ndarray[np.double_t, ndim=1, mode="c"] z_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] d_com_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sin_ra_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] cos_ra_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sin_dec_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] cos_dec_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] w_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] e_1_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] e_2_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] z_l_max_s_np,
    np.ndarray[np.int64_t, ndim=1, mode="c"] healpix_id_s_np,

    # HEALPix configuration
    int nside_healpix,
    str order_healpix_str, # Pass as Python string

    # Optional lens data for sigma_crit_eff
    bint has_sigma_crit_eff,
    int n_z_bins_l,
    np.ndarray[np.double_t, ndim=1, mode="c"] sigma_crit_eff_l_np,

    # Optional source data
    np.ndarray[np.int32_t, ndim=1, mode="c"] z_bin_s_np,

    # Optional source properties
    bint has_m_s, np.ndarray[np.double_t, ndim=1, mode="c"] m_s_np,
    bint has_e_rms_s, np.ndarray[np.double_t, ndim=1, mode="c"] e_rms_s_np,
    bint has_R_2_s, np.ndarray[np.double_t, ndim=1, mode="c"] R_2_s_np,
    bint has_R_matrix_s,
    np.ndarray[np.double_t, ndim=1, mode="c"] R_11_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] R_12_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] R_21_s_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] R_22_s_np,

    # Binning information
    np.ndarray[np.double_t, ndim=1, mode="c"] dist_3d_sq_bins_np,
    int n_bins,

    # Configuration
    bint comoving,
    float weighting,

    # Output arrays (pre-allocated by Python, modified in place)
    np.ndarray[np.int64_t, ndim=1, mode="c"] sum_1_r_np, # Matches long long in C++
    np.ndarray[np.double_t, ndim=1, mode="c"] sum_w_ls_r_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sum_w_ls_e_t_r_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sum_w_ls_e_t_sigma_crit_r_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sum_w_ls_sigma_crit_r_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sum_w_ls_z_s_r_np,
    # Optional output arrays (can be None from Python)
    np.ndarray[np.double_t, ndim=1, mode="c"] sum_w_ls_m_r_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sum_w_ls_1_minus_e_rms_sq_r_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sum_w_ls_A_p_R_2_r_np,
    np.ndarray[np.double_t, ndim=1, mode="c"] sum_w_ls_R_T_r_np,

    int n_gpus = 1
    ):

    cdef _precompute_cuda.TableData c_table_data
    cdef int status

    # Populate lens data
    c_table_data.z_l = <double*>z_l_np.data
    c_table_data.d_com_l = <double*>d_com_l_np.data
    c_table_data.sin_ra_l = <double*>sin_ra_l_np.data
    c_table_data.cos_ra_l = <double*>cos_ra_l_np.data
    c_table_data.sin_dec_l = <double*>sin_dec_l_np.data
    c_table_data.cos_dec_l = <double*>cos_dec_l_np.data
    c_table_data.n_lenses = z_l_np.shape[0] # C++ TableData has int n_lenses
    c_table_data.healpix_id_l = <long*>healpix_id_l_np.data

    # Populate source data
    c_table_data.z_s = <double*>z_s_np.data
    c_table_data.d_com_s = <double*>d_com_s_np.data
    c_table_data.sin_ra_s = <double*>sin_ra_s_np.data
    c_table_data.cos_ra_s = <double*>cos_ra_s_np.data
    c_table_data.sin_dec_s = <double*>sin_dec_s_np.data
    c_table_data.cos_dec_s = <double*>cos_dec_s_np.data
    c_table_data.w_s = <double*>w_s_np.data
    c_table_data.e_1_s = <double*>e_1_s_np.data
    c_table_data.e_2_s = <double*>e_2_s_np.data
    c_table_data.z_l_max_s = <double*>z_l_max_s_np.data
    c_table_data.n_sources = z_s_np.shape[0] # C++ TableData has int n_sources
    c_table_data.healpix_id_s = <long*>healpix_id_s_np.data

    # HEALPix configuration
    c_table_data.nside_healpix = nside_healpix
    cdef bytes order_bytes = order_healpix_str.encode('UTF-8')
    c_table_data.order_healpix = <const char*>order_bytes

    # Optional lens data
    c_table_data.has_sigma_crit_eff = has_sigma_crit_eff
    c_table_data.n_z_bins_l = n_z_bins_l if has_sigma_crit_eff else 0
    c_table_data.sigma_crit_eff_l = <double*>sigma_crit_eff_l_np.data if sigma_crit_eff_l_np is not None and has_sigma_crit_eff else NULL

    # Optional source data
    c_table_data.z_bin_s = <int*>z_bin_s_np.data if z_bin_s_np is not None else NULL # z_bin_s is always needed if has_sigma_crit_eff, but C++ checks pointer

    # Optional source properties
    c_table_data.has_m_s = has_m_s
    c_table_data.m_s = <double*>m_s_np.data if m_s_np is not None and has_m_s else NULL

    c_table_data.has_e_rms_s = has_e_rms_s
    c_table_data.e_rms_s = <double*>e_rms_s_np.data if e_rms_s_np is not None and has_e_rms_s else NULL

    c_table_data.has_R_2_s = has_R_2_s
    c_table_data.R_2_s = <double*>R_2_s_np.data if R_2_s_np is not None and has_R_2_s else NULL

    c_table_data.has_R_matrix_s = has_R_matrix_s
    c_table_data.R_11_s = <double*>R_11_s_np.data if R_11_s_np is not None and has_R_matrix_s else NULL
    c_table_data.R_12_s = <double*>R_12_s_np.data if R_12_s_np is not None and has_R_matrix_s else NULL
    c_table_data.R_21_s = <double*>R_21_s_np.data if R_21_s_np is not None and has_R_matrix_s else NULL
    c_table_data.R_22_s = <double*>R_22_s_np.data if R_22_s_np is not None and has_R_matrix_s else NULL

    # Binning information
    c_table_data.dist_3d_sq_bins = <double*>dist_3d_sq_bins_np.data
    c_table_data.n_bins = n_bins

    # Configuration
    c_table_data.comoving = comoving
    c_table_data.weighting = weighting

    # Output arrays
    c_table_data.sum_1_r = <long long*>sum_1_r_np.data
    c_table_data.sum_w_ls_r = <double*>sum_w_ls_r_np.data
    c_table_data.sum_w_ls_e_t_r = <double*>sum_w_ls_e_t_r_np.data
    c_table_data.sum_w_ls_e_t_sigma_crit_r = <double*>sum_w_ls_e_t_sigma_crit_r_np.data
    c_table_data.sum_w_ls_sigma_crit_r = <double*>sum_w_ls_sigma_crit_r_np.data
    c_table_data.sum_w_ls_z_s_r = <double*>sum_w_ls_z_s_r_np.data

    c_table_data.sum_w_ls_m_r = <double*>sum_w_ls_m_r_np.data if sum_w_ls_m_r_np is not None and has_m_s else NULL
    c_table_data.sum_w_ls_1_minus_e_rms_sq_r = <double*>sum_w_ls_1_minus_e_rms_sq_r_np.data if sum_w_ls_1_minus_e_rms_sq_r_np is not None and has_e_rms_s else NULL
    c_table_data.sum_w_ls_A_p_R_2_r = <double*>sum_w_ls_A_p_R_2_r_np.data if sum_w_ls_A_p_R_2_r_np is not None and has_R_2_s else NULL
    c_table_data.sum_w_ls_R_T_r = <double*>sum_w_ls_R_T_r_np.data if sum_w_ls_R_T_r_np is not None and has_R_matrix_s else NULL

    # Call the C++ function
    status = _precompute_cuda.precompute_cuda_interface(&c_table_data, n_gpus)

    # order_bytes goes out of scope here, C++ side must copy the string if needed beyond the call.
    # The C++ TableData in precompute_interface.h has std::string order_healpix,
    # so it will make a copy from the char*. If it were char*, it would need to copy.

    if status != 0:
        # Consider checking cudaGetLastError() or specific error codes if C++ returns more info
        raise RuntimeError("Error in precompute_cuda_interface: C++ function returned status %d" % status)

    # No return value needed if Python arrays are modified in-place and status is checked.
    # Or, return status if Python side wants to check it.
    # For now, let's assume void-like behavior on success, exception on failure.
    return None # Explicitly return None for clarity
