"""Module for pre-computing lensing results."""

import multiprocessing as mp
import numbers
import numpy as np
import queue as Queue
import warnings


from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.units import UnitConversionError
from astropy_healpix import HEALPix
from scipy.interpolate import interp1d

from .physics import critical_surface_density
from .physics import effective_critical_surface_density
from .precompute_engine import precompute_engine

try:
    from ._precompute_cuda import precompute_gpu_wrapper
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


__all__ = ["photo_z_dilution_factor", "mean_photo_z_offset", "precompute"]


def photo_z_dilution_factor(z_l, table_c, cosmology, weighting=-2,
                            lens_source_cut=0):
    """Calculate the photo-z delta sigma bias as a function of lens redshift.

    Parameters
    ----------
    z_l : float or numpy.ndarray
        Redshift(s) of the lens.
    table_c : astropy.table.Table
        Photometric redshift calibration catalog.
    cosmology : astropy.cosmology
        Cosmology to assume for calculations.
    weighting : float, optional
        The exponent of weighting of each lens-source pair by the critical
        surface density. A natural choice is -2 which minimizes shape noise.
        Default is -2.
    lens_source_cut : None, float or numpy.ndarray, optional
        Determine the lens-source redshift separation cut. If None, no cut is
        applied. If a float, determines the minimum redshift separation between
        lens and source redshift for lens-source pairs to be used. If an array,
        it has to be the same length as the source table and determines the
        maximum lens redshift for a lens-source pair to be used. Default is 0.

    Returns
    -------
    f_bias : float or numpy.ndarray
        The photo-z bias factor, `f_bias`, for the lens redshift(s).

    """
    if lens_source_cut is None:
        z_l_max = np.repeat(np.amax(z_l) + 1, len(table_c))
    elif isinstance(lens_source_cut, numbers.Number):
        z_l_max = table_c['z'] - lens_source_cut
    else:
        z_l_max = np.array(lens_source_cut)

    z_s = table_c['z']
    z_s_true = table_c['z_true']
    d_l = cosmology.comoving_transverse_distance(z_l).to(u.Mpc).value
    d_s = cosmology.comoving_transverse_distance(table_c['z']).to(u.Mpc).value
    d_s_true = cosmology.comoving_transverse_distance(
        table_c['z_true']).to(u.Mpc).value
    w = table_c['w_sys'] * table_c['w']

    if hasattr(z_l, '__len__'):
        shape = (len(z_l), len(table_c))
        z_s = np.tile(z_s, len(z_l)).reshape(shape)
        z_s_true = np.tile(z_s_true, len(z_l)).reshape(shape)
        d_s = np.tile(d_s, len(z_l)).reshape(shape)
        d_s_true = np.tile(d_s_true, len(z_l)).reshape(shape)
        z_l_max = np.tile(z_l_max, len(z_l)).reshape(shape)
        w = np.tile(w, len(z_l)).reshape(shape)
        z_l = np.repeat(z_l, len(table_c)).reshape(shape)
        d_l = np.repeat(d_l, len(table_c)).reshape(shape)

    sigma_crit_phot = critical_surface_density(z_l, z_s, d_l=d_l, d_s=d_s)
    sigma_crit_true = critical_surface_density(z_l, z_s_true, d_l=d_l,
                                               d_s=d_s_true)

    mask = (z_l_max < z_l) | (z_l > z_s)

    if np.any(np.all(mask, axis=-1)):
        warnings.warn('Could not find valid calibration sources for some ' +
                      'lens redshifts. The f_bias correction may be ' +
                      'undefined.', RuntimeWarning)

    return (np.sum((w * sigma_crit_phot**weighting) * (~mask), axis=-1) /
            np.sum((w * sigma_crit_phot**(weighting + 1) / sigma_crit_true) *
                   (~mask), axis=-1))


def mean_photo_z_offset(z_l, table_c, cosmology, weighting=-2,
                        lens_source_cut=0):
    """Calculate the mean offset of source photometric redshifts.

    Parameters
    ----------
    z_l : float or numpy.ndarray
        Redshift(s) of the lens.
    table_c : astropy.table.Table, optional
        Photometric redshift calibration catalog.
    cosmology : astropy.cosmology
        Cosmology to assume for calculations.
    weighting : float, optional
        The exponent of weighting of each lens-source pair by the critical
        surface density. A natural choice is -2 which minimizes shape noise.
        Default is -2.
    lens_source_cut : None, float or numpy.ndarray, optional
        Determine the lens-source redshift separation cut. If None, no cut is
        applied. If a float, determines the minimum redshift separation between
        lens and source redshift for lens-source pairs to be used. If an array,
        it has to be the same length as the source table and determines the
        maximum lens redshift for a lens-source pair to be used. Default is 0.

    Returns
    -------
    dz : float or numpy.ndarray
        The mean source redshift offset for the lens redshift(s).

    """
    if lens_source_cut is None:
        z_l_max = np.repeat(np.amax(z_l) + 1, len(table_c))
    elif isinstance(lens_source_cut, numbers.Number):
        z_l_max = table_c['z'] - lens_source_cut
    else:
        z_l_max = np.array(lens_source_cut)

    z_s = table_c['z']
    z_s_true = table_c['z_true']
    d_l = cosmology.comoving_transverse_distance(z_l).to(u.Mpc).value
    d_s = cosmology.comoving_transverse_distance(table_c['z']).to(
        u.Mpc).value
    w = table_c['w_sys'] * table_c['w']

    if not np.isscalar(z_l):
        shape = (len(z_l), len(table_c))
        z_s = np.tile(z_s, len(z_l)).reshape(shape)
        z_s_true = np.tile(z_s_true, len(z_l)).reshape(shape)
        d_s = np.tile(d_s, len(z_l)).reshape(shape)
        z_l_max = np.tile(z_l_max, len(z_l)).reshape(shape)
        w = np.tile(w, len(z_l)).reshape(shape)
        z_l = np.repeat(z_l, len(table_c)).reshape(shape)
        d_l = np.repeat(d_l, len(table_c)).reshape(shape)

    sigma_crit = critical_surface_density(z_l, z_s, d_l=d_l, d_s=d_s)
    w = w * sigma_crit**weighting

    mask = (z_l_max < z_l) | (z_l > z_s)

    return np.sum((z_s - z_s_true) * w * (~mask), axis=-1) / np.sum(
        w * (~mask), axis=-1)


def get_raw_multiprocessing_array(array):
    """Convert a numpy array into a shared-memory multiprocessing array.

    Parameters
    ----------
    array : numpy.ndarray or None
        Input array.

    Returns
    -------
    array_mp : multiprocessing.RawArray or None
        Output array. None if input is None.

    """
    if array is None:
        return None

    array_mp = mp.RawArray('l' if np.issubdtype(array.dtype, np.integer) else
                           'd', len(array))
    array_np = np.ctypeslib.as_array(array_mp)
    array_np[:] = array

    return array_mp


def precompute(
        table_l, table_s, bins, table_c=None, table_n=None,
        cosmology=FlatLambdaCDM(H0=100, Om0=0.3), comoving=True,
        weighting=-2, lens_source_cut=0, nside=256, n_jobs=1,
        progress_bar=False, use_gpu=False, force_shared=False, force_global=False):
    """For all lenses in the catalog, precompute the lensing statistics.

    Parameters
    ----------
    table_l : astropy.table.Table
        Catalog of lenses.
    table_s : astropy.table.Table
        Catalog of sources.
    bins : numpy.ndarray or astropy.units.quantity.Quantity
        Bins in radius to use for the stacking. If a numpy array, bins are
        assumed to be in Mpc. If an astropy quantity, one can pass both length
        units, e.g. kpc and Mpc, as well as angular units, i.e. deg and rad.
    table_c : astropy.table.Table, optional
        Additional photometric redshift calibration catalog. If provided, this
        will be used to statistically correct the photometric source redshifts
        and critical surface densities. Default is None.
    table_n : astropy.table.Table, optional
        Source redshift distributions. If provided, this will be used to
        compute mean source redshifts and critical surface densities. These
        mean quantities would be used instead the individual photometric
        redshift estimates. Default is None.
    cosmology : astropy.cosmology, optional
        Cosmology to assume for calculations. Default is a flat LambdaCDM
        cosmology with h=1 and Om0=0.3.
    comoving : boolean, optional
        Whether to use comoving or physical quantities for radial bins (if
        given in physical units) and the excess surface density. Default is
        True.
    weighting : float, optional
        The exponent of weighting of each lens-source pair by the critical
        surface density. A natural choice is -2 which minimizes shape noise.
        Default is -2.
    lens_source_cut : None, float or numpy.ndarray, optional
        Determine the lens-source redshift separation cut. If None, no cut is
        applied. If a float, determines the minimum redshift separation between
        lens and source redshift for lens-source pairs to be used. If an array,
        it has to be the same length as the source table and determines the
        maximum lens redshift for a lens-source pair to be used. Default is 0.
    nside : int, optional
        dsigma uses pixelization to group nearby lenses together and process
        them simultaneously. This parameter determines the number of pixels.
        It has to be a power of 2. May impact performance. Default is 256.
    n_jobs : int, optional
        Number of jobs to run at the same time. Default is 1. In case use_gpu=True,
        n_jobs will determine the number of GPUs to use.
    progress_bar : boolean, option
        Whether to show a progress bar for the main loop over lens pixels.
        Default is False.
    use_gpu : bool, optional
        If True, attempt to use the GPU-accelerated precomputation engine.
        If False or GPU is not available, falls back to the CPU engine.
        Default is False.
    force_shared : bool, optional
        If True, force the GPU computation to use shared memory. If the
        required max_k doesn't fit in shared memory, nside will be reduced
        until it does. Only effective when use_gpu=True. Default is False.
    force_global : bool, optional
        If True, force the GPU computation to use global memory instead of
        shared memory. Only effective when use_gpu=True. Default is False.

    Returns
    -------
    table_l : astropy.table.Table
        Lens catalog with the pre-computation results attached to the table.

    Raises
    ------
    ValueError
        If there are problems in the input.

    """
    try:
        assert cosmology.Ok0 == 0
    except AssertionError:
        raise ValueError('dsigma does not support non-flat cosmologies.')

    if np.any(table_l['z'] < 0):
        raise ValueError('Input lens redshifts must all be non-negative.')
    if not isinstance(nside, int) or not np.isin(nside, 2**np.arange(15)):
        raise ValueError('nside must be a positive power of 2. Received ' +
                         '{}.'.format(nside))
    if not isinstance(n_jobs, int) or n_jobs < 1:
        raise ValueError('Number of jobs must be positive integer. Received ' +
                         '{}.'.format(n_jobs))

    # Validate memory mode flags
    if force_shared and force_global:
        raise ValueError('force_shared and force_global cannot both be True.')

    # GPU memory management: check max_k requirements and adjust nside if needed
    if use_gpu and force_shared:
        # Import the check_max_k function for early nside adjustment
        if GPU_AVAILABLE:
            from ._precompute_cuda import check_max_k
            
            # Estimate maximum distance bins to calculate max_k
            # We need to do this before HEALPix pixelization 
            z_l_min = np.min(table_l['z'])
            d_com_l_min = cosmology.comoving_transverse_distance(z_l_min).to(u.Mpc).value
            # TODO: We calculate max_k for the worst-case lens, which we assume to be the one closest to the observer.
            # If in the future lenses behind the turnover point of the angular diameter distence are included,
            # in case of comoving=False, this could be inaccurate. 
            # Convert bins to theta_bins (estimate)
            if not isinstance(bins, u.quantity.Quantity):
                bins_quantity = bins * u.Mpc
            else:
                bins_quantity = bins
                
            try:
                theta_bins_max = np.amax(bins_quantity.to(u.rad).value)
            except UnitConversionError:
                theta_bins_max = np.amax(bins_quantity.to(u.Mpc).value / d_com_l_min)
                if not comoving:
                    theta_bins_max *= (1 + z_l_min)
            
            # Estimate maximum dist_3d_sq for max_k calculation
            max_dist_3d_sq_estimate = min(4 * np.sin(theta_bins_max / 2.0)**2, 2.0)
            
            # Check max_k requirements and adjust nside if necessary
            max_k_result = check_max_k(nside, len(bins) - 1, max_dist_3d_sq_estimate, len(table_l), True)
            original_nside = nside
            nside = max_k_result['adjusted_nside']
            
            if nside != original_nside:
                print(f"force_shared=True: Reduced nside from {original_nside} to {nside} "
                      f"to fit max_k={max_k_result['max_k']} in shared memory")
        else:
            warnings.warn(
                "force_shared=True requested but GPU not available. Ignoring flag.",
                RuntimeWarning)

    if table_n is not None:
        if 'z_bin' not in table_s.colnames:
            raise ValueError('To use source redshift distributions, the ' +
                             'source table needs to have a `z_bin` column.')
        if not np.issubdtype(table_s['z_bin'].data.dtype, int) or np.amin(
                table_s['z_bin']) < 0:
            raise ValueError('The `z_bin` column in the source table must ' +
                             'contain only non-negative integers.')
        if np.amax(table_s['z_bin']) > table_n['n'].data.shape[1]:
            raise ValueError('The source table contains more redshift bins ' +
                             'than where passed via the nz argument.')

    hp = HEALPix(nside, order='ring')
    pix_l = hp.lonlat_to_healpix(table_l['ra'] * u.deg, table_l['dec'] * u.deg)
    pix_s = hp.lonlat_to_healpix(table_s['ra'] * u.deg, table_s['dec'] * u.deg)
    argsort_pix_l = np.argsort(pix_l)
    argsort_pix_s = np.argsort(pix_s)
    u_pix_l, n_pix_l = np.unique(pix_l, return_counts=True)
    u_pix_l = np.ascontiguousarray(u_pix_l)
    n_pix_l = np.ascontiguousarray(np.cumsum(n_pix_l))
    u_pix_s, n_pix_s = np.unique(pix_s, return_counts=True)
    u_pix_s = np.ascontiguousarray(u_pix_s)
    n_pix_s = np.ascontiguousarray(np.cumsum(n_pix_s))

    table_engine_l = {}
    table_engine_s = {}

    table_engine_l['z'] = np.ascontiguousarray(
        table_l['z'][argsort_pix_l], dtype=np.float64)
    table_engine_s['z'] = np.ascontiguousarray(
        table_s['z'][argsort_pix_s], dtype=np.float64)

    for f, f_name in zip([np.sin, np.cos], ['sin', 'cos']):
        for table, argsort_pix, table_engine in zip(
                [table_l, table_s], [argsort_pix_l, argsort_pix_s],
                [table_engine_l, table_engine_s]):
            for angle in ['ra', 'dec']:
                table_engine['{} {}'.format(f_name, angle)] =\
                    np.ascontiguousarray(f(np.deg2rad(table[angle]))[
                        argsort_pix])

    for key in ['w', 'e_1', 'e_2', 'm', 'e_rms', 'R_2', 'R_11', 'R_22',
                'R_12', 'R_21']:
        if key in table_s.colnames:
            table_engine_s[key] = np.ascontiguousarray(
                table_s[key][argsort_pix_s], dtype=np.float64)

    if lens_source_cut is None:
        z_l_max = np.repeat(np.amax(table_l['z']) + 1, len(table_s))
    elif isinstance(lens_source_cut, numbers.Number):
        z_l_max = table_s['z'] - lens_source_cut
    else:
        z_l_max = np.array(lens_source_cut)

    table_engine_s['z_l_max'] = np.ascontiguousarray(
        z_l_max[argsort_pix_s], dtype=np.float64)

    for table, argsort_pix, table_engine in zip(
            [table_l, table_s], [argsort_pix_l, argsort_pix_s],
            [table_engine_l, table_engine_s]):

        z_min = np.amin(table['z'])
        z_max = np.amax(table['z'])
        z_interp = np.linspace(
            z_min, z_max, max(10, int((z_max - z_min) / 0.0001)))

        table_engine['d_com'] = np.ascontiguousarray(interp1d(
            z_interp, cosmology.comoving_transverse_distance(z_interp).to(
                u.Mpc).value)(table['z'])[argsort_pix])

    if table_c is not None and table_n is None:
        z_min = np.amin(table_l['z'])
        z_max = np.amax(table_l['z'])
        z_interp = np.linspace(
            z_min, z_max, max(10, int((z_max - z_min) / 0.001)))
        f_bias_interp = photo_z_dilution_factor(
            z_interp, table_c, cosmology, weighting=weighting,
            lens_source_cut=lens_source_cut)
        f_bias_interp = interp1d(
            z_interp, f_bias_interp, kind='cubic', bounds_error=False,
            fill_value=(f_bias_interp[0], f_bias_interp[-1]))
        table_engine_l['f_bias'] = np.ascontiguousarray(
            f_bias_interp(np.array(table_engine_l['z'])), dtype=np.float64)
        dz_s_interp = mean_photo_z_offset(
            z_interp, table_c=table_c, cosmology=cosmology,
            weighting=weighting)
        dz_s_interp = interp1d(
            z_interp, dz_s_interp, kind='cubic', bounds_error=False,
            fill_value=(dz_s_interp[0], dz_s_interp[-1]))
        table_engine_l['delta z_s'] = np.ascontiguousarray(
            dz_s_interp(np.array(table_engine_l['z'])), dtype=np.float64)

    elif table_c is None and table_n is not None:
        n_bins = table_n['n'].data.shape[1]
        sigma_crit_eff = np.zeros(len(table_l) * n_bins, dtype=np.float64)
        z_mean = np.zeros(n_bins, dtype=np.float64)
        for i in range(n_bins):
            z_min = np.amin(table_l['z'])
            z_max = min(np.amax(table_l['z']),
                        np.amax(table_n['z'][table_n['n'][:, i] > 0]))
            z_interp = np.linspace(
                z_min, z_max, max(10, int((z_max - z_min) / 0.001)))

            sigma_crit_eff_inv_interp = effective_critical_surface_density(
                z_interp, table_n['z'], table_n['n'][:, i],
                cosmology=cosmology, comoving=comoving)**-1
            sigma_crit_eff_inv_interp = interp1d(
                z_interp, sigma_crit_eff_inv_interp, kind='cubic',
                bounds_error=False,
                fill_value=(sigma_crit_eff_inv_interp[0],
                            sigma_crit_eff_inv_interp[-1]))
            sigma_crit_eff_inv_interp = sigma_crit_eff_inv_interp(
                np.array(table_engine_l['z']))
            sigma_crit_eff_interp = np.repeat(np.inf, len(table_l))
            mask = sigma_crit_eff_inv_interp == 0
            sigma_crit_eff_interp[~mask] = sigma_crit_eff_inv_interp[~mask]**-1
            sigma_crit_eff[i::n_bins] = sigma_crit_eff_interp
            z_mean[i] = np.average(table_n['z'], weights=table_n['n'][:, i])
        table_engine_l['sigma_crit_eff'] = np.ascontiguousarray(
            sigma_crit_eff, dtype=np.float64)
        table_engine_s['z_bin'] = np.ascontiguousarray(
            table_s['z_bin'][argsort_pix_s], dtype=int)
        # Overwrite the photometric redshifts in the source table. These
        # redshifts will be used to compute the mean source redshifts for each
        # lens.
        table_engine_s['z'] = np.ascontiguousarray(
            z_mean[table_s['z_bin']][argsort_pix_s], dtype=np.float64)

    elif table_c is not None and table_s is not None:
        raise ValueError('table_c and table_n cannot both be given.')

    # Create arrays that will hold the final results.
    table_engine_r = {}
    n_results = len(table_l) * (len(bins) - 1)

    key_list = ['sum 1', 'sum w_ls', 'sum w_ls e_t', 'sum w_ls z_s',
                'sum w_ls e_t sigma_crit', 'sum w_ls sigma_crit']

    if 'm' in table_s.colnames:
        key_list.append('sum w_ls m')

    if 'e_rms' in table_s.colnames:
        key_list.append('sum w_ls (1 - e_rms^2)')

    if 'R_2' in table_s.colnames:
        key_list.append('sum w_ls A p(R_2=0.3)')

    if (('R_11' in table_s.colnames) and ('R_12' in table_s.colnames) and
            ('R_21' in table_s.colnames) and ('R_22' in table_s.colnames)):
        key_list.append('sum w_ls R_T')

    for key in key_list:
        table_engine_r[key] = np.ascontiguousarray(
            np.zeros(n_results, dtype=(
                np.int64 if key == 'sum 1' else np.float64)))

    z_l = np.array(table_engine_l['z'])
    d_com_l = np.array(table_engine_l['d_com'])

    if not isinstance(bins, u.quantity.Quantity):
        bins = bins * u.Mpc

    try:
        theta_bins = np.tile(bins.to(u.rad).value, len(table_l))
    except UnitConversionError:
        theta_bins = (np.tile(bins.to(u.Mpc).value, len(table_l)) /
                      np.repeat(d_com_l, len(bins))).flatten()
        if not comoving:
            theta_bins *= (1 + np.repeat(z_l, len(bins))).flatten()

    dist_3d_sq_bins = np.minimum(4 * np.sin(theta_bins / 2.0)**2, 2.0)

    if use_gpu:
        if not GPU_AVAILABLE:
            warnings.warn(
                "GPU requested (use_gpu=True) but GPU extensions not found "
                "or failed to import. Falling back to CPU.", RuntimeWarning)
            # Fall through to CPU path by not setting use_gpu to False here,
            # rather let the original CPU logic handle it.
            # The structure below will naturally fall to the CPU part.
        else:
            # Prepare data for precompute_gpu_wrapper
            # Ensure all arrays are standard NumPy arrays, not RawArray
            # (RawArray conversion happens later for CPU if n_jobs > 1)

            # Lens data
            z_l_np = table_engine_l['z']
            d_com_l_np = table_engine_l['d_com']
            sin_ra_l_np = table_engine_l['sin ra']
            cos_ra_l_np = table_engine_l['cos ra']
            sin_dec_l_np = table_engine_l['sin dec']
            cos_dec_l_np = table_engine_l['cos dec']
            # Sorted HEALPix IDs for lenses
            healpix_id_l_np = pix_l[argsort_pix_l].astype(np.int64)

            # Source data
            z_s_np = table_engine_s['z']
            d_com_s_np = table_engine_s['d_com']
            sin_ra_s_np = table_engine_s['sin ra']
            cos_ra_s_np = table_engine_s['cos ra']
            sin_dec_s_np = table_engine_s['sin dec']
            cos_dec_s_np = table_engine_s['cos dec']
            w_s_np = table_engine_s['w']
            e_1_s_np = table_engine_s['e_1']
            e_2_s_np = table_engine_s['e_2']
            z_l_max_s_np = table_engine_s['z_l_max']
            # Sorted HEALPix IDs for sources
            healpix_id_s_np = pix_s[argsort_pix_s].astype(np.int64)

            # Optional data
            has_sigma_crit_eff = 'sigma_crit_eff' in table_engine_l
            sigma_crit_eff_l_np = table_engine_l.get('sigma_crit_eff')
            # n_z_bins_l calculation:
            n_z_bins_l_val = 0
            if has_sigma_crit_eff and sigma_crit_eff_l_np is not None and len(table_l) > 0:
                n_z_bins_l_val = sigma_crit_eff_l_np.shape[0] // len(table_l)

            z_bin_s_np = table_engine_s.get('z_bin') # Will be int32 if from test, else needs cast

            has_m_s = 'm' in table_engine_s
            m_s_np = table_engine_s.get('m')
            has_e_rms_s = 'e_rms' in table_engine_s
            e_rms_s_np = table_engine_s.get('e_rms')
            has_R_2_s = 'R_2' in table_engine_s
            R_2_s_np = table_engine_s.get('R_2')
            has_R_matrix_s = ('R_11' in table_engine_s and
                              'R_12' in table_engine_s and
                              'R_21' in table_engine_s and
                              'R_22' in table_engine_s)
            R_11_s_np = table_engine_s.get('R_11')
            R_12_s_np = table_engine_s.get('R_12')
            R_21_s_np = table_engine_s.get('R_21')
            R_22_s_np = table_engine_s.get('R_22')

            # Output arrays
            sum_1_r_np = table_engine_r['sum 1']
            sum_w_ls_r_np = table_engine_r['sum w_ls']
            sum_w_ls_e_t_r_np = table_engine_r['sum w_ls e_t']
            sum_w_ls_e_t_sigma_crit_r_np = table_engine_r['sum w_ls e_t sigma_crit']
            sum_w_ls_sigma_crit_r_np = table_engine_r['sum w_ls sigma_crit']
            sum_w_ls_z_s_r_np = table_engine_r['sum w_ls z_s']

            sum_w_ls_m_r_np = table_engine_r.get('sum w_ls m')
            sum_w_ls_1_minus_e_rms_sq_r_np = table_engine_r.get('sum w_ls (1 - e_rms^2)')
            sum_w_ls_A_p_R_2_r_np = table_engine_r.get('sum w_ls A p(R_2=0.3)')
            sum_w_ls_R_T_r_np = table_engine_r.get('sum w_ls R_T')

            # Assuming order_healpix is 'ring' as per HEALPix default in this file
            order_healpix_str = "ring"

            precompute_gpu_wrapper(
                z_l_np, d_com_l_np, sin_ra_l_np, cos_ra_l_np, sin_dec_l_np, cos_dec_l_np,
                healpix_id_l_np,
                z_s_np, d_com_s_np, sin_ra_s_np, cos_ra_s_np, sin_dec_s_np, cos_dec_s_np,
                w_s_np, e_1_s_np, e_2_s_np, z_l_max_s_np, healpix_id_s_np,
                nside, order_healpix_str, # nside_healpix, order_healpix_str
                has_sigma_crit_eff, n_z_bins_l_val,
                sigma_crit_eff_l_np.astype(np.double) if sigma_crit_eff_l_np is not None else None,
                z_bin_s_np.astype(np.int32) if z_bin_s_np is not None else None,
                has_m_s, m_s_np.astype(np.double) if m_s_np is not None else None,
                has_e_rms_s, e_rms_s_np.astype(np.double) if e_rms_s_np is not None else None,
                has_R_2_s, R_2_s_np.astype(np.double) if R_2_s_np is not None else None,
                has_R_matrix_s,
                R_11_s_np.astype(np.double) if R_11_s_np is not None else None,
                R_12_s_np.astype(np.double) if R_12_s_np is not None else None,
                R_21_s_np.astype(np.double) if R_21_s_np is not None else None,
                R_22_s_np.astype(np.double) if R_22_s_np is not None else None,
                dist_3d_sq_bins, len(bins) - 1, # dist_3d_sq_bins_np, n_bins
                comoving, float(weighting), # comoving, weighting
                sum_1_r_np, sum_w_ls_r_np, sum_w_ls_e_t_r_np,
                sum_w_ls_e_t_sigma_crit_r_np, sum_w_ls_sigma_crit_r_np,
                sum_w_ls_z_s_r_np,
                sum_w_ls_m_r_np, sum_w_ls_1_minus_e_rms_sq_r_np,
                sum_w_ls_A_p_R_2_r_np, sum_w_ls_R_T_r_np,
                n_gpus=n_jobs,
                force_shared=force_shared,
                force_global=force_global,
                verbose=progress_bar
            )
            # Results are in table_engine_r, processing below will handle them.

    # Conditional CPU execution (original logic)
    if not use_gpu or not GPU_AVAILABLE:
        # When running in parrallel, replace numpy arrays with shared-memory
        # multiprocessing arrays.
        current_dist_3d_sq_bins = dist_3d_sq_bins # Keep original numpy array if GPU path failed
        if n_jobs > 1:
            current_dist_3d_sq_bins = get_raw_multiprocessing_array(dist_3d_sq_bins)
            # Convert all table_engine arrays to RawArray for multiprocessing
            # This needs to be done carefully if GPU path modified them (it shouldn't if it ran)
            # Assuming table_engine_l/s/r are still numpy arrays if GPU path was chosen but failed before this point
            # or if GPU path was not chosen at all.
            for table_engine_dict_cpu in [table_engine_l, table_engine_s, table_engine_r]:
                for key_cpu in table_engine_dict_cpu.keys():
                    if isinstance(table_engine_dict_cpu[key_cpu], np.ndarray): # Ensure it's an ndarray
                        table_engine_dict_cpu[key_cpu] = get_raw_multiprocessing_array(
                            table_engine_dict_cpu[key_cpu])
        else: # Ensure current_dist_3d_sq_bins is the numpy version for single job CPU
            current_dist_3d_sq_bins = dist_3d_sq_bins


        # Create a queue that holds all the pixels containing lenses.
        if n_jobs == 1:
            queue = Queue.Queue()
        else:
            queue = mp.Queue()

        for i in range(len(u_pix_l)):
            queue.put(i)

        args = (u_pix_l, n_pix_l, u_pix_s, n_pix_s, current_dist_3d_sq_bins,
                table_engine_l, table_engine_s, table_engine_r, bins, comoving,
                weighting, nside, queue, progress_bar)

        if n_jobs == 1:
            precompute_engine(*args)
        else:
            processes = []
            # args_list needs to be created for each process if progress_bar is selective
            for i in range(n_jobs):
                # Only show progress bar for the first job
                current_progress_bar = progress_bar if i == 0 else False
                current_args = list(args)
                current_args[-1] = current_progress_bar # Update progress_bar argument

                process = mp.Process(target=precompute_engine, args=tuple(current_args))
                process.start()
                processes.append(process)
            for i in range(n_jobs):
                processes[i].join()

    inv_argsort_pix_l = np.argsort(argsort_pix_l)
    for key in table_engine_r.keys():
        # Ensure data from RawArray is converted back if necessary,
        # or if GPU path ran, it's already numpy array.
        result_array = np.array(table_engine_r[key])
        table_l[key] = result_array.reshape(
            len(table_l), len(bins) - 1)[inv_argsort_pix_l]

    table_l['sum w_ls z_l'] = table_l['z'][:, np.newaxis] * table_l['sum w_ls']

    if 'f_bias' in table_engine_l.keys():
        table_l['sum w_ls sigma_crit f_bias'] = (
            np.array(table_engine_l['f_bias'])[inv_argsort_pix_l][
                :, np.newaxis] * table_l['sum w_ls sigma_crit'])
        table_l['sum w_ls e_t sigma_crit f_bias'] = (
            np.array(table_engine_l['f_bias'])[inv_argsort_pix_l][
                :, np.newaxis] * table_l['sum w_ls e_t sigma_crit'])

    if 'delta z_s' in table_engine_l.keys():
        table_l['sum w_ls z_s'] = (
            table_l['sum w_ls z_s'] - table_l['sum w_ls'] * np.array(
                table_engine_l['delta z_s'])[inv_argsort_pix_l][:, np.newaxis])

    table_l.meta['bins'] = bins
    table_l.meta['comoving'] = comoving
    table_l.meta['H0'] = cosmology.H0.value
    table_l.meta['Ok0'] = cosmology.Ok0
    table_l.meta['Om0'] = cosmology.Om0
    table_l.meta['weighting'] = weighting

    return table_l
