"""Module with functions specific to the DECADE Survey."""

import numpy as np

__all__ = ['default_version', 'known_versions', 'e_2_convention',
           'default_column_keys', 'tomographic_redshift_bin',
           'multiplicative_shear_bias', 'shear_response']

default_version = 'DR1'
known_versions = ['DR1']
e_2_convention = 'standard'


def default_column_keys(version=default_version):
    """Return a dictionary of default column keys.

    Parameters
    ----------
    version : string or None, optional
        Version of the catalog.

    Returns
    -------
    keys : dict
        Dictionary of default column keys.

    Raises
    ------
    ValueError
        If `version` does not correspond to a known catalog version.

    """
    if version == 'DR1':
        keys = {
            'ra': 'RA',
            'dec': 'Dec',
            'e_1': 'MCAL_G_1_NOSHEAR',
            'e_2': 'MCAL_G_2_NOSHEAR',
            'w': 'MCAL_W_NOSHEAR',
            'R_11': 'R_11',
            'R_12': 'R_12',
            'R_21': 'R_21',
            'R_22': 'R_22'}
    else:
        raise ValueError(
            "Unkown version of DECADE. Supported versions are {}.".format(
                known_versions))

    return keys

def multiplicative_shear_bias(z_bin, gal_cap, version=default_version):
    """Return the multiplicative shear bias.

    For DES Y3, we can define a blending-related multiplicative shear bias.
    This function returns the multiplicative bias :math:`m` as a function
    of the bin. The values can be computed from the blending-corrected Y3
    redshift distributions and are, as expected, very similar to the values in
    Table 4 in MacCrann et al. (2022) where they were calculated for mock
    catalogs.

    Parameters
    ----------
    z_bin : numpy.ndarray
        Tomographic redshift bin.
    gal_cap : string
        Galacyic cap [NGC,SGC]
    version : string, optional
        Which catalog version to use.

    Returns
    -------
    m : numpy.ndarray
        The multiplicative shear bias corresponding to each tomographic bin.

    Raises
    ------
    ValueError
        If the `version` does not correspond to a known catalog version or
        multiplicative shear biases cannot be defined for this version of the
        catalog.

    """
    mbias = {'NGC': np.array([-.92, -1.90, -4.00, -3.73]), 'SGC': np.array([-1.33, -2.26, -3.67, -5.72])}

    if version == 'DR1':
        m = mbias[gal_cap] * 0.01
        return np.where(z_bin != -1, m[z_bin], np.nan)
    else:
        raise ValueError(
            "Unkown version of DECADE. Supported versions are {}.".format(
                known_versions))

def shear_response(table_s, tomographic_bin, include_selection_response=True, version=default_version):
    sel = (table_s['MCAL_SEL_NOSHEAR'] == tomographic_bin + 1)
    R = np.zeros((2, 2, np.sum(sel)))
    R_sel = np.zeros((2, 2)) if include_selection_response else None
    for i in range(2):
        for j in range(2):
            # if i!=j:
            #     print("Skipping off-diagonal element of DECADE shear response")
            #     continue
            R[i, j] = (table_s[f'MCAL_G_{i+1}_{j+1}P'][sel] - table_s[f'MCAL_G_{i+1}_{j+1}M'][sel]) / 0.02
            if include_selection_response:
                sp = (table_s[f'MCAL_SEL_{j+1}P'] == tomographic_bin + 1)
                sm = (table_s[f'MCAL_SEL_{j+1}M'] == tomographic_bin + 1)
                R_sel[i, j] = (np.average(table_s[f'MCAL_G_{i+1}_NOSHEAR'][sp], weights=table_s[f'MCAL_W_{j+1}P'][sp]) -
                    np.average(table_s[f'MCAL_G_{i+1}_NOSHEAR'][sm], weights=table_s[f'MCAL_W_{j+1}M'][sm])) / 0.02
    return R, R_sel
