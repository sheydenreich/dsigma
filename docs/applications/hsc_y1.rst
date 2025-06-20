Hyper Suprime-Cam (HSC)
=======================

.. note::
    This guide has not been inspected or endorsed by the HSC collaboration.

This tutorial will teach us how to cross-correlate BOSS lens galaxies with lensing catalogs from the HSC survey.

Downloading the Data
--------------------

First, we need to download the necessary HSC data files. Head to the `HSC data release site <https://hsc-release.mtk.nao.ac.jp/doc/>`_ and register for an account if you haven't done so already. As of September 2020, the only publicly available data set is part of the public data release 2 (PDR2) and goes back to the internal S16A release.

The following SQL script will return all the necessary data from the `CAS search <https://hsc-release.mtk.nao.ac.jp/datasearch/>`_. Make sure to select PDR1 as the release; otherwise, the SQL command will fail. Also, choose FITS as the output format.

.. code-block:: sql

    SELECT
        meas.object_id, meas.ira, meas.idec,
        meas2.ishape_hsm_regauss_e1, meas2.ishape_hsm_regauss_e2,
        meas2.ishape_hsm_regauss_resolution,
        photoz_ephor_ab.photoz_best, photoz_ephor_ab.photoz_err68_min,
        photoz_ephor_ab.photoz_risk_best, photoz_ephor_ab.photoz_std_best,
        weaklensing_hsm_regauss.ishape_hsm_regauss_derived_shape_weight,
        weaklensing_hsm_regauss.ishape_hsm_regauss_derived_shear_bias_m,
        weaklensing_hsm_regauss.ishape_hsm_regauss_derived_rms_e

    FROM
        s16a_wide.weaklensing_hsm_regauss
        LEFT JOIN s16a_wide.meas USING (object_id)
        LEFT JOIN s16a_wide.meas2 USING (object_id)
    	LEFT JOIN s16a_wide.photoz_ephor_ab USING (object_id)

    ORDER BY meas.object_id

As you can see, we will use the Ephor Afterburner photometric redshifts in this application. But you're free to use any other photometric redshift estimate available in the HSC database. Also, we neglect additive shear biases, which is fine because we will use random catalogs to correct those. In the following, this tutorial assumes that the above source catalog is saved as :code:`hsc_sources.fits` in the working directory.

In addition to the source catalog, we need a calibration catalog to correct for eventual biases stemming from using shallow photometric redshift point estimates. The relevant files can be downloaded using the following links: `1 <https://hsc-release.mtk.nao.ac.jp/archive/filetree/cosmos_photoz_catalog_reweighted_to_s16a_shape_catalog/Afterburner_reweighted_COSMOS_photoz_FDFC.fits>`_, `2 <https://hsc-release.mtk.nao.ac.jp/archive/filetree/cosmos_photoz_catalog_reweighted_to_s16a_shape_catalog/ephor_ab/pdf-s17a_wide-9812.cat.fits>`_, `3 <https://hsc-release.mtk.nao.ac.jp/archive/filetree/cosmos_photoz_catalog_reweighted_to_s16a_shape_catalog/ephor_ab/pdf-s17a_wide-9813.cat.fits>`_.

Preparing the Data
------------------

First, we must put the data into a format easily understandable by :code:`dsigma`. There are several helper functions to make this easy.

.. code-block:: python

    import numpy as np
    from astropy import units as u
    from astropy.table import Table, vstack, join
    from dsigma.helpers import dsigma_table

    table_s = Table.read('hsc_sources.fits')
    table_s = dsigma_table(table_s, 'source', survey='HSC')

    table_c_1 = vstack([Table.read('pdf-s17a_wide-9812.cat.fits'),
                        Table.read('pdf-s17a_wide-9813.cat.fits')])
    for key in table_c_1.colnames:
        table_c_1.rename_column(key, key.lower())
    table_c_2 = Table.read('Afterburner_reweighted_COSMOS_photoz_FDFC.fits')
    table_c_2.rename_column('S17a_objid', 'id')
    table_c = join(table_c_1, table_c_2, keys='id')
    table_c = dsigma_table(table_c, 'calibration', w_sys='SOM_weight',
                           w='weight_source', z_true='COSMOS_photoz', survey='HSC')

Precomputing the Signal
-----------------------

We will now run the computationally expensive precomputation phase. Here, we first define the lens-source separation cuts. We require that :math:`z_l < z_{s, \rm min}` and :math:`z_l + 0.1 < z_s`. Afterward, we run the actual precomputation.

.. code-block:: python

    from astropy.cosmology import Planck15
    from dsigma.precompute import precompute

    rp_bins = np.logspace(-1, 1.6, 14)
    precompute(table_l, table_s, rp_bins, cosmology=Planck15, comoving=True,
               table_c=table_c, lens_source_cut=0.1, progress_bar=True)
    precompute(table_r, table_s, rp_bins, cosmology=Planck15, comoving=True,
               table_c=table_c, lens_source_cut=0.1, progress_bar=True)

Stacking the Signal
-------------------

The total galaxy-galaxy lensing signal can be obtained with the following code. It first filters out all BOSS galaxies for which we couldn't find any source galaxy nearby. Then we divide it into jackknife samples that we will later use to estimate uncertainties. Finally, we stack the lensing signal in 4 different BOSS redshift bins and save the data.

We choose to include all the necessary corrections factors. The shear responsivity correction and multiplicative shear correction are the most important and necessary. The selection bias corrections do not dramatically impact the signal but are also required for HSC data. The photo-z dilution correction is not strictly necessary but highly recommended. Finally, random subtraction is also highly recommended but not consistently applied. Note that we don't use a boost correction, but this would also be possible.

.. code-block:: python

    from dsigma.jackknife import compute_jackknife_fields, jackknife_resampling
    from dsigma.stacking import excess_surface_density

    # Drop all lenses and randoms that did not have any nearby source.
    table_l = table_l[np.sum(table_l['sum 1'], axis=1) > 0]
    table_r = table_r[np.sum(table_r['sum 1'], axis=1) > 0]

    centers = compute_jackknife_fields(
        table_l, 100, weights=np.sum(table_l['sum 1'], axis=1))
    compute_jackknife_fields(table_r, centers)

    z_bins = np.array([0.15, 0.31, 0.43, 0.54, 0.70])

    for lens_bin in range(len(z_bins) - 1):
        mask_l = ((z_bins[lens_bin] <= table_l['z']) &
                  (table_l['z'] < z_bins[lens_bin + 1]))
        mask_r = ((z_bins[lens_bin] <= table_r['z']) &
                  (table_r['z'] < z_bins[lens_bin + 1]))

        kwargs = {'return_table': True,
                  'scalar_shear_response_correction': True,
                  'shear_responsivity_correction': True,
                  'hsc_selection_bias_correction': True,
                  'boost_correction': False, 'random_subtraction': True,
                  'photo_z_dilution_correction': True,
                  'table_r': table_r[mask_r]}

        result = excess_surface_density(table_l[mask_l], **kwargs)
        kwargs['return_table'] = False
        result['ds_err'] = np.sqrt(np.diag(jackknife_resampling(
            excess_surface_density, table_l[mask_l], **kwargs)))

        result.write('hsc_{}.csv'.format(lens_bin))

Acknowledgments
---------------

When using the above data and algorithms, please make sure to cite `Mandelbaum et al. (2018a) <https://ui.adsabs.harvard.edu/abs/2018PASJ...70S..25M/abstract>`_ and `Mandelbaum et al. (2018b) <https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.3170M/abstract>`_.
