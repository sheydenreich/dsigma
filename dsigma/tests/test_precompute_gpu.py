import unittest
import numpy as np
import numpy.testing as npt
from astropy.table import Table
import astropy.units as u

# Attempt to import healpy for HEALPix operations
try:
    import healpy as hp
    HEALPY_FOUND = True
except ImportError:
    HEALPY_FOUND = False
    # print("Warning: healpy not found. HEALPix ID generation in tests will be mocked/simplified.")

# CPU engine (assuming it can be imported like this in the test environment)
try:
    from dsigma.precompute_engine import precompute_engine as precompute_cpu_engine
    CPU_ENGINE_FOUND = True
except ImportError:
    CPU_ENGINE_FOUND = False
    # print("Warning: dsigma.precompute_engine could not be imported. CPU comparison will be skipped.")

# GPU wrapper
from dsigma._precompute_cuda import precompute_gpu_wrapper


class TestPrecomputeGPU(unittest.TestCase):

    def _generate_test_data(self, n_lenses=100, n_sources=1000, n_bins=10, nside=32,
                              with_m=False, with_e_rms=False, with_r2=False,
                              with_r_matrix=False, with_sigma_crit_eff=False,
                              seed=0):
        np.random.seed(seed)

        # Basic properties
        table_l = Table()
        table_l['z_l'] = np.random.uniform(0.1, 0.5, n_lenses)
        table_l['ra_l'] = np.random.uniform(0, 360, n_lenses) # degrees
        table_l['dec_l'] = np.random.uniform(-90, 90, n_lenses) # degrees

        table_s = Table()
        table_s['z_s'] = np.random.uniform(0.2, 2.0, n_sources)
        table_s['ra_s'] = np.random.uniform(0, 360, n_sources) # degrees
        table_s['dec_s'] = np.random.uniform(-90, 90, n_sources) # degrees
        table_s['w_s'] = np.random.uniform(0.5, 2.0, n_sources)
        table_s['e_1_s'] = np.random.normal(0, 0.05, n_sources)
        table_s['e_2_s'] = np.random.normal(0, 0.05, n_sources)

        # Ensure z_l < z_s for a majority of pairs for realistic z_l_max_s
        # A simple z_l_max_s generation: sources can be lensed by lenses up to z_s - 0.1
        table_s['z_l_max_s'] = np.maximum(0, table_s['z_s'] - 0.1)


        # HEALPix IDs
        order = 'ring' # Consistent with C++ default if not specified otherwise
        if HEALPY_FOUND:
            table_l['healpix_id_l'] = hp.ang2pix(nside, table_l['ra_l'], table_l['dec_l'], lonlat=True, nest= (order=='nested'))
            table_s['healpix_id_s'] = hp.ang2pix(nside, table_s['ra_s'], table_s['dec_s'], lonlat=True, nest= (order=='nested'))
        else: # Fallback if healpy is not found
            table_l['healpix_id_l'] = np.random.randint(0, hp.nside2npix(nside) if HEALPY_FOUND else nside*nside*12, n_lenses)
            table_s['healpix_id_s'] = np.random.randint(0, hp.nside2npix(nside) if HEALPY_FOUND else nside*nside*12, n_sources)

        # Sort by HEALPix ID (required by both CPU and GPU implementations)
        l_sort_idx = np.argsort(table_l['healpix_id_l'])
        table_l = table_l[l_sort_idx]

        s_sort_idx = np.argsort(table_s['healpix_id_s'])
        table_s = table_s[s_sort_idx]

        # Unique pixel info for CPU engine
        u_pix_l, u_pix_l_indices, u_pix_l_counts = np.unique(table_l['healpix_id_l'], return_index=True, return_counts=True)
        n_pix_l_in = np.cumsum(u_pix_l_counts) # Cumulative counts

        u_pix_s, u_pix_s_indices, u_pix_s_counts = np.unique(table_s['healpix_id_s'], return_index=True, return_counts=True)
        n_pix_s_in = np.cumsum(u_pix_s_counts)


        # Derived trigonometric and comoving distance arrays (as doubles)
        d_com_l_np = np.random.uniform(100, 1000, n_lenses).astype(np.double) # Mock comoving distances
        d_com_s_np = np.random.uniform(200, 2000, n_sources).astype(np.double)
        # Ensure d_com_s > d_com_l for many pairs for physical scenarios
        for i in range(n_lenses): # Simple way to ensure some physical pairs
            if n_sources > i : d_com_s_np[i] = max(d_com_s_np[i], d_com_l_np[i % n_lenses] + 100)


        ra_l_rad = np.deg2rad(table_l['ra_l']).astype(np.double)
        dec_l_rad = np.deg2rad(table_l['dec_l']).astype(np.double)
        sin_ra_l_np = np.sin(ra_l_rad)
        cos_ra_l_np = np.cos(ra_l_rad)
        sin_dec_l_np = np.sin(dec_l_rad)
        cos_dec_l_np = np.cos(dec_l_rad)

        ra_s_rad = np.deg2rad(table_s['ra_s']).astype(np.double)
        dec_s_rad = np.deg2rad(table_s['dec_s']).astype(np.double)
        sin_ra_s_np = np.sin(ra_s_rad)
        cos_ra_s_np = np.cos(ra_s_rad)
        sin_dec_s_np = np.sin(dec_s_rad)
        cos_dec_s_np = np.cos(dec_s_rad)

        # Binning
        # For each lens, n_bins+1 bin edges. Last edge is max_dist for that lens.
        dist_3d_sq_bins_np = np.zeros((n_lenses, n_bins + 1), dtype=np.double)
        for i in range(n_lenses):
            # Create some monotonically increasing bin edges
            max_dist_sq_lens = np.random.uniform(50**2, 200**2) # Max squared distance for this lens
            dist_3d_sq_bins_np[i, :] = np.sort(np.random.uniform(0, max_dist_sq_lens, n_bins + 1))
            dist_3d_sq_bins_np[i, 0] = 0 # Ensure first bin starts at 0
            dist_3d_sq_bins_np[i, n_bins] = max_dist_sq_lens # Ensure last bin edge is max_dist_sq_lens
        dist_3d_sq_bins_np = dist_3d_sq_bins_np.flatten()


        # Optional data
        n_z_bins_for_sources = 5 # For sigma_crit_eff
        sigma_crit_eff_l_np = None
        z_bin_s_np = None
        if with_sigma_crit_eff:
            sigma_crit_eff_l_np = np.random.uniform(1e14, 1e15, (n_lenses, n_z_bins_for_sources)).astype(np.double).flatten()
            z_bin_s_np = np.random.randint(0, n_z_bins_for_sources, n_sources).astype(np.int32)
            table_s['z_bin'] = z_bin_s_np # For CPU engine dict

        m_s_np, e_rms_s_np, R_2_s_np = None, None, None
        R_11_s_np, R_12_s_np, R_21_s_np, R_22_s_np = None, None, None, None

        if with_m:
            m_s_np = np.random.normal(0.01, 0.02, n_sources).astype(np.double)
            table_s['m'] = m_s_np
        if with_e_rms:
            e_rms_s_np = np.random.uniform(0.1, 0.3, n_sources).astype(np.double)
            table_s['e_rms'] = e_rms_s_np
        if with_r2:
            R_2_s_np = np.random.uniform(0.2, 0.4, n_sources).astype(np.double)
            table_s['R_2'] = R_2_s_np
        if with_r_matrix:
            R_11_s_np = np.random.normal(1, 0.05, n_sources).astype(np.double)
            R_12_s_np = np.random.normal(0, 0.01, n_sources).astype(np.double)
            R_21_s_np = np.random.normal(0, 0.01, n_sources).astype(np.double)
            R_22_s_np = np.random.normal(1, 0.05, n_sources).astype(np.double)
            table_s['R_11'] = R_11_s_np
            table_s['R_12'] = R_12_s_np
            table_s['R_21'] = R_21_s_np
            table_s['R_22'] = R_22_s_np

        # Prepare dicts for CPU engine (as it takes table-like dicts for sources/lenses)
        table_l_dict_cpu = {
            'z': table_l['z_l'].astype(np.double), 'd_com': d_com_l_np,
            'sin ra': sin_ra_l_np, 'cos ra': cos_ra_l_np,
            'sin dec': sin_dec_l_np, 'cos dec': cos_dec_l_np
        }
        if with_sigma_crit_eff: # sigma_crit_eff_l is passed differently to CPU engine
             # CPU engine expects sigma_crit_eff per lens, not flattened for all z_bins yet.
             # This part needs to align with how precompute_cpu_engine expects it.
             # For now, pass the flattened one, knowing it might not be right for CPU.
            table_l_dict_cpu['sigma_crit_eff'] = sigma_crit_eff_l_np # Placeholder

        table_s_dict_cpu = {
            'z': table_s['z_s'].astype(np.double), 'd_com': d_com_s_np,
            'sin ra': sin_ra_s_np, 'cos ra': cos_ra_s_np,
            'sin dec': sin_dec_s_np, 'cos dec': cos_dec_s_np,
            'w': table_s['w_s'].astype(np.double),
            'e_1': table_s['e_1_s'].astype(np.double),
            'e_2': table_s['e_2_s'].astype(np.double),
            'z_l_max': table_s['z_l_max_s'].astype(np.double)
        }
        if 'z_bin' in table_s.colnames: table_s_dict_cpu['z_bin'] = table_s['z_bin'].astype(np.int32)
        if 'm' in table_s.colnames: table_s_dict_cpu['m'] = table_s['m'].astype(np.double)
        if 'e_rms' in table_s.colnames: table_s_dict_cpu['e_rms'] = table_s['e_rms'].astype(np.double)
        if 'R_2' in table_s.colnames: table_s_dict_cpu['R_2'] = table_s['R_2'].astype(np.double)
        if 'R_11' in table_s.colnames:
            table_s_dict_cpu['R_11'] = table_s['R_11'].astype(np.double)
            table_s_dict_cpu['R_12'] = table_s['R_12'].astype(np.double)
            table_s_dict_cpu['R_21'] = table_s['R_21'].astype(np.double)
            table_s_dict_cpu['R_22'] = table_s['R_22'].astype(np.double)

        return {
            "z_l_np": table_l['z_l'].astype(np.double), "d_com_l_np": d_com_l_np,
            "sin_ra_l_np": sin_ra_l_np, "cos_ra_l_np": cos_ra_l_np,
            "sin_dec_l_np": sin_dec_l_np, "cos_dec_l_np": cos_dec_l_np,
            "healpix_id_l_np": table_l['healpix_id_l'].astype(np.int64),
            "z_s_np": table_s['z_s'].astype(np.double), "d_com_s_np": d_com_s_np,
            "sin_ra_s_np": sin_ra_s_np, "cos_ra_s_np": cos_ra_s_np,
            "sin_dec_s_np": sin_dec_s_np, "cos_dec_s_np": cos_dec_s_np,
            "w_s_np": table_s['w_s'].astype(np.double),
            "e_1_s_np": table_s['e_1_s'].astype(np.double),
            "e_2_s_np": table_s['e_2_s'].astype(np.double),
            "z_l_max_s_np": table_s['z_l_max_s'].astype(np.double),
            "healpix_id_s_np": table_s['healpix_id_s'].astype(np.int64),
            "nside_healpix": nside, "order_healpix_str": order,
            "has_sigma_crit_eff": with_sigma_crit_eff, "n_z_bins_l": n_z_bins_for_sources,
            "sigma_crit_eff_l_np": sigma_crit_eff_l_np, "z_bin_s_np": z_bin_s_np,
            "has_m_s": with_m, "m_s_np": m_s_np,
            "has_e_rms_s": with_e_rms, "e_rms_s_np": e_rms_s_np,
            "has_R_2_s": with_r2, "R_2_s_np": R_2_s_np,
            "has_R_matrix_s": with_r_matrix,
            "R_11_s_np": R_11_s_np, "R_12_s_np": R_12_s_np,
            "R_21_s_np": R_21_s_np, "R_22_s_np": R_22_s_np,
            "dist_3d_sq_bins_np": dist_3d_sq_bins_np, "n_bins": n_bins,
            # CPU engine specific inputs
            "u_pix_l_cpu": u_pix_l.astype(np.int64), "n_pix_l_in_cpu": n_pix_l_in.astype(np.int64),
            "u_pix_s_cpu": u_pix_s.astype(np.int64), "n_pix_s_in_cpu": n_pix_s_in.astype(np.int64),
            "table_l_dict_cpu": table_l_dict_cpu, "table_s_dict_cpu": table_s_dict_cpu
        }

    def _initialize_output_arrays(self, n_lenses, n_bins, data_dict):
        size = n_lenses * n_bins
        outputs = {
            "sum_1_r_np": np.zeros(size, dtype=np.int64),
            "sum_w_ls_r_np": np.zeros(size, dtype=np.double),
            "sum_w_ls_e_t_r_np": np.zeros(size, dtype=np.double),
            "sum_w_ls_e_t_sigma_crit_r_np": np.zeros(size, dtype=np.double),
            "sum_w_ls_sigma_crit_r_np": np.zeros(size, dtype=np.double),
            "sum_w_ls_z_s_r_np": np.zeros(size, dtype=np.double),
            "sum_w_ls_m_r_np": np.zeros(size, dtype=np.double) if data_dict["has_m_s"] else None,
            "sum_w_ls_1_minus_e_rms_sq_r_np": np.zeros(size, dtype=np.double) if data_dict["has_e_rms_s"] else None,
            "sum_w_ls_A_p_R_2_r_np": np.zeros(size, dtype=np.double) if data_dict["has_R_2_s"] else None,
            "sum_w_ls_R_T_r_np": np.zeros(size, dtype=np.double) if data_dict["has_R_matrix_s"] else None,
        }
        return outputs

    def test_basic_comparison(self):
        n_lenses, n_sources, n_bins, nside = 3, 5, 2, 4 # Small numbers for quick test
        # n_lenses, n_sources, n_bins, nside = 100, 1000, 10, 32 # Larger test

        data = self._generate_test_data(n_lenses, n_sources, n_bins, nside,
                                          with_m=True, with_e_rms=True, with_r2=True,
                                          with_r_matrix=True, with_sigma_crit_eff=True)

        # Output arrays for GPU
        out_gpu = self._initialize_output_arrays(n_lenses, n_bins, data)

        # Call GPU wrapper
        precompute_gpu_wrapper(
            z_l_np=data["z_l_np"], d_com_l_np=data["d_com_l_np"],
            sin_ra_l_np=data["sin_ra_l_np"], cos_ra_l_np=data["cos_ra_l_np"],
            sin_dec_l_np=data["sin_dec_l_np"], cos_dec_l_np=data["cos_dec_l_np"],
            healpix_id_l_np=data["healpix_id_l_np"],
            z_s_np=data["z_s_np"], d_com_s_np=data["d_com_s_np"],
            sin_ra_s_np=data["sin_ra_s_np"], cos_ra_s_np=data["cos_ra_s_np"],
            sin_dec_s_np=data["sin_dec_s_np"], cos_dec_s_np=data["cos_dec_s_np"],
            w_s_np=data["w_s_np"], e_1_s_np=data["e_1_s_np"], e_2_s_np=data["e_2_s_np"],
            z_l_max_s_np=data["z_l_max_s_np"], healpix_id_s_np=data["healpix_id_s_np"],
            nside_healpix=data["nside_healpix"], order_healpix_str=data["order_healpix_str"],
            has_sigma_crit_eff=data["has_sigma_crit_eff"], n_z_bins_l=data["n_z_bins_l"],
            sigma_crit_eff_l_np=data["sigma_crit_eff_l_np"], z_bin_s_np=data["z_bin_s_np"],
            has_m_s=data["has_m_s"], m_s_np=data["m_s_np"],
            has_e_rms_s=data["has_e_rms_s"], e_rms_s_np=data["e_rms_s_np"],
            has_R_2_s=data["has_R_2_s"], R_2_s_np=data["R_2_s_np"],
            has_R_matrix_s=data["has_R_matrix_s"],
            R_11_s_np=data["R_11_s_np"], R_12_s_np=data["R_12_s_np"],
            R_21_s_np=data["R_21_s_np"], R_22_s_np=data["R_22_s_np"],
            dist_3d_sq_bins_np=data["dist_3d_sq_bins_np"], n_bins=data["n_bins"],
            comoving=False, weighting=0.0, # Example config
            sum_1_r_np=out_gpu["sum_1_r_np"], sum_w_ls_r_np=out_gpu["sum_w_ls_r_np"],
            sum_w_ls_e_t_r_np=out_gpu["sum_w_ls_e_t_r_np"],
            sum_w_ls_e_t_sigma_crit_r_np=out_gpu["sum_w_ls_e_t_sigma_crit_r_np"],
            sum_w_ls_sigma_crit_r_np=out_gpu["sum_w_ls_sigma_crit_r_np"],
            sum_w_ls_z_s_r_np=out_gpu["sum_w_ls_z_s_r_np"],
            sum_w_ls_m_r_np=out_gpu["sum_w_ls_m_r_np"],
            sum_w_ls_1_minus_e_rms_sq_r_np=out_gpu["sum_w_ls_1_minus_e_rms_sq_r_np"],
            sum_w_ls_A_p_R_2_r_np=out_gpu["sum_w_ls_A_p_R_2_r_np"],
            sum_w_ls_R_T_r_np=out_gpu["sum_w_ls_R_T_r_np"],
            n_gpus=1
        )

        # Output arrays for CPU
        out_cpu = self._initialize_output_arrays(n_lenses, n_bins, data)
        table_r_dict_cpu = { # This is how precompute_cpu_engine expects outputs
            'sum_1': out_cpu["sum_1_r_np"],
            'sum_w_ls': out_cpu["sum_w_ls_r_np"],
            'sum_w_ls_e_t': out_cpu["sum_w_ls_e_t_r_np"],
            'sum_w_ls_e_t_sigma_crit': out_cpu["sum_w_ls_e_t_sigma_crit_r_np"],
            'sum_w_ls_sigma_crit': out_cpu["sum_w_ls_sigma_crit_r_np"],
            'sum_w_ls_z_s': out_cpu["sum_w_ls_z_s_r_np"],
        }
        if data["has_m_s"]: table_r_dict_cpu['sum_w_ls_m'] = out_cpu["sum_w_ls_m_r_np"]
        if data["has_e_rms_s"]: table_r_dict_cpu['sum_w_ls_1_minus_e_rms_sq'] = out_cpu["sum_w_ls_1_minus_e_rms_sq_r_np"]
        if data["has_R_2_s"]: table_r_dict_cpu['sum_w_ls_A_p_R_2'] = out_cpu["sum_w_ls_A_p_R_2_r_np"]
        if data["has_R_matrix_s"]: table_r_dict_cpu['sum_w_ls_R_T'] = out_cpu["sum_w_ls_R_T_r_np"]

        if CPU_ENGINE_FOUND:
            # Call CPU engine
            # Note: sigma_crit_eff handling for CPU engine might need adjustment
            # The CPU engine's sigma_crit_eff might expect a different shape or be handled internally
            # by dsigma.precompute.precompute if has_sigma_crit_eff is True.
            # This direct call to precompute_cpu_engine might need careful alignment.
            precompute_cpu_engine(
                u_pix_l=data["u_pix_l_cpu"], n_pix_l_in=data["n_pix_l_in_cpu"],
                u_pix_s=data["u_pix_s_cpu"], n_pix_s_in=data["n_pix_s_in_cpu"],
                dist_3d_sq_bins_in=data["dist_3d_sq_bins_np"], # CPU engine expects it flat
                table_l_dict=data["table_l_dict_cpu"],
                table_s_dict=data["table_s_dict_cpu"],
                table_r_dict=table_r_dict_cpu,
                bins_dummy=np.arange(n_bins + 1), # For indexing, not bin values
                comoving=False, weighting=0.0, # Must match GPU call
                nside=data["nside_healpix"],
                # Optional arguments for CPU engine
                has_sigma_crit_eff=data["has_sigma_crit_eff"],
                n_z_bins=data["n_z_bins_l"] if data["has_sigma_crit_eff"] else 0, # n_z_bins for sigma_crit_eff
                has_m=data["has_m_s"], has_e_rms=data["has_e_rms_s"],
                has_R_2=data["has_R_2_s"], has_R_matrix=data["has_R_matrix_s"],
                queue=None, progress_bar=False
            )

            # Compare results
            npt.assert_equal(out_gpu["sum_1_r_np"], out_cpu["sum_1_r_np"])
            for key in out_gpu:
                if key == "sum_1_r_np": continue # Already checked
                if out_gpu[key] is not None: # Only compare if array exists
                    self.assertIsNotNone(out_cpu[key], f"CPU output for {key} is None but GPU is not.")
                    npt.assert_allclose(out_gpu[key], out_cpu[key], rtol=1e-5, atol=1e-8,
                                        err_msg=f"Mismatch in {key}")
        else:
            self.skipTest("CPU engine not found, skipping comparison.")

if __name__ == '__main__':
    unittest.main()
