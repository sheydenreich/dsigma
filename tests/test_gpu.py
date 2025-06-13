from astropy.table import Table
import numpy as np
from time import time
import astropy.units as u

from dsigma.precompute import precompute
from fixtures import test_catalogs


def test_precompute(test_catalogs):
    table_l, table_s = test_catalogs

    # Precompute the fields
    n_bins = 30
    theta_bins = np.logspace(0, 1, n_bins + 1) * u.deg
    startt = time()
    precomputed_table_gpu = precompute(table_l.copy(), table_s, theta_bins,
                                                  use_gpu=True)
    time1 = time() - startt
    precomputed_table_cpu = precompute(table_l.copy(), table_s, theta_bins,
                                                  use_gpu=False,progress_bar=True)
    time2 = time() - time1

    print(f"GPU precomputation time: {time1:.4f} seconds")
    print(f"CPU precomputation time: {time2:.4f} seconds")
    print(f"Speedup: {time2 / time1:.2f}x")
    # Check that the GPU and CPU results are the same
    # for col in precomputed_table_gpu.colnames:
    #     assert np.allclose(precomputed_table_gpu[col], precomputed_table_cpu[col])
    return precomputed_table_gpu, precomputed_table_cpu

if __name__ == "__main__":
    n_l, n_s = 100000, 100000

    np.random.seed(0)

    table_l = Table()
    table_l['ra'] = np.random.random(n_l) * 2 * np.pi
    table_l['dec'] = np.rad2deg(np.arcsin(2 * np.random.random(n_l) - 1))
    table_l['z'] = np.random.random(n_l) * 0.2 + 0.1
    table_l['w_sys'] = 1.0

    table_s = Table()
    table_s['ra'] = np.random.random(n_s) * 2 * np.pi
    table_s['dec'] = np.rad2deg(np.arcsin(2 * np.random.random(n_s) - 1))
    table_s['z'] = np.random.random(n_s) * 0.5 + 0.1
    table_s['w'] = np.random.random(n_s) * 0.2 + 0.9
    table_s['e_1'] = np.random.normal(loc=0, scale=0.2, size=n_s)
    table_s['e_2'] = np.random.normal(loc=0, scale=0.2, size=n_s)


    precomputed_table_gpu, precomputed_table_cpu = test_precompute((table_l, table_s))

    for col in precomputed_table_gpu.colnames:
        print(f"{col}: GPU mean = {np.mean(precomputed_table_gpu[col]):.4e}, "
              f"CPU mean = {np.mean(precomputed_table_cpu[col]):.4e}"
              f", ratio: {np.mean(precomputed_table_gpu[col]) / np.mean(precomputed_table_cpu[col]):.4f}"
              f", match: {np.allclose(precomputed_table_gpu[col], precomputed_table_cpu[col])}")