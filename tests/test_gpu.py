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
    time1 = time()
    precomputed_table_cpu = precompute(table_l.copy(), table_s, theta_bins,
                                                  use_gpu=False)
    time2 = time()

    print(f"GPU precomputation time: {time1 - startt:.4f} seconds")
    print(f"CPU precomputation time: {time2 - time1:.4f} seconds")
    print(f"Speedup: {(time2 - time1) / (time1 - startt):.2f}x")
    # Check that the GPU and CPU results are the same
    for col in precomputed_table_gpu.colnames:
        assert np.allclose(precomputed_table_gpu[col], precomputed_table_cpu[col])
    return

