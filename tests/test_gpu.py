from astropy.table import Table
import numpy as np
from time import time
import astropy.units as u

from dsigma.precompute import precompute
from fixtures import test_catalogs


def test_precompute(test_catalogs, assert_close=True, gpu_node=False):
    table_l, table_s = test_catalogs

    if gpu_node:
        n_jobs_gpu = 4
        n_jobs_cpu = 128
    else:
        n_jobs_gpu = 1
        n_jobs_cpu = 20
    # Precompute the fields
    n_bins = 30
    theta_bins = np.logspace(0, 1, n_bins + 1) * u.deg
    times = []
    methods = []
    times.append(time())
    precomputed_table_gpu = precompute(table_l.copy(), table_s, theta_bins,
                                                  use_gpu=True, n_jobs=n_jobs_gpu, nside=64)
    times.append(time())
    methods.append("GPU")
    precomputed_table_gpu_shared = precompute(table_l.copy(), table_s, theta_bins,
                                                  use_gpu=True, n_jobs=n_jobs_gpu, force_shared=True)
    times.append(time())
    methods.append("GPU shared")
    precomputed_table_gpu_global = precompute(table_l.copy(), table_s, theta_bins,
                                                  use_gpu=True, n_jobs=n_jobs_gpu, force_global=True)
    times.append(time())
    methods.append("GPU global")
    precomputed_table_cpu = precompute(table_l.copy(), table_s, theta_bins,
                                                  use_gpu=False, n_jobs=n_jobs_cpu)
    times.append(time())
    methods.append("CPU")

    for i in range(len(times)-1):
        print(f"{methods[i]} precomputation time: {times[i+1] - times[i]:.4f} seconds")
        print(f"Speedup: {(times[-1] - times[-2]) / (times[i+1] - times[i]):.2f}x")

    # Check that the GPU and CPU results are the same
    if assert_close:
        for col in precomputed_table_gpu.colnames:
            assert np.allclose(precomputed_table_gpu[col], precomputed_table_cpu[col])
            assert np.allclose(precomputed_table_gpu_shared[col], precomputed_table_cpu[col])
            assert np.allclose(precomputed_table_gpu_global[col], precomputed_table_cpu[col])
        return 
    else:
        return precomputed_table_gpu, precomputed_table_gpu_shared, precomputed_table_gpu_global, precomputed_table_cpu

if __name__ == "__main__":
    import numpy as np
    from astropy.table import Table


    n_l, n_s = 10**6, 10**7

    np.random.seed(0)

    table_l = Table()
    table_l['ra'] = np.random.random(n_l) * 360.0
    table_l['dec'] = np.rad2deg(np.arcsin(2 * np.random.random(n_l) - 1))
    table_l['z'] = np.random.random(n_l) * 0.2 + 0.1
    table_l['w_sys'] = 1.0

    table_s = Table()
    table_s['ra'] = np.random.random(n_s) * 360.0
    table_s['dec'] = np.rad2deg(np.arcsin(2 * np.random.random(n_s) - 1))
    table_s['z'] = np.random.random(n_s) * 0.5 + 0.1
    table_s['w'] = np.random.random(n_s) * 0.2 + 0.9
    table_s['e_1'] = np.random.normal(loc=0, scale=0.2, size=n_s)
    table_s['e_2'] = np.random.normal(loc=0, scale=0.2, size=n_s)

    import sys
    gpu_node = ('--gpu-node' in sys.argv)
    print(f"Running on GPU node: {gpu_node}")
    tab_l_gpu, tab_l_gpu_shared, tab_l_gpu_global, tab_l_cpu = test_precompute((table_l, table_s),assert_close=False,gpu_node = gpu_node)

    # print(tab_l_gpu)
    # print(tab_l_cpu)

    for col in tab_l_gpu.colnames:
        print(col, np.allclose(tab_l_gpu[col], tab_l_cpu[col]), np.allclose(tab_l_gpu_shared[col], tab_l_cpu[col]), np.allclose(tab_l_gpu_global[col], tab_l_cpu[col]))
        if not np.allclose(tab_l_gpu[col], tab_l_cpu[col]):
            print(np.mean(tab_l_gpu[col]),np.mean(tab_l_cpu[col]),np.mean(tab_l_gpu[col])/np.mean(tab_l_cpu[col]))
        if not np.allclose(tab_l_gpu_shared[col], tab_l_cpu[col]):
            print(np.mean(tab_l_gpu_shared[col]),np.mean(tab_l_cpu[col]),np.mean(tab_l_gpu_shared[col])/np.mean(tab_l_cpu[col]))
        if not np.allclose(tab_l_gpu_global[col], tab_l_cpu[col]):
            print(np.mean(tab_l_gpu_global[col]),np.mean(tab_l_cpu[col]),np.mean(tab_l_gpu_global[col])/np.mean(tab_l_cpu[col]))
        print("*"*100)


