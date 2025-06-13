import platform
import numpy

from Cython.Build import cythonize
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get the conda environment's include directory
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix is None:
    raise RuntimeError("This package requires a conda environment. "
                       "CONDA_PREFIX is not set.")
conda_include_path = os.path.join(conda_prefix, "include")

extra_compile_args = ['-O3']

if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
    extra_compile_args += ['-march=native']

setup(
    ext_modules=cythonize([Extension(
        'dsigma.precompute_engine', ['dsigma/precompute_engine.pyx'],
        extra_compile_args=extra_compile_args)]) + [
        CUDAExtension(
            name='dsigma._precompute_cuda',
            sources=[
                'dsigma/precompute_engine_cuda.cu',
                'dsigma/_precompute_cuda.pyx',
                'dsigma/precompute_interface.cu',
                'dsigma/cuda_host_utils.cpp',
                'dsigma/healpix_gpu.cu',
                'dsigma/kdtree_search_gpu.cu',
            ],
            # Add this line below
            include_dirs=['dsigma', numpy.get_include(),conda_include_path,
                          os.path.join(conda_prefix,"include/healpix_cxx/")],
            extra_compile_args={'cxx': ['-O3','-g'], 'nvcc': ['-O3','-g','-rdc=true']},
            libraries=['cudart', 'healpix_cxx'],
            define_macros=[('HEALPIX_FOUND', '1')],
                            )
    ],
    cmdclass={'build_ext': BuildExtension}
)
