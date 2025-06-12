import platform

from Cython.Build import cythonize
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

extra_compile_args = ['-O3']

if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
    extra_compile_args += ['-march=native']

setup(
    ext_modules=cythonize([Extension(
        'dsigma.precompute_engine', ['dsigma/precompute_engine.pyx'],
        extra_compile_args=extra_compile_args)]) + [
        CppExtension(
            name='dsigma._precompute_cuda',
            sources=[
                'dsigma/precompute_engine_cuda.cu',
                'dsigma/_precompute_cuda.pyx',
                'dsigma/precompute_interface.cpp',
                'dsigma/cuda_host_utils.cpp',
            ],
            extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
            libraries=['cudart'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
