"""
Setup script for building custom CUDA kernels as PyTorch extensions.

Usage:
    python kernels/setup.py install
    
Or for development:
    python kernels/setup.py develop
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Get CUDA architecture
cuda_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;8.9;9.0")

setup(
    name='llm_cuda_kernels',
    version='1.0.0',
    description='Custom CUDA kernels for LLM serving',
    ext_modules=[
        CUDAExtension(
            name='llm_cuda_kernels',
            sources=[
                'cuda_kernels.cpp',
                'flash_attention_v2.cu',
                'fused_matmul.cu',
                'quantized_linear.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--use_fast_math',
                    '-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '-U__CUDA_NO_HALF2_OPERATORS__',
                    '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                ] + [f'-gencode=arch=compute_{arch.replace(".", "")},code=sm_{arch.replace(".", "")}' 
                     for arch in cuda_arch_list.split(';')]
            },
            include_dirs=[
                '/usr/local/cuda/include',
            ],
            library_dirs=[
                '/usr/local/cuda/lib64',
            ],
            libraries=[
                'cuda',
                'cudart',
                'cublas',
            ],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
    ],
)
