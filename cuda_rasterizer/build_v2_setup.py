"""
SAR-GS CUDA Rasterizer V2 编译脚本

使用方法:
    python build_v2_setup.py build_ext --inplace

基于PyTorch的CUDA扩展编译系统。
"""

import os
import sys

os.environ['USE_NINJA'] = '0'
os.environ['TORCH_CUDA_ARCH_LIST'] = '9.0;8.9;8.6;8.0;7.5;7.0'

import torch
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = Path(__file__).parent.absolute()
CUDA_SOURCES = [ROOT_DIR / "rasterizer_impl_v2.cu"]

print("=" * 60)
print("SAR-GS CUDA Rasterizer V2 编译脚本")
print("=" * 60)
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA_HOME: {torch.utils.cpp_extension.CUDA_HOME}")

if not torch.cuda.is_available():
    print("警告: CUDA不可用，但仍尝试编译...")

try:
    print(f"可用GPU: {torch.cuda.get_device_name(0)}")
except:
    print("无可用GPU")

os.environ['USE_NINJA'] = '0'

extension = CUDAExtension(
    name="sar_gs.cuda_rasterizer.cuda_rasterizer_sar",
    sources=[str(s) for s in CUDA_SOURCES],
    extra_compile_args={
        'cxx': ['/O2', '/MD', '/std:c++17', '/DNOMINMAX', '/wd4251', '/wd4244', '/wd4267'],
        'nvcc': [
            '-O3',
            '--use_fast_math',
            '-gencode=arch=compute_120,code=sm_120',
            '-gencode=arch=compute_90,code=sm_90',
            '-Xcompiler=/MD',
            '-std=c++17',
            '-DNOMINMAX',
        ]
    },
    extra_link_args=['/LIBPATH:"C:\\Users\\LIU\\anaconda3\\envs\\SARGS\\lib\\site-packages\\torch\\lib"']
)

setup(
    name="sar_gs_cuda_rasterizer_v2",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=[extension],
    cmdclass={'build_ext': BuildExtension},
    description="SAR-GS CUDA Rasterizer V2",
)
