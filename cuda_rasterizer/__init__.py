"""
SAR-GS CUDA Rasterizer Python绑定

该模块提供CUDA加速的SAR前向渲染功能。

核心公式:
I(na,nr) = Σ_i [ Si × σi × γi,ipp(na,nr) × Ωi,s(na,nr) ]
Ωi,s = Π_{j: Zj < Zi} (1 - γj,s × σj)
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
import importlib.util
import os
import sys

_pyd_path = os.path.join(os.path.dirname(__file__), 'cuda_rasterizer_sar.cp39-win_amd64.pyd')

CUDA_AVAILABLE = False
CUDASARRasterizer = None
render_sar_func = None
render_sar_backward = None

def _load_cuda_module():
    global CUDA_AVAILABLE, CUDASARRasterizer, render_sar_func, render_sar_backward

    if not torch.cuda.is_available():
        print("CUDA不可用")
        CUDA_AVAILABLE = False
        return

    try:
        if os.path.exists(_pyd_path):
            spec = importlib.util.spec_from_file_location("cuda_rasterizer_sar", _pyd_path)
            if spec and spec.loader:
                cuda_module = importlib.util.module_from_spec(spec)
                sys.modules['cuda_rasterizer_sar'] = cuda_module
                spec.loader.exec_module(cuda_module)
                render_sar_func = cuda_module.render_sar
                render_sar_backward = getattr(cuda_module, 'render_sar_backward', None)
                CUDA_AVAILABLE = True
                print("CUDA模块加载成功")
                return
    except Exception as e:
        print(f"直接加载失败: {e}")

    try:
        from torch.utils.cpp_extension import load
        os.chdir(os.path.dirname(__file__))
        cuda_module = load(
            name='cuda_rasterizer_sar',
            sources=['rasterizer_impl_v2.cu'],
            verbose=True
        )
        render_sar_func = cuda_module.render_sar
        render_sar_backward = getattr(cuda_module, 'render_sar_backward', None)
        CUDA_AVAILABLE = True
        print("CUDA模块编译加载成功")
    except Exception as e:
        print(f"CUDA模块加载失败: {e}")
        CUDA_AVAILABLE = False

_load_cuda_module()


class GaussianDataCPU:
    """CPU端高斯数据结构"""

    def __init__(
        self,
        means: torch.Tensor,
        covs: torch.Tensor,
        transmittance: torch.Tensor,
        scattering: torch.Tensor,
        radar_position: Tuple[float, float, float],
        track_angle: float,
        incidence_angle: float,
        azimuth_angle: float = 0.0,
        range_resolution: float = 1.0,
        azimuth_resolution: float = 1.0,
        range_samples: int = 64,
        azimuth_samples: int = 64,
    ):
        self.means = means
        self.covs = covs
        self.transmittance = transmittance
        self.scattering = scattering
        self.radar_position = radar_position
        self.track_angle = track_angle
        self.incidence_angle = incidence_angle
        self.azimuth_angle = azimuth_angle
        self.range_resolution = range_resolution
        self.azimuth_resolution = azimuth_resolution
        self.range_samples = range_samples
        self.azimuth_samples = azimuth_samples

    def to_device(self, device: torch.device) -> 'GaussianDataGPU':
        return GaussianDataGPU(
            means=self.means.to(device),
            covs=self.covs.to(device),
            transmittance=self.transmittance.to(device),
            scattering=self.scattering.to(device),
            radar_position=self.radar_position,
            track_angle=self.track_angle,
            incidence_angle=self.incidence_angle,
            azimuth_angle=self.azimuth_angle,
            range_resolution=self.range_resolution,
            azimuth_resolution=self.azimuth_resolution,
            range_samples=self.range_samples,
            azimuth_samples=self.azimuth_samples,
        )


class GaussianDataGPU:
    """GPU端高斯数据结构"""

    def __init__(
        self,
        means: torch.Tensor,
        covs: torch.Tensor,
        transmittance: torch.Tensor,
        scattering: torch.Tensor,
        radar_position: Tuple[float, float, float],
        track_angle: float,
        incidence_angle: float,
        azimuth_angle: float = 0.0,
        range_resolution: float = 1.0,
        azimuth_resolution: float = 1.0,
        range_samples: int = 64,
        azimuth_samples: int = 64,
    ):
        self.means = means
        self.covs = covs
        self.transmittance = transmittance
        self.scattering = scattering
        self.radar_position = radar_position
        self.track_angle = track_angle
        self.incidence_angle = incidence_angle
        self.azimuth_angle = azimuth_angle
        self.range_resolution = range_resolution
        self.azimuth_resolution = azimuth_resolution
        self.range_samples = range_samples
        self.azimuth_samples = azimuth_samples


def render_sar(
    gaussian_data: GaussianDataGPU,
    output_height: int = 512,
    output_width: int = 512,
) -> Tuple[torch.Tensor, Dict]:
    """
    SAR前向渲染函数

    Args:
        gaussian_data: GPU端高斯数据
        output_height: 输出图像高度
        output_width: 输出图像宽度

    Returns:
        rendered_image: [H, W] 渲染的SAR图像
        aux_data: 辅助数据字典
    """
    if not CUDA_AVAILABLE or render_sar_func is None:
        raise RuntimeError("CUDA渲染器不可用")

    output = render_sar_func(
        gaussian_data.means,
        gaussian_data.covs,
        gaussian_data.transmittance,
        gaussian_data.scattering,
        gaussian_data.radar_position[0],
        gaussian_data.radar_position[1],
        gaussian_data.radar_position[2],
        gaussian_data.track_angle,
        gaussian_data.incidence_angle,
        gaussian_data.azimuth_angle,
        gaussian_data.range_resolution,
        gaussian_data.azimuth_resolution,
        gaussian_data.range_samples,
        gaussian_data.azimuth_samples,
    )

    aux_data = {}

    return output, aux_data


__all__ = [
    'CUDA_AVAILABLE',
    'CUDASARRasterizer',
    'render_sar',
    'GaussianDataCPU',
    'GaussianDataGPU',
]