"""
渲染管道模块

管理渲染器实例的创建、缓存和渲染逻辑
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from scene.dataset_readers import SARCameraInfo, RadarParams
from gaussian_model import GaussianModel


@dataclass
class RenderResult:
    """渲染结果"""
    image: torch.Tensor
    camera_key: Tuple


class RenderPipeline:
    """渲染管道 - 管理渲染器和协方差缓存"""

    def __init__(self, range_samples: int = 128, azimuth_samples: int = 128):
        """初始化渲染管道

        Args:
            range_samples: 距离向采样数（固定128）
            azimuth_samples: 方位向采样数（固定128）
        """
        self.range_samples = range_samples
        self.azimuth_samples = azimuth_samples
        self._renderers: Dict[Tuple, object] = {}
        self._device: Optional[torch.device] = None

    def set_device(self, device: torch.device):
        """设置计算设备

        Args:
            device: torch.device
        """
        self._device = device

    def render(
        self,
        model: GaussianModel,
        camera: SARCameraInfo,
        norm_factor: float = 1.0
    ) -> torch.Tensor:
        """渲染单张SAR图像

        Args:
            model: 高斯模型
            camera: 相机信息
            norm_factor: 归一化因子

        Returns:
            torch.Tensor: 渲染图像
        """
        renderer = self._get_or_create_renderer(camera)
        cov_full = model.compute_covariance_full()

        rendered = renderer(
            model._means,
            cov_full,
            model.get_opacity().squeeze(-1),
            model.get_active_sh_coeffs()
        )

        if rendered.shape != camera.image.shape:
            rendered = torch.nn.functional.interpolate(
                rendered.unsqueeze(0).unsqueeze(0),
                size=camera.image.shape,
                mode='bilinear',
                align_corners=False
            ).squeeze()

        return rendered

    def _get_or_create_renderer(self, camera: SARCameraInfo):
        """获取或创建渲染器（带缓存）

        Args:
            camera: 相机信息

        Returns:
            渲染器实例
        """
        key = self._make_camera_key(camera.radar_params)

        if key not in self._renderers:
            self._renderers[key] = self._create_renderer(camera)

        return self._renderers[key]

    def _create_renderer(self, camera: SARCameraInfo):
        """创建新的渲染器

        Args:
            camera: 相机信息

        Returns:
            渲染器实例
        """
        rp = camera.radar_params

        theta_rad = np.deg2rad(rp.incidence_angle)
        phi_rad = np.deg2rad(rp.azimuth_angle)
        sin_beta = np.sin(theta_rad) * np.cos(phi_rad)
        beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))
        tan_beta = np.tan(beta)
        alpha = np.deg2rad(rp.track_angle)

        radar_x = rp.radar_altitude * tan_beta * np.sin(alpha)
        radar_y = -rp.radar_altitude * tan_beta * np.cos(alpha)
        radar_z = rp.radar_altitude

        from cuda_rasterizer.rasterizer_autograd import SARRasterizer

        renderer = SARRasterizer(
            radar_x=radar_x,
            radar_y=radar_y,
            radar_z=radar_z,
            track_angle=rp.track_angle,
            incidence_angle=rp.incidence_angle,
            azimuth_angle=rp.azimuth_angle,
            range_resolution=rp.range_resolution,
            azimuth_resolution=rp.azimuth_resolution,
            range_samples=self.range_samples,
            azimuth_samples=self.azimuth_samples
        )

        if self._device is not None:
            renderer = renderer.to(self._device)

        return renderer

    def _make_camera_key(self, rp: RadarParams) -> Tuple:
        """生成相机唯一标识键

        Args:
            rp: 雷达参数

        Returns:
            tuple: 唯一键
        """
        return (
            rp.incidence_angle,
            rp.track_angle,
            rp.azimuth_angle,
            rp.range_resolution,
            rp.azimuth_resolution
        )

    def get_renderer_count(self) -> int:
        """获取当前缓存的渲染器数量

        Returns:
            int: 缓存的渲染器数量
        """
        return len(self._renderers)

    def clear_cache(self):
        """清空渲染器缓存"""
        self._renderers.clear()
