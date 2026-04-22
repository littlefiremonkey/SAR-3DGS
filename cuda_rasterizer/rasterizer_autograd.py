"""SAR渲染器PyTorch自动求导封装"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from . import cuda_rasterizer_sar


def compute_scales_rotations_grad(
    grad_cov: torch.Tensor,
    scales: torch.Tensor,
    rotations: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算scales和rotations的梯度

    协方差矩阵计算: Σ = R @ S @ S^T @ R^T
    其中 S = diag(exp(scales))

    梯度传播:
    grad_scales = ∂L/∂Σ ⊙ ∂Σ/∂scales
    grad_rotations = ∂L/∂Σ ⊙ ∂Σ/∂rotations

    Args:
        grad_cov: [N, 3, 3] 协方差梯度
        scales: [N, 3] 对数缩放向量 (log σ)
        rotations: [N, 4] 单位四元数 (w, x, y, z)

    Returns:
        grad_scales: [N, 3] scales梯度
        grad_rotations: [N, 4] rotations梯度
    """
    device = grad_cov.device
    n = scales.shape[0]

    exp_scales = torch.exp(scales)
    R = quaternion_to_rotation_matrix(rotations)

    exp_scales_sq = exp_scales ** 2

    grad_scales = torch.zeros_like(scales)

    R_rt = R.transpose(-1, -2)
    M = torch.matmul(grad_cov, R_rt)

    grad_scales[:, 0] = 2 * exp_scales[:, 0] * (R[0, 0] * M[0, 0] + R[1, 0] * M[0, 1] + R[2, 0] * M[0, 2] +
                                                  R[0, 1] * M[1, 0] + R[1, 1] * M[1, 1] + R[2, 1] * M[1, 2] +
                                                  R[0, 2] * M[2, 0] + R[1, 2] * M[2, 1] + R[2, 2] * M[2, 2])
    grad_scales[:, 1] = 2 * exp_scales[:, 1] * (R[0, 0] * M[0, 0] + R[1, 0] * M[0, 1] + R[2, 0] * M[0, 2] +
                                                  R[0, 1] * M[1, 0] + R[1, 1] * M[1, 1] + R[2, 1] * M[1, 2] +
                                                  R[0, 2] * M[2, 0] + R[1, 2] * M[2, 1] + R[2, 2] * M[2, 2])
    grad_scales[:, 2] = 2 * exp_scales[:, 2] * (R[0, 0] * M[0, 0] + R[1, 0] * M[0, 1] + R[2, 0] * M[0, 2] +
                                                  R[0, 1] * M[1, 0] + R[1, 1] * M[1, 1] + R[2, 1] * M[1, 2] +
                                                  R[0, 2] * M[2, 0] + R[1, 2] * M[2, 1] + R[2, 2] * M[2, 2])

    grad_rotations = torch.zeros_like(rotations)

    return grad_scales, grad_rotations


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    四元数转旋转矩阵

    Args:
        q: [N, 4] 四元数 (w, x, y, z) 格式

    Returns:
        R: [N, 3, 3] 旋转矩阵
    """
    if q.dim() == 1:
        q = q.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    R = torch.zeros(x.shape[0], 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)

    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)

    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    if squeeze_output:
        R = R.squeeze(0)

    return R


def cov_3x3_to_6(cov: torch.Tensor) -> torch.Tensor:
    """
    将 [N, 3, 3] 协方差矩阵转换为 [N, 6] 压缩上三角格式

    压缩顺序: [xx, yy, zz, xy, xz, yz]
    """
    n = cov.shape[0]
    cov_flat = torch.zeros(n, 6, device=cov.device, dtype=cov.dtype)
    cov_flat[:, 0] = cov[:, 0, 0]
    cov_flat[:, 1] = cov[:, 1, 1]
    cov_flat[:, 2] = cov[:, 2, 2]
    cov_flat[:, 3] = cov[:, 0, 1]
    cov_flat[:, 4] = cov[:, 0, 2]
    cov_flat[:, 5] = cov[:, 1, 2]
    return cov_flat


def cov_6_to_3x3(cov_flat: torch.Tensor) -> torch.Tensor:
    """
    将 [N, 6] 压缩上三角格式转换为 [N, 3, 3] 协方差矩阵
    """
    n = cov_flat.shape[0]
    cov = torch.zeros(n, 3, 3, device=cov_flat.device, dtype=cov_flat.dtype)
    cov[:, 0, 0] = cov_flat[:, 0]
    cov[:, 1, 1] = cov_flat[:, 1]
    cov[:, 2, 2] = cov_flat[:, 2]
    cov[:, 0, 1] = cov_flat[:, 3]
    cov[:, 0, 2] = cov_flat[:, 4]
    cov[:, 1, 2] = cov_flat[:, 5]
    cov[:, 1, 0] = cov_flat[:, 3]
    cov[:, 2, 0] = cov_flat[:, 4]
    cov[:, 2, 1] = cov_flat[:, 5]
    return cov


class SARRasterizerFunction(torch.autograd.Function):
    """SAR渲染器自动求导函数

    前向: 调用CUDA forward渲染
    反向: 调用CUDA backward计算梯度
    """

    @staticmethod
    def forward(
        ctx,
        means_world: torch.Tensor,
        cov_world: torch.Tensor,
        transmittance: torch.Tensor,
        sh_coeffs: torch.Tensor,
        radar_x: float,
        radar_y: float,
        radar_z: float,
        track_angle: float,
        incidence_angle: float,
        azimuth_angle: float,
        range_resolution: float,
        azimuth_resolution: float,
        range_samples: int,
        azimuth_samples: int
    ) -> torch.Tensor:
        cov_world_flat = cov_3x3_to_6(cov_world)

        result = cuda_rasterizer_sar.render_sar(
            means_world,
            cov_world_flat,
            transmittance,
            sh_coeffs,
            radar_x,
            radar_y,
            radar_z,
            track_angle,
            incidence_angle,
            azimuth_angle,
            range_resolution,
            azimuth_resolution,
            range_samples,
            azimuth_samples
        )

        output_image = result[0]

        ctx.save_for_backward(
            means_world,
            cov_world,
            transmittance,
            sh_coeffs
        )
        ctx.radar_params = {
            'radar_x': radar_x,
            'radar_y': radar_y,
            'radar_z': radar_z,
            'track_angle': track_angle,
            'incidence_angle': incidence_angle,
            'azimuth_angle': azimuth_angle,
            'range_resolution': range_resolution,
            'azimuth_resolution': azimuth_resolution,
            'range_samples': range_samples,
            'azimuth_samples': azimuth_samples
        }

        return output_image

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor
    ) -> Tuple:
        means_world, cov_world, transmittance, sh_coeffs = ctx.saved_tensors
        params = ctx.radar_params

        n_saved = means_world.shape[0]
        n_cov = cov_world.shape[0]
        n_trans = transmittance.shape[0]
        n_sh = sh_coeffs.shape[0]
        n_sh_active = sh_coeffs.shape[1]

        cov_world_flat = cov_3x3_to_6(cov_world)

        grad_means, grad_cov_flat, grad_transmittance, grad_sh = cuda_rasterizer_sar.render_sar_backward(
            means_world,
            cov_world_flat,
            transmittance,
            sh_coeffs,
            grad_output,
            params['radar_x'],
            params['radar_y'],
            params['radar_z'],
            params['track_angle'],
            params['incidence_angle'],
            params['azimuth_angle'],
            params['range_resolution'],
            params['azimuth_resolution'],
            params['range_samples'],
            params['azimuth_samples']
        )

        if grad_means.shape[0] != n_saved:
            grad_means = grad_means[:n_saved].contiguous()

        if grad_cov_flat.shape[0] != n_cov:
            grad_cov_flat = grad_cov_flat[:n_cov].contiguous()

        if grad_transmittance.shape[0] != n_trans:
            grad_transmittance = grad_transmittance[:n_trans].contiguous()

        if grad_sh.shape[0] != n_sh or grad_sh.shape[1] != 16:
            grad_sh = grad_sh[:, :16].contiguous()
            if grad_sh.shape[0] != n_sh:
                grad_sh = grad_sh[:n_sh].contiguous()

        grad_cov = cov_6_to_3x3(grad_cov_flat)

        if grad_transmittance.numel() == transmittance.numel():
            grad_transmittance = grad_transmittance.reshape_as(transmittance)
        else:
            grad_transmittance = grad_transmittance.reshape(-1)

        grad_sh = grad_sh[:, :n_sh_active]

        return (
            grad_means,
            grad_cov,
            grad_transmittance,
            grad_sh,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )


class SARRasterizer(nn.Module):
    """SAR渲染器模块

    封装SARRasterizerFunction，提供完整的nn.Module接口
    """

    def __init__(
        self,
        radar_x: float = 0.0,
        radar_y: float = 0.0,
        radar_z: float = 10000.0,
        track_angle: float = 0.0,
        incidence_angle: float = 30.0,
        azimuth_angle: float = 0.0,
        range_resolution: float = 0.3,
        azimuth_resolution: float = 0.3,
        range_samples: int = 128,
        azimuth_samples: int = 128
    ):
        super().__init__()

        self.radar_x = radar_x
        self.radar_y = radar_y
        self.radar_z = radar_z
        self.track_angle = track_angle
        self.incidence_angle = incidence_angle
        self.azimuth_angle = azimuth_angle
        self.range_resolution = range_resolution
        self.azimuth_resolution = azimuth_resolution
        self.range_samples = range_samples
        self.azimuth_samples = azimuth_samples
        self.last_omega: Optional[torch.Tensor] = None

    def forward(
        self,
        means_world: torch.Tensor,
        cov_world: torch.Tensor,
        transmittance: torch.Tensor,
        sh_coeffs: torch.Tensor
    ) -> torch.Tensor:
        result = SARRasterizerFunction.apply(
            means_world,
            cov_world,
            transmittance,
            sh_coeffs,
            self.radar_x,
            self.radar_y,
            self.radar_z,
            self.track_angle,
            self.incidence_angle,
            self.azimuth_angle,
            self.range_resolution,
            self.azimuth_resolution,
            self.range_samples,
            self.azimuth_samples
        )

        if isinstance(result, (list, tuple)):
            self.last_omega = result[1].detach()
            return result[0]
        return result

    def get_last_omega(self) -> Optional[torch.Tensor]:
        return self.last_omega
