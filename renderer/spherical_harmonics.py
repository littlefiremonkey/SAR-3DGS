"""
球谐函数展开模块 - SAR散射强度计算

适用于单极化SAR幅度图像，使用3阶球谐展开建模各向异性散射。

数学原理:
    I(θ, φ) = Σₗ₌₀³ Σₘ₌₋ₗˡ cₗₘ · Yₗₘ(θ, φ)

其中:
    - θ: 极角 (入射方向与目标法线的夹角)
    - φ: 方位角
    - cₗₘ: 可学习的球谐系数
    - Yₗₘ: 球谐基函数

特点:
    - 3阶展开，共16个系数
    - 简化模型: 不需要法线信息
    - 仅依赖雷达视线方向
    - 全可微分设计，支持端到端训练
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class SphericalHarmonics:
    """
    球谐函数计算器

    用于计算3阶球谐基函数值
    """

    SH_C0 = 0.28209479177387814
    SH_C1 = 0.4886025119029199
    SH_C2 = 1.0925484305920792
    SH_C20 = 0.31539156525252005
    SH_C22 = 0.5462742152960396
    SH_C3 = 0.5900435899266435
    SH_C30 = 0.4570459564643759
    SH_C31 = 2.890611442640554
    SH_C32 = 0.3731763325901154
    SH_C33 = 1.4453057113204849

    @staticmethod
    def compute_sh_basis(theta: torch.Tensor, phi: torch.Tensor, max_degree: int = 3) -> torch.Tensor:
        """
        计算球谐基函数值

        Args:
            theta: 极角 [N] 或 [N, ...], 范围 [0, π]
            phi: 方位角 [N] 或 [N, ...], 范围 [0, 2π]
            max_degree: 最大阶数，默认3阶

        Returns:
            SH值 [N, K], K = (max_degree+1)²
        """
        device = theta.device
        dtype = theta.dtype

        original_shape = theta.shape
        theta_flat = theta.flatten()
        phi_flat = phi.flatten()

        num_gaussians = theta_flat.shape[0]
        num_coeffs = (max_degree + 1) ** 2
        sh = torch.zeros((num_gaussians, num_coeffs), device=device, dtype=dtype)

        cos_theta = torch.cos(theta_flat)
        sin_theta = torch.sin(theta_flat)
        cos_phi = torch.cos(phi_flat)
        sin_phi = torch.sin(phi_flat)

        sh[:, 0] = SphericalHarmonics.SH_C0

        idx = 1

        if max_degree >= 1:
            sh[:, idx] = SphericalHarmonics.SH_C1 * sin_theta * sin_phi
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C1 * cos_theta
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C1 * sin_theta * cos_phi
            idx += 1

        if max_degree >= 2:
            cos_2theta = torch.cos(2 * theta_flat)
            sin_2theta = torch.sin(2 * theta_flat)
            sin_2phi = torch.sin(2 * phi_flat)
            cos_2phi = torch.cos(2 * phi_flat)

            sh[:, idx] = SphericalHarmonics.SH_C2 * sin_theta * sin_theta * sin_2phi
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C2 * sin_theta * cos_theta * sin_phi
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C20 * (3.0 * cos_theta * cos_theta - 1.0)
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C2 * sin_theta * cos_theta * cos_phi
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C2 * sin_theta * sin_theta * cos_2phi
            idx += 1

        if max_degree >= 3:
            sin_3phi = torch.sin(3 * phi_flat)
            cos_3phi = torch.cos(3 * phi_flat)

            sh[:, idx] = SphericalHarmonics.SH_C33 * sin_theta * sin_theta * sin_theta * sin_3phi
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C32 * sin_theta * sin_theta * cos_theta * sin_phi * cos_phi
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C30 * sin_theta * (5.0 * cos_theta * cos_theta - 1.0) * sin_phi
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C3 * (5.0 * cos_theta * cos_theta * cos_theta - 3.0 * cos_theta) / 2.0
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C30 * sin_theta * (5.0 * cos_theta * cos_theta - 1.0) * cos_phi
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C32 * sin_theta * sin_theta * cos_theta * cos_2phi
            idx += 1
            sh[:, idx] = SphericalHarmonics.SH_C33 * sin_theta * sin_theta * sin_theta * cos_3phi

        sh = sh.view(*original_shape, num_coeffs)

        return sh

    @staticmethod
    def get_sh_index(l: int, m: int) -> int:
        """获取球谐系数的索引"""
        return l * (l + 1) + m


class SHRenderer(nn.Module):
    """
    球谐渲染器 - 计算高斯散射强度

    用于SAR成像中，将3D高斯投影到2D图像平面后的散射强度计算。

    与3DGS的关系:
        - 3DGS: 使用SH计算RGB颜色，依赖视角方向
        - SAR-GS: 使用SH计算散射强度，依赖雷达视线方向

    散射模型 (简化模型):
        I = Σₗₘ cₗₘ · Yₗₘ(θ, φ)
    """

    def __init__(self, sh_degree: int = 3):
        """
        初始化球谐渲染器

        Args:
            sh_degree: 球谐展开阶数，默认3阶
        """
        super().__init__()
        self.sh_degree = sh_degree
        self.num_coeffs = (sh_degree + 1) ** 2

    def forward(
        self,
        view_dirs: torch.Tensor,
        sh_coeffs: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算散射强度

        Args:
            view_dirs: 雷达视线方向 [N, 3] 或 [B, N, 3]
                      归一化方向向量，从目标指向雷达
            sh_coeffs: 球谐系数 [N, K] 或 [B, N, K]
                      K = (sh_degree+1)²

        Returns:
            散射强度 [N] 或 [B, N]
        """
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-8)

        theta, phi = self._cartesian_to_spherical(view_dirs)

        sh_basis = SphericalHarmonics.compute_sh_basis(theta, phi, self.sh_degree)

        intensity = torch.sum(sh_coeffs * sh_basis, dim=-1)

        return intensity

    def _cartesian_to_spherical(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        笛卡尔坐标转球坐标

        Args:
            xyz: 笛卡尔坐标 [N, 3]

        Returns:
            theta: 极角 [N], 范围 [0, π]
            phi: 方位角 [N], 范围 [0, 2π]
        """
        x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

        r = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)

        cos_theta = torch.clamp(z / r, -1.0, 1.0)
        theta = torch.acos(cos_theta)

        phi = torch.atan2(y, x)
        phi = torch.where(phi < 0, phi + 2 * np.pi, phi)

        return theta, phi

    def get_dc_component(self, sh_coeffs: torch.Tensor) -> torch.Tensor:
        """
        获取直流分量 (l=0)

        直流分量代表平均散射强度，与视角无关。
        """
        dc = sh_coeffs[..., 0] * SphericalHarmonics.SH_C0
        return dc


def create_sh_coeffs(
    num_gaussians: int,
    sh_degree: int = 3,
    init_method: str = 'random',
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """
    创建初始化球谐系数

    Args:
        num_gaussians: 高斯数量
        sh_degree: 球谐阶数
        init_method: 初始化方法 ['random', 'dc_only', 'zeros']
        device: 设备

    Returns:
        球谐系数 [num_gaussians, (sh_degree+1)²]
    """
    num_coeffs = (sh_degree + 1) ** 2

    if init_method == 'random':
        coeffs = torch.randn(num_gaussians, num_coeffs, device=device) * 0.01
    elif init_method == 'dc_only':
        coeffs = torch.zeros(num_gaussians, num_coeffs, device=device)
        coeffs[:, 0] = 1.0
    elif init_method == 'zeros':
        coeffs = torch.zeros(num_gaussians, num_coeffs, device=device)
    else:
        raise ValueError(f"Unknown init_method: {init_method}")

    return coeffs


def test_spherical_harmonics():
    """测试球谐函数计算"""
    print("=" * 60)
    print("Testing Spherical Harmonics Module")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    num_points = 1000

    print("Test 1: Fixed view direction (top-down)")
    view_dirs = torch.tensor([[0, 0, -1]], device=device).repeat(num_points, 1)
    print(f"  View direction: {view_dirs[0].cpu().numpy()}")

    sh_coeffs = create_sh_coeffs(num_points, sh_degree=3, init_method='dc_only', device=device)
    print(f"  SH coeffs shape: {sh_coeffs.shape}")
    print(f"  DC value: {sh_coeffs[0, 0].item():.4f}")

    renderer = SHRenderer(sh_degree=3).to(device)

    with torch.no_grad():
        intensity = renderer(view_dirs, sh_coeffs)

    print(f"  Computed intensity: {intensity[0].item():.4f}")
    print(f"  Expected (DC only): {SphericalHarmonics.SH_C0 * sh_coeffs[0, 0].item():.4f}")
    print()

    print("Test 2: Varying azimuth angle")
    angles = torch.linspace(0, 2 * np.pi, num_points)
    view_dirs = torch.stack([
        torch.sin(angles),
        torch.cos(angles),
        -torch.ones_like(angles) * 0.707,
    ], dim=-1).to(device)

    sh_coeffs = torch.zeros(num_points, 16, device=device)
    sh_coeffs[:, 0] = 1.0
    sh_coeffs[:, 3] = 0.5

    intensity = renderer(view_dirs, sh_coeffs)
    print(f"  Intensity range: [{intensity.min().item():.4f}, {intensity.max().item():.4f}]")
    print()

    print("Test 3: Gradient check")
    view_dirs.requires_grad_(True)
    sh_coeffs_grad = sh_coeffs.clone().requires_grad_(True)

    intensity = renderer(view_dirs, sh_coeffs_grad)
    loss = intensity.sum()
    loss.backward()

    has_view_grad = view_dirs.grad is not None and torch.any(view_dirs.grad != 0)
    has_coeff_grad = sh_coeffs_grad.grad is not None and torch.any(sh_coeffs_grad.grad != 0)

    print(f"  View direction gradient: {'OK' if has_view_grad else 'FAIL'}")
    print(f"  SH coefficients gradient: {'OK' if has_coeff_grad else 'FAIL'}")
    print()

    print("Test 4: SH basis computation")
    theta = torch.tensor([np.pi / 4])
    phi = torch.tensor([np.pi / 4])
    sh_basis = SphericalHarmonics.compute_sh_basis(theta, phi, max_degree=3)
    print(f"  SH basis values (theta=pi/4, phi=pi/4):")
    print(f"  Shape: {sh_basis.shape}")
    print(f"  DC (l=0): {sh_basis[0, 0].item():.6f}")
    print(f"  Y10 (l=1,m=0): {sh_basis[0, 1].item():.6f}")
    print(f"  Y11 (l=1,m=1): {sh_basis[0, 2].item():.6f}")

    print()
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_spherical_harmonics()
