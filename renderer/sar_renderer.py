"""
===================================================================================================
SAR Gaussian Splatting - 统一前向渲染管线
===================================================================================================

【设计依据】: 前向渲染原理（新设计）.md

【Alpha混合渲染公式】:
I(na, nr) = Σ_i [ Ti × αi × Si ]

其中:
- Si(β, α): 球谐散射强度，由等效入射角β和航迹角α决定
- αi = σi × γi(na, nr): Alpha值（透射率 × IPP投影密度）
- Ti = Π_{j: Rj < Ri} (1 - αj): 累积透射率（按斜距从近到远排序）

【物理意义】:
- Ti = 1: 无前方遮挡，能量完全到达
- Ti = 0: 被前方高斯完全遮挡
- 0 < Ti < 1: 部分被前方高斯遮挡

【与光学3DGS统一性】:
- 光学3DGS: C = Σ_i Ti × αi × ci
- SAR-GS:     I = Σ_i Ti × αi × Si
- 两者数学形式完全一致

【管线流程】:
Stage 1: 坐标变换 (世界 → 雷达)
Stage 2: IPP投影计算 (3D雷达坐标 → 2D成像平面)
Stage 3: Alpha计算 (α = γ × σ)
Stage 4: 按Rmin升序排序 (近处高斯先处理)
Stage 5: Alpha混合 (从近到远累积透射率)
Stage 6: 球谐散射计算
Stage 7: 统一渲染合成

===================================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from .spherical_harmonics import SphericalHarmonics


@dataclass
class SARRenderParams:
    """
    SAR渲染参数

    【参数说明】:
    - incidence_angle θ: 入射角(度)
    - azimuth_angle φ: 斜视角方位角(度)
    - radar_altitude H: 雷达高度(米)

    【有效入射角计算】:
    sinβ = sinθ × cosφ
    Rc = H / cosβ
    """
    incidence_angle: float = 30.0
    azimuth_angle: float = 0.0
    radar_altitude: float = 5000.0

    azimuth_resolution: float = 1.0
    range_resolution: float = 1.0
    azimuth_samples: int = 512
    range_samples: int = 512

    @property
    def track_angle(self) -> float:
        """航迹角α (度)"""
        return 0.0

    @property
    def incidence_angle_rad(self) -> float:
        """入射角弧度值"""
        return np.deg2rad(self.incidence_angle)

    @property
    def azimuth_angle_rad(self) -> float:
        """斜视角弧度值"""
        return np.deg2rad(self.azimuth_angle)

    @property
    def beta(self) -> float:
        """
        有效入射角β: sinβ = sinθ × cosφ
        """
        sin_theta = np.sin(self.incidence_angle_rad)
        cos_phi = np.cos(self.azimuth_angle_rad)
        sin_beta = sin_theta * cos_phi
        sin_beta = np.clip(sin_beta, -1.0, 1.0)
        return np.arcsin(sin_beta)

    @property
    def beta_rad(self) -> float:
        """有效入射角β的弧度值"""
        return self.beta

    @property
    def beta_deg(self) -> float:
        """有效入射角β的度数值"""
        return np.rad2deg(self.beta)

    @property
    def Rc(self) -> float:
        """
        参考斜距 Rc = H / cosβ
        """
        return self.radar_altitude / np.cos(self.beta)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        图像尺寸 (H, W) = (距离向像素数, 方位向像素数)
        """
        return (self.range_samples, self.azimuth_samples)

    def get_radar_position(self) -> torch.Tensor:
        """
        获取雷达位置 (当track_angle=0时)

        P_radar,w = (H·tanβ·sinα, -H·tanβ·cosα, H)

        Returns:
            [3] 雷达在世界坐标系中的位置
        """
        H = self.radar_altitude
        tan_beta = np.tan(self.beta)
        alpha_rad = 0.0

        x = H * tan_beta * np.sin(alpha_rad)
        y = -H * tan_beta * np.cos(alpha_rad)
        z = H

        return torch.tensor([x, y, z], dtype=torch.float32)

    def compute_radar_position(self, track_angle: float) -> torch.Tensor:
        """
        计算指定航迹角的雷达位置

        Args:
            track_angle: 航迹角α (度)

        Returns:
            [3] 雷达在世界坐标系中的位置
        """
        H = self.radar_altitude
        tan_beta = np.tan(self.beta)
        alpha_rad = np.deg2rad(track_angle)

        x = H * tan_beta * np.sin(alpha_rad)
        y = -H * tan_beta * np.cos(alpha_rad)
        z = H

        return torch.tensor([x, y, z], dtype=torch.float32)


class CoordinateTransformer:
    """
    坐标变换模块
    实现: 世界坐标系W → 雷达坐标系R

    【旋转矩阵 R_w→r】:
    R_w→r = | cosα       sinα      0     |
            | cosβ·sinα  -cosβ·cosα  -sinβ |
            | -sinβ·sinα  sinβ·cosα  -cosβ |

    【变换公式】:
    P_r = R_w→r · (P_w - P_radar,w)
    Σ_r = R_w→r · Σ_w · R_w→r^T
    """

    def __init__(self, beta_rad: float, alpha_rad: float = 0.0):
        """
        Args:
            beta_rad: 有效入射角β (弧度)
            alpha_rad: 航迹角α (弧度)
        """
        self.beta = beta_rad
        self.alpha = alpha_rad
        self._R_w2r_cache = None
        self._cache_alpha = None

        self._R_w2r = self._compute_rotation_matrix()

    @property
    def R_w2r(self) -> torch.Tensor:
        """缓存的旋转矩阵"""
        if self._R_w2r_cache is None or self._cache_alpha != self.alpha:
            self._R_w2r_cache = self._compute_rotation_matrix()
            self._cache_alpha = self.alpha
        return self._R_w2r_cache

    def _compute_rotation_matrix(self) -> torch.Tensor:
        """
        计算旋转矩阵 R_w→r

        Returns:
            R_w2r: [3, 3] 旋转矩阵
        """
        cos_a = float(np.cos(self.alpha))
        sin_a = float(np.sin(self.alpha))
        cos_b = float(np.cos(self.beta))
        sin_b = float(np.sin(self.beta))

        R = torch.tensor([
            [cos_a,                   sin_a,                  0.0],
            [cos_b * sin_a,           -cos_b * cos_a,         -sin_b],
            [-sin_b * sin_a,          sin_b * cos_a,          -cos_b]
        ], dtype=torch.float32)

        return R

    def world_to_radar(
        self,
        points_world: torch.Tensor,
        radar_position: torch.Tensor
    ) -> torch.Tensor:
        """
        世界坐标 → 雷达坐标

        P_r = R_w→r · (P_w - P_radar,w)

        Args:
            points_world: [N, 3] 世界坐标系中的点
            radar_position: [3] 雷达在世界坐标系中的位置

        Returns:
            points_radar: [N, 3] 雷达坐标系中的点 (Xr, Yr, Zr)
        """
        if points_world.dim() == 1:
            points_world = points_world.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        device = points_world.device
        R_w2r = self.R_w2r.to(device)

        relative_pos = points_world - radar_position.unsqueeze(0).to(device)
        points_radar = torch.matmul(relative_pos, R_w2r.T)

        if squeeze_output:
            points_radar = points_radar.squeeze(0)

        return points_radar

    def transform_covariance(
        self,
        cov_world: torch.Tensor
    ) -> torch.Tensor:
        """
        协方差矩阵变换: 世界坐标 → 雷达坐标

        Σ_r = R_w→r · Σ_w · R_w→r^T

        Args:
            cov_world: [N, 3, 3] 或 [3, 3] 世界坐标系协方差

        Returns:
            cov_radar: [N, 3, 3] 或 [3, 3] 雷达坐标系协方差
        """
        if cov_world.dim() == 2:
            cov_world = cov_world.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        device = cov_world.device
        R = self.R_w2r.unsqueeze(0).to(device)
        R_T = R.transpose(-1, -2)

        temp = torch.matmul(R, cov_world)
        cov_radar = torch.matmul(temp, R_T)

        if squeeze_output:
            cov_radar = cov_radar.squeeze(0)

        return cov_radar


class UnifiedProjector:
    """
    统一投影模块
    处理 IPP投影 (斜距投影)

    【IPP投影 (斜距投影)】:
    Rmin = √(Yr² + Zr² + ε²), 其中 ε = 1e-6
    r = Rmin/ρr + Nr/2 - Rc/ρr
    c = Xr/ρa + Na/2

    雅可比矩阵 J_ipp:
    J_ipp = | 0    Yr/(ρr·Rmin)  Zr/(ρr·Rmin) |
            | 1/ρa       0                0        |

    【排序说明】:
    - 按Rmin升序排序（Rmin小的=近处高斯先混合）
    - 与光学3DGS的深度排序一致
    """

    def __init__(
        self,
        reference_range: float,
        azimuth_resolution: float = 1.0,
        range_resolution: float = 1.0,
        range_samples: int = 512,
        azimuth_samples: int = 512,
        epsilon: float = 1e-6
    ):
        """
        Args:
            reference_range: 参考斜距Rc
            azimuth_resolution: 方位向分辨率ρa (米/像素)
            range_resolution: 距离向分辨率ρr (米/像素)
            range_samples: 距离向采样点数Nr
            azimuth_samples: 方位向采样点数Na
            epsilon: 数值稳定常数 (防止Rmin=0)
        """
        self.Rc = reference_range
        self.rho_a = azimuth_resolution
        self.rho_r = range_resolution
        self.Nr = range_samples
        self.Na = azimuth_samples
        self.eps = epsilon

    def project_to_ipp(
        self,
        points_radar: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        3D雷达坐标 → 2D IPP坐标

        Args:
            points_radar: [N, 3] 雷达坐标系中的3D坐标 (Xr, Yr, Zr)

        Returns:
            ipp_coords: [N, 2] IPP坐标 (r, c)
            Rmin: [N] 最短斜距 √(Yr² + Zr²)
            jacobian_ipp: [N, 2, 3] IPP投影雅可比矩阵
        """
        Xr = points_radar[:, 0]
        Yr = points_radar[:, 1]
        Zr = points_radar[:, 2]

        Rmin = torch.sqrt(Yr**2 + Zr**2 + self.eps**2)

        r = Rmin / self.rho_r + self.Nr / 2.0 - self.Rc / self.rho_r
        c = Xr / self.rho_a + self.Na / 2.0

        ipp_coords = torch.stack([r, c], dim=-1)

        jacobian_ipp = self._compute_ipp_jacobian(Yr, Zr, Rmin)

        return ipp_coords, Rmin, jacobian_ipp

    def _compute_ipp_jacobian(
        self,
        Yr: torch.Tensor,
        Zr: torch.Tensor,
        Rmin: torch.Tensor
    ) -> torch.Tensor:
        """
        计算IPP投影的雅可比矩阵

        J_ipp = | 0    Yr/(ρr·Rmin)  Zr/(ρr·Rmin) |
                | 1/ρa       0                0        |
        其中 Rmin = √(Yr² + Zr² + ε²), ε = 1e-6

        Args:
            Yr: [N] 雷达Y坐标
            Zr: [N] 雷达Z坐标
            Rmin: [N] 最短斜距 (已添加ε²)

        Returns:
            J_ipp: [N, 2, 3]
        """
        N = Yr.shape[0]
        device = Yr.device

        J = torch.zeros(N, 2, 3, device=device, dtype=Yr.dtype)

        J[:, 0, 1] = Yr / (self.rho_r * Rmin)
        J[:, 0, 2] = Zr / (self.rho_r * Rmin)

        J[:, 1, 0] = 1.0 / self.rho_a

        return J

    def project_covariance_ipp(
        self,
        cov_radar: torch.Tensor,
        jacobian_ipp: torch.Tensor
    ) -> torch.Tensor:
        """
        投影协方差到IPP: Σ_ipp = J_ipp · Σ_r · J_ipp^T

        Args:
            cov_radar: [N, 3, 3] 雷达坐标系协方差
            jacobian_ipp: [N, 2, 3] IPP投影雅可比

        Returns:
            cov_ipp: [N, 2, 2] IPP 2D协方差
        """
        J_T = jacobian_ipp.transpose(-1, -2)
        temp = torch.matmul(jacobian_ipp, cov_radar)
        cov_ipp = torch.matmul(temp, J_T)
        return cov_ipp


class GaussianDensityCalculator:
    """
    2D高斯密度值计算器

    【定义】:
    对于3D高斯基元 Gi，其2D投影密度值定义为归一化的高斯概率密度:

    γ(na, nr) = (1 / (2π × det(Σ'))) × exp( -½ × (x - μ')ᵀ × Σ'⁻¹ × (x - μ') )

    其中:
    - μ': 高斯中心在投影平面上的2D坐标
    - Σ': 3D协方差矩阵经投影变换后的2D协方差矩阵
    - x: 投影平面上的像素坐标
    """

    @staticmethod
    def compute_2d_gaussian_density(
        pixel_coords: torch.Tensor,
        gaussian_mean: torch.Tensor,
        gaussian_cov: torch.Tensor
    ) -> torch.Tensor:
        """
        计算2D高斯在指定像素处的密度值

        Args:
            pixel_coords: [H, W, 2] 像素坐标网格 (r, c) 或 (x, y)
            gaussian_mean: [2] 高斯均值
            gaussian_cov: [2, 2] 高斯协方差矩阵

        Returns:
            density: [H, W] 密度值
        """
        H, W = pixel_coords.shape[:2]
        device = pixel_coords.device

        diff = pixel_coords - gaussian_mean.unsqueeze(0).unsqueeze(0)

        cov_inv = torch.inverse(gaussian_cov)

        mahal = torch.einsum('hwi,ij,hwj->hw', diff, cov_inv, diff)

        cov_det = torch.det(gaussian_cov)

        norm_factor = 1.0 / (2.0 * np.pi * cov_det)

        density = norm_factor * torch.exp(-0.5 * mahal)

        return density

    @staticmethod
    def compute_gaussian_influence_range(
        gaussian_cov: torch.Tensor,
        num_std: float = 3.0
    ) -> Tuple[float, float]:
        """
        计算高斯的影响范围 (基于协方差矩阵的特征值)

        Args:
            gaussian_cov: [2, 2] 高斯协方差矩阵
            num_std: 倍数标准差 (默认3σ覆盖99.7%)

        Returns:
            range_x, range_y: X和Y方向的半径
        """
        eigenvalues = torch.linalg.eigvalsh(gaussian_cov)
        range_x = num_std * torch.sqrt(eigenvalues[0]).item()
        range_y = num_std * torch.sqrt(eigenvalues[1]).item()
        return range_x, range_y


class UnifiedSARRenderer(nn.Module):
    """
    统一SAR渲染器 (Alpha混合版本)

    【核心公式】:
    I(na, nr) = Σ_i [ Ti × αi × Si ]
    其中:
    - αi = σi × γi(na, nr): Alpha值（透射率 × IPP投影密度）
    - Ti = Π_{j: Rj < Ri} (1 - αj): 累积透射率（按Rmin从近到远）
    - Si(β, α): 球谐散射强度

    【与光学3DGS统一性】:
    - 光学3DGS: C = Σ_i Ti × αi × ci
    - SAR-GS:     I = Σ_i Ti × αi × Si

    【管线流程】:
    1. 坐标变换: 世界坐标 → 雷达坐标
    2. IPP投影: 3D雷达坐标 → 2D IPP坐标 + 协方差
    3. Alpha计算: α = σ × γ (透射率 × IPP密度)
    4. 按Rmin升序排序: Rmin小的(近处)先处理
    5. Alpha混合: 从近到远累积Ti = ∏(1-αj)
    6. 球谐散射: 计算Si(β, α)
    7. 统一合成: I = Σ Ti × αi × Si

    【设计考量】:
    - 完全可微: 支持端到端训练
    - CUDA友好: 模块化设计便于GPU并行化
    - 数值稳定: 添加ε防止Rmin=0
    - 与光学3DGS数学形式完全一致
    """

    def __init__(self, params: SARRenderParams):
        """
        Args:
            params: SAR渲染参数
        """
        super().__init__()
        self.params = params

        self.coord_transform = CoordinateTransformer(
            beta_rad=params.beta_rad,
            alpha_rad=np.deg2rad(params.track_angle)
        )

        self.projector = UnifiedProjector(
            reference_range=params.Rc,
            azimuth_resolution=params.azimuth_resolution,
            range_resolution=params.range_resolution,
            range_samples=params.range_samples,
            azimuth_samples=params.azimuth_samples
        )

        self.H, self.W = params.image_size

        from .spherical_harmonics import SHRenderer
        self.sh_renderer = SHRenderer(sh_degree=3)

        self._cached_beta = None
        self._cached_alpha = None
        self._cached_sh_basis = None
        self._cached_active_degree = None

    def _update_sh_basis_cache(self, beta_rad: float, alpha_rad: float, active_degree: int):
        """预计算并缓存SH基函数

        当视角参数(beta, alpha)或SH阶数(active_degree)变化时，重新计算并缓存SH基函数。

        Args:
            beta_rad: 有效入射角β (弧度)
            alpha_rad: 航迹角α (弧度)
            active_degree: 当前激活的SH阶数
        """
        view_changed = (self._cached_beta != beta_rad or self._cached_alpha != alpha_rad)
        degree_changed = (self._cached_active_degree != active_degree)

        if view_changed:
            view_dir = torch.tensor([
                np.sin(beta_rad) * np.cos(alpha_rad),
                np.sin(beta_rad) * np.sin(alpha_rad),
                -np.cos(beta_rad)
            ], dtype=torch.float32)

            theta, phi = self.sh_renderer._cartesian_to_spherical(view_dir.unsqueeze(0))
            self._cached_sh_basis = SphericalHarmonics.compute_sh_basis(
                theta, phi, max_degree=3
            ).squeeze(0)

            self._cached_beta = beta_rad
            self._cached_alpha = alpha_rad
            self._cached_active_degree = active_degree
        elif degree_changed:
            self._cached_active_degree = active_degree

    def forward(
        self,
        gaussian_means: torch.Tensor,
        gaussian_cov: torch.Tensor,
        sh_coeffs: torch.Tensor,
        transmittance: torch.Tensor,
        radar_position: Optional[torch.Tensor] = None,
        track_angle: float = 0.0,
        active_sh_degree: int = 3
    ) -> Tuple[torch.Tensor, Dict]:
        """
        统一前向渲染

        Args:
            gaussian_means: [N, 3] 高斯均值 (世界坐标)
            gaussian_cov: [N, 3, 3] 高斯协方差 (世界坐标)
            sh_coeffs: [N, K] 球谐系数 (散射强度)
            transmittance: [N] 透射率 σ (等价于opacity, 0~1)
            radar_position: [3] 雷达位置
            track_angle: 航迹角α (度)
            active_sh_degree: 当前激活的SH阶数

        Returns:
            rendered_image: [H, W] 渲染图像
            aux_data: 辅助数据字典
        """
        if radar_position is None:
            radar_position = self.params.get_radar_position()

        if track_angle != 0.0:
            alpha_rad = np.deg2rad(track_angle)
            beta_rad = self.params.beta_rad
            self.coord_transform = CoordinateTransformer(
                beta_rad=beta_rad,
                alpha_rad=alpha_rad
            )
            radar_position = self.params.compute_radar_position(track_angle)

        radar_position = radar_position.to(gaussian_means.device)

        points_radar = self.coord_transform.world_to_radar(gaussian_means, radar_position)
        cov_radar = self.coord_transform.transform_covariance(gaussian_cov)

        ipp_coords, Rmin, jacobian_ipp = self.projector.project_to_ipp(points_radar)
        cov_ipp = self.projector.project_covariance_ipp(cov_radar, jacobian_ipp)

        beta_rad = self.params.beta_rad
        alpha_rad = np.deg2rad(track_angle) if track_angle != 0.0 else 0.0

        scattering = self._compute_scattering_cached(
            beta_rad, alpha_rad, sh_coeffs, active_sh_degree
        )

        rendered_image = self._alpha_blend_render(
            ipp_coords,
            cov_ipp,
            Rmin,
            scattering,
            transmittance
        )

        aux_data = {
            'ipp_coords': ipp_coords,
            'Rmin': Rmin,
            'points_radar': points_radar,
            'scattering': scattering,
            'transmittance': transmittance
        }

        return rendered_image, aux_data

    def _compute_scattering_cached(
        self,
        beta_rad: float,
        alpha_rad: float,
        sh_coeffs: torch.Tensor,
        active_degree: int
    ) -> torch.Tensor:
        """使用缓存的SH基函数计算散射强度

        Args:
            beta_rad: 有效入射角β (弧度)
            alpha_rad: 航迹角α (弧度)
            sh_coeffs: [N, K] 球谐系数
            active_degree: 当前激活的SH阶数

        Returns:
            scattering: [N] 散射强度
        """
        self._update_sh_basis_cache(beta_rad, alpha_rad, active_degree)

        active_K = (active_degree + 1) ** 2
        sh_basis_active = self._cached_sh_basis[:active_K]

        N = sh_coeffs.shape[0]
        sh_basis = sh_basis_active.unsqueeze(0).expand(N, -1)

        scattering = torch.sum(sh_coeffs * sh_basis, dim=-1)
        return scattering

    def _alpha_blend_render(
        self,
        ipp_coords: torch.Tensor,
        cov_ipp: torch.Tensor,
        Rmin: torch.Tensor,
        scattering: torch.Tensor,
        transmittance: torch.Tensor
    ) -> torch.Tensor:
        """
        Alpha混合渲染: I = Σ_i Ti × αi × Si

        【算法步骤】:
        1. 有效高斯掩码 (IPP坐标在图像范围内)
        2. 按Rmin升序排序 (小=近处先混合, 与光学3DGS一致)
        3. 对每个高斯:
           a. 计算αi = σi × γi (Alpha值)
           b. 累积透射率 Ti = ∏_{j: Rj < Ri} (1 - αj)
           c. 累加贡献: Ti × αi × Si

        【与光学3DGS统一性】:
        - 光学3DGS: C = Σ Ti × αi × ci
        - SAR-GS:     I = Σ Ti × αi × Si

        Args:
            ipp_coords: [N, 2] IPP坐标 (r, c)  r=距离向(行), c=方位向(列)
            cov_ipp: [N, 2, 2] IPP 2D协方差
            Rmin: [N] 最短斜距 √(Yr² + Zr² + ε²)
            scattering: [N] 散射强度 Si
            transmittance: [N] 透射率 σ

        Returns:
            image: [H, W] 渲染图像
        """
        H, W = self.H, self.W
        device = ipp_coords.device
        N = ipp_coords.shape[0]

        valid_mask = self._compute_valid_mask(ipp_coords)
        valid_indices = torch.where(valid_mask)[0]
        num_valid = len(valid_indices)

        if num_valid == 0:
            return torch.zeros(H, W, device=device)

        ipp_valid = ipp_coords[valid_mask]
        cov_ipp_valid = cov_ipp[valid_mask]
        Rmin_valid = Rmin[valid_mask]
        scatter_valid = scattering[valid_mask]
        transmit_valid = transmittance[valid_mask]

        sorted_indices = torch.argsort(Rmin_valid, descending=False)
        sorted_indices = sorted_indices.tolist()

        r_coords = ipp_valid[:, 0]
        c_coords = ipp_valid[:, 1]

        alpha_map = torch.zeros(num_valid, H, W, device=device, dtype=torch.float32)

        for idx, orig_i in enumerate(sorted_indices):
            cov_ipp_i = cov_ipp_valid[orig_i]
            range_r, range_c = GaussianDensityCalculator.compute_gaussian_influence_range(cov_ipp_i, num_std=3.0)

            r_center = r_coords[orig_i].item()
            c_center = c_coords[orig_i].item()

            r_min = max(0, int(r_center - range_r) - 1)
            r_max = min(W, int(r_center + range_r) + 2)
            c_min = max(0, int(c_center - range_c) - 1)
            c_max = min(H, int(c_center + range_c) + 2)

            if r_min < r_max and c_min < c_max:
                r_size = r_max - r_min
                c_size = c_max - c_min

                r_values = torch.arange(r_min, r_max, device=device, dtype=torch.float32)
                c_values = torch.arange(c_min, c_max, device=device, dtype=torch.float32)

                r_grid = r_values.unsqueeze(1).expand(r_size, c_size)
                c_grid = c_values.unsqueeze(0).expand(r_size, c_size)

                diff_r = r_grid - ipp_valid[orig_i, 0]
                diff_c = c_grid - ipp_valid[orig_i, 1]

                diff = torch.stack([diff_r, diff_c], dim=-1)

                cov_inv = torch.inverse(cov_ipp_i)
                mahal = torch.sum(diff * torch.matmul(diff, cov_inv), dim=-1)
                cov_det = torch.det(cov_ipp_i)
                norm_factor = 1.0 / (2.0 * np.pi * cov_det)

                gamma = norm_factor * torch.exp(-0.5 * mahal)

                sigma = transmit_valid[orig_i]
                alpha = (sigma * gamma).clamp(0.0, 1.0)

                alpha_map[idx, c_min:c_max, r_min:r_max] = alpha.T

        T = torch.ones(H, W, device=device, dtype=torch.float32)
        image = torch.zeros(H, W, device=device, dtype=torch.float32)

        for idx, orig_i in enumerate(sorted_indices):
            alpha_i = alpha_map[idx]
            scatter_i = scatter_valid[orig_i]

            image = image + T * alpha_i * scatter_i
            T = T * (1.0 - alpha_i)

        return image

    def _compute_valid_mask(self, ipp_coords: torch.Tensor) -> torch.Tensor:
        """
        计算有效高斯掩码 (IPP坐标在图像范围内)

        Args:
            ipp_coords: [N, 2] IPP坐标 (r, c)

        Returns:
            valid_mask: [N] 布尔掩码
        """
        r = ipp_coords[:, 0]
        c = ipp_coords[:, 1]

        valid = (r >= 0) & (r < self.params.range_samples) & \
                (c >= 0) & (c < self.params.azimuth_samples)

        return valid


class SARRenderer(nn.Module):
    """
    SAR渲染器封装类 (兼容旧接口)

    提供与旧代码兼容的接口，同时使用新的统一渲染管线
    """

    def __init__(self, params: SARRenderParams):
        """
        Args:
            params: SAR渲染参数
        """
        super().__init__()
        self.params = params
        self.unified_renderer = UnifiedSARRenderer(params)

    def forward(
        self,
        gaussian_means: torch.Tensor,
        gaussian_cov: torch.Tensor,
        sh_coeffs: torch.Tensor,
        radar_position: Optional[torch.Tensor] = None,
        transmittance: Optional[torch.Tensor] = None,
        track_angle: float = 0.0,
        compute_shadow: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        SAR渲染

        Args:
            gaussian_means: [N, 3] 高斯均值
            gaussian_cov: [N, 3, 3] 高斯协方差
            sh_coeffs: [N, K] 球谐系数
            radar_position: [3] 雷达位置
            transmittance: [N] 透射率 (默认使用opacity)
            track_angle: 航迹角α (度)
            compute_shadow: 是否计算阴影 (本版本忽略，始终计算)

        Returns:
            rendered_image: [H, W] 渲染图像
            aux_data: 辅助数据
        """
        if transmittance is None:
            if hasattr(self, '_opacities'):
                transmittance = self._opacities
            else:
                transmittance = torch.ones(gaussian_means.shape[0], device=gaussian_means.device) * 0.8

        rendered_image, aux_data = self.unified_renderer(
            gaussian_means,
            gaussian_cov,
            sh_coeffs,
            transmittance,
            radar_position,
            track_angle
        )

        aux_data['intensity'] = rendered_image
        aux_data['shadow'] = torch.ones_like(rendered_image)

        return rendered_image, aux_data
