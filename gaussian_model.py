"""
3D高斯模型 - Gaussian Splatting for SAR

【核心思想】
使用一组3D高斯分布来表示场景,每个高斯包含:
- 位置 (means): [N, 3] 3D空间中的均值坐标 (xyz)
- 缩放 (scales): [N, 3] 对数缩放向量, 通过exp()激活确保正值
- 旋转 (rotations): [N, 4] 单位四元数 (w, x, y, z格式)
- 不透明度 (opacities): [N, 1] 不透明度, 通过sigmoid激活约束到[0,1]
- 球谐系数 (sh_coeffs): [N, K] 球谐展开系数

【协方差矩阵计算】
Σ = R @ S @ S^T @ R^T
其中:
    - S = diag(exp(scales))  缩放矩阵
    - R = quaternion_to_rotation(rotations)  旋转矩阵
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class Quaternion:
    """四元数工具类"""
    @staticmethod
    def normalize(q: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(q, dim=-1, keepdim=True)
        return q / (norm + 1e-8)

    @staticmethod
    def to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
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


class GaussianModel(nn.Module):
    """3D高斯模型"""

    def __init__(
        self,
        sh_degree: int = 3,
        init_num_gaussians: int = 5000
    ):
        super().__init__()
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self.init_num_gaussians = init_num_gaussians
        self._active_sh_degree = 0

        self._means = None
        self._scales = None
        self._rotations = None
        self._opacities = None
        self._sh_coeffs = None

        self._cov_cache = None
        self._cov_cache_size = None

    @property
    def active_sh_degree(self) -> int:
        return self._active_sh_degree

    @active_sh_degree.setter
    def active_sh_degree(self, value: int):
        self._active_sh_degree = min(value, self.sh_degree)

    @property
    def num_gaussians(self) -> int:
        return self._means.shape[0] if self._means is not None else 0

    @property
    def means(self) -> nn.Parameter:
        return self._means

    @property
    def scales(self) -> nn.Parameter:
        return self._scales

    @property
    def rotations(self) -> nn.Parameter:
        return self._rotations

    @property
    def opacities(self) -> nn.Parameter:
        return self._opacities

    @property
    def sh_coeffs(self) -> nn.Parameter:
        return self._sh_coeffs

    def _initialize_params(self, num_gaussians: int, device: torch.device):
        self._means = nn.Parameter(
            torch.zeros(num_gaussians, 3, device=device),
            requires_grad=True
        )
        self._scales = nn.Parameter(
            torch.zeros(num_gaussians, 3, device=device),
            requires_grad=True
        )
        self._rotations = nn.Parameter(
            torch.zeros(num_gaussians, 4, device=device),
            requires_grad=True
        )
        self._opacities = nn.Parameter(
            torch.zeros(num_gaussians, 1, device=device),
            requires_grad=True
        )
        self._sh_coeffs = nn.Parameter(
            torch.zeros(num_gaussians, self.num_sh_coeffs, device=device),
            requires_grad=True
        )

    def initialize_random(
        self,
        scene_bounds: Tuple[float, float, float, float, float, float],
        num_gaussians: Optional[int] = None,
        init_scale: float = 0.5,
        init_opacity: float = 0.5,
        device: Optional[torch.device] = None,
        add_ground_plane: bool = False,
        ground_scale: float = 0.05,
        ground_opacity: float = 0.3,
        ground_ratio: float = 0.3,
        gaussian_std: float = None
    ):
        """在指定场景边界内按高斯分布初始化高斯（越靠近原点越密集）

        Args:
            scene_bounds: (x_min, x_max, y_min, y_max, z_min, z_max)
            num_gaussians: 高斯数量
            init_scale: 初始缩放
            init_opacity: 初始不透明度
            device: 计算设备
            add_ground_plane: 是否在地面添加密集小高斯
            ground_scale: 地面高斯的初始缩放
            ground_opacity: 地面高斯的初始不透明度
            ground_ratio: 地面高斯占总高斯的比例
            gaussian_std: 高斯分布标准差，默认为场景边界最大范围的四分之一
        """
        if num_gaussians is None:
            num_gaussians = self.init_num_gaussians

        if device is None:
            device = self._means.device if self._means is not None else torch.device('cpu')

        x_min, x_max, y_min, y_max, z_min, z_max = scene_bounds

        x_range = x_max - x_min
        y_range = y_max - y_min
        L = max(x_range, y_range)

        if gaussian_std is None:
            gaussian_std = L / 4

        num_ground = int(num_gaussians * ground_ratio) if add_ground_plane else 0
        num_scene = num_gaussians - num_ground

        total_gaussians = num_gaussians
        self._initialize_params(total_gaussians, device)

        with torch.no_grad():
            if num_scene > 0:
                x_coords = np.random.normal(0, gaussian_std, num_scene)
                y_coords = np.random.normal(0, gaussian_std, num_scene)
                z_coords = np.random.normal(0, gaussian_std / 2, num_scene)

                x_coords = np.clip(x_coords, x_min, x_max)
                y_coords = np.clip(y_coords, y_min, y_max)
                z_coords = np.clip(z_coords, z_min, z_max)

                self._means[:num_scene, 0] = torch.from_numpy(x_coords).float().to(device)
                self._means[:num_scene, 1] = torch.from_numpy(y_coords).float().to(device)
                self._means[:num_scene, 2] = torch.from_numpy(z_coords).float().to(device)

                self._scales[:num_scene].copy_(torch.log(torch.ones(num_scene, 3, device=device) * init_scale))

                self._rotations[:num_scene].copy_(torch.zeros(num_scene, 4, device=device))
                self._rotations[:num_scene, 0] = 1.0

                self._opacities[:num_scene].copy_(torch.full((num_scene, 1), init_opacity, device=device))

                self._sh_coeffs[:num_scene].copy_(torch.zeros(num_scene, self.num_sh_coeffs, device=device))
                self._sh_coeffs[:num_scene, 0] = 0.3

            if num_ground > 0:
                ground_x_min = -2 * L
                ground_x_max = 2 * L
                ground_y_min = -2 * L
                ground_y_max = 2 * L

                x_coords = np.random.uniform(ground_x_min, ground_x_max, num_ground)
                y_coords = np.random.uniform(ground_y_min, ground_y_max, num_ground)
                z_coords = np.zeros(num_ground)

                self._means[num_scene:, 0] = torch.from_numpy(x_coords).float().to(device)
                self._means[num_scene:, 1] = torch.from_numpy(y_coords).float().to(device)
                self._means[num_scene:, 2] = torch.from_numpy(z_coords).float().to(device)

                self._scales[num_scene:].copy_(torch.log(torch.ones(num_ground, 3, device=device) * ground_scale))

                self._rotations[num_scene:].copy_(torch.zeros(num_ground, 4, device=device))
                self._rotations[num_scene:, 0] = 1.0

                self._opacities[num_scene:].copy_(torch.full((num_ground, 1), ground_opacity, device=device))

                self._sh_coeffs[num_scene:].copy_(torch.zeros(num_ground, self.num_sh_coeffs, device=device))
                self._sh_coeffs[num_scene:, 0] = 0.2

    def initialize_from_points(
        self,
        points: torch.Tensor,
        init_scale: float = 0.01,
        init_opacity: float = 0.5,
        device: Optional[torch.device] = None
    ):
        """从点云初始化

        Args:
            points: [N, 3] 点云坐标
            init_scale: 初始缩放
            init_opacity: 初始不透明度
            device: 计算设备
        """
        if device is None:
            device = points.device

        num_points = points.shape[0]
        self._initialize_params(num_points, device)

        with torch.no_grad():
            self._means.copy_(points)
            self._scales.copy_(torch.log(torch.ones(num_points, 3, device=device) * init_scale))
            self._rotations.copy_(torch.zeros(num_points, 4, device=device))
            self._rotations[:, 0] = 1.0
            self._opacities.copy_(torch.full((num_points, 1), init_opacity, device=device))
            self._sh_coeffs.copy_(torch.zeros(num_points, self.num_sh_coeffs, device=device))
            self._sh_coeffs[:, 0] = 0.1

    def compute_covariance(self) -> torch.Tensor:
        """计算协方差矩阵的上三角元素

        Returns:
            torch.Tensor: [N, 6] 协方差矩阵上三角元素
        """
        scales = torch.exp(self._scales)
        R = Quaternion.to_rotation_matrix(self._rotations)
        S = torch.zeros_like(R)
        for i in range(scales.shape[0]):
            S[i] = torch.diag(scales[i])

        Sigma = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)

        cov_6 = torch.zeros(self.num_gaussians, 6, device=Sigma.device, dtype=Sigma.dtype)
        cov_6[:, 0] = Sigma[:, 0, 0]
        cov_6[:, 1] = Sigma[:, 0, 1]
        cov_6[:, 2] = Sigma[:, 0, 2]
        cov_6[:, 3] = Sigma[:, 1, 1]
        cov_6[:, 4] = Sigma[:, 1, 2]
        cov_6[:, 5] = Sigma[:, 2, 2]

        return cov_6

    def compute_covariance_full(self) -> torch.Tensor:
        """计算完整的协方差矩阵（带缓存）

        Returns:
            torch.Tensor: [N, 3, 3] 协方差矩阵
        """
        current_size = self._means.shape[0]
        scales_id = id(self._scales)
        rotations_id = id(self._rotations)

        if (self._cov_cache is None or
            self._cov_cache_size != current_size or
            self._scales is None or self._rotations is None):

            scales = torch.exp(self._scales)
            R = Quaternion.to_rotation_matrix(self._rotations)
            S = torch.zeros_like(R)
            for i in range(scales.shape[0]):
                S[i] = torch.diag(scales[i])

            Sigma = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
            self._cov_cache = Sigma
            self._cov_cache_size = current_size
            self._cov_cache_scales_id = scales_id
            self._cov_cache_rotations_id = rotations_id
            return Sigma

        if (self._cov_cache_scales_id == scales_id and
            self._cov_cache_rotations_id == rotations_id and
            self._cov_cache_size == current_size):
            return self._cov_cache

        scales = torch.exp(self._scales)
        R = Quaternion.to_rotation_matrix(self._rotations)
        S = torch.zeros_like(R)
        for i in range(scales.shape[0]):
            S[i] = torch.diag(scales[i])

        Sigma = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
        self._cov_cache = Sigma
        self._cov_cache_size = current_size
        self._cov_cache_scales_id = scales_id
        self._cov_cache_rotations_id = rotations_id
        return Sigma

    def get_opacity(self) -> torch.Tensor:
        """获取sigmoid激活后的不透明度"""
        return torch.sigmoid(self._opacities)

    def get_active_sh_coeffs(self) -> torch.Tensor:
        """获取当前激活的球谐系数"""
        active_coeffs = (self._active_sh_degree + 1) ** 2
        return self._sh_coeffs[:, :active_coeffs]

    def get_covariance_full(self) -> torch.Tensor:
        """获取完整的协方差矩阵

        Returns:
            torch.Tensor: [N, 3, 3] 协方差矩阵
        """
        scales = torch.exp(self._scales)
        R = Quaternion.to_rotation_matrix(self._rotations)
        S = torch.zeros_like(R)
        for i in range(scales.shape[0]):
            S[i] = torch.diag(scales[i])
        Sigma = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
        return Sigma

    def save_checkpoint(self, file_path: str, iteration: int = 0, optimizer_state: dict = None):
        """保存检查点

        Args:
            file_path: 保存路径
            iteration: 当前迭代次数
            optimizer_state: 优化器状态字典
        """
        checkpoint = {
            'iteration': iteration,
            'means': self._means.data.cpu(),
            'scales': self._scales.data.cpu(),
            'rotations': self._rotations.data.cpu(),
            'opacities': self._opacities.data.cpu(),
            'sh_coeffs': self._sh_coeffs.data.cpu(),
            'sh_degree': self.sh_degree,
            'active_sh_degree': self._active_sh_degree,
            'num_gaussians': self.num_gaussians
        }
        if optimizer_state is not None:
            checkpoint['optimizer_state'] = optimizer_state
        torch.save(checkpoint, file_path)

    def load_checkpoint(self, file_path: str, device: torch.device = None):
        """加载检查点

        Args:
            file_path: 检查点路径
            device: 加载设备

        Returns:
            dict: 检查点信息 {'iteration': int, 'optimizer_state': dict or None}
        """
        if device is None:
            device = torch.device('cpu')
        checkpoint = torch.load(file_path, map_location=device)

        num_gaussians = checkpoint['means'].shape[0]
        sh_degree = checkpoint.get('sh_degree', 3)
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2
        self._active_sh_degree = checkpoint.get('active_sh_degree', 0)

        self._initialize_params(num_gaussians, device)

        self._means = torch.nn.Parameter(checkpoint['means'].to(device), requires_grad=True)
        self._scales = torch.nn.Parameter(checkpoint['scales'].to(device), requires_grad=True)
        self._rotations = torch.nn.Parameter(checkpoint['rotations'].to(device), requires_grad=True)
        self._opacities = torch.nn.Parameter(checkpoint['opacities'].to(device), requires_grad=True)
        self._sh_coeffs = torch.nn.Parameter(checkpoint['sh_coeffs'].to(device), requires_grad=True)

        result = {
            'iteration': checkpoint.get('iteration', 0),
            'optimizer_state': checkpoint.get('optimizer_state', None)
        }
        return result

    def _expand_params(self, new_total: int):
        """扩展参数张量以容纳更多高斯"""
        device = self._means.device
        old_total = self._means.shape[0]

        new_means = torch.zeros(new_total, 3, device=device)
        new_scales = torch.zeros(new_total, 3, device=device)
        new_rotations = torch.zeros(new_total, 4, device=device)
        new_opacities = torch.zeros(new_total, 1, device=device)
        new_sh_coeffs = torch.zeros(new_total, self.num_sh_coeffs, device=device)

        new_means[:old_total] = self._means.data
        new_scales[:old_total] = self._scales.data
        new_rotations[:old_total] = self._rotations.data
        new_opacities[:old_total] = self._opacities.data
        new_sh_coeffs[:old_total] = self._sh_coeffs.data

        self._means = torch.nn.Parameter(new_means, requires_grad=True)
        self._scales = torch.nn.Parameter(new_scales, requires_grad=True)
        self._rotations = torch.nn.Parameter(new_rotations, requires_grad=True)
        self._opacities = torch.nn.Parameter(new_opacities, requires_grad=True)
        self._sh_coeffs = torch.nn.Parameter(new_sh_coeffs, requires_grad=True)

    def densify_and_prune(
        self,
        grad_means: torch.Tensor,
        grad_scales: torch.Tensor,
        grad_opacities: torch.Tensor,
        grad_threshold: float = 0.0001,
        size_threshold: float = 0.01,
        large_scale_threshold: float = 2.0,
        opacity_threshold: float = 0.05,
        max_gaussians: int = 500000
    ):
        """致密化和剪枝

        根据梯度信息进行:
        - 分裂: 大高斯+大梯度 -> 分裂为多个小高斯
        - 克隆: 小高斯+大梯度 -> 复制到欠重建区域
        - 剪枝: 低不透明度或大体积 -> 移除

        Args:
            grad_means: [N, 3] 位置梯度
            grad_scales: [N, 3] 缩放梯度
            grad_opacities: [N, 1] 不透明度梯度
            grad_threshold: 梯度阈值
            size_threshold: 小尺度阈值
            large_scale_threshold: 大尺度阈值
            opacity_threshold: 剪枝不透明度阈值
            max_gaussians: 最大高斯数量
        """
        with torch.no_grad():
            scales = torch.exp(self._scales)
            scale_norms = scales.norm(dim=-1)
            grad_norms = grad_means.norm(dim=-1)

            small_mask = scale_norms < size_threshold
            large_mask = scale_norms > large_scale_threshold
            high_grad_mask = grad_norms > grad_threshold
            low_opacity_mask = (torch.sigmoid(self._opacities) < opacity_threshold).squeeze(-1)

            clone_mask = small_mask & high_grad_mask
            split_mask = large_mask & high_grad_mask
            prune_mask = low_opacity_mask | large_mask

            num_clones = clone_mask.sum().item()
            num_splits = split_mask.sum().item()
            num_prunes = prune_mask.sum().item()

            del scales, scale_norms, grad_norms
            del small_mask, large_mask, high_grad_mask, low_opacity_mask

            saved_means = self._means.data.clone()
            saved_scales = self._scales.data.clone()
            saved_rotations = self._rotations.data.clone()
            saved_opacities = self._opacities.data.clone()
            saved_sh_coeffs = self._sh_coeffs.data.clone()
            saved_grad_means = grad_means.clone()
            saved_scales_for_split = torch.exp(self._scales.data.clone())

            if num_prunes > 0:
                keep_mask = ~prune_mask
                self._means = torch.nn.Parameter(self._means.data[keep_mask].clone())
                self._scales = torch.nn.Parameter(self._scales.data[keep_mask].clone())
                self._rotations = torch.nn.Parameter(self._rotations.data[keep_mask].clone())
                self._opacities = torch.nn.Parameter(self._opacities.data[keep_mask].clone())
                self._sh_coeffs = torch.nn.Parameter(self._sh_coeffs.data[keep_mask].clone())

            current_num = self.num_gaussians
            new_gaussians = []

            if num_clones > 0 and current_num + num_clones < max_gaussians:
                clone_indices = torch.where(clone_mask)[0]
                for idx in clone_indices:
                    new_mean = saved_means[idx] + saved_grad_means[idx] * 0.01
                    new_scale = saved_scales[idx].clone()
                    new_rotation = saved_rotations[idx].clone()
                    new_opacity = saved_opacities[idx].clone()
                    new_sh = saved_sh_coeffs[idx].clone()

                    new_gaussians.append((
                        new_mean.unsqueeze(0),
                        new_scale.unsqueeze(0),
                        new_rotation.unsqueeze(0),
                        new_opacity.unsqueeze(0),
                        new_sh.unsqueeze(0)
                    ))

            if num_splits > 0 and current_num + len(new_gaussians) + num_splits < max_gaussians:
                split_indices = torch.where(split_mask)[0]
                for idx in split_indices:
                    old_scale = saved_scales_for_split[idx]
                    new_scale_value = old_scale * 0.8

                    for angle in [0, np.pi]:
                        new_mean = saved_means[idx].clone()
                        offset = torch.tensor([new_scale_value[0] * 0.5 * np.cos(angle),
                                              new_scale_value[1] * 0.5 * np.sin(angle),
                                              0.0], device=new_mean.device)
                        new_mean = new_mean + offset

                        new_scale = torch.log(new_scale_value.unsqueeze(0))
                        new_rotation = saved_rotations[idx].clone()
                        new_opacity = saved_opacities[idx].clone()
                        new_sh = saved_sh_coeffs[idx].clone()

                        new_gaussians.append((
                            new_mean.unsqueeze(0),
                            new_scale,
                            new_rotation.unsqueeze(0),
                            new_opacity.unsqueeze(0),
                            new_sh.unsqueeze(0)
                        ))

            if new_gaussians:
                total_new = len(new_gaussians)
                new_means = torch.cat([g[0] for g in new_gaussians], dim=0)
                new_scales = torch.cat([g[1] for g in new_gaussians], dim=0)
                new_rotations = torch.cat([g[2] for g in new_gaussians], dim=0)
                new_opacities = torch.cat([g[3] for g in new_gaussians], dim=0)
                new_sh_coeffs = torch.cat([g[4] for g in new_gaussians], dim=0)

                combined_means = torch.cat([self._means.data, new_means], dim=0)
                combined_scales = torch.cat([self._scales.data, new_scales], dim=0)
                combined_rotations = torch.cat([self._rotations.data, new_rotations], dim=0)
                combined_opacities = torch.cat([self._opacities.data, new_opacities], dim=0)
                combined_sh_coeffs = torch.cat([self._sh_coeffs.data, new_sh_coeffs], dim=0)

                self._means = torch.nn.Parameter(combined_means, requires_grad=True)
                self._scales = torch.nn.Parameter(combined_scales, requires_grad=True)
                self._rotations = torch.nn.Parameter(combined_rotations, requires_grad=True)
                self._opacities = torch.nn.Parameter(combined_opacities, requires_grad=True)
                self._sh_coeffs = torch.nn.Parameter(combined_sh_coeffs, requires_grad=True)

                return num_clones, num_splits, num_prunes, total_new

        return num_clones, num_splits, num_prunes, 0