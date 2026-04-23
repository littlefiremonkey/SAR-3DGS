"""
SAR-GS损失函数模块

包含:
- L1损失
- D-SSIM损失 (结构相似性差异)
- 组合损失 (L1 + DSSIM, 比例0.9:0.1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1损失

    Args:
        pred: 预测图像 [H, W] 或 [B, H, W]
        target: 目标图像 [H, W] 或 [B, H, W]

    Returns:
        torch.Tensor: L1损失值
    """
    return torch.abs(pred - target).mean()


def weighted_l1_loss(pred: torch.Tensor, target: torch.Tensor, mode: str = 'linear') -> torch.Tensor:
    """加权L1损失

    对像素值大的部位进行加权增强

    Args:
        pred: 预测图像
        target: 目标图像
        mode: 加权模式
            - 'linear': 线性加权，权重 = 1 + target / max(target)
            - 'square': 平方加权，权重 = 1 + (target / max(target))^2
            - 'sqrt': 平方根加权，权重 = 1 + sqrt(target / max(target))

    Returns:
        torch.Tensor: 加权L1损失值
    """
    diff = torch.abs(pred - target)
    target_max = target.max()
    if target_max > 0:
        target_normalized = target / target_max
    else:
        return diff.mean()

    if mode == 'linear':
        weights = 1.0 + target_normalized
    elif mode == 'square':
        weights = 1.0 + target_normalized ** 2
    elif mode == 'sqrt':
        weights = 1.0 + torch.sqrt(target_normalized + 1e-8)
    else:
        weights = 1.0 + target_normalized

    weighted_diff = diff * weights
    return weighted_diff.mean()


def gaussian_kernel(size: int, sigma: float = 1.5) -> torch.Tensor:
    """创建高斯核

    Args:
        size: 核大小
        sigma: 高斯标准差

    Returns:
        torch.Tensor: [size, size] 高斯核
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g.outer(g)
    return kernel


def dssim_loss(pred: torch.Tensor, target: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """D-SSIM损失 (结构相似性差异)

    Args:
        pred: 预测图像 [H, W] 或 [B, H, W]
        target: 目标图像 [H, W] 或 [B, H, W]
        window_size: 滑动窗口大小

    Returns:
        torch.Tensor: D-SSIM损失值 (0-1范围)
    """
    if pred.dim() == 2:
        pred = pred.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
    elif pred.dim() == 3:
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
    elif pred.dim() == 4:
        pass
    else:
        raise ValueError(f"输入维度错误: pred.dim()={pred.dim()}")

    device = pred.device
    kernel = gaussian_kernel(window_size, 1.5).to(device)
    kernel = kernel.view(1, 1, window_size, window_size)
    kernel = kernel.repeat(pred.size(1), 1, 1, 1)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = F.conv2d(pred, kernel, padding=window_size // 2, groups=pred.size(1))
    mu_target = F.conv2d(target, kernel, padding=window_size // 2, groups=pred.size(1))

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred ** 2, kernel, padding=window_size // 2, groups=pred.size(1)) - mu_pred_sq
    sigma_target_sq = F.conv2d(target ** 2, kernel, padding=window_size // 2, groups=pred.size(1)) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, kernel, padding=window_size // 2, groups=pred.size(1)) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))

    return 1.0 - ssim_map.mean()


class CombinedLoss(nn.Module):
    """组合损失模块

    L1 + D-SSIM组合，比例0.9:0.1，支持加权L1
    """

    _cached_kernel = None
    _cached_kernel_size = None

    def __init__(
        self,
        l1_weight: float = 0.9,
        ssim_weight: float = 0.1,
        window_size: int = 5,
        use_weighted_l1: bool = False,
        l1_weight_mode: str = 'linear'
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        self.use_weighted_l1 = use_weighted_l1
        self.l1_weight_mode = l1_weight_mode

    def _get_kernel(self, device: torch.device) -> torch.Tensor:
        if CombinedLoss._cached_kernel is None or CombinedLoss._cached_kernel_size != self.window_size:
            CombinedLoss._cached_kernel = gaussian_kernel(self.window_size, 1.5)
            CombinedLoss._cached_kernel_size = self.window_size
        kernel = CombinedLoss._cached_kernel.to(device)
        return kernel.view(1, 1, self.window_size, self.window_size)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算组合损失

        Args:
            pred: 预测图像
            target: 目标图像

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: 总损失和损失详情
        """
        if self.use_weighted_l1:
            loss_l1 = weighted_l1_loss(pred, target, mode=self.l1_weight_mode)
        else:
            loss_l1 = l1_loss(pred, target)

        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        elif pred.dim() != 4:
            raise ValueError(f"输入维度错误: pred.dim()={pred.dim()}")

        device = pred.device
        kernel = self._get_kernel(device)
        kernel = kernel.repeat(pred.size(1), 1, 1, 1)

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_pred = F.conv2d(pred, kernel, padding=self.window_size // 2, groups=pred.size(1))
        mu_target = F.conv2d(target, kernel, padding=self.window_size // 2, groups=pred.size(1))

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, kernel, padding=self.window_size // 2, groups=pred.size(1)) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, kernel, padding=self.window_size // 2, groups=pred.size(1)) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, kernel, padding=self.window_size // 2, groups=pred.size(1)) - mu_pred_target

        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        loss_ssim = 1.0 - ssim_map.mean()

        total_loss = self.l1_weight * loss_l1 + self.ssim_weight * loss_ssim

        loss_l1_val = loss_l1.item() if loss_l1.numel() == 1 else loss_l1.mean().item()
        loss_ssim_val = loss_ssim.item() if loss_ssim.numel() == 1 else loss_ssim.mean().item()
        total_loss_val = total_loss.item() if total_loss.numel() == 1 else total_loss.mean().item()

        return total_loss, {
            'l1': loss_l1_val,
            'dssim': loss_ssim_val,
            'total': total_loss_val,
            'l1_weight': self.l1_weight,
            'dssim_weight': self.ssim_weight,
            'weighted_l1': self.use_weighted_l1
        }


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    l1_weight: float = 0.9,
    ssim_weight: float = 0.1,
    window_size: int = 11,
    use_weighted_l1: bool = False,
    l1_weight_mode: str = 'linear'
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """组合损失函数

    Args:
        pred: 预测图像
        target: 目标图像
        l1_weight: L1损失权重 (默认0.9)
        ssim_weight: D-SSIM损失权重 (默认0.1)
        window_size: SSIM窗口大小
        use_weighted_l1: 是否使用加权L1 (默认False)
        l1_weight_mode: 加权L1的加权模式 (默认'linear')

    Returns:
        Tuple[torch.Tensor, Dict[str, float]]: 总损失和损失详情
    """
    loss_fn = CombinedLoss(l1_weight, ssim_weight, window_size, use_weighted_l1, l1_weight_mode)
    return loss_fn(pred, target)