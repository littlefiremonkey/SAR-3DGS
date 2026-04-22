"""
致密化和剪枝管理模块

管理高斯模型的致密化（clone/split）和剪枝操作
"""

from dataclasses import dataclass
from typing import Optional, Callable

import torch

from gaussian_model import GaussianModel
from training_strategies import DensifyConfig, PruneConfig


@dataclass
class DensifyPruneResult:
    """致密化和剪枝操作结果"""
    num_clones: int = 0
    num_splits: int = 0
    num_prunes: int = 0
    num_new_gaussians: int = 0


@dataclass
class GradientInfo:
    """梯度信息"""
    means: torch.Tensor
    scales: torch.Tensor
    opacities: torch.Tensor


class DensifyPruneManager:
    """致密化和剪枝管理器"""

    def __init__(
        self,
        densify_config: DensifyConfig,
        prune_config: PruneConfig
    ):
        """初始化

        Args:
            densify_config: 致密化配置
            prune_config: 剪枝配置
        """
        self.densify = densify_config
        self.prune = prune_config

        self._last_prune_iter = 0
        self._last_opacity_reset_iter = 0

    def should_densify_or_prune(
        self,
        iteration: int,
        current_epoch: int,
        num_gaussians: int
    ) -> bool:
        """判断是否应该执行致密化/剪枝

        Args:
            iteration: 当前迭代次数
            current_epoch: 当前epoch
            num_gaussians: 当前高斯数量

        Returns:
            bool: 是否应该执行
        """
        if not self.densify.enabled:
            return False

        if iteration <= self.densify.start_iter:
            return False

        if iteration % self.densify.interval != 0:
            return False

        if current_epoch <= self._last_prune_iter + 3:
            return False

        if num_gaussians >= self.densify.max_gaussians:
            return False

        return True

    def should_reset_opacity(
        self,
        iteration: int,
        opacity_reset_interval: int
    ) -> bool:
        """判断是否应该重置不透明度

        Args:
            iteration: 当前迭代次数
            opacity_reset_interval: 重置间隔

        Returns:
            bool: 是否应该重置
        """
        return (
            iteration > self.densify.interval * 20 and
            iteration % opacity_reset_interval == 0
        )

    def execute(
        self,
        model: GaussianModel,
        grads: GradientInfo,
        optimizer: torch.optim.Optimizer
    ) -> DensifyPruneResult:
        """执行致密化和剪枝

        Args:
            model: 高斯模型
            grads: 梯度信息
            optimizer: 优化器

        Returns:
            DensifyPruneResult: 操作结果
        """
        result = model.densify_and_prune(
            grad_means=grads.means,
            grad_scales=grads.scales,
            grad_opacities=grads.opacities,
            grad_threshold=self.densify.grad_threshold,
            size_threshold=self.densify.clone_threshold,
            large_scale_threshold=self.prune.scale_threshold,
            opacity_threshold=self.prune.opacity_threshold,
            max_gaussians=self.densify.max_gaussians
        )

        num_clones, num_splits, num_prunes, num_new = result

        self._last_prune_iter += 1

        return DensifyPruneResult(
            num_clones=num_clones,
            num_splits=num_splits,
            num_prunes=num_prunes,
            num_new_gaussians=num_new
        )

    def reset_opacity(
        self,
        model: GaussianModel,
        init_opacity_logit: float
    ):
        """重置模型不透明度

        Args:
            model: 高斯模型
            init_opacity_logit: 初始不透明度的logit值
        """
        model._opacities.data.fill_(init_opacity_logit)
        self._last_opacity_reset_iter = 0

    def get_last_prune_iter(self) -> int:
        """获取上次剪枝的迭代数"""
        return self._last_prune_iter

    def get_last_opacity_reset_iter(self) -> int:
        """获取上次不透明度重置的迭代数"""
        return self._last_opacity_reset_iter
