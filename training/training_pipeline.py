"""
核心训练管道模块

整合渲染、损失计算、优化、致密化和剪枝逻辑
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch

from gaussian_model import GaussianModel
from losses import CombinedLoss
from training_strategies import DensifyConfig, PruneConfig
from training.render_pipeline import RenderPipeline
from training.densify_prune import DensifyPruneManager, DensifyPruneResult, GradientInfo


@dataclass
class EpochResult:
    """Epoch训练结果"""
    epoch: int
    iteration: int
    avg_loss: float
    avg_l1: float
    avg_ssim: float
    num_gaussians: int
    clone_count: int
    split_count: int
    prune_count: int
    time_seconds: float


@dataclass
class TrainStepResult:
    """单步训练结果"""
    loss: float
    l1_loss: float
    ssim_loss: float
    densify_prune: Optional[DensifyPruneResult]


class TrainingPipeline:
    """核心训练管道"""

    def __init__(
        self,
        gaussian_model: GaussianModel,
        optimizer: torch.optim.Optimizer,
        loss_fn: CombinedLoss,
        densify_config: DensifyConfig,
        prune_config: PruneConfig,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        stop_callback: Optional[Callable[[], bool]] = None
    ):
        """初始化训练管道

        Args:
            gaussian_model: 高斯模型
            optimizer: 优化器
            loss_fn: 损失函数
            densify_config: 致密化配置
            prune_config: 剪枝配置
            scheduler: 学习率调度器
            stop_callback: 停止训练回调，返回True表示应该停止
        """
        self.model = gaussian_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.stop_callback = stop_callback

        self.render_pipeline = RenderPipeline()
        self.densify_prune_manager = DensifyPruneManager(densify_config, prune_config)

        self.iteration = 0
        self.epoch = 0
        self.loss_history: List[float] = []

        self._epoch_l1_losses: List[float] = []
        self._epoch_ssim_losses: List[float] = []
        self._epoch_clone_count = 0
        self._epoch_split_count = 0
        self._epoch_prune_count = 0

        self._init_opacity_logit: Optional[float] = None
        self._opacity_reset_interval: Optional[int] = None

    def set_init_opacity(self, logit: float, reset_interval: int):
        """设置初始不透明度参数

        Args:
            logit: 不透明度的logit值
            reset_interval: 重置间隔
        """
        self._init_opacity_logit = logit
        self._opacity_reset_interval = reset_interval

    def set_device(self, device: torch.device):
        """设置计算设备

        Args:
            device: torch.device
        """
        self.render_pipeline.set_device(device)

    def train_epoch(
        self,
        dataset,
        epoch_indices: Optional[np.ndarray] = None
    ) -> EpochResult:
        """执行一个epoch的训练

        Args:
            dataset: 数据集
            epoch_indices: 指定训练顺序的索引数组，默认随机打乱

        Returns:
            EpochResult: epoch结果
        """
        epoch_start_time = time.time()

        if epoch_indices is None:
            epoch_indices = np.random.permutation(len(dataset))

        self._reset_epoch_stats()

        for idx in epoch_indices:
            if self._should_stop():
                break

            camera = dataset.get_camera(idx)
            step_result = self.train_step(camera)

            self._accumulate_epoch_stats(step_result)
            self.loss_history.append(step_result.loss)

            self.iteration += 1

            self._handle_opacity_reset()

        self.epoch += 1

        return self._build_epoch_result(epoch_start_time)

    def train_step(self, camera) -> TrainStepResult:
        """单步训练

        Args:
            camera: 相机信息

        Returns:
            TrainStepResult: 训练结果
        """
        norm_factor = camera.normalization_factor
        target_image = camera.image.to(next(self.model.parameters()).device) / norm_factor

        rendered = self.render_pipeline.render(
            self.model, camera, norm_factor
        )

        loss, loss_dict = self.loss_fn(rendered, target_image)

        self.optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            grads = GradientInfo(
                means=self.model._means.grad,
                scales=self.model._scales.grad,
                opacities=self.model._opacities.grad
            )

            densify_prune_result = None
            if self.densify_prune_manager.should_densify_or_prune(
                self.iteration, self.epoch, self.model.num_gaussians
            ):
                if grads.means is not None:
                    grads.means = grads.means.detach()
                if grads.scales is not None:
                    grads.scales = grads.scales.detach()
                if grads.opacities is not None:
                    grads.opacities = grads.opacities.detach()

                densify_prune_result = self.densify_prune_manager.execute(
                    self.model, grads, self.optimizer
                )

                self._epoch_clone_count += densify_prune_result.num_clones
                self._epoch_split_count += densify_prune_result.num_splits
                self._epoch_prune_count += densify_prune_result.num_prunes

                torch.cuda.empty_cache()

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return TrainStepResult(
            loss=loss_dict['total'],
            l1_loss=loss_dict['l1'],
            ssim_loss=loss_dict['dssim'],
            densify_prune=densify_prune_result
        )

    def _should_stop(self) -> bool:
        """检查是否应该停止训练"""
        if self.stop_callback is not None:
            return self.stop_callback()
        return False

    def _reset_epoch_stats(self):
        """重置epoch统计"""
        self._epoch_l1_losses = []
        self._epoch_ssim_losses = []
        self._epoch_clone_count = 0
        self._epoch_split_count = 0
        self._epoch_prune_count = 0

    def _accumulate_epoch_stats(self, result: TrainStepResult):
        """累积epoch统计"""
        self._epoch_l1_losses.append(result.l1_loss)
        self._epoch_ssim_losses.append(result.ssim_loss)

    def _handle_opacity_reset(self):
        """处理不透明度重置"""
        if self._init_opacity_logit is None or self._opacity_reset_interval is None:
            return

        if self.densify_prune_manager.should_reset_opacity(
            self.iteration, self._opacity_reset_interval
        ):
            self.model._opacities.data.fill_(self._init_opacity_logit)

    def _build_epoch_result(self, start_time: float) -> EpochResult:
        """构建epoch结果"""
        avg_loss = np.mean(self.loss_history[-len(self._epoch_l1_losses):]) if self._epoch_l1_losses else 0.0
        avg_l1 = np.mean(self._epoch_l1_losses) if self._epoch_l1_losses else 0.0
        avg_ssim = np.mean(self._epoch_ssim_losses) if self._epoch_ssim_losses else 0.0

        return EpochResult(
            epoch=self.epoch,
            iteration=self.iteration,
            avg_loss=avg_loss,
            avg_l1=avg_l1,
            avg_ssim=avg_ssim,
            num_gaussians=self.model.num_gaussians,
            clone_count=self._epoch_clone_count,
            split_count=self._epoch_split_count,
            prune_count=self._epoch_prune_count,
            time_seconds=time.time() - start_time
        )

    def get_loss_history(self) -> List[float]:
        """获取损失历史"""
        return self.loss_history

    def get_sh_degree(self) -> int:
        """获取当前SH degree"""
        return self.model.active_sh_degree

    def set_sh_degree(self, degree: int):
        """设置SH degree"""
        self.model.active_sh_degree = degree
