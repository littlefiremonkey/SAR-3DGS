"""
SAR-GS训练策略配置模块

致密化策略:
- 分裂: 大高斯(尺度>split_threshold) + 大梯度 -> 分裂为多个小高斯
- 克隆: 小高斯(尺度<clone_threshold) + 大梯度 -> 复制到欠重建区域

剪枝策略:
- 尺寸剪枝: scale > scale_threshold 的高斯被剪枝
- 透明度剪枝: opacity < opacity_threshold 的高斯被剪枝
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DensifyConfig:
    """致密化配置"""
    enabled: bool = True
    interval: int = 100
    start_iter: int = 500

    grad_threshold: float = 0.0001
    clone_threshold: float = 0.01
    split_threshold: float = 2.0

    split_factor: float = 0.8
    clone_scale: float = 0.01
    max_gaussians: int = 500000


@dataclass
class PruneConfig:
    """剪枝配置"""
    enabled: bool = True
    interval: int = 100
    start_iter: int = 500

    opacity_threshold: float = 0.05
    scale_threshold: float = 2.0
    min_gaussians: int = 1000


@dataclass
class TrainingConfig:
    """完整训练配置"""
    densify: DensifyConfig = None
    prune: PruneConfig = None

    l1_weight: float = 0.9
    ssim_weight: float = 0.1

    learning_rate: float = 0.001
    lr_schedule: bool = True
    lr_decay_factor: float = 0.5
    lr_decay_interval: int = 10000

    opacity_reset_interval: int = 3000
    opacity_reset_value: float = 0.5

    max_iterations: int = 50000
    log_interval: int = 100
    save_interval: int = 5000

    init_num_gaussians: int = 5000
    init_scale: float = 0.5
    init_opacity: float = 0.5

    sh_degree: int = 3

    def __post_init__(self):
        if self.densify is None:
            self.densify = DensifyConfig()
        if self.prune is None:
            self.prune = PruneConfig()


def get_default_training_config() -> TrainingConfig:
    """获取默认训练配置"""
    return TrainingConfig()