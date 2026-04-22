"""
SAR-GS Training Pipeline Module

模块化训练管道，包含：
- RenderPipeline: 渲染相关逻辑
- DensifyPruneManager: 致密化和剪枝管理
- TrainingPipeline: 核心训练管道
"""

from .render_pipeline import RenderPipeline
from .densify_prune import DensifyPruneManager, DensifyPruneResult
from .training_pipeline import TrainingPipeline

__all__ = [
    'RenderPipeline',
    'DensifyPruneManager',
    'DensifyPruneResult',
    'TrainingPipeline',
]
