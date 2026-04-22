"""
SAR Gaussian Splatting 模块

可导入的模块:
- GaussianModel: 3D高斯模型
- SARRenderParams, SARRenderer: 渲染器
- SphericalHarmonics, SHRenderer: 球谐函数渲染
- SARSceneDataset: 数据加载
- CombinedLoss: 损失函数
- DensifyConfig, PruneConfig, TrainingConfig: 训练配置
"""

from .gaussian_model import GaussianModel, Quaternion

from .renderer import (
    SARRenderParams,
    SARRenderer,
    CoordinateTransformer,
    UnifiedProjector,
    GaussianDensityCalculator,
    UnifiedSARRenderer,
    SphericalHarmonics,
    SHRenderer,
    create_sh_coeffs,
)

from .scene import (
    SARSceneDataset,
    RadarParams,
    SARCameraInfo,
    compute_scene_bounds_from_dataset,
)

from .losses import (
    CombinedLoss,
    l1_loss,
    dssim_loss,
    combined_loss,
    weighted_l1_loss,
)

from .training_strategies import (
    DensifyConfig,
    PruneConfig,
    TrainingConfig,
    get_default_training_config,
)

__all__ = [
    'GaussianModel',
    'Quaternion',
    'SARRenderParams',
    'SARRenderer',
    'CoordinateTransformer',
    'UnifiedProjector',
    'GaussianDensityCalculator',
    'UnifiedSARRenderer',
    'SphericalHarmonics',
    'SHRenderer',
    'create_sh_coeffs',
    'SARSceneDataset',
    'RadarParams',
    'SARCameraInfo',
    'compute_scene_bounds_from_dataset',
    'CombinedLoss',
    'l1_loss',
    'dssim_loss',
    'combined_loss',
    'weighted_l1_loss',
    'DensifyConfig',
    'PruneConfig',
    'TrainingConfig',
    'get_default_training_config',
]