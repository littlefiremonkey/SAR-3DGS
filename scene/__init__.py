"""
SAR场景模块

提供SAR数据加载、相机参数管理、场景边界计算等功能
"""

from .dataset_readers import (
    RadarParams,
    SARCameraInfo,
    SARSceneDataset,
    parse_sar_filename,
    load_image_as_tensor,
    compute_scene_bounds_from_dataset
)

__all__ = [
    'RadarParams',
    'SARCameraInfo',
    'SARSceneDataset',
    'parse_sar_filename',
    'load_image_as_tensor',
    'compute_scene_bounds_from_dataset',
]