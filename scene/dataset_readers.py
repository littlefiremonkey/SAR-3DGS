"""
SAR场景数据加载模块

支持用户指定路径下的SAR图像数据加载
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

import torch
import numpy as np
from PIL import Image


@dataclass
class RadarParams:
    """雷达参数"""
    incidence_angle: float
    track_angle: float
    azimuth_angle: float = 0.0
    radar_altitude: float = 10000.0
    range_resolution: float = 0.3
    azimuth_resolution: float = 0.3


@dataclass
class SARCameraInfo:
    """SAR相机/视角信息"""
    image_path: str
    radar_params: RadarParams
    image: Optional[torch.Tensor] = None
    range_pixels: int = 0
    azimuth_pixels: int = 0
    normalization_factor: float = 255.0

    @property
    def incidence_angle(self) -> float:
        return self.radar_params.incidence_angle

    @property
    def track_angle(self) -> float:
        return self.radar_params.track_angle

    @property
    def radar_height(self) -> float:
        return self.radar_params.radar_altitude


def parse_sar_filename(filename: str) -> Optional[Dict]:
    """解析SAR图像文件名获取雷达参数

    支持格式:
    - 新格式: inc_{incidence}-track_{track}-height_{height}-squint_{squint}-rr_{range_res}-ar_{azimuth_res}.png
    - 旧格式: elev-{incidence}-azim-{track}-A.png

    Args:
        filename: 文件名

    Returns:
        dict: 包含解析出的参数，或None
    """
    new_pattern = r'inc_([n\d.]+)-track_([n\d.]+)-height_(\d+)-squint_([n\d.]+)-rr_([\d.]+)-ar_([\d.]+)\.png'
    old_pattern = r'elev-([\d.]+)-azim-(-?[\d.]+)-A\.png'

    new_match = re.match(new_pattern, filename)
    if new_match:
        def parse_signed(val: str) -> float:
            if val.startswith('n'):
                return -float(val[1:])
            return float(val)
        return {
            'incidence': float(new_match.group(1)),
            'track': parse_signed(new_match.group(2)),
            'height': float(new_match.group(3)),
            'squint': parse_signed(new_match.group(4)),
            'range_resolution': float(new_match.group(5)),
            'azimuth_resolution': float(new_match.group(6)),
            'format': 'new'
        }

    old_match = re.match(old_pattern, filename)
    if old_match:
        return {
            'incidence': float(old_match.group(1)),
            'track': float(old_match.group(2)),
            'height': 10000.0,
            'squint': 0.0,
            'range_resolution': 0.3,
            'azimuth_resolution': 0.3,
            'format': 'old'
        }

    return None


def load_image_as_tensor(image_path: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """加载图像并转为张量

    Args:
        image_path: 图像路径

    Returns:
        torch.Tensor: [H, W] 灰度图像
        Tuple[int, int]: (距离向像素数, 方位向像素数)
    """
    img = Image.open(image_path).convert('L')
    img_np = np.array(img, dtype=np.float32)
    return torch.from_numpy(img_np), (img_np.shape[0], img_np.shape[1])


class SARSceneDataset:
    """SAR场景数据集

    从用户指定路径加载SAR图像
    """

    def __init__(
        self,
        data_path: str,
        load_images: bool = True
    ):
        """
        Args:
            data_path: 数据路径（可以是目录或图像文件）
            load_images: 是否立即加载图像到内存
        """
        self.data_path = data_path
        self.cameras: List[SARCameraInfo] = []
        self.scene_normalization: Dict = {}

        self._scan_and_load(data_path, load_images)
        self._compute_scene_normalization()

    def _scan_and_load(self, data_path: str, load_images: bool):
        """扫描路径并加载相机信息"""
        path = Path(data_path)

        if path.is_file():
            self._add_image_file(path, load_images)
        elif path.is_dir():
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff']:
                for img_file in sorted(path.glob(ext)):
                    self._add_image_file(img_file, load_images)

        if not self.cameras:
            raise ValueError(f"未找到有效图像文件: {data_path}")

    def _add_image_file(self, img_path: Path, load_images: bool):
        """添加单个图像文件"""
        parsed = parse_sar_filename(img_path.name)
        if parsed is None:
            return

        radar_params = RadarParams(
            incidence_angle=parsed['incidence'],
            track_angle=parsed['track'],
            azimuth_angle=parsed.get('squint', 0.0),
            radar_altitude=parsed.get('height', 10000.0),
            range_resolution=parsed.get('range_resolution', 0.3),
            azimuth_resolution=parsed.get('azimuth_resolution', 0.3)
        )

        img = Image.open(img_path)
        range_pixels, azimuth_pixels = img.size

        camera_info = SARCameraInfo(
            image_path=str(img_path),
            radar_params=radar_params,
            range_pixels=range_pixels,
            azimuth_pixels=azimuth_pixels
        )

        if load_images:
            img_tensor, _ = load_image_as_tensor(str(img_path))
            camera_info.image = img_tensor
            camera_info.normalization_factor = float(img_tensor.max()) if img_tensor.max() > 0 else 255.0

        self.cameras.append(camera_info)

    def _compute_scene_normalization(self):
        """计算场景归一化参数"""
        if not self.cameras:
            return

        radar_heights = [c.radar_height for c in self.cameras]
        incidences = [c.incidence_angle for c in self.cameras]

        h = np.mean(radar_heights)

        range_pixels = max(c.range_pixels for c in self.cameras)
        azimuth_pixels = max(c.azimuth_pixels for c in self.cameras)
        range_res = self.cameras[0].radar_params.range_resolution
        azimuth_res = self.cameras[0].radar_params.azimuth_resolution

        range_span = range_pixels * range_res
        azimuth_span = azimuth_pixels * azimuth_res

        L = max(range_span, azimuth_span)
        scene_scale = L / 2.0

        self.scene_normalization = {
            'scene_center': [0.0, 0.0, 0.0],
            'scene_scale': float(scene_scale),
            'range_span': float(range_span),
            'azimuth_span': float(azimuth_span),
            'radar_height': float(h),
            'num_cameras': len(self.cameras),
            'incidence_range': [float(min(incidences)), float(max(incidences))],
            'track_range': [float(min(c.track_angle for c in self.cameras)), float(max(c.track_angle for c in self.cameras))]
        }

    def get_camera(self, index: int) -> SARCameraInfo:
        """获取指定索引的相机"""
        return self.cameras[index]

    def __len__(self) -> int:
        return len(self.cameras)

    def split_train_val(self, train_ratio: float = 0.8) -> Tuple['SARSceneDataset', 'SARSceneDataset']:
        """分割训练集和验证集

        Args:
            train_ratio: 训练集比例

        Returns:
            Tuple[SARSceneDataset, SARSceneDataset]: 训练集和验证集
        """
        num_cameras = len(self.cameras)
        num_train = int(num_cameras * train_ratio)

        indices = np.random.permutation(num_cameras)
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]

        train_dataset = SARSceneDataset.__new__(SARSceneDataset)
        train_dataset.data_path = self.data_path
        train_dataset.cameras = [self.cameras[i] for i in train_indices]
        train_dataset.scene_normalization = self.scene_normalization

        val_dataset = SARSceneDataset.__new__(SARSceneDataset)
        val_dataset.data_path = self.data_path
        val_dataset.cameras = [self.cameras[i] for i in val_indices]
        val_dataset.scene_normalization = self.scene_normalization

        return train_dataset, val_dataset


def compute_scene_bounds_from_dataset(dataset: SARSceneDataset, z_ratio: float = 0.25) -> Tuple[float, float, float, float, float, float]:
    """从数据集计算场景边界

    Args:
        dataset: SAR场景数据集
        z_ratio: Z轴范围相对于X/Y范围的比例

    Returns:
        Tuple: (x_min, x_max, y_min, y_max, z_min, z_max)
    """
    norm = dataset.scene_normalization
    L = norm['scene_scale'] * 2

    quarter_L = L / 5.0

    return (-quarter_L, quarter_L, -quarter_L, quarter_L, 0.0, quarter_L)