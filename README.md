# SAR Gaussian Splatting (sar_gs) 项目详解

## 目录
1. [项目概述](#1-项目概述)
2. [数学原理](#2-数学原理)
3. [模块架构](#3-模块架构)
4. [渲染器架构](#4-渲染器架构)
5. [核心公式对照表](#5-核心公式对照表)
6. [使用示例](#6-使用示例)

---

## 1. 项目概述

### 1.1 项目目标
将 **3D Gaussian Splatting (3DGS)** 框架应用于 **合成孔径雷达(SAR)** 图像合成，通过3D高斯分布表示场景，实现端到端的SAR图像渲染和训练。

### 1.2 核心思想
- 使用一组3D高斯分布来表示场景
- 每个高斯包含位置、形状（协方差）、不透明度、球谐系数
- 通过渲染将这些3D高斯投影到SAR图像平面
- 针对SAR成像的几何特性进行物理建模

### 1.3 项目结构
```
sar_gs_v2-temp/
├── __init__.py                     # 模块导出
├── gaussian_model.py               # 3D高斯模型定义
├── losses.py                       # 损失函数
├── train.py                       # 命令行训练脚本
├── train_gui.py                   # GUI训练界面
├── training_strategies.py         # 致密化/剪枝配置
├── scene/
│   ├── __init__.py
│   └── dataset_readers.py         # MSTAR数据加载
├── training/
│   ├── __init__.py
│   ├── training_pipeline.py       # 训练管线
│   ├── render_pipeline.py         # 渲染管线（含缓存）
│   └── densify_prune.py          # 致密化/剪枝管理
└── cuda_rasterizer/
    ├── __init__.py               # CUDA Python绑定
    ├── build_v2_setup.py          # CUDA编译脚本
    ├── rasterizer_autograd.py     # PyTorch autograd封装
    └── rasterizer_impl_v2.cu      # CUDA渲染核实现
```

---

## 2. 数学原理

### 2.1 坐标系定义

#### 世界坐标系 W (ENU)
- 原点: 场景中心 O_w
- 轴向: X_w东, Y_w北, Z_w天
- 目标坐标: P_w = (x_w, y_w, z_w)

#### 雷达坐标系 R (航迹-视线系)
- 原点: 雷达相位中心 O_r
- 轴向:
  - X_r: 航迹方向, 单位矢量 u_x
  - Z_r: 视线方向, 由 O_r 指向场景中心
  - Y_r: 由右手定则 Y_r = Z_r × X_r

### 2.2 关键参数

| 参数 | 符号 | 定义 |
|------|------|------|
| 入射角 | θ | 雷达波束与地面法线的夹角 |
| 斜视角 | φ | 偏离正侧视的角度 |
| 航迹角 | α | X_r 与 X_w 的夹角 |
| 等效入射角 | β | sin(β) = sin(θ) · cos(φ) |
| 雷达高度 | H | 雷达平台高度 |
| 参考斜距 | R_c | R_c = H / cos(β) |

### 2.3 雷达位置计算

```
P_radar,w = (H * tan(β) * sin(α), -H * tan(β) * cos(α), H)
```

### 2.4 旋转矩阵 R<sub>w→r</sub>

```
R_w→r = | cos(α)          sin(α)          0      |
        | cos(β)*sin(α)  -cos(β)*cos(α)  -sin(β) |
        |-sin(β)*sin(α)   sin(β)*cos(α)  -cos(β) |
```

### 2.5 坐标变换

```
P_r = R_w→r · (P_w - P_radar,w)
Σ_r = R_w→r · Σ_w · R_w→r^T
```

### 2.6 斜距投影

**最短斜距**: R_min = √(Y_r² + Z_r²)

**距离向像素坐标**: r = R_min / ρ_r + N_r / 2 - R_c / ρ_r

**方位向像素坐标**: c = X_r / ρ_a + N_a / 2

### 2.7 阴影投影

阴影平面: Z_r = 0

---

## 3. 模块架构

### 3.1 高斯模型 (gaussian_model.py)

**功能**: 3D高斯分布表示

**可学习参数**:
| 参数 | 形状 | 描述 |
|------|------|------|
| means | [N, 3] | 高斯中心坐标 |
| scales | [N, 3] | 对数缩放向量 log(σ) |
| rotations | [N, 4] | 单位四元数 (w,x,y,z) |
| opacities | [N, 1] | 不透明度 (sigmoid激活) |
| sh_coeffs | [N, K] | 球谐系数 K=(degree+1)² |

**协方差矩阵计算**:
```python
# Σ = R @ S @ S^T @ R^T
scales = torch.exp(self._scales)
S = torch.diag_embed(scales)
R = Quaternion.to_rotation_matrix(self._rotations)
cov = R @ S @ S_T @ R.transpose(-1, -2)
```

### 3.2 球谐函数 (CUDA实现)

**功能**: GPU内部实时计算球谐散射强度

**散射模型**:
```
I(θ, φ) = Σ(l=0→3) Σ(m=-l→l) c_lm · Y_lm(θ, φ)
```

**Leaky ReLU激活**: 为保证梯度流畅性，负散射值通过Leaky ReLU压缩（α=0.1）

### 3.3 CUDA渲染器 (cuda_rasterizer/)

**功能**: GPU版本加速SAR渲染

**核心组件**:
- `rasterizer_autograd.py`: PyTorch autograd封装
- `rasterizer_impl_v2.cu`: CUDA kernel实现
  - `build_tile_gaussian_list_kernel`: 构建tile高斯列表
  - `render_kernel`: 渲染kernel
  - `compute_sh_scattering`: GPU内部球谐计算
  - `compute_gaussian_density_2d`: 2D高斯密度计算

---

## 4. 渲染器架构

### 4.1 渲染流程

```
GaussianModel (Python)
    ↓ 生成高斯参数
RenderPipeline.render()
    ↓ 渲染器缓存管理
SARRasterizer.forward() (CUDA)
    ↓ 1. 数据传输到GPU
    ↓ 2. build_tile_gaussian_list_kernel (GPU)
    ↓    - 坐标变换、球谐计算、密度计算
    ↓ 3. render_kernel (GPU)
    ↓    - tile内高斯并行渲染
    → 输出SAR图像
```

### 4.2 SH阶数自适应

渲染器自动检测SH阶数变化，清空缓存确保正确渲染：
- 初始阶数由GUI设置
- 训练过程中根据迭代次数/Epoch自动调整
- 阶数变化时自动清空渲染器缓存

### 4.3 核心渲染公式

渲染物理模型：

```
I(n_a, n_r) = Σ_i [ S_i(β, α) × σ_i × γ_i,ipp(n_a, n_r) × Ω_i,s(n_a, n_r) ]
```

其中:
- S_i(β, α): 球谐散射强度（Leaky ReLU激活）
- σ_i: 透射率（不透明度）
- γ_i,ipp: IPP投影密度值
- Ω_i,s: 遮挡因子 = Π(j: Z_j < Z_i) [1 - γ_j,s × σ_j]

### 4.4 渲染缓存机制

为提升训练效率，渲染器实现缓存机制：
- 按雷达视角参数（入射角、方位角）缓存渲染器
- SH阶数变化时自动清空缓存
- 支持渲染器数量上限控制

---

## 5. 核心公式对照表

### 5.1 参数计算

| 公式 | 代码位置 |
|------|----------|
| sin(β) = sin(θ) · cos(φ) | `scene.dataset_readers.py` |
| R_c = H / cos(β) | `scene.dataset_readers.py` |
| P_radar,w = (H·tan(β)·sin(α), -H·tan(β)·cos(α), H) | `scene.dataset_readers.py` |

### 5.2 坐标变换

| 公式 | 代码位置 |
|------|----------|
| R_w→r 矩阵 | `CUDA rasterizer_impl_v2.cu` |
| P_r = R_w→r · (P_w - P_radar,w) | `CUDA rasterizer_impl_v2.cu` |
| Σ_r = R_w→r · Σ_w · R_w→r^T | `CUDA rasterizer_impl_v2.cu` |

### 5.3 投影映射

| 公式 | 代码位置 |
|------|----------|
| R_min = √(Y_r² + Z_r²) | `CUDA rasterizer_impl_v2.cu` |
| r = R_min/ρ_r + N_r/2 - R_c/ρ_r | `CUDA rasterizer_impl_v2.cu` |
| c = X_r/ρ_a + N_a/2 | `CUDA rasterizer_impl_v2.cu` |

### 5.4 渲染公式

| 公式 | 代码位置 |
|------|----------|
| I = Σ T_i × α_i × S_i | `CUDA rasterizer_impl_v2.cu` |

---

## 6. 使用示例

### 6.1 GPU渲染
```python
from sar_gs.cuda_rasterizer import cuda_rasterizer_sar
import torch

# 准备GPU上的高斯数据
means_t = torch.randn(1000, 3).cuda() * 100
cov_t = torch.eye(3).unsqueeze(0).expand(1000, -1, -1).cuda() * 10
trans_t = torch.sigmoid(torch.randn(1000)).cuda()
sh_t = torch.randn(1000, 16).cuda() * 0.1

# 渲染
output = cuda_rasterizer_sar.render_sar(
    means_t, cov_t, trans_t, sh_t,
    radar_x, radar_y, radar_z,
    track_angle, incidence_angle, azimuth_angle,
    range_resolution, azimuth_resolution,
    range_samples, azimuth_samples
)
```

### 6.2 训练流程
```python
from sar_gs import GaussianModel
from sar_gs.training import TrainingPipeline
from sar_gs.losses import CombinedLoss

model = GaussianModel(sh_degree=3)
loss_fn = CombinedLoss()
pipeline = TrainingPipeline(model, loss_fn)

# 训练
for camera in dataloader:
    result = pipeline.train_step(camera)
```

---

## 7. 3DGS训练管线

### 7.1 项目结构（训练相关）
```
sar_gs_v2-temp/
├── __init__.py                     # 模块导出
├── gaussian_model.py               # 3D高斯模型
├── losses.py                      # L1 + D-SSIM损失函数
├── training_strategies.py         # 致密化/剪枝配置
├── train.py                       # 命令行训练脚本
├── train_gui.py                   # GUI训练界面
├── scene/
│   ├── __init__.py
│   └── dataset_readers.py         # MSTAR数据加载
├── training/
│   ├── __init__.py
│   ├── training_pipeline.py       # 训练管线
│   ├── render_pipeline.py         # 渲染管线（含缓存）
│   └── densify_prune.py          # 致密化/剪枝管理
└── cuda_rasterizer/               # CUDA渲染器
```

### 7.2 核心模块

#### 数据加载 (scene/dataset_readers.py)
- `RadarParams`: 雷达参数（入射角、方位角、高度、分辨率）
- `SARCameraInfo`: 单视角SAR信息
- `SARSceneDataset`: 数据集加载器，自动解析文件名参数

#### 高斯模型 (gaussian_model.py)
**可训练参数**:
| 参数 | 形状 | 学习率倍率 | 说明 |
|------|------|----------|------|
| means | [N, 3] | 1.0x | 位置坐标 |
| scales | [N, 3] | 0.01x | 对数尺寸 log(σ) |
| rotations | [N, 4] | 0.001x | 四元数旋转 |
| opacities | [N, 1] | 1.0x | 不透明度 |
| sh_coeffs | [N, K] | 0.01x | 球谐系数 |

#### 损失函数 (losses.py)
- **L1损失** (权重0.9): 像素级强度差异
- **D-SSIM损失** (权重0.1): 结构相似性差异

#### 训练策略 (training_strategies.py)
**致密化**:
- 分裂: 大尺度高斯(>split_threshold) + 大梯度 → 分裂为多个小高斯
- 克隆: 小尺度高斯(<clone_threshold) + 大梯度 → 复制到欠重建区域

**剪枝**:
- 不透明度 < opacity_threshold → 移除
- 尺度 > scale_threshold → 移除

### 7.3 渲染管线（Alpha混合）

```
渲染公式: I_final = Σ_i [ T_i × α_i × I_i ]
其中:
    - α_i = opacity_i × gaussian_density_i  (高斯密度投影值)
    - T_i = Π_{j<i} (1 - α_j)  (累积透射率)
```

**渲染流程**:
1. 高斯从雷达出发按斜距排序
2. 沿斜距方向从前向后alpha混合
3. 近距离高斯先混合，远距离后混合

### 7.4 初始化

**L的计算**:
```
L = max(距离向像素数 × 距离分辨率, 方位向像素数 × 方位分辨率)
```

**场景边界** (立方体):
- X, Y ∈ [-L/2, L/2]
- Z ∈ [0, L/4]

**参数初始化**:
- 位置: 均匀随机分布
- 尺寸: log(0.5) × 3 (各向同性)
- 旋转: 单位四元数 (1,0,0,0)
- 不透明度: sigmoid(0) = 0.5
- 球谐系数: 仅直流分量=1

### 7.5 使用方法

#### 命令行训练
```bash
python train.py --data_path <数据路径> --output_dir ./output/training
```

#### GUI训练
```bash
python train_gui.py
```

**关键参数**:
| 参数 | 默认值 | 说明 |
|------|-------|------|
| --init_num_gaussians | 5000 | 初始高斯数量 |
| --init_scale | 0.5 | 初始尺寸(对数空间) |
| --init_opacity | 0.5 | 初始不透明度 |
| --max_iterations | 50000 | 最大迭代次数 |
| --batch_size | 4 | 每步处理图像数 |
| --learning_rate | 0.001 | 位置学习率 |
| --densify_interval | 100 | 致密化间隔 |
| --prune_interval | 100 | 剪枝间隔 |

### 7.6 输出

```
output/training/
├── viz_epoch_0001/
│   ├── gaussian_distribution.png   # 高斯分布可视化
│   └── render_comparison.png       # 渲染对比图
├── checkpoint_epoch_XXXX.pth       # 中间检查点
└── final_model.pth                 # 最终模型
```

### 7.7 数据格式

**图像文件名格式**:
```
inc_{入射角}-track_{方位角}-height_{高度}-squint_{斜视角}-rr_{距离分辨率}-ar_{方位分辨率}.png
```

**示例**:
```
inc_75.0000-track_100.2248-height_10000-squint_0.0-rr_0.3-ar_0.3.png
```
