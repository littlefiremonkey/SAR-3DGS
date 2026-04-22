"""
SAR-GS主训练脚本

从多视角SAR图像进行3DGS三维重建

使用方法:
    python train.py --data_path <数据路径> --output_dir <输出目录>
"""

import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from pathlib import Path
from typing import List, Tuple, Dict

from scene.dataset_readers import SARSceneDataset, compute_scene_bounds_from_dataset
from gaussian_model import GaussianModel
from losses import CombinedLoss
from training_strategies import get_default_training_config, DensifyConfig, PruneConfig

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True

    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:
    HAS_MATPLOTLIB = False


def parse_args():
    parser = argparse.ArgumentParser(description='SAR-GS训练')
    parser.add_argument('--data_path', type=str, required=True,
                        help='SAR图像数据路径')
    parser.add_argument('--output_dir', type=str, default='./output/training',
                        help='输出目录')
    parser.add_argument('--max_iterations', type=int, default=50000,
                        help='最大迭代次数')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='每个epoch处理的图像数量')
    parser.add_argument('--init_num_gaussians', type=int, default=5000,
                        help='初始高斯数量')
    parser.add_argument('--init_scale', type=float, default=0.5,
                        help='初始高斯尺度')
    parser.add_argument('--init_opacity', type=float, default=0.5,
                        help='初始高斯不透明度')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--grad_threshold', type=float, default=0.0001,
                        help='致密化梯度阈值')
    parser.add_argument('--clone_threshold', type=float, default=0.01,
                        help='克隆阈值(小尺度)')
    parser.add_argument('--split_threshold', type=float, default=2.0,
                        help='分裂阈值(大尺度)')
    parser.add_argument('--opacity_threshold', type=float, default=0.05,
                        help='剪枝不透明度阈值')
    parser.add_argument('--densify_interval', type=int, default=100,
                        help='致密化间隔')
    parser.add_argument('--prune_interval', type=int, default=100,
                        help='剪枝间隔')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='日志输出间隔')
    parser.add_argument('--save_interval', type=int, default=5000,
                        help='保存模型间隔')
    parser.add_argument('--viz_interval', type=int, default=1,
                        help='可视化输出间隔(epoch)')
    parser.add_argument('--viz_num_views', type=int, default=8,
                        help='对比可视化使用的视角数量')
    parser.add_argument('--device', type=str, default='cuda',
                        help='计算设备')
    return parser.parse_args()


def create_densify_and_prune_configs(args):
    densify_config = DensifyConfig(
        enabled=True,
        interval=args.densify_interval,
        start_iter=500,
        grad_threshold=args.grad_threshold,
        clone_threshold=args.clone_threshold,
        split_threshold=args.split_threshold
    )

    prune_config = PruneConfig(
        enabled=True,
        interval=args.prune_interval,
        start_iter=500,
        opacity_threshold=args.opacity_threshold
    )

    return densify_config, prune_config


def render_single_view(
    gaussian_model: GaussianModel,
    camera,
    device: torch.device,
    renderer=None
) -> torch.Tensor:
    """渲染单个视角

    Args:
        gaussian_model: 高斯模型
        camera: 相机视角
        device: 计算设备
        renderer: 可选的预创建渲染器，如果为None则创建新的

    Returns:
        torch.Tensor: 渲染图像
    """
    means = gaussian_model._means
    cov_full = gaussian_model.compute_covariance_full()
    opacity = gaussian_model.get_opacity()
    sh_coeffs = gaussian_model.get_active_sh_coeffs()

    radar_params = camera.radar_params
    theta_rad = np.deg2rad(radar_params.incidence_angle)
    phi_rad = np.deg2rad(radar_params.azimuth_angle)
    sin_beta = np.sin(theta_rad) * np.cos(phi_rad)
    beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))
    tan_beta = np.tan(beta)
    alpha = np.deg2rad(radar_params.track_angle)
    radar_altitude = radar_params.radar_altitude

    radar_x = radar_altitude * tan_beta * np.sin(alpha)
    radar_y = -radar_altitude * tan_beta * np.cos(alpha)
    radar_z = radar_altitude

    if renderer is None:
        from cuda_rasterizer.rasterizer_autograd import SARRasterizer
        renderer = SARRasterizer(
            radar_x=radar_x,
            radar_y=radar_y,
            radar_z=radar_z,
            track_angle=radar_params.track_angle,
            incidence_angle=radar_params.incidence_angle,
            azimuth_angle=radar_params.azimuth_angle,
            range_resolution=radar_params.range_resolution,
            azimuth_resolution=radar_params.azimuth_resolution,
            range_samples=128,
            azimuth_samples=128
        ).to(device)

    rendered = renderer(means, cov_full, opacity.squeeze(-1), sh_coeffs)
    return rendered


def render_view_for_saving(
    gaussian_model: GaussianModel,
    camera,
    device: torch.device
) -> Tuple[torch.Tensor, Dict]:
    """渲染单个视角用于保存（带完整参数）

    Args:
        gaussian_model: 高斯模型
        camera: 相机视角
        device: 计算设备

    Returns:
        Tuple[渲染图像, 相机信息字典]
    """
    from cuda_rasterizer.rasterizer_autograd import SARRasterizer

    means = gaussian_model._means
    cov_full = gaussian_model.compute_covariance_full()
    opacity = gaussian_model.get_opacity()
    sh_coeffs = gaussian_model.get_active_sh_coeffs()

    radar_params = camera.radar_params
    theta_rad = np.deg2rad(radar_params.incidence_angle)
    phi_rad = np.deg2rad(radar_params.azimuth_angle)
    sin_beta = np.sin(theta_rad) * np.cos(phi_rad)
    beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))
    tan_beta = np.tan(beta)
    alpha = np.deg2rad(radar_params.track_angle)
    radar_altitude = radar_params.radar_altitude

    radar_x = radar_altitude * tan_beta * np.sin(alpha)
    radar_y = -radar_altitude * tan_beta * np.cos(alpha)
    radar_z = radar_altitude

    renderer = SARRasterizer(
        radar_x=radar_x,
        radar_y=radar_y,
        radar_z=radar_z,
        track_angle=radar_params.track_angle,
        incidence_angle=radar_params.incidence_angle,
        azimuth_angle=radar_params.azimuth_angle,
        range_resolution=radar_params.range_resolution,
        azimuth_resolution=radar_params.azimuth_resolution,
        range_samples=128,
        azimuth_samples=128
    ).to(device)

    rendered = renderer(means, cov_full, opacity.squeeze(-1), sh_coeffs)

    camera_info = {
        'radar_x': radar_x,
        'radar_y': radar_y,
        'radar_z': radar_z,
        'incidence_angle': radar_params.incidence_angle,
        'azimuth_angle': radar_params.azimuth_angle,
        'range_resolution': radar_params.range_resolution,
        'azimuth_resolution': radar_params.azimuth_resolution,
        'range_samples': 128,
        'azimuth_samples': 128,
    }

    return rendered, camera_info


def visualize_gaussian_distribution(
    gaussian_model: GaussianModel,
    output_path: Path,
    title: str = "高斯分布可视化"
):
    """可视化高斯分布的位置、不透明度、尺寸

    包含:
    - 3D视角下高斯分布（颜色表示不透明度）
    - 3D视角下高斯分布（颜色表示尺寸）
    - 不透明度分布直方图
    - 尺寸分布直方图

    Args:
        gaussian_model: 高斯模型
        output_path: 输出路径
        title: 标题
    """
    if not HAS_MATPLOTLIB:
        print("  警告: matplotlib未安装，跳过高斯分布可视化")
        return

    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("  警告: mpl_toolkits未安装，跳过高斯分布可视化")
        return

    means = gaussian_model.means.detach().cpu().numpy()
    scales = torch.exp(gaussian_model._scales).detach().cpu().numpy()
    opacities = torch.sigmoid(gaussian_model._opacities).detach().cpu().numpy()
    scale_norms = np.linalg.norm(scales, axis=1)
    total_gaussians = gaussian_model.num_gaussians

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{title}\n高斯总数: {total_gaussians}', fontsize=14)

    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    scatter1 = ax1.scatter(means[:, 0], means[:, 1], means[:, 2],
                           c=opacities[:, 0], s=2, cmap='viridis', alpha=0.6)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D视图 (颜色=不透明度)')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, label='Opacity')

    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    scatter2 = ax2.scatter(means[:, 0], means[:, 1], means[:, 2],
                           c=scale_norms, s=2, cmap='plasma', alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D视图 (颜色=尺寸)')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, label='Scale Norm')

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(opacities[:, 0], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('不透明度')
    ax3.set_ylabel('数量')
    ax3.set_title(f'不透明度分布 (总数: {total_gaussians})')
    ax3.axvline(x=np.mean(opacities[:, 0]), color='red', linestyle='--',
                label=f'均值: {np.mean(opacities[:, 0]):.3f}')
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(scale_norms, bins=50, color='darkorange', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('尺寸')
    ax4.set_ylabel('数量')
    ax4.set_title(f'尺寸分布 (总数: {total_gaussians})')
    ax4.axvline(x=np.mean(scale_norms), color='red', linestyle='--',
                label=f'均值: {np.mean(scale_norms):.3f}')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  已保存高斯分布可视化: {output_path}")


def save_epoch_renders(
    gaussian_model: GaussianModel,
    dataset: SARSceneDataset,
    output_dir: Path,
    epoch: int,
    device: torch.device
):
    """保存当前epoch的渲染结果图像

    每个视角的图像包含：真实图像、渲染图像、叠加对比

    Args:
        gaussian_model: 高斯模型
        dataset: 数据集
        output_dir: 输出目录
        epoch: 当前epoch
        device: 计算设备
    """
    if not HAS_MATPLOTLIB:
        print("  警告: matplotlib未安装，跳过渲染图像保存")
        return

    gaussian_model.eval()
    all_cameras = list(range(len(dataset)))
    num_to_render = min(10, len(all_cameras))

    if len(all_cameras) <= num_to_render:
        selected_indices = all_cameras
    else:
        step = len(all_cameras) / num_to_render
        selected_indices = [int(i * step) for i in range(num_to_render)]

    with torch.no_grad():
        for selected_idx in selected_indices:
            camera = dataset.get_camera(selected_idx)
            target_image = camera.image.cpu().numpy()

            rendered_image, _ = render_view_for_saving(gaussian_model, camera, device)
            rendered_image = rendered_image.detach().cpu().numpy()

            if target_image.shape != rendered_image.shape:
                from scipy.ndimage import zoom
                zoom_factors = (target_image.shape[0] / rendered_image.shape[0],
                              target_image.shape[1] / rendered_image.shape[1])
                rendered_image = zoom(rendered_image, zoom_factors)

            target_image_min = target_image.min()
            target_image_max = target_image.max()
            if target_image_max > target_image_min:
                target_image_norm = (target_image - target_image_min) / (target_image_max - target_image_min)
            else:
                target_image_norm = target_image

            rendered_image_norm = np.clip(rendered_image, 0, 1)
            overlay = np.clip(target_image_norm + rendered_image_norm, 0, 1)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(f'Epoch {epoch:04d} - 视角 {selected_idx} (入射角: {camera.incidence_angle:.1f}°, 方位角: {camera.azimuth_angle:.1f}°)', fontsize=12)

            axes[0].imshow(target_image_norm, cmap='gray', vmin=0, vmax=1)
            axes[0].set_title('真实图像')
            axes[0].axis('off')

            axes[1].imshow(rendered_image_norm, cmap='gray', vmin=0, vmax=1)
            axes[1].set_title('渲染图像')
            axes[1].axis('off')

            axes[2].imshow(overlay, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('叠加对比')
            axes[2].axis('off')

            plt.tight_layout()
            save_path = output_dir / f"epoch_{epoch:04d}_view_{selected_idx:02d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    gaussian_model.train()
    print(f"  已保存渲染图像到: {output_dir}")


def get_visualization_indices(
    dataset: SARSceneDataset,
    num_views: int
) -> List[int]:
    """获取用于可视化的视角索引

    Args:
        dataset: 数据集
        num_views: 视角数量

    Returns:
        List[int]: 均匀分布的视角索引
    """
    num_cameras = len(dataset)
    if num_cameras <= num_views:
        return list(range(num_cameras))

    step = num_cameras / num_views
    indices = [int(i * step) for i in range(num_views)]
    return indices


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print(f"\n加载数据集: {args.data_path}")
    dataset = SARSceneDataset(args.data_path, load_images=True)
    print(f"  图像数量: {len(dataset)}")
    print(f"  场景归一化: {dataset.scene_normalization}")

    scene_bounds = compute_scene_bounds_from_dataset(dataset, z_ratio=0.25)
    print(f"  场景边界: {scene_bounds}")

    print(f"\n初始化高斯模型...")
    gaussian_model = GaussianModel(
        sh_degree=3,
        init_num_gaussians=args.init_num_gaussians
    ).to(device)

    gaussian_model.initialize_random(
        scene_bounds=scene_bounds,
        num_gaussians=args.init_num_gaussians,
        init_scale=args.init_scale,
        init_opacity=args.init_opacity,
        device=device
    )
    print(f"  初始高斯数量: {gaussian_model.num_gaussians}")

    loss_fn = CombinedLoss(l1_weight=0.9, ssim_weight=0.1).to(device)

    optimizer = optim.Adam([
        {'params': [gaussian_model._means], 'lr': args.learning_rate},
        {'params': [gaussian_model._scales], 'lr': args.learning_rate * 0.01},
        {'params': [gaussian_model._rotations], 'lr': args.learning_rate * 0.001},
        {'params': [gaussian_model._opacities], 'lr': args.learning_rate},
        {'params': [gaussian_model._sh_coeffs], 'lr': args.learning_rate * 0.01},
    ])

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.densify_interval * 10,
        gamma=0.5
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    render_output_dir = output_dir / 'renders'
    render_output_dir.mkdir(parents=True, exist_ok=True)

    viz_output_dir = output_dir / 'visualizations'
    viz_output_dir.mkdir(parents=True, exist_ok=True)

    viz_indices = get_visualization_indices(dataset, args.viz_num_views)
    print(f"\n可视化使用的视角索引: {viz_indices}")

    from cuda_rasterizer.rasterizer_autograd import SARRasterizer
    theta_rad = np.deg2rad(30.0)
    phi_rad = np.deg2rad(0.0)
    sin_beta = np.sin(theta_rad) * np.cos(phi_rad)
    beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))
    tan_beta = np.tan(beta)
    alpha = np.deg2rad(0.0)
    radar_altitude = 10000.0

    radar_x = radar_altitude * tan_beta * np.sin(alpha)
    radar_y = -radar_altitude * tan_beta * np.cos(alpha)
    radar_z = radar_altitude

    renderer = SARRasterizer(
        radar_x=radar_x,
        radar_y=radar_y,
        radar_z=radar_z,
        track_angle=0.0,
        incidence_angle=30.0,
        azimuth_angle=0.0,
        range_resolution=0.3,
        azimuth_resolution=0.3,
        range_samples=128,
        azimuth_samples=128
    ).to(device)

    print(f"\n开始训练...")
    iteration = 0
    epoch = 0
    loss_history = []

    densify_config, prune_config = create_densify_and_prune_configs(args)

    while iteration < args.max_iterations:
        epoch += 1
        epoch_indices = np.random.permutation(len(dataset))

        for batch_start in range(0, len(epoch_indices), args.batch_size):
            batch_indices = epoch_indices[batch_start:batch_start + args.batch_size]

            for idx in batch_indices:
                if iteration >= args.max_iterations:
                    break

                camera = dataset.get_camera(idx)
                target_image = camera.image.to(device)

                radar_params = camera.radar_params
                renderer.radar_x = 0.0
                renderer.radar_y = -radar_params.radar_altitude * np.tan(np.pi/2 - np.deg2rad(radar_params.incidence_angle))
                renderer.radar_z = radar_params.radar_altitude
                renderer.track_angle = radar_params.track_angle
                renderer.incidence_angle = radar_params.incidence_angle
                renderer.azimuth_angle = radar_params.azimuth_angle
                renderer.range_resolution = radar_params.range_resolution
                renderer.azimuth_resolution = radar_params.azimuth_resolution
                renderer.range_samples = 128
                renderer.azimuth_samples = 128

                rendered_image = render_single_view(gaussian_model, camera, device, renderer)

                if rendered_image.shape != target_image.shape:
                    rendered_image = torch.nn.functional.interpolate(
                        rendered_image.unsqueeze(0).unsqueeze(0),
                        size=target_image.shape,
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()

                loss, loss_dict = loss_fn(rendered_image, target_image)

                optimizer.zero_grad()
                loss.backward()

                with torch.no_grad():
                    grad_means = gaussian_model._means.grad
                    grad_scales = gaussian_model._scales.grad
                    grad_opacities = gaussian_model._opacities.grad

                    if iteration > prune_config.start_iter and iteration % prune_config.interval == 0:
                        gaussian_model.densify_and_prune(
                            grad_means=grad_means,
                            grad_scales=grad_scales,
                            grad_opacities=grad_opacities,
                            grad_threshold=densify_config.grad_threshold,
                            size_threshold=densify_config.clone_threshold,
                            large_scale_threshold=densify_config.split_threshold,
                            opacity_threshold=prune_config.opacity_threshold
                        )

                    if iteration > args.densify_interval * 20 and iteration % 3000 == 0:
                        gaussian_model._opacities.data.fill_(args.init_opacity)

                optimizer.step()
                scheduler.step()

                if iteration % args.log_interval == 0:
                    print(f"  Epoch {epoch:04d} Iter {iteration:06d} | Loss: {loss_dict['total']:.6f} "
                          f"(L1: {loss_dict['l1']:.6f}, DSSIM: {loss_dict['dssim']:.6f}) | "
                          f"Gaussians: {gaussian_model.num_gaussians}")

                loss_history.append(loss_dict['total'])
                iteration += 1

        if epoch % args.viz_interval == 0:
            print(f"\n  Epoch {epoch:04d} 可视化输出:")

            gaussian_viz_path = viz_output_dir / f'gaussian_distribution_epoch_{epoch:04d}.png'
            visualize_gaussian_distribution(
                gaussian_model,
                gaussian_viz_path,
                title=f"Epoch {epoch} - 高斯分布"
            )

            save_epoch_renders(gaussian_model, dataset, render_output_dir, epoch, device)

        if epoch % args.save_interval == 0 and epoch > 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'iteration': iteration,
                'means': gaussian_model.means.data,
                'scales': gaussian_model.scales.data,
                'rotations': gaussian_model.rotations.data,
                'opacities': gaussian_model.opacities.data,
                'sh_coeffs': gaussian_model.sh_coeffs.data,
                'loss_history': loss_history,
            }, checkpoint_path)
            print(f"    已保存检查点: {checkpoint_path}")

    final_path = output_dir / 'final_model.pth'
    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'means': gaussian_model.means.data,
        'scales': gaussian_model.scales.data,
        'rotations': gaussian_model.rotations.data,
        'opacities': gaussian_model.opacities.data,
        'sh_coeffs': gaussian_model.sh_coeffs.data,
        'loss_history': loss_history,
    }, final_path)
    print(f"\n训练完成! 模型已保存: {final_path}")


if __name__ == '__main__':
    args = parse_args()
    train(args)