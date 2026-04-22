"""
SAR_GS 渲染性能基准测试脚本
对比 CUDA V2 Alpha 混合渲染管线在不同雷达参数下的性能
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import scipy.io
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class RadarParams:
    incidence_angle: float
    azimuth_angle: float
    track_angle: float

    def __str__(self):
        return f"Incidence={self.incidence_angle}deg, Azimuth={self.azimuth_angle}deg, Track={self.track_angle}deg"


@dataclass
class BenchmarkConfig:
    pointcloud_path: str
    cloud_scale: float = 3.0
    altitude: float = 10000.0
    azimuth_resolution: float = 3.5
    range_resolution: float = 3.5
    azimuth_samples: int = 128
    range_samples: int = 128
    output_dir: str = r"c:\Users\LIU\Desktop\gs_render\sar_gs\output\benchmark"


def load_pointcloud_from_mat(file_path: str, device: torch.device) -> torch.Tensor:
    """从.mat文件加载点云"""
    mat_data = scipy.io.loadmat(file_path)

    points = None
    for key in ['points', 'pointcloud', 'xyz', 'coords', 'data']:
        if key in mat_data:
            points = mat_data[key]
            break

    if points is None:
        for key in mat_data.keys():
            if not key.startswith('__'):
                potential = mat_data[key]
                if isinstance(potential, np.ndarray) and potential.ndim == 2 and potential.shape[1] == 3:
                    points = potential
                    break

    if points is None:
        raise ValueError(f"Cannot find valid point cloud data in .mat file: {file_path}")

    if points.shape[1] != 3:
        if points.shape[0] == 3:
            points = points.T
        else:
            raise ValueError(f"Point cloud should be [N,3], got {points.shape}")

    points = points.astype(np.float32)

    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()

    points[:, 0] = (points[:, 0] - (x_max - x_min) / 2) / 1
    points[:, 1] = (points[:, 1] - (y_max - y_min) / 2) / 1
    points[:, 2] = (points[:, 2] - z_min) / 1

    return torch.from_numpy(points).to(device)


def create_gaussian_ball_from_pointcloud(
    points: torch.Tensor,
    scale: float = 5.0,
    device: torch.device = None
):
    """从点云创建高斯分布"""
    if device is None:
        device = points.device

    points = points.to(device)
    num_points = points.shape[0]

    from sar_gs import GaussianModel

    fill_indices = torch.randint(0, num_points, (num_points // 3,))
    fill_points = points[fill_indices] + torch.randn(num_points // 3, 3, device=device) * scale * 0.5
    all_points = torch.cat([points, fill_points], dim=0)

    gaussian_model = GaussianModel(sh_degree=3)
    gaussian_model.initialize_from_points(all_points, init_scale=0.01 * scale)

    num_gaussians = gaussian_model.num_gaussians
    gaussian_model._scales.data = torch.log(torch.ones(num_gaussians, 3, device=device) * scale)
    gaussian_model._opacities.data.fill_(0.8)
    gaussian_model._sh_coeffs.data.zero_()
    gaussian_model._sh_coeffs.data[:, 0] = 1.0

    return gaussian_model


def compute_radar_position(
    altitude: float,
    track_angle: float,
    incidence_angle: float,
    azimuth_angle: float
) -> Tuple[float, float, float, float, float]:
    """计算雷达位置和相关角度参数"""
    theta_rad = np.deg2rad(incidence_angle)
    phi_rad = np.deg2rad(azimuth_angle)
    alpha = np.deg2rad(track_angle)

    sin_beta = np.sin(theta_rad) * np.cos(phi_rad)
    beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))
    tan_beta = np.tan(beta)
    Rc = altitude / np.cos(beta)

    radar_x = altitude * tan_beta * np.sin(alpha)
    radar_y = -altitude * tan_beta * np.cos(alpha)
    radar_z = altitude

    return radar_x, radar_y, radar_z, beta, Rc


def compute_ipp_coords(
    means_np: np.ndarray,
    radar_x: float,
    radar_y: float,
    radar_z: float,
    track_angle: float,
    incidence_angle: float,
    azimuth_angle: float,
    range_resolution: float,
    azimuth_resolution: float,
    range_samples: int,
    azimuth_samples: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算IPP坐标"""
    theta_rad = np.deg2rad(incidence_angle)
    phi_rad = np.deg2rad(azimuth_angle)
    alpha = np.deg2rad(track_angle)

    sin_beta = np.sin(theta_rad) * np.cos(phi_rad)
    beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))
    Rc = radar_z / np.cos(beta)

    dx = means_np[:, 0] - radar_x
    dy = means_np[:, 1] - radar_y
    dz = means_np[:, 2] - radar_z

    cos_a, sin_a = np.cos(alpha), np.sin(alpha)
    cos_b, sin_b = np.cos(beta), np.sin(beta)

    xr = cos_a * dx + sin_a * dy
    yr = cos_b * sin_a * dx - cos_b * cos_a * dy - sin_b * dz
    zr = -sin_b * sin_a * dx + sin_b * cos_a * dy - cos_b * dz

    Rmin = np.sqrt(yr**2 + zr**2)
    r_coords = Rmin / range_resolution + range_samples / 2 - Rc / range_resolution
    c_coords = xr / azimuth_resolution + azimuth_samples / 2

    valid_mask = (r_coords >= 0) & (r_coords < range_samples) & (c_coords >= 0) & (c_coords < azimuth_samples)

    return r_coords, c_coords, valid_mask


def render_cuda_v2(
    means_t: torch.Tensor,
    cov_t: torch.Tensor,
    trans_t: torch.Tensor,
    sh_t: torch.Tensor,
    radar_x: float,
    radar_y: float,
    radar_z: float,
    track_angle: float,
    incidence_angle: float,
    azimuth_angle: float,
    range_resolution: float,
    azimuth_resolution: float,
    range_samples: int,
    azimuth_samples: int,
    warmup: bool = True
) -> Tuple[torch.Tensor, float]:
    """使用CUDA V2渲染器（Alpha混合）并返回结果和时间"""
    from sar_gs.cuda_rasterizer import cuda_rasterizer_sar

    if warmup:
        _, _ = cuda_rasterizer_sar.render_sar(
            means_t[:10], cov_t[:10], trans_t[:10], sh_t[:10],
            radar_x, radar_y, radar_z,
            track_angle, incidence_angle, azimuth_angle,
            range_resolution, azimuth_resolution,
            range_samples, azimuth_samples
        )
        torch.cuda.synchronize()

    start = time.perf_counter()
    output, _ = cuda_rasterizer_sar.render_sar(
        means_t, cov_t, trans_t, sh_t,
        radar_x, radar_y, radar_z,
        track_angle, incidence_angle, azimuth_angle,
        range_resolution, azimuth_resolution,
        range_samples, azimuth_samples
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return output, elapsed


def render_cpu_v2(
    means: torch.Tensor,
    cov: torch.Tensor,
    sh: torch.Tensor,
    transmittance: torch.Tensor,
    radar_params: RadarParams,
    range_resolution: float,
    azimuth_resolution: float,
    range_samples: int,
    azimuth_samples: int,
    altitude: float = 10000.0
) -> Tuple[torch.Tensor, float]:
    """使用CPU渲染器并返回结果和时间"""
    from sar_gs import SARRenderer, SARRenderParams

    params = SARRenderParams(
        incidence_angle=radar_params.incidence_angle,
        azimuth_angle=radar_params.azimuth_angle,
        radar_altitude=altitude,
        azimuth_resolution=azimuth_resolution,
        range_resolution=range_resolution,
        azimuth_samples=azimuth_samples,
        range_samples=range_samples
    )

    renderer = SARRenderer(params)

    start = time.perf_counter()
    with torch.no_grad():
        rendered_cpu, _ = renderer(
            means,
            cov,
            sh,
            radar_position=None,
            transmittance=transmittance,
            track_angle=radar_params.track_angle,
            compute_shadow=True
        )
    elapsed = time.perf_counter() - start

    return rendered_cpu, elapsed


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results: List[dict] = []

    def setup(self) -> Tuple[torch.device, 'GaussianModel']:
        """初始化设置"""
        np.random.seed(42)
        torch.cuda.empty_cache()
        device = torch.device('cuda:0')

        print("=" * 70)
        print("SAR_GS Rendering Performance Benchmark (CUDA V2 - Alpha Blending)")
        print("=" * 70)

        print(f"\nLoading point cloud: {self.config.pointcloud_path}")
        points = load_pointcloud_from_mat(self.config.pointcloud_path, device)
        print(f"  Point count: {points.shape[0]}")

        print(f"\nCreating Gaussian distribution (scale={self.config.cloud_scale})...")
        gaussian_model = create_gaussian_ball_from_pointcloud(
            points,
            scale=self.config.cloud_scale,
            device=device
        )
        print(f"  Gaussian count: {gaussian_model.num_gaussians}")

        os.makedirs(self.config.output_dir, exist_ok=True)

        return device, gaussian_model

    def run_single_radar_config(
        self,
        device: torch.device,
        gaussian_model: 'GaussianModel',
        radar_params: RadarParams,
        render_mode: str = 'gpu',
        num_iterations: int = 10
    ) -> dict:
        """运行单个雷达配置的性能测试"""
        means = gaussian_model._means
        cov_full = gaussian_model.compute_covariance_full()
        transmittance = gaussian_model.get_opacity().squeeze(-1)
        sh = gaussian_model.get_active_sh_coeffs()

        cov_3x3 = cov_full.detach().cpu().numpy()
        cov_6 = cov_3x3[:, [0, 1, 2, 0, 0, 1], [0, 1, 2, 1, 2, 2]].copy()
        means_np = means.detach().cpu().numpy()

        radar_x, radar_y, radar_z, beta, Rc = compute_radar_position(
            self.config.altitude,
            radar_params.track_angle,
            radar_params.incidence_angle,
            radar_params.azimuth_angle
        )

        r_coords, c_coords, valid_mask = compute_ipp_coords(
            means_np, radar_x, radar_y, radar_z,
            radar_params.track_angle, radar_params.incidence_angle, radar_params.azimuth_angle,
            self.config.range_resolution, self.config.azimuth_resolution,
            self.config.range_samples, self.config.azimuth_samples
        )

        print(f"\n{'=' * 70}")
        print(f"Radar Config: {radar_params}")
        print(f"{'=' * 70}")
        print(f"  Valid Gaussians: {valid_mask.sum()} / {means_np.shape[0]}")
        print(f"  r range: [{r_coords.min():.2f}, {r_coords.max():.2f}]")
        print(f"  c range: [{c_coords.min():.2f}, {c_coords.max():.2f}]")

        means_t = torch.from_numpy(means_np).float().to(device)
        cov_t = torch.from_numpy(cov_6).float().to(device)
        trans_t = torch.from_numpy(transmittance.detach().cpu().numpy()).float().to(device)
        sh_t = torch.from_numpy(sh.detach().cpu().numpy()).float().to(device)

        result = {
            'params': radar_params,
            'valid_gaussians': valid_mask.sum(),
            'renders': []
        }

        if render_mode in ['gpu', 'both']:
            print(f"\n  [GPU Rendering] ({num_iterations} iterations)")
            gpu_times = []
            output_cuda = None

            for i in range(num_iterations):
                output_cuda, t = render_cuda_v2(
                    means_t, cov_t, trans_t, sh_t,
                    radar_x, radar_y, radar_z,
                    radar_params.track_angle,
                    radar_params.incidence_angle,
                    radar_params.azimuth_angle,
                    self.config.range_resolution,
                    self.config.azimuth_resolution,
                    self.config.range_samples,
                    self.config.azimuth_samples,
                    warmup=(i == 0)
                )
                gpu_times.append(t * 1000)
                if i == num_iterations - 1:
                    output_np = output_cuda.cpu().numpy()
                    print(f"    Output range: [{output_np.min():.6f}, {output_np.max():.6f}]")
                    print(f"    Non-zero pixels: {np.count_nonzero(output_np)}")

            avg_gpu_time = np.mean(gpu_times)
            std_gpu_time = np.std(gpu_times)
            print(f"    Avg time: {avg_gpu_time:.2f} ms (std: {std_gpu_time:.2f} ms)")

            result['time_cuda_avg'] = avg_gpu_time / 1000
            result['time_cuda_std'] = std_gpu_time / 1000
            result['cuda_output'] = output_cuda

            cuda_filename = f"GPU_inc{radar_params.incidence_angle:.1f}_az{radar_params.azimuth_angle:.1f}_tr{radar_params.track_angle:.1f}.png"
            cuda_path = os.path.join(self.config.output_dir, cuda_filename)
            img = Image.fromarray((np.clip(output_np / (output_np.max() + 1e-8) * 255, 0, 255)).astype(np.uint8))
            img.save(cuda_path)
            print(f"    Saved: {cuda_filename}")

        if render_mode in ['cpu', 'both']:
            print(f"\n  [CPU Rendering] ({num_iterations} iterations)")
            cpu_times = []
            output_cpu = None

            for i in range(num_iterations):
                output_cpu, t = render_cpu_v2(
                    means, cov_full, sh, transmittance, radar_params,
                    self.config.range_resolution,
                    self.config.azimuth_resolution,
                    self.config.range_samples,
                    self.config.azimuth_samples,
                    self.config.altitude
                )
                cpu_times.append(t * 1000)
                if i == num_iterations - 1:
                    output_np = output_cpu.cpu().numpy()
                    print(f"    Output range: [{output_np.min():.6f}, {output_np.max():.6f}]")
                    print(f"    Non-zero pixels: {np.count_nonzero(output_np)}")

            avg_cpu_time = np.mean(cpu_times)
            std_cpu_time = np.std(cpu_times)
            print(f"    Avg time: {avg_cpu_time:.2f} ms (std: {std_cpu_time:.2f} ms)")

            result['time_cpu_avg'] = avg_cpu_time / 1000
            result['time_cpu_std'] = std_cpu_time / 1000
            result['cpu_output'] = output_cpu

            cpu_filename = f"CPU_inc{radar_params.incidence_angle:.1f}_az{radar_params.azimuth_angle:.1f}_tr{radar_params.track_angle:.1f}.png"
            cpu_path = os.path.join(self.config.output_dir, cpu_filename)
            img = Image.fromarray((np.clip(output_np / (output_np.max() + 1e-8) * 255, 0, 255)).astype(np.uint8))
            img.save(cpu_path)
            print(f"    Saved: {cpu_filename}")

        if render_mode == 'both':
            speedup = result['time_cpu_avg'] / result['time_cuda_avg']
            print(f"\n  [Performance Comparison]")
            print(f"    Speedup: {speedup:.2f}x")
            result['speedup'] = speedup

        return result

    def run_benchmark(
        self,
        radar_configs: List[RadarParams],
        render_mode: str = 'gpu',
        num_iterations: int = 10
    ):
        """运行完整的基准测试"""
        device, gaussian_model = self.setup()

        self.results = []
        for params in radar_configs:
            result = self.run_single_radar_config(
                device, gaussian_model, params, render_mode, num_iterations
            )
            self.results.append(result)

        self.print_summary(render_mode)
        self.save_summary(render_mode)

    def print_summary(self, render_mode: str):
        """打印汇总结果"""
        print(f"\n{'=' * 70}")
        print("Summary Results")
        print(f"{'=' * 70}")

        if render_mode == 'both':
            print(f"{'Config':<50} {'GPU (ms)':>12} {'CPU (ms)':>12} {'Speedup':>10}")
            print(f"{'-' * 50} {'-' * 12} {'-' * 12} {'-' * 10}")
            for r in self.results:
                params_str = str(r['params'])
                print(f"{params_str:<50} {r['time_cuda_avg']*1000:>11.2f} {r['time_cpu_avg']*1000:>11.2f} {r['speedup']:>9.2f}x")
            avg_speedup = np.mean([r['speedup'] for r in self.results])
            total_cuda = sum(r['time_cuda_avg'] for r in self.results)
            total_cpu = sum(r['time_cpu_avg'] for r in self.results)
            print(f"{'-' * 50} {'-' * 12} {'-' * 12} {'-' * 10}")
            print(f"{'Total/Avg':<50} {total_cuda*1000:>11.2f} {total_cpu*1000:>11.2f} {avg_speedup:>9.2f}x")

        elif render_mode == 'gpu':
            print(f"{'Config':<50} {'GPU (ms)':>12}")
            print(f"{'-' * 50} {'-' * 12}")
            for r in self.results:
                params_str = str(r['params'])
                print(f"{params_str:<50} {r['time_cuda_avg']*1000:>11.2f}")
            total_cuda = sum(r['time_cuda_avg'] for r in self.results)
            avg_cuda = np.mean([r['time_cuda_avg'] for r in self.results])
            print(f"{'-' * 50} {'-' * 12}")
            print(f"{'Total/Avg':<50} {total_cuda*1000:>11.2f}")

        elif render_mode == 'cpu':
            print(f"{'Config':<50} {'CPU (ms)':>12}")
            print(f"{'-' * 50} {'-' * 12}")
            for r in self.results:
                params_str = str(r['params'])
                print(f"{params_str:<50} {r['time_cpu_avg']*1000:>11.2f}")
            total_cpu = sum(r['time_cpu_avg'] for r in self.results)
            avg_cpu = np.mean([r['time_cpu_avg'] for r in self.results])
            print(f"{'-' * 50} {'-' * 12}")
            print(f"{'Total/Avg':<50} {total_cpu*1000:>11.2f}")

    def save_summary(self, render_mode: str):
        """保存汇总结果到文件"""
        summary_path = os.path.join(self.config.output_dir, 'benchmark_summary.txt')

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"SAR_GS Rendering Performance Benchmark - Mode: {render_mode.upper()}\n")
            f.write("CUDA V2 - Alpha Blending Rendering Pipeline\n")
            f.write("=" * 70 + "\n\n")

            if render_mode == 'both':
                f.write(f"{'Config':<50} {'GPU (ms)':>12} {'CPU (ms)':>12} {'Speedup':>10}\n")
                f.write(f"{'-' * 50} {'-' * 12} {'-' * 12} {'-' * 10}\n")
                for r in self.results:
                    params_str = str(r['params'])
                    f.write(f"{params_str:<50} {r['time_cuda_avg']*1000:>11.2f} {r['time_cpu_avg']*1000:>11.2f} {r['speedup']:>9.2f}x\n")
                avg_speedup = np.mean([r['speedup'] for r in self.results])
                total_cuda = sum(r['time_cuda_avg'] for r in self.results)
                total_cpu = sum(r['time_cpu_avg'] for r in self.results)
                f.write(f"{'-' * 50} {'-' * 12} {'-' * 12} {'-' * 10}\n")
                f.write(f"{'Total/Avg':<50} {total_cuda*1000:>11.2f} {total_cpu*1000:>11.2f} {avg_speedup:>9.2f}x\n")

            elif render_mode == 'gpu':
                f.write(f"{'Config':<50} {'GPU (ms)':>12}\n")
                f.write(f"{'-' * 50} {'-' * 12}\n")
                for r in self.results:
                    params_str = str(r['params'])
                    f.write(f"{params_str:<50} {r['time_cuda_avg']*1000:>11.2f}\n")
                total_cuda = sum(r['time_cuda_avg'] for r in self.results)
                avg_cuda = np.mean([r['time_cuda_avg'] for r in self.results])
                f.write(f"{'-' * 50} {'-' * 12}\n")
                f.write(f"{'Total/Avg':<50} {total_cuda*1000:>11.2f}\n")

            elif render_mode == 'cpu':
                f.write(f"{'Config':<50} {'CPU (ms)':>12}\n")
                f.write(f"{'-' * 50} {'-' * 12}\n")
                for r in self.results:
                    params_str = str(r['params'])
                    f.write(f"{params_str:<50} {r['time_cpu_avg']*1000:>11.2f}\n")
                total_cpu = sum(r['time_cpu_avg'] for r in self.results)
                avg_cpu = np.mean([r['time_cpu_avg'] for r in self.results])
                f.write(f"{'-' * 50} {'-' * 12}\n")
                f.write(f"{'Total/Avg':<50} {total_cpu*1000:>11.2f}\n")

        print(f"\nSummary saved to: {summary_path}")


def run_default_benchmark():
    """运行默认配置的基准测试"""
    DEFAULT_CONFIG = BenchmarkConfig(
        pointcloud_path=r'C:\Users\LIU\Desktop\CAD舰船下采样点云\052D点云(1529个).mat',
        cloud_scale=3,
        altitude=10000.0,
        azimuth_resolution=3.5,
        range_resolution=3.5,
        azimuth_samples=128,
        range_samples=128,
        output_dir=r'c:\Users\LIU\Desktop\gs_render\sar_gs\output\benchmark'
    )

    DEFAULT_RADAR_CONFIGS = [
        RadarParams(incidence_angle=30.0, azimuth_angle=0.0, track_angle=0),
        RadarParams(incidence_angle=30.0, azimuth_angle=0.0, track_angle=25),
        RadarParams(incidence_angle=30.0, azimuth_angle=0.0, track_angle=50),
        RadarParams(incidence_angle=30.0, azimuth_angle=0.0, track_angle=75),
    ]

    RENDER_MODE = 'gpu'
    NUM_ITERATIONS = 10

    runner = BenchmarkRunner(DEFAULT_CONFIG)
    runner.run_benchmark(
        radar_configs=DEFAULT_RADAR_CONFIGS,
        render_mode=RENDER_MODE,
        num_iterations=NUM_ITERATIONS
    )


if __name__ == '__main__':
    run_default_benchmark()