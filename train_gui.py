"""
SAR-GS训练GUI界面

使用方法:
    python train_gui.py
"""

import os
import sys
import threading
import time
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from typing import Optional

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    HAS_MATPLOTLIB = True

    matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:
    HAS_MATPLOTLIB = False

import numpy as np
import torch
import torch.optim as optim

from scene.dataset_readers import SARSceneDataset, compute_scene_bounds_from_dataset
from gaussian_model import GaussianModel
from losses import CombinedLoss
from training_strategies import DensifyConfig, PruneConfig

try:
    from cuda_rasterizer import cuda_rasterizer_sar
except ImportError:
    cuda_rasterizer_sar = None


class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tooltip, text=self.text, justify=tk.LEFT,
                        background="#ffffe0", relief=tk.SOLID, borderwidth=1)
        label.pack()

    def hide_tip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class LogText(scrolledtext.ScrolledText):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.configure(state='disabled')

    def append(self, text):
        self.configure(state='normal')
        self.insert(tk.END, text)
        self.see(tk.END)
        self.configure(state='disabled')


class SARGSTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("SAR-GS 训练界面")
        self.root.geometry("1100x800")

        self.training_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_training = False

        self._setup_styles()
        self._create_widgets()
        self._bind_shortcuts()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')

        style.configure('Title.TLabel', font=('Microsoft YaHei', 12, 'bold'))
        style.configure('Section.TLabelframe', font=('Microsoft YaHei', 10, 'bold'))
        style.configure('Section.TLabelframe.Label', font=('Microsoft YaHei', 10, 'bold'))

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        self.training_tab = ttk.Frame(notebook)
        self.visualization_tab = ttk.Frame(notebook)
        notebook.add(self.training_tab, text='训练参数')
        notebook.add(self.visualization_tab, text='可视化预览')

        self._create_training_tab()
        self._create_visualization_tab()

        self.button_frame = ttk.Frame(main_frame)
        self.button_frame.pack(fill=tk.X, pady=(10, 0))

        self.start_button = ttk.Button(
            self.button_frame, text="开始训练", command=self.start_training, style='Accent.TButton'
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.continue_button = ttk.Button(
            self.button_frame, text="继续训练", command=self.continue_training
        )
        self.continue_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            self.button_frame, text="停止训练", command=self.stop_training, state='disabled'
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.status_label = ttk.Label(self.button_frame, text="就绪", foreground='green')
        self.status_label.pack(side=tk.LEFT, padx=20)

    def _create_training_tab(self):
        canvas = tk.Canvas(self.training_tab)
        scrollbar = ttk.Scrollbar(self.training_tab, orient="vertical", command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        scroll_frame.columnconfigure(0, weight=1)

        self.log_frame = ttk.LabelFrame(scroll_frame, text='训练日志', padding="5")
        self.log_frame.grid(row=0, column=1, rowspan=20, sticky='nsew', padx=10, pady=5)

        self.log_text = LogText(self.log_frame, height=20, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        row = 1

        path_frame = ttk.LabelFrame(scroll_frame, text='路径设置', padding="10")
        path_frame.grid(row=row, column=0, sticky='ew', padx=10, pady=5)
        row += 1

        ttk.Label(path_frame, text="数据路径:").grid(row=0, column=0, sticky='w', pady=2)
        self.data_path_var = tk.StringVar(value='C:/Users/LIU/Desktop/gs_render/sar_gs_v2-temp/data/MSTAR/17PNG_TRAIN/T-72 - 副本')
        ttk.Entry(path_frame, textvariable=self.data_path_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(path_frame, text="浏览...", command=lambda: self.browse_folder(self.data_path_var)).grid(row=0, column=2)

        ttk.Label(path_frame, text="输出目录:").grid(row=1, column=0, sticky='w', pady=2)
        self.output_dir_var = tk.StringVar(value='C:/Users/LIU/Desktop/gs_render/sar_gs_v2-temp/output/training')
        ttk.Entry(path_frame, textvariable=self.output_dir_var, width=50).grid(row=1, column=1, padx=5)
        ttk.Button(path_frame, text="浏览...", command=lambda: self.browse_folder(self.output_dir_var)).grid(row=1, column=2)

        init_frame = ttk.LabelFrame(scroll_frame, text='高斯初始化参数', padding="10")
        init_frame.grid(row=row, column=0, sticky='ew', padx=10, pady=5)
        row += 1

        self.init_num_gaussians_var = tk.IntVar(value=50000)
        self.init_scale_var = tk.DoubleVar(value=0.3)
        self.init_opacity_var = tk.DoubleVar(value=0.3)

        self.init_num_gaussians_entry = ttk.Entry(init_frame, textvariable=self.init_num_gaussians_var, width=15)
        self.init_num_gaussians_entry.grid(row=0, column=1, sticky='w', padx=5)
        ttk.Label(init_frame, text="初始高斯数量:").grid(row=0, column=0, sticky='w', pady=2)
        ToolTip(self.init_num_gaussians_entry, "初始化时的高斯分布数量")

        self.init_scale_entry = ttk.Entry(init_frame, textvariable=self.init_scale_var, width=15)
        self.init_scale_entry.grid(row=0, column=3, sticky='w', padx=5)
        ttk.Label(init_frame, text="初始尺寸:").grid(row=0, column=2, sticky='w', pady=2, padx=(20, 0))
        ToolTip(self.init_scale_entry, "高斯的初始尺度（对数空间）")

        self.init_opacity_entry = ttk.Entry(init_frame, textvariable=self.init_opacity_var, width=15)
        self.init_opacity_entry.grid(row=1, column=1, sticky='w', padx=5)
        ttk.Label(init_frame, text="初始不透明度:").grid(row=1, column=0, sticky='w', pady=2)
        ToolTip(self.init_opacity_entry, "高斯的初始不透明度")

        ttk.Label(init_frame, text="SH阶数:").grid(row=1, column=2, sticky='w', pady=2, padx=(20, 0))
        self.sh_degree_var = tk.IntVar(value=3)
        ttk.Entry(init_frame, textvariable=self.sh_degree_var, width=15).grid(row=1, column=3, sticky='w', padx=5)

        self.add_ground_plane_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(init_frame, text="添加地面高斯", variable=self.add_ground_plane_var).grid(row=2, column=0, sticky='w', pady=5, columnspan=2)

        ttk.Label(init_frame, text="高斯标准差:").grid(row=3, column=0, sticky='w', pady=2)
        self.gaussian_std_var = tk.DoubleVar(value=5.0)
        gaussian_std_entry = ttk.Entry(init_frame, textvariable=self.gaussian_std_var, width=15)
        gaussian_std_entry.grid(row=3, column=1, sticky='w', padx=5)
        ToolTip(gaussian_std_entry, "高斯分布标准差，越小越集中在原点附近")

        train_frame = ttk.LabelFrame(scroll_frame, text='训练参数', padding="10")
        train_frame.grid(row=row, column=0, sticky='ew', padx=10, pady=5)
        row += 1

        ttk.Label(train_frame, text="最大迭代次数:").grid(row=0, column=0, sticky='w', pady=2)
        self.max_iterations_var = tk.IntVar(value=100000)
        ttk.Entry(train_frame, textvariable=self.max_iterations_var, width=15).grid(row=0, column=1, sticky='w', padx=5)

        ttk.Label(train_frame, text="学习率-位置:").grid(row=1, column=0, sticky='w', pady=2)
        self.lr_position_var = tk.DoubleVar(value=0.005)
        ttk.Entry(train_frame, textvariable=self.lr_position_var, width=15).grid(row=1, column=1, sticky='w', padx=5)

        ttk.Label(train_frame, text="学习率-缩放:").grid(row=1, column=2, sticky='w', pady=2, padx=(20, 0))
        self.lr_scale_var = tk.DoubleVar(value=0.005)
        ttk.Entry(train_frame, textvariable=self.lr_scale_var, width=15).grid(row=1, column=3, sticky='w', padx=5)

        ttk.Label(train_frame, text="学习率-旋转:").grid(row=2, column=0, sticky='w', pady=2)
        self.lr_rotation_var = tk.DoubleVar(value=0.005)
        ttk.Entry(train_frame, textvariable=self.lr_rotation_var, width=15).grid(row=2, column=1, sticky='w', padx=5)

        ttk.Label(train_frame, text="学习率-不透明度:").grid(row=2, column=2, sticky='w', pady=2, padx=(20, 0))
        self.lr_opacity_var = tk.DoubleVar(value=0.1)
        ttk.Entry(train_frame, textvariable=self.lr_opacity_var, width=15).grid(row=2, column=3, sticky='w', padx=5)

        ttk.Label(train_frame, text="学习率-SH函数:").grid(row=3, column=0, sticky='w', pady=2)
        self.lr_sh_var = tk.DoubleVar(value=0.01)
        ttk.Entry(train_frame, textvariable=self.lr_sh_var, width=15).grid(row=3, column=1, sticky='w', padx=5)

        ttk.Label(train_frame, text="保存间隔:").grid(row=3, column=2, sticky='w', pady=2, padx=(20, 0))
        self.save_interval_var = tk.IntVar(value=300)
        ttk.Entry(train_frame, textvariable=self.save_interval_var, width=15).grid(row=3, column=3, sticky='w', padx=5)

        self.use_weighted_l1_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(train_frame, text="使用加权L1", variable=self.use_weighted_l1_var).grid(row=4, column=0, sticky='w', pady=2)

        ttk.Label(train_frame, text="L1权重:").grid(row=4, column=1, sticky='w', pady=2, padx=(20, 0))
        self.l1_weight_var = tk.DoubleVar(value=90.0)
        ttk.Entry(train_frame, textvariable=self.l1_weight_var, width=10).grid(row=4, column=2, sticky='w', padx=5)

        ttk.Label(train_frame, text="SSIM权重:").grid(row=4, column=3, sticky='w', pady=2, padx=(20, 0))
        self.ssim_weight_var = tk.DoubleVar(value=5.0)
        ttk.Entry(train_frame, textvariable=self.ssim_weight_var, width=10).grid(row=4, column=4, sticky='w', padx=5)

        ttk.Label(train_frame, text="加权模式:").grid(row=5, column=0, sticky='w', pady=2)
        self.l1_weight_mode_var = tk.StringVar(value='linear')
        l1_weight_mode_combo = ttk.Combobox(train_frame, textvariable=self.l1_weight_mode_var, width=12, state='readonly')
        l1_weight_mode_combo['values'] = ('linear', 'square', 'sqrt')
        l1_weight_mode_combo.grid(row=5, column=1, sticky='w', padx=5)

        densify_frame = ttk.LabelFrame(scroll_frame, text='致密化参数', padding="10")
        densify_frame.grid(row=row, column=0, sticky='ew', padx=10, pady=5)
        row += 1

        self.densify_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(densify_frame, text="启用致密化", variable=self.densify_enabled_var).grid(row=0, column=0, sticky='w', columnspan=2)

        ttk.Label(densify_frame, text="致密化间隔:").grid(row=1, column=0, sticky='w', pady=2)
        self.densify_interval_var = tk.IntVar(value=100)
        ttk.Entry(densify_frame, textvariable=self.densify_interval_var, width=15).grid(row=1, column=1, sticky='w', padx=5)

        ttk.Label(densify_frame, text="起始迭代:").grid(row=1, column=2, sticky='w', pady=2, padx=(20, 0))
        self.densify_start_iter_var = tk.IntVar(value=200)
        ttk.Entry(densify_frame, textvariable=self.densify_start_iter_var, width=15).grid(row=1, column=3, sticky='w', padx=5)

        ttk.Label(densify_frame, text="梯度阈值:").grid(row=2, column=0, sticky='w', pady=2)
        self.densify_grad_threshold_var = tk.DoubleVar(value=0.0001)
        ttk.Entry(densify_frame, textvariable=self.densify_grad_threshold_var, width=15).grid(row=2, column=1, sticky='w', padx=5)

        ttk.Label(densify_frame, text="克隆阈值(小尺度):").grid(row=2, column=2, sticky='w', pady=2, padx=(20, 0))
        self.clone_threshold_var = tk.DoubleVar(value=0.3)
        ttk.Entry(densify_frame, textvariable=self.clone_threshold_var, width=15).grid(row=2, column=3, sticky='w', padx=5)

        ttk.Label(densify_frame, text="分裂阈值(大尺度):").grid(row=3, column=0, sticky='w', pady=2)
        self.split_threshold_var = tk.DoubleVar(value=1.5)
        ttk.Entry(densify_frame, textvariable=self.split_threshold_var, width=15).grid(row=3, column=1, sticky='w', padx=5)

        prune_frame = ttk.LabelFrame(scroll_frame, text='剪枝参数', padding="10")
        prune_frame.grid(row=row, column=0, sticky='ew', padx=10, pady=5)
        row += 1

        self.prune_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prune_frame, text="启用剪枝", variable=self.prune_enabled_var).grid(row=0, column=0, sticky='w', columnspan=2)

        ttk.Label(prune_frame, text="剪枝间隔:").grid(row=1, column=0, sticky='w', pady=2)
        self.prune_interval_var = tk.IntVar(value=50)
        ttk.Entry(prune_frame, textvariable=self.prune_interval_var, width=15).grid(row=1, column=1, sticky='w', padx=5)

        ttk.Label(prune_frame, text="起始迭代:").grid(row=1, column=2, sticky='w', pady=2, padx=(20, 0))
        self.prune_start_iter_var = tk.IntVar(value=300)
        ttk.Entry(prune_frame, textvariable=self.prune_start_iter_var, width=15).grid(row=1, column=3, sticky='w', padx=5)

        ttk.Label(prune_frame, text="不透明度阈值:").grid(row=2, column=0, sticky='w', pady=2)
        self.opacity_threshold_var = tk.DoubleVar(value=0.25)
        ttk.Entry(prune_frame, textvariable=self.opacity_threshold_var, width=15).grid(row=2, column=1, sticky='w', padx=5)

        ttk.Label(prune_frame, text="不透明度重置间隔:").grid(row=2, column=2, sticky='w', pady=2, padx=(20, 0))
        self.opacity_reset_interval_var = tk.IntVar(value=350)
        ttk.Entry(prune_frame, textvariable=self.opacity_reset_interval_var, width=15).grid(row=2, column=3, sticky='w', padx=5)

        ttk.Label(prune_frame, text="尺寸阈值:").grid(row=3, column=0, sticky='w', pady=2)
        self.size_threshold_var = tk.DoubleVar(value=2.0)
        size_threshold_entry = ttk.Entry(prune_frame, textvariable=self.size_threshold_var, width=15)
        size_threshold_entry.grid(row=3, column=1, sticky='w', padx=5)
        ToolTip(size_threshold_entry, "超过此尺寸的高斯将被剪枝")

        self.scatter_prune_enabled_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(prune_frame, text="启用散射剪枝", variable=self.scatter_prune_enabled_var).grid(row=4, column=0, sticky='w', columnspan=2)

        ttk.Label(prune_frame, text="散射剪枝阈值:").grid(row=5, column=0, sticky='w', pady=2)
        self.scatter_threshold_var = tk.DoubleVar(value=0.2)
        scatter_threshold_entry = ttk.Entry(prune_frame, textvariable=self.scatter_threshold_var, width=15)
        scatter_threshold_entry.grid(row=5, column=1, sticky='w', padx=5)
        ToolTip(scatter_threshold_entry, "高斯投影到所有视角的像素值均小于真实图像*阈值时剪枝")

        ttk.Label(prune_frame, text="散射剪枝间隔:").grid(row=5, column=2, sticky='w', pady=2, padx=(20, 0))
        self.scatter_interval_var = tk.IntVar(value=30)
        scatter_interval_entry = ttk.Entry(prune_frame, textvariable=self.scatter_interval_var, width=15)
        scatter_interval_entry.grid(row=5, column=3, sticky='w', padx=5)
        ToolTip(scatter_interval_entry, "执行散射剪枝的迭代间隔")

    def _create_visualization_tab(self):
        frame = ttk.Frame(self.visualization_tab, padding="10")
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="检查点可视化", font='Title.TLabel').pack(anchor='w', pady=(0, 10))

        load_frame = ttk.LabelFrame(frame, text='加载检查点', padding="10")
        load_frame.pack(fill=tk.X, pady=(0, 10))

        self.checkpoint_path_var = tk.StringVar()
        ttk.Entry(load_frame, textvariable=self.checkpoint_path_var, width=50).pack(side=tk.LEFT, padx=5)
        ttk.Button(load_frame, text="浏览...", command=lambda: self._browse_checkpoint(self.checkpoint_path_var)).pack(side=tk.LEFT)
        ttk.Button(load_frame, text="加载", command=self._load_checkpoint_for_viz).pack(side=tk.LEFT, padx=5)

        info_frame = ttk.LabelFrame(frame, text='模型信息', padding="10")
        info_frame.pack(fill=tk.X, pady=(0, 10))

        self.viz_model_info_label = ttk.Label(info_frame, text="未加载模型", foreground='gray')
        self.viz_model_info_label.pack(anchor='w')

        filter_frame = ttk.LabelFrame(frame, text='筛选条件', padding="10")
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        opacity_filter_frame = ttk.Frame(filter_frame)
        opacity_filter_frame.pack(fill=tk.X, pady=2)
        ttk.Label(opacity_filter_frame, text="不透明度范围:").pack(side=tk.LEFT)
        self.opacity_min_var = tk.StringVar(value="0")
        self.opacity_max_var = tk.StringVar(value="1")
        ttk.Entry(opacity_filter_frame, textvariable=self.opacity_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(opacity_filter_frame, text="~").pack(side=tk.LEFT)
        ttk.Entry(opacity_filter_frame, textvariable=self.opacity_max_var, width=8).pack(side=tk.LEFT, padx=2)

        scale_filter_frame = ttk.Frame(filter_frame)
        scale_filter_frame.pack(fill=tk.X, pady=2)
        ttk.Label(scale_filter_frame, text="尺寸范围:").pack(side=tk.LEFT)
        self.scale_min_var = tk.StringVar(value="0")
        self.scale_max_var = tk.StringVar(value="10")
        ttk.Entry(scale_filter_frame, textvariable=self.scale_min_var, width=8).pack(side=tk.LEFT, padx=2)
        ttk.Label(scale_filter_frame, text="~").pack(side=tk.LEFT)
        ttk.Entry(scale_filter_frame, textvariable=self.scale_max_var, width=8).pack(side=tk.LEFT, padx=2)

        ttk.Button(filter_frame, text="更新视图", command=self._update_viz_view).pack(pady=5)

        view_mode_frame = ttk.Frame(filter_frame)
        view_mode_frame.pack(fill=tk.X, pady=2)
        ttk.Label(view_mode_frame, text="显示模式:").pack(side=tk.LEFT)
        self.viz_view_mode_var = tk.StringVar(value="point")
        ttk.Radiobutton(view_mode_frame, text="点云", variable=self.viz_view_mode_var, value="point").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_mode_frame, text="球体", variable=self.viz_view_mode_var, value="sphere").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_mode_frame, text="椭球", variable=self.viz_view_mode_var, value="ellipsoid").pack(side=tk.LEFT, padx=5)

        sphere_scale_frame = ttk.Frame(filter_frame)
        sphere_scale_frame.pack(fill=tk.X, pady=2)
        ttk.Label(sphere_scale_frame, text="球体/椭球缩放:").pack(side=tk.LEFT)
        self.viz_sphere_scale_var = tk.DoubleVar(value=1.0)
        sphere_scale_entry = ttk.Entry(sphere_scale_frame, textvariable=self.viz_sphere_scale_var, width=8)
        sphere_scale_entry.pack(side=tk.LEFT, padx=5)
        ToolTip(sphere_scale_entry, "球体或椭球的显示缩放倍数")

        self.hist_frame = ttk.Frame(frame)
        self.hist_frame.pack(fill=tk.BOTH, expand=True)

        self.opacity_hist_ax = None
        self.scale_hist_ax = None

        fig_container = ttk.Frame(self.hist_frame)
        fig_container.pack(fill=tk.BOTH, expand=True)

        self.fig_canvas = None
        self.fig_agg = None

    def _browse_checkpoint(self, var):
        file_path = filedialog.askopenfilename(
            title="选择检查点文件",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if file_path:
            var.set(file_path)

    def _load_checkpoint_for_viz(self):
        file_path = self.checkpoint_path_var.get()
        if not file_path:
            self.log("请选择检查点文件")
            return

        try:
            self.log(f"加载检查点: {file_path}")
            checkpoint = torch.load(file_path, map_location='cpu')

            means = checkpoint['means']
            if means.dim() > 2:
                means = means.reshape(-1, 3)
            elif means.dim() == 1:
                means = means.reshape(-1, 3)

            opacities = torch.sigmoid(checkpoint['opacities'])
            if opacities.dim() > 1:
                opacities = opacities.reshape(-1)
            opacities = opacities.numpy()

            scales = torch.exp(checkpoint['scales'])
            if scales.dim() > 2:
                scales = scales.reshape(-1, 3)
            elif scales.dim() == 1:
                scales = scales.reshape(-1, 3)
            scales = scales.numpy()
            scale_mags = np.sqrt((scales ** 2).sum(axis=1))

            self.viz_checkpoint_data = {
                'means': means,
                'opacities': opacities,
                'scales': scales,
                'scale_mags': scale_mags,
                'rotations': checkpoint.get('rotations'),
                'iteration': checkpoint.get('iteration', 0)
            }

            self.viz_model_info_label.config(
                text=f"迭代次数: {checkpoint.get('iteration', 'N/A')}, "
                     f"高斯数量: {means.shape[0]}",
                foreground='black'
            )
            self.log(f"检查点加载成功")

            self._update_histograms()
            self._update_viz_view()

        except Exception as e:
            self.log(f"加载检查点失败: {str(e)}")
            import traceback
            self.log(traceback.format_exc())

    def _update_histograms(self):
        if not hasattr(self, 'viz_checkpoint_data'):
            return

        data = self.viz_checkpoint_data
        opacities = data['opacities']
        scale_mags = data['scale_mags']

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if self.fig_canvas is None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.hist(opacities, bins=50, color='blue', alpha=0.7)
            ax1.set_title('Opacity Distribution')
            ax1.set_xlabel('Opacity')
            ax1.set_ylabel('Count')

            ax2.hist(scale_mags, bins=50, color='green', alpha=0.7)
            ax2.set_title('Scale Magnitude Distribution')
            ax2.set_xlabel('Scale Magnitude')
            ax2.set_ylabel('Count')

            plt.tight_layout()

            self.fig_canvas = FigureCanvasTkAgg(fig, master=self.hist_frame)
            self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ax1, ax2 = self.fig_canvas.figure.axes
            ax1.clear()
            ax2.clear()
            ax1.hist(opacities, bins=50, color='blue', alpha=0.7)
            ax1.set_title('Opacity Distribution')
            ax1.set_xlabel('Opacity')
            ax1.set_ylabel('Count')
            ax2.hist(scale_mags, bins=50, color='green', alpha=0.7)
            ax2.set_title('Scale Magnitude Distribution')
            ax2.set_xlabel('Scale Magnitude')
            ax2.set_ylabel('Count')
            self.fig_canvas.draw()

    def _update_viz_view(self):
        if not hasattr(self, 'viz_checkpoint_data'):
            return

        try:
            opacity_min = float(self.opacity_min_var.get())
            opacity_max = float(self.opacity_max_var.get())
            scale_min = float(self.scale_min_var.get())
            scale_max = float(self.scale_max_var.get())
        except ValueError:
            self.log("筛选条件输入无效")
            return

        data = self.viz_checkpoint_data
        means = data['means']
        opacities = data['opacities']
        scales = data['scales']
        scale_mags = data['scale_mags']
        rotations = data.get('rotations')
        view_mode = self.viz_view_mode_var.get()
        sphere_scale = self.viz_sphere_scale_var.get()

        mask = (opacities >= opacity_min) & (opacities <= opacity_max) & \
               (scale_mags >= scale_min) & (scale_mags <= scale_max)

        if mask.ndim > 1:
            mask = mask.flatten()

        filtered_means = means[mask]
        filtered_scales = scales[mask]
        z_values = filtered_means[:, 2]

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        num_samples = min(len(filtered_means), 5000)
        if num_samples < len(filtered_means):
            indices = np.random.choice(len(filtered_means), num_samples, replace=False)
            sample_means = filtered_means[indices]
            sample_scales = filtered_scales[indices]
            sample_z = z_values[indices]
        else:
            sample_means = filtered_means
            sample_scales = filtered_scales
            sample_z = z_values

        z_min, z_max = sample_z.min(), sample_z.max()
        norm = plt.Normalize(z_min, z_max) if z_max > z_min else plt.Normalize(0, 1)

        if view_mode == 'point':
            scatter = ax.scatter(
                sample_means[:, 0],
                sample_means[:, 1],
                sample_means[:, 2],
                c=sample_z,
                cmap='jet',
                s=2,
                alpha=0.8,
                norm=norm
            )
        elif view_mode == 'sphere':
            u = np.linspace(0, 2 * np.pi, 12)
            v = np.linspace(0, np.pi, 8)
            u_mesh, v_mesh = np.meshgrid(u, v, indexing='ij')

            for i in range(len(sample_means)):
                mx = float(sample_means[i, 0])
                my = float(sample_means[i, 1])
                mz = float(sample_means[i, 2])
                r = float(sample_scales[i].mean()) * sphere_scale * 0.5
                color_val = plt.cm.jet(norm(sample_z[i]))
                x_surf = np.cos(u_mesh) * np.sin(v_mesh) * r + mx
                y_surf = np.sin(u_mesh) * np.sin(v_mesh) * r + my
                z_surf = np.cos(v_mesh) * r + mz
                ax.plot_surface(
                    x_surf, y_surf, z_surf,
                    color=color_val, alpha=0.6, shade=False
                )
        elif view_mode == 'ellipsoid' and rotations is not None:
            filtered_rotations = rotations[mask]
            if len(filtered_rotations) > num_samples:
                sample_rotations = filtered_rotations[indices]
            else:
                sample_rotations = filtered_rotations

            u = np.linspace(0, 2 * np.pi, 12)
            v = np.linspace(0, np.pi, 8)
            u_mesh, v_mesh = np.meshgrid(u, v, indexing='ij')

            for i in range(len(sample_means)):
                mx = float(sample_means[i, 0])
                my = float(sample_means[i, 1])
                mz = float(sample_means[i, 2])
                sx = float(sample_scales[i, 0]) * sphere_scale * 0.5
                sy = float(sample_scales[i, 1]) * sphere_scale * 0.5
                sz = float(sample_scales[i, 2]) * sphere_scale * 0.5
                quat = sample_rotations[i]
                quat = quat / (np.linalg.norm(quat) + 1e-8)
                w, x, y, z = quat[0], quat[1], quat[2], quat[3]
                R = np.array([
                    [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                    [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                    [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
                ])
                bx = np.cos(u_mesh) * np.sin(v_mesh)
                by = np.sin(u_mesh) * np.sin(v_mesh)
                bz = np.cos(v_mesh)
                x_surf = (bx * sx) * R[0, 0] + (by * sy) * R[0, 1] + (bz * sz) * R[0, 2] + mx
                y_surf = (bx * sx) * R[1, 0] + (by * sy) * R[1, 1] + (bz * sz) * R[1, 2] + my
                z_surf = (bx * sx) * R[2, 0] + (by * sy) * R[2, 1] + (bz * sz) * R[2, 2] + mz
                color_val = plt.cm.jet(norm(sample_z[i]))
                ax.plot_surface(
                    x_surf, y_surf, z_surf,
                    color=color_val, alpha=0.6, shade=False
                )

        x_min, x_max = sample_means[:, 0].min(), sample_means[:, 0].max()
        y_min, y_max = sample_means[:, 1].min(), sample_means[:, 1].max()
        z_min_ax, z_max_ax = sample_means[:, 2].min(), sample_means[:, 2].max()

        max_range = max(x_max - x_min, y_max - y_min, z_max_ax - z_min_ax) / 2 * 1.1
        center_x = (x_max + x_min) / 2
        center_y = (y_max + y_min) / 2
        center_z = (z_max_ax + z_min_ax) / 2

        ax.set_xlim(center_x - max_range, center_x + max_range)
        ax.set_ylim(center_y - max_range, center_y + max_range)
        ax.set_zlim(center_z - max_range, center_z + max_range)
        ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Gaussian Distribution - {view_mode} mode (Filtered: {mask.sum()}/{len(mask)})')

        cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax, shrink=0.6)
        cbar.set_label('Z Height')

        top = tk.Toplevel(self.root)
        top.title("3D Gaussian View")
        top.geometry("900x700")

        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, top)
        toolbar.update()

        def on_mouse_wheel(event):
            scale_factor = 0.9 if event.step > 0 else 1.1
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            cur_zlim = ax.get_zlim()
            x_range = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            y_range = (cur_ylim[1] - cur_ylim[0]) * scale_factor
            z_range = (cur_zlim[1] - cur_zlim[0]) * scale_factor
            ax.set_xlim([cur_xlim[0] - (x_range - (cur_xlim[1] - cur_xlim[0]))/2,
                         cur_xlim[1] + (x_range - (cur_xlim[1] - cur_xlim[0]))/2])
            ax.set_ylim([cur_ylim[0] - (y_range - (cur_ylim[1] - cur_ylim[0]))/2,
                         cur_ylim[1] + (y_range - (cur_ylim[1] - cur_ylim[0]))/2])
            ax.set_zlim([cur_zlim[0] - (z_range - (cur_zlim[1] - cur_zlim[0]))/2,
                         cur_zlim[1] + (z_range - (cur_zlim[1] - cur_zlim[0]))/2])
            canvas.draw()

        fig.canvas.mpl_connect('scroll_event', on_mouse_wheel)

        plt.close(fig)

    def _bind_shortcuts(self):
        self.root.bind('<Control-s>', lambda e: self.start_training())
        self.root.bind('<Control-q>', lambda e: self.on_closing())

    def browse_folder(self, var):
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)

    def load_checkpoint(self):
        file_path = filedialog.askopenfilename(
            title="选择模型检查点",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if file_path:
            self.log(f"加载检查点: {file_path}")
            checkpoint = torch.load(file_path, map_location='cpu')
            self.model_info_label.config(
                text=f"迭代次数: {checkpoint.get('iteration', 'N/A')}, "
                     f"高斯数量: {checkpoint['means'].shape[0]}",
                foreground='black'
            )
            self.log(f"检查点加载成功")

    def get_densify_config(self):
        return DensifyConfig(
            enabled=self.densify_enabled_var.get(),
            interval=self.densify_interval_var.get(),
            start_iter=self.densify_start_iter_var.get(),
            grad_threshold=self.densify_grad_threshold_var.get(),
            clone_threshold=self.clone_threshold_var.get(),
            split_threshold=self.split_threshold_var.get()
        )

    def get_prune_config(self):
        return PruneConfig(
            enabled=self.prune_enabled_var.get(),
            interval=self.prune_interval_var.get(),
            start_iter=self.prune_start_iter_var.get(),
            opacity_threshold=self.opacity_threshold_var.get(),
            scale_threshold=self.size_threshold_var.get()
        )

    def log(self, message: str):
        import datetime
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        self.log_text.append(f"{log_message}\n")
        print(log_message)

    def start_training(self):
        if self.is_training:
            self.log("训练正在进行中...")
            return

        self.log("=" * 50)
        self.log("开始训练...")

        self.stop_event.clear()
        self.training_thread = threading.Thread(target=self._training_worker, daemon=True)
        self.training_thread.start()

    def continue_training(self):
        if self.is_training:
            self.log("训练正在进行中...")
            return

        checkpoint_path = filedialog.askopenfilename(
            title="选择检查点文件",
            filetypes=[("PyTorch files", "*.pth"), ("All files", "*.*")]
        )
        if not checkpoint_path:
            return

        self.log("=" * 50)
        self.log(f"从检查点继续训练: {checkpoint_path}")

        self.stop_event.clear()
        self.checkpoint_path = checkpoint_path
        self.training_thread = threading.Thread(target=self._training_worker, daemon=True, kwargs={'resume_from_checkpoint': True})
        self.training_thread.start()

    def stop_training(self):
        if not self.is_training:
            return
        self.log("正在停止训练...")
        self.stop_event.set()

    def _training_worker(self, resume_from_checkpoint=False):
        self.is_training = True
        self.start_button.config(state='disabled')
        self.continue_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.status_label.config(text="训练中...", foreground="orange")

        self._train_checkpoint_info = {'stopped': False, 'completed': False}
        try:
            self._train(resume_from_checkpoint=resume_from_checkpoint)
            self._train_checkpoint_info['completed'] = True
        except Exception as e:
            self.log(f"训练出错: {str(e)}")
            self.log(traceback.format_exc())
            self._emergency_checkpoint_save()
        finally:
            self.is_training = False
            if not self._train_checkpoint_info.get('stopped') and not self._train_checkpoint_info.get('completed'):
                self._emergency_checkpoint_save()
            self.start_button.config(state='normal')
            self.continue_button.config(state='normal')
            self.stop_button.config(state='disabled')
            self.status_label.config(text="就绪", foreground="green")

    def _emergency_checkpoint_save(self):
        if not hasattr(self, '_train_checkpoint_info') or self._train_checkpoint_info is None:
            self.log("无可用的训练状态信息，无法保存紧急检查点")
            return

        info = self._train_checkpoint_info
        if info.get('stopped'):
            return

        try:
            gaussian_model = info['gaussian_model']
            optimizer = info['optimizer']
            iteration = info['iteration']
            output_dir = info['output_dir']

            emergency_path = output_dir / f'emergency_checkpoint_iter_{iteration}.pth'
            gaussian_model.save_checkpoint(str(emergency_path), iteration=iteration, optimizer_state=optimizer.state_dict())
            self.log(f"紧急检查点已保存: {emergency_path}")
            info['stopped'] = True
        except Exception as e:
            self.log(f"保存紧急检查点失败: {str(e)}")

    def _train(self, resume_from_checkpoint=False):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log(f"使用设备: {device}")

        data_path = self.data_path_var.get()
        self.log(f"加载数据集: {data_path}")

        dataset = SARSceneDataset(data_path, load_images=True)
        self.log(f"图像数量: {len(dataset)}")
        self.log(f"场景归一化: {dataset.scene_normalization}")

        scene_bounds = compute_scene_bounds_from_dataset(dataset, z_ratio=0.25)
        self.log(f"场景边界: {scene_bounds}")

        self.log("初始化高斯模型...")
        gaussian_model = GaussianModel(
            sh_degree=self.sh_degree_var.get(),
            init_num_gaussians=self.init_num_gaussians_var.get()
        ).to(device)

        start_iteration = 0
        checkpoint_info = {'iteration': 0, 'optimizer_state': None}
        if resume_from_checkpoint and hasattr(self, 'checkpoint_path'):
            self.log(f"从检查点加载: {self.checkpoint_path}")
            checkpoint_info = gaussian_model.load_checkpoint(self.checkpoint_path, device)
            start_iteration = checkpoint_info['iteration']
            self.log(f"已加载检查点，从迭代 {start_iteration} 继续训练")
        else:
            init_opacity_logit = torch.logit(torch.tensor(self.init_opacity_var.get())).item()
            gaussian_model.initialize_random(
                scene_bounds=scene_bounds,
                num_gaussians=self.init_num_gaussians_var.get(),
                init_scale=self.init_scale_var.get(),
                init_opacity=init_opacity_logit,
                device=device,
                add_ground_plane=self.add_ground_plane_var.get(),
                ground_scale=0.15,
                ground_opacity=0.3,
                ground_ratio=0.3,
                gaussian_std=self.gaussian_std_var.get()
            )
        self.log(f"初始高斯数量: {gaussian_model.num_gaussians}")
        if resume_from_checkpoint:
            init_opacity_actual = torch.sigmoid(gaussian_model._opacities).mean().item()
            self.log(f"从检查点加载不透明度: 平均={init_opacity_actual:.4f}")
        else:
            init_opacity_actual = torch.sigmoid(gaussian_model._opacities).mean().item()
            self.log(f"初始不透明度: GUI设置={self.init_opacity_var.get():.4f}, logit={init_opacity_logit:.4f}, 实际值={init_opacity_actual:.4f}")

        loss_fn = CombinedLoss(
            l1_weight=self.l1_weight_var.get(),
            ssim_weight=self.ssim_weight_var.get(),
            use_weighted_l1=self.use_weighted_l1_var.get(),
            l1_weight_mode=self.l1_weight_mode_var.get()
        ).to(device)

        optimizer = optim.Adam([
            {'params': [gaussian_model._means], 'lr': self.lr_position_var.get()},
            {'params': [gaussian_model._scales], 'lr': self.lr_scale_var.get()},
            {'params': [gaussian_model._rotations], 'lr': self.lr_rotation_var.get()},
            {'params': [gaussian_model._opacities], 'lr': self.lr_opacity_var.get()},
            {'params': [gaussian_model._sh_coeffs], 'lr': self.lr_sh_var.get()},
        ])

        if resume_from_checkpoint and hasattr(self, 'checkpoint_path'):
            if checkpoint_info.get('optimizer_state') is not None:
                optimizer.load_state_dict(checkpoint_info['optimizer_state'])
                self.log("优化器状态已恢复")

        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.densify_interval_var.get() * 10,
            gamma=0.5
        )

        output_dir = Path(self.output_dir_var.get()).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        render_output_dir = output_dir / 'renders'
        render_output_dir.mkdir(parents=True, exist_ok=True)

        viz_output_dir = output_dir / 'visualizations'
        viz_output_dir.mkdir(parents=True, exist_ok=True)

        viz_indices = self._get_visualization_indices(dataset, 8)
        self.log(f"可视化视角索引: {viz_indices}")

        densify_config = self.get_densify_config()
        prune_config = self.get_prune_config()

        max_iterations = self.max_iterations_var.get()
        log_interval = 1
        save_interval = self.save_interval_var.get()
        viz_interval = 1
        opacity_reset_interval = self.opacity_reset_interval_var.get()
        init_opacity = torch.logit(torch.tensor(self.init_opacity_var.get())).item()

        self.log("开始训练循环...")
        iteration = start_iteration
        epoch = 0
        self._train_checkpoint_info = {
            'gaussian_model': gaussian_model,
            'optimizer': optimizer,
            'iteration': iteration,
            'output_dir': output_dir
        }
        loss_history = []
        self.log(f"初始不透明度设置: GUI={self.init_opacity_var.get():.4f}, logit={init_opacity:.4f}")
        self.log(f"不透明度重置间隔: {opacity_reset_interval} 次迭代")
        last_opacity_reset_epoch = 0
        epoch_clone_count = 0
        epoch_split_count = 0
        epoch_prune_count = 0
        epoch_scatter_prune_count = 0
        epoch_l1_losses = []
        epoch_ssim_losses = []
        last_scatter_prune_iter = -self.scatter_interval_var.get()

        precomputed_covs = {}

        if self.scatter_prune_enabled_var.get():
            num_scatter_init = self._scatter_prune(gaussian_model, dataset, device, self.scatter_threshold_var.get(), iteration, show_progress=True)
            self.log(f"初始化散射剪枝完成，移除 {num_scatter_init} 个高斯，剩余 {gaussian_model.num_gaussians} 个高斯")
            epoch_scatter_prune_count += num_scatter_init
            last_scatter_prune_iter = iteration

            if num_scatter_init > 0:
                precomputed_covs.clear()
                optimizer = optim.Adam([
                    {'params': [gaussian_model._means], 'lr': self.lr_position_var.get()},
                    {'params': [gaussian_model._scales], 'lr': self.lr_scale_var.get()},
                    {'params': [gaussian_model._rotations], 'lr': self.lr_rotation_var.get()},
                    {'params': [gaussian_model._opacities], 'lr': self.lr_opacity_var.get()},
                    {'params': [gaussian_model._sh_coeffs], 'lr': self.lr_sh_var.get()},
                ])
        else:
            self.log("散射剪枝已禁用")

        while iteration < max_iterations:
            if self.stop_event.is_set():
                self.log("训练被用户停止，正在保存检查点...")
                self._emergency_checkpoint_save()
                self._train_checkpoint_info['stopped'] = True
                break

            epoch_based_degree = epoch // 10
            iteration_based_degree = iteration // 300
            new_sh_degree = min(max(epoch_based_degree, iteration_based_degree), self.sh_degree_var.get())
            if new_sh_degree != gaussian_model.active_sh_degree:
                gaussian_model.active_sh_degree = new_sh_degree
                active_coeffs = (new_sh_degree + 1) ** 2
                self.log(f"[Epoch {epoch}/Iter {iteration}] SH系数激活: degree={new_sh_degree}, 系数数量={active_coeffs}")

            if iteration % 500 == 0 and iteration > 0:
                means_data = gaussian_model._means.data
                mean_dist = torch.sqrt(means_data[:, 0]**2 + means_data[:, 1]**2 + means_data[:, 2]**2)
                self.log(f"[诊断 迭代{iteration}] 均值到原点距离: mean={mean_dist.mean():.4f}, max={mean_dist.max():.4f}, min={mean_dist.min():.4f}")

                grad_means_check = gaussian_model._means.grad
                if grad_means_check is not None:
                    grad_mean_sum = grad_means_check.mean(dim=0)
                    self.log(f"[诊断 迭代{iteration}] 梯度均值: {grad_mean_sum.cpu().numpy()}")

            epoch += 1
            epoch_start_time = time.time()
            epoch_indices = np.random.permutation(len(dataset))
            epoch_losses = []

            precomputed_renderers = {}
            precomputed_covs = {}

            for idx in epoch_indices:
                if iteration >= max_iterations or self.stop_event.is_set():
                    break

                camera = dataset.get_camera(idx)
                norm_factor = camera.normalization_factor
                target_image = camera.image.to(device) / norm_factor

                radar_params = camera.radar_params
                camera_key = (radar_params.incidence_angle, radar_params.track_angle, radar_params.azimuth_angle, radar_params.range_resolution, radar_params.azimuth_resolution)

                if camera_key not in precomputed_renderers:
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
                    precomputed_renderers[camera_key] = (renderer, radar_params)

                renderer, radar_params = precomputed_renderers[camera_key]

                if id(gaussian_model._means) not in precomputed_covs:
                    cov_full = gaussian_model.compute_covariance_full()
                    precomputed_covs[id(gaussian_model._means)] = cov_full.detach()
                else:
                    cov_full = precomputed_covs[id(gaussian_model._means)]

                opacity = gaussian_model.get_opacity()
                sh_coeffs = gaussian_model.get_active_sh_coeffs()
                means = gaussian_model._means

                rendered_image = renderer(means, cov_full, opacity.squeeze(-1), sh_coeffs)

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

                    if grad_means is not None:
                        grad_means = grad_means.detach()
                    if grad_scales is not None:
                        grad_scales = grad_scales.detach()
                    if grad_opacities is not None:
                        grad_opacities = grad_opacities.detach()

                    if iteration > prune_config.start_iter and iteration % prune_config.interval == 0 and epoch > last_opacity_reset_epoch + 3:
                        num_clones, num_splits, num_prunes, _ = gaussian_model.densify_and_prune(
                            grad_means=grad_means,
                            grad_scales=grad_scales,
                            grad_opacities=grad_opacities,
                            grad_threshold=densify_config.grad_threshold,
                            size_threshold=densify_config.clone_threshold,
                            large_scale_threshold=prune_config.scale_threshold,
                            opacity_threshold=prune_config.opacity_threshold
                        )
                        epoch_clone_count += num_clones
                        epoch_split_count += num_splits
                        epoch_prune_count += num_prunes

                        del grad_means, grad_scales, grad_opacities
                        torch.cuda.empty_cache()

                        if num_clones > 0 or num_splits > 0 or num_prunes > 0:
                            precomputed_covs.clear()
                            optimizer = optim.Adam([
                                {'params': [gaussian_model._means], 'lr': self.lr_position_var.get()},
                                {'params': [gaussian_model._scales], 'lr': self.lr_scale_var.get()},
                                {'params': [gaussian_model._rotations], 'lr': self.lr_rotation_var.get()},
                                {'params': [gaussian_model._opacities], 'lr': self.lr_opacity_var.get()},
                                {'params': [gaussian_model._sh_coeffs], 'lr': self.lr_sh_var.get()},
                            ])

                    if self.scatter_prune_enabled_var.get() and iteration - last_scatter_prune_iter >= self.scatter_interval_var.get():
                        num_scatter = self._scatter_prune(gaussian_model, dataset, device, self.scatter_threshold_var.get(), iteration)
                        epoch_scatter_prune_count += num_scatter
                        last_scatter_prune_iter = iteration

                        if num_scatter > 0:
                            precomputed_covs.clear()
                            optimizer = optim.Adam([
                                {'params': [gaussian_model._means], 'lr': self.lr_position_var.get()},
                                {'params': [gaussian_model._scales], 'lr': self.lr_scale_var.get()},
                                {'params': [gaussian_model._rotations], 'lr': self.lr_rotation_var.get()},
                                {'params': [gaussian_model._opacities], 'lr': self.lr_opacity_var.get()},
                                {'params': [gaussian_model._sh_coeffs], 'lr': self.lr_sh_var.get()},
                            ])

                    if iteration > densify_config.interval * 20 and iteration % opacity_reset_interval == 0:
                        gaussian_model._opacities.data.fill_(init_opacity)
                        last_opacity_reset_epoch = epoch
                        actual_opacity = torch.sigmoid(gaussian_model._opacities).mean().item()
                        self.log(f"[迭代 {iteration}] 不透明度重置为: logit={init_opacity}, 实际值={actual_opacity:.4f}")

                optimizer.step()
                scheduler.step()

                epoch_losses.append(loss_dict['total'])
                epoch_l1_losses.append(loss_dict['l1'])
                epoch_ssim_losses.append(loss_dict['dssim'])
                loss_history.append(loss_dict['total'])
                iteration += 1
                self._train_checkpoint_info['iteration'] = iteration

            epoch_time = time.time() - epoch_start_time
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
            avg_l1 = np.mean(epoch_l1_losses) if epoch_l1_losses else 0.0
            avg_ssim = np.mean(epoch_ssim_losses) if epoch_ssim_losses else 0.0
            total_prune = epoch_prune_count + epoch_scatter_prune_count
            self.log(f"Epoch {epoch:04d} | Avg Loss: {avg_loss:.6f} (L1: {avg_l1:.6f}, DSSIM: {avg_ssim:.6f}) | "
                     f"Gaussians: {gaussian_model.num_gaussians} | "
                     f"Clone: {epoch_clone_count} | Split: {epoch_split_count} | Prune: {total_prune} | Time: {epoch_time:.1f}s")

            epoch_clone_count = 0
            epoch_split_count = 0
            epoch_prune_count = 0
            epoch_scatter_prune_count = 0
            epoch_l1_losses = []
            epoch_ssim_losses = []

            if epoch % viz_interval == 0:
                self._save_visualization(output_dir, render_output_dir, viz_output_dir, epoch, gaussian_model, dataset, viz_indices)

            if iteration % save_interval == 0 and iteration > 0:
                checkpoint_path = output_dir / f'checkpoint_iter_{iteration}.pth'
                gaussian_model.save_checkpoint(str(checkpoint_path), iteration=iteration, optimizer_state=optimizer.state_dict())
                self.log(f"已保存检查点: {checkpoint_path}")

        final_path = output_dir / 'final_model.pth'
        gaussian_model.save_checkpoint(str(final_path), iteration=iteration, optimizer_state=optimizer.state_dict())
        self.log(f"训练完成! 模型已保存: {final_path}")

    def _get_visualization_indices(self, dataset, num_views):
        num_cameras = len(dataset)
        if num_cameras <= num_views:
            return list(range(num_cameras))
        step = num_cameras / num_views
        return [int(i * step) for i in range(num_views)]

    def _scatter_prune(self, gaussian_model, dataset, device, threshold, iteration, show_progress=False):
        return self._scatter_prune_python(gaussian_model, dataset, device, threshold, iteration, show_progress=show_progress)

    def _scatter_prune_python(self, gaussian_model, dataset, device, threshold, iteration, show_progress=False):
        means = gaussian_model._means.detach()
        opacities = torch.sigmoid(gaussian_model._opacities.detach())
        num_gaussians = len(means)

        if num_gaussians == 0:
            return 0

        has_significant = torch.zeros(num_gaussians, dtype=torch.bool, device=device)
        total_views = len(dataset)

        valid_mask = opacities.squeeze(-1) > 0.01
        valid_indices = valid_mask.nonzero(as_tuple=True)[0]

        if len(valid_indices) == 0:
            return 0

        valid_means = means[valid_indices]

        for idx in range(total_views):
            camera = dataset.get_camera(idx)
            target_image = camera.image.to(device)

            rendered_image, _ = self._render_view_for_saving(gaussian_model, camera, device)
            rendered_image = rendered_image.detach()

            if rendered_image.shape != target_image.shape:
                rendered_image = torch.nn.functional.interpolate(
                    rendered_image.unsqueeze(0).unsqueeze(0),
                    size=target_image.shape,
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

            target_max = target_image.max()
            threshold_value = target_max * threshold

            radar_params = camera.radar_params
            theta_rad = np.deg2rad(radar_params.incidence_angle)
            phi_rad = np.deg2rad(radar_params.azimuth_angle)
            alpha = np.deg2rad(radar_params.track_angle)

            sin_beta = np.sin(theta_rad) * np.cos(phi_rad)
            beta = np.arcsin(np.clip(sin_beta, -1.0, 1.0))
            tan_beta = np.tan(beta)
            radar_altitude = radar_params.radar_altitude

            radar_x = radar_altitude * tan_beta * np.sin(alpha)
            radar_y = -radar_altitude * tan_beta * np.cos(alpha)
            radar_z = radar_altitude

            Rc = radar_altitude / np.cos(beta)

            cos_a, sin_a = np.cos(alpha), np.sin(alpha)
            cos_b, sin_b = np.cos(beta), np.sin(beta)

            dx_all = valid_means[:, 0] - radar_x
            dy_all = valid_means[:, 1] - radar_y
            dz_all = valid_means[:, 2] - radar_z

            xr_all = cos_a * dx_all + sin_a * dy_all
            yr_all = cos_b * sin_a * dx_all - cos_b * cos_a * dy_all - sin_b * dz_all
            zr_all = -sin_b * sin_a * dx_all + sin_b * cos_a * dy_all - cos_b * dz_all

            Rmin_all = torch.sqrt(yr_all**2 + zr_all**2)
            r_coords = Rmin_all / radar_params.range_resolution + rendered_image.shape[0] / 2 - Rc / radar_params.range_resolution
            c_coords = xr_all / radar_params.azimuth_resolution + rendered_image.shape[1] / 2

            r_coords = torch.clamp(r_coords.long(), 0, rendered_image.shape[0] - 1)
            c_coords = torch.clamp(c_coords.long(), 0, rendered_image.shape[1] - 1)

            pixel_values = target_image[r_coords, c_coords]
            significant_gauss_indices = valid_indices[pixel_values > threshold_value]
            has_significant[significant_gauss_indices] = True

            if show_progress:
                progress = (idx + 1) / total_views * 100
                self.log(f"散射剪枝进度: {idx + 1}/{total_views} ({progress:.1f}%)")

        prune_mask = ~has_significant
        num_pruned = prune_mask.sum().item()

        if num_pruned > 0:
            keep_mask = ~prune_mask
            gaussian_model._means = torch.nn.Parameter(gaussian_model._means.data[keep_mask].clone())
            gaussian_model._scales = torch.nn.Parameter(gaussian_model._scales.data[keep_mask].clone())
            gaussian_model._rotations = torch.nn.Parameter(gaussian_model._rotations.data[keep_mask].clone())
            gaussian_model._opacities = torch.nn.Parameter(gaussian_model._opacities.data[keep_mask].clone())
            gaussian_model._sh_coeffs = torch.nn.Parameter(gaussian_model._sh_coeffs.data[keep_mask].clone())

        return num_pruned

    def _render_view_for_saving(self, gaussian_model, camera, device):
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
        return rendered, {}

    def _save_visualization(self, output_dir, render_output_dir, viz_output_dir, epoch, gaussian_model, dataset, viz_indices):
        if not HAS_MATPLOTLIB:
            return

        try:
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            self.log("警告: mpl_toolkits未安装，跳过高斯分布可视化")
            return

        try:
            means = gaussian_model.means.detach().cpu().numpy()
            scales = torch.exp(gaussian_model._scales).detach().cpu().numpy()
            opacities = torch.sigmoid(gaussian_model._opacities).detach().cpu().numpy()
            scale_norms = np.linalg.norm(scales, axis=1)
            total_gaussians = gaussian_model.num_gaussians

            fig = plt.figure(figsize=(20, 10))
            fig.suptitle(f'Epoch {epoch} - Gaussian Distribution\nTotal Gaussians: {total_gaussians}', fontsize=14)

            ax1 = fig.add_subplot(2, 4, 1)
            scatter1 = ax1.scatter(means[:, 0], means[:, 1], c=opacities[:, 0], s=2, cmap='viridis', alpha=0.6)
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_title('XY View (Opacity)')
            plt.colorbar(scatter1, ax=ax1, label='Opacity')
            ax1.set_aspect('equal', adjustable='box')

            ax2 = fig.add_subplot(2, 4, 2)
            scatter2 = ax2.scatter(means[:, 0], means[:, 2], c=opacities[:, 0], s=2, cmap='viridis', alpha=0.6)
            ax2.set_xlabel('X')
            ax2.set_ylabel('Z')
            ax2.set_title('XZ View (Opacity)')
            plt.colorbar(scatter2, ax=ax2, label='Opacity')
            ax2.set_aspect('equal', adjustable='box')

            ax3 = fig.add_subplot(2, 4, 3)
            scatter3 = ax3.scatter(means[:, 1], means[:, 2], c=opacities[:, 0], s=2, cmap='viridis', alpha=0.6)
            ax3.set_xlabel('Y')
            ax3.set_ylabel('Z')
            ax3.set_title('YZ View (Opacity)')
            plt.colorbar(scatter3, ax=ax3, label='Opacity')
            ax3.set_aspect('equal', adjustable='box')

            ax4 = fig.add_subplot(2, 4, 4)
            ax4.hist(opacities[:, 0], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax4.set_xlabel('Opacity')
            ax4.set_ylabel('Count')
            ax4.set_title(f'Opacity Distribution (Total: {total_gaussians})')
            ax4.axvline(x=np.mean(opacities[:, 0]), color='red', linestyle='--',
                        label=f'Mean: {np.mean(opacities[:, 0]):.3f}')
            ax4.legend()

            ax5 = fig.add_subplot(2, 4, 5)
            scatter5 = ax5.scatter(means[:, 0], means[:, 1], c=scale_norms, s=2, cmap='plasma', alpha=0.6)
            ax5.set_xlabel('X')
            ax5.set_ylabel('Y')
            ax5.set_title('XY View (Size)')
            plt.colorbar(scatter5, ax=ax5, label='Size')
            ax5.set_aspect('equal', adjustable='box')

            ax6 = fig.add_subplot(2, 4, 6)
            scatter6 = ax6.scatter(means[:, 0], means[:, 2], c=scale_norms, s=2, cmap='plasma', alpha=0.6)
            ax6.set_xlabel('X')
            ax6.set_ylabel('Z')
            ax6.set_title('XZ View (Size)')
            plt.colorbar(scatter6, ax=ax6, label='Size')
            ax6.set_aspect('equal', adjustable='box')

            ax7 = fig.add_subplot(2, 4, 7)
            scatter7 = ax7.scatter(means[:, 1], means[:, 2], c=scale_norms, s=2, cmap='plasma', alpha=0.6)
            ax7.set_xlabel('Y')
            ax7.set_ylabel('Z')
            ax7.set_title('YZ View (Size)')
            plt.colorbar(scatter7, ax=ax7, label='Size')
            ax7.set_aspect('equal', adjustable='box')

            ax8 = fig.add_subplot(2, 4, 8)
            ax8.hist(scale_norms, bins=50, color='darkorange', edgecolor='black', alpha=0.7)
            ax8.set_xlabel('Size')
            ax8.set_ylabel('Count')
            ax8.set_title(f'Size Distribution (Total: {total_gaussians})')
            ax8.axvline(x=np.mean(scale_norms), color='red', linestyle='--',
                        label=f'Mean: {np.mean(scale_norms):.3f}')
            ax8.legend()

            plt.tight_layout()
            gaussian_viz_path = viz_output_dir / f'gaussian_distribution_epoch_{epoch:04d}.png'
            plt.savefig(gaussian_viz_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            self.log(f"可视化保存失败: {str(e)}")

        try:
            device = gaussian_model._means.device
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
                    norm_factor = camera.normalization_factor
                    target_image = camera.image.cpu().numpy() / norm_factor

                    rendered_image, _ = self._render_view_for_saving(gaussian_model, camera, device)
                    rendered_image = rendered_image.detach().cpu().numpy()

                    if target_image.shape != rendered_image.shape:
                        from scipy.ndimage import zoom
                        zoom_factors = (target_image.shape[0] / rendered_image.shape[0],
                                      target_image.shape[1] / rendered_image.shape[1])
                        rendered_image = zoom(rendered_image, zoom_factors)

                    target_image_min = float(target_image.min())
                    target_image_max = float(target_image.max())
                    target_image_mean = float(target_image.mean())
                    rendered_image_min = float(rendered_image.min())
                    rendered_image_max = float(rendered_image.max())
                    rendered_image_mean = float(rendered_image.mean())

                    target_image_norm = (target_image - target_image_min) / (target_image_max - target_image_min) if target_image_max > target_image_min else target_image
                    rendered_image_norm = (rendered_image - rendered_image_min) / (rendered_image_max - rendered_image_min) if rendered_image_max > rendered_image_min else rendered_image

                    overlay = np.stack([
                        target_image_norm,
                        rendered_image_norm,
                        target_image_norm * 0.5 + rendered_image_norm * 0.5
                    ], axis=-1)

                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    fig.suptitle(f'Epoch {epoch:04d} - View {selected_idx} (Incidence: {camera.incidence_angle:.1f}deg, Track: {camera.track_angle:.1f}deg)', fontsize=12)

                    axes[0].imshow(target_image_norm, cmap='gray', vmin=0, vmax=1)
                    axes[0].set_title(f'Real Image\nMin: {target_image_min:.4f}, Max: {target_image_max:.4f}, Mean: {target_image_mean:.4f}')
                    axes[0].axis('off')

                    axes[1].imshow(rendered_image_norm, cmap='gray', vmin=0, vmax=1)
                    axes[1].set_title(f'Reendered Image\nMin: {rendered_image_min:.4f}, Max: {rendered_image_max:.4f}, Mean: {rendered_image_mean:.4f}')
                    axes[1].axis('off')

                    axes[2].imshow(overlay)
                    axes[2].set_title('Overlay (R=Real, G=Rendered)')
                    axes[2].axis('off')

                    plt.tight_layout()
                    save_path = render_output_dir / f"epoch_{epoch:04d}_view_{selected_idx:02d}.png"
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    plt.close('all')

            gaussian_model.train()
        except Exception as e:
            self.log(f"渲染图像保存失败: {str(e)}")

    def on_closing(self):
        if self.is_training:
            if tk.messagebox.askokcancel("退出", "训练正在进行中，确定要退出吗？"):
                self.stop_event.set()
                self.root.destroy()
        else:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = SARGSTrainingGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()