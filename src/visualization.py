"""
水文模型率定可视化工具包

提供用于水文模型参数率定与分析的综合可视化功能，
包括水文过程线、收敛曲线、散点图、箱线图、Pareto前沿、热力图、参数演化和流量历时曲线。
所有图形均采用学术论文级别的排版质量。
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Glyph.*missing from.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Font.*does not have a glyph.*")

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

import matplotlib
matplotlib.use('Agg')  # 设置非交互式后端，适用于服务器环境

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ========== 字体设置 ==========
import os
import matplotlib.font_manager as fm

# 尝试在 WSL/Linux 环境下加载 Windows 挂载的或系统的中文字体文件
possible_font_paths = [
    '/mnt/c/Windows/Fonts/simhei.ttf',
    '/mnt/c/Windows/Fonts/SimHei.ttf',
    '/mnt/c/Windows/Fonts/msyh.ttc',      # 微软雅黑
    '/mnt/c/Windows/Fonts/msyh.ttf',
    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc', # 文泉驿微米黑
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', # Noto Sans CJK
]

for path in possible_font_paths:
    if os.path.exists(path):
        try:
            fm.fontManager.addfont(path)
        except Exception:
            pass

# ========== Seaborn 样式和配色 ==========
sns.set_style('whitegrid', rc={
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
})
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 专业配色方案
COLORS = [
    '#1f77b4',  # 蓝色
    '#d62728',  # 红色
    '#2ca02c',  # 绿色
    '#ff7f0e',  # 橙色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
    '#bcbd22',  # 黄绿色
    '#17becf',  # 青色
]

# 不同线型，用于区分多条曲线
LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']


def plot_hydrograph(dates, rainfall, obs, sim, title='', save_path=None):
    """
    绘制水文过程线图（双轴图）。

    上方为倒置的降雨柱状图，下方为实测与模拟径流过程线对比。

    参数:
        dates: 日期序列（x轴）。
        rainfall: 降雨量序列，与 dates 等长。
        obs: 实测径流序列。
        sim: 模拟径流序列。
        title: 图形标题，默认为空。
        save_path: 保存路径，若提供则保存图形并关闭，默认 None。

    返回:
        fig, (ax1, ax2): 图形对象和坐标轴元组，便于进一步自定义。
    """
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # ---- 上方：降雨倒置柱状图 ----
    ax_rain = ax1.twinx()
    ax_rain.bar(
        dates, rainfall,
        color='#4a90d9', alpha=0.6, width=0.8,
        label='降雨量'
    )
    ax_rain.set_ylim(max(rainfall) * 2.5, 0)  # 倒置y轴，柱状图从顶部向下
    ax_rain.set_ylabel('降雨量 (mm)', fontsize=12, color='#4a90d9')
    ax_rain.tick_params(axis='y', labelcolor='#4a90d9')

    # ---- 下方：实测与模拟径流 ----
    ax1.plot(
        dates, obs,
        color='black', linewidth=1.5, label='实测径流',
        zorder=3
    )
    ax1.plot(
        dates, sim,
        color='#d62728', linewidth=1.2, linestyle='--', label='模拟径流',
        alpha=0.85, zorder=2
    )

    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_ylabel('径流 ($m^3/s$)', fontsize=12)
    ax1.set_title(title if title else '水文过程线对比图', fontsize=14, fontweight='bold')

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax_rain.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2, labels_1 + labels_2,
        loc='upper right', fontsize=11, framealpha=0.9
    )

    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, (ax1, ax_rain)


def plot_convergence(histories, labels, title='', save_path=None):
    """
    绘制多条收敛曲线对比图。

    参数:
        histories: 收敛历史列表的列表，每个子列表为一个算法的收敛序列。
        labels: 各算法名称列表，与 histories 一一对应。
        title: 图形标题，默认为空。
        save_path: 保存路径，若提供则保存图形并关闭，默认 None。

    返回:
        fig, ax: 图形对象和坐标轴，便于进一步自定义。
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (history, label) in enumerate(zip(histories, labels)):
        color = COLORS[i % len(COLORS)]
        linestyle = LINESTYLES[i % len(LINESTYLES)]
        ax.plot(
            history, label=label,
            color=color, linestyle=linestyle, linewidth=1.8
        )

    # 判断是否需要对数坐标（值域跨越超过2个数量级）
    all_values = []
    for h in histories:
        all_values.extend([v for v in h if v > 0])
    if len(all_values) > 0:
        val_range = max(all_values) / max(min(all_values), 1e-15)
        if val_range > 100:
            ax.set_yscale('log')
            ax.set_ylabel('目标函数值（对数尺度）', fontsize=12)
        else:
            ax.set_ylabel('目标函数值', fontsize=12)
    else:
        ax.set_ylabel('目标函数值', fontsize=12)

    ax.set_xlabel('迭代次数', fontsize=12)
    ax.set_title(title if title else '算法收敛曲线对比', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, ax


def plot_scatter(obs, sim, title='', save_path=None):
    """
    绘制实测值与模拟值的散点对比图。

    包含 1:1 参考线、线性回归线和 R² 值标注。

    参数:
        obs: 实测值数组。
        sim: 模拟值数组。
        title: 图形标题，默认为空。
        save_path: 保存路径，若提供则保存图形并关闭，默认 None。

    返回:
        fig, ax: 图形对象和坐标轴，便于进一步自定义。
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8, 8))

    # 散点图
    ax.scatter(obs, sim, c='#1f77b4', alpha=0.5, s=20, edgecolors='none', label='数据点')

    # 确定坐标范围
    all_vals = np.concatenate([obs, sim])
    val_min = np.min(all_vals)
    val_max = np.max(all_vals)
    margin = (val_max - val_min) * 0.05
    plot_min = val_min - margin
    plot_max = val_max + margin

    # 1:1 参考线
    ax.plot(
        [plot_min, plot_max], [plot_min, plot_max],
        'k--', linewidth=1.2, label='1:1 线', alpha=0.7
    )

    # 线性回归
    coeffs = np.polyfit(obs, sim, 1)
    regression_line = np.poly1d(coeffs)
    x_fit = np.linspace(plot_min, plot_max, 100)
    ax.plot(x_fit, regression_line(x_fit), color='#d62728', linewidth=1.5, label='回归线')

    # 计算 R²
    ss_res = np.sum((sim - obs) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # R² 标注
    ax.text(
        0.05, 0.95, f'$R^2$ = {r_squared:.4f}',
        transform=ax.transAxes, fontsize=13,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8)
    )

    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_xlabel('实测值', fontsize=12)
    ax.set_ylabel('模拟值', fontsize=12)
    ax.set_title(title if title else '实测值 vs 模拟值', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, ax


def plot_boxplot(data_dict, ylabel='', title='', save_path=None):
    """
    绘制算法对比箱线图。

    使用 seaborn 绘制箱线图，叠加带抖动的散点以展示数据分布。

    参数:
        data_dict: 字典，键为算法名称，值为对应的数值列表。
        ylabel: y轴标签，默认为空。
        title: 图形标题，默认为空。
        save_path: 保存路径，若提供则保存图形并关闭，默认 None。

    返回:
        fig, ax: 图形对象和坐标轴，便于进一步自定义。
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # 将字典数据转化为长格式列表
    plot_data = []
    plot_labels = []
    for name, values in data_dict.items():
        plot_data.extend(values)
        plot_labels.extend([name] * len(values))

    # 绘制箱线图
    sns.boxplot(
        x=plot_labels, y=plot_data, ax=ax,
        palette=COLORS[:len(data_dict)],
        width=0.5, linewidth=1.5,
        fliersize=0  # 隐藏默认异常值点，使用 stripplot 替代
    )

    # 叠加带抖动的散点
    sns.stripplot(
        x=plot_labels, y=plot_data, ax=ax,
        color='black', size=4, alpha=0.4, jitter=0.2
    )

    ax.set_ylabel(ylabel if ylabel else '值', fontsize=12)
    ax.set_xlabel('算法', fontsize=12)
    ax.set_title(title if title else '算法对比箱线图', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, ax


def plot_pareto_front(pareto_front, obj_names=None, title='', save_path=None):
    """
    绘制二维 Pareto 前沿散点图。

    若可能，自动识别并高亮拐点（knee point）。

    参数:
        pareto_front: Pareto 前沿点集，形状为 (n_points, 2) 的数组。
        obj_names: 两个目标函数的名称列表，默认 ['目标1', '目标2']。
        title: 图形标题，默认为空。
        save_path: 保存路径，若提供则保存图形并关闭，默认 None。

    返回:
        fig, ax: 图形对象和坐标轴，便于进一步自定义。
    """
    pareto_front = np.asarray(pareto_front, dtype=np.float64)
    if obj_names is None:
        obj_names = ['目标1', '目标2']

    fig, ax = plt.subplots(figsize=(9, 7))

    # 按第一个目标排序以绘制连线
    sorted_indices = np.argsort(pareto_front[:, 0])
    pf_sorted = pareto_front[sorted_indices]

    # 绘制 Pareto 前沿散点和连线
    ax.plot(
        pf_sorted[:, 0], pf_sorted[:, 1],
        'o-', color=COLORS[0], markersize=6, linewidth=1.5,
        markerfacecolor=COLORS[0], markeredgecolor='white', markeredgewidth=0.8,
        label='Pareto 前沿'
    )

    # 尝试识别拐点（距离对角线最远的点）
    if len(pf_sorted) >= 3:
        try:
            # 归一化到 [0, 1]
            f1_min, f1_max = pf_sorted[:, 0].min(), pf_sorted[:, 0].max()
            f2_min, f2_max = pf_sorted[:, 1].min(), pf_sorted[:, 1].max()
            f1_range = f1_max - f1_min if f1_max > f1_min else 1.0
            f2_range = f2_max - f2_min if f2_max > f2_min else 1.0

            f1_norm = (pf_sorted[:, 0] - f1_min) / f1_range
            f2_norm = (pf_sorted[:, 1] - f2_min) / f2_range

            # 拐点：距对角线 (0,1)-(1,0) 最远的点
            # 直线方程: x + y - 1 = 0，距离 = |x + y - 1| / sqrt(2)
            distances = np.abs(f1_norm + f2_norm - 1.0) / np.sqrt(2)
            knee_idx = np.argmax(distances)

            ax.scatter(
                pf_sorted[knee_idx, 0], pf_sorted[knee_idx, 1],
                s=200, c='#d62728', marker='*', zorder=5,
                edgecolors='black', linewidths=0.8,
                label=f'拐点'
            )
        except Exception:
            pass  # 拐点识别失败时静默跳过

    ax.set_xlabel(obj_names[0], fontsize=12)
    ax.set_ylabel(obj_names[1], fontsize=12)
    ax.set_title(title if title else 'Pareto 前沿', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, ax


def plot_heatmap(data, xlabels, ylabels, title='', save_path=None):
    """
    绘制带数值标注的热力图。

    参数:
        data: 二维数组或矩阵，形状为 (n_rows, n_cols)。
        xlabels: x轴标签列表（列标签）。
        ylabels: y轴标签列表（行标签）。
        title: 图形标题，默认为空。
        save_path: 保存路径，若提供则保存图形并关闭，默认 None。

    返回:
        fig, ax: 图形对象和坐标轴，便于进一步自定义。
    """
    data = np.asarray(data, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(max(8, len(xlabels) * 1.2), max(6, len(ylabels) * 0.8)))

    sns.heatmap(
        data, ax=ax,
        xticklabels=xlabels,
        yticklabels=ylabels,
        annot=True, fmt='.3f',
        cmap='YlOrRd',
        linewidths=0.5, linecolor='white',
        cbar_kws={'label': '数值'},
        square=False
    )

    ax.set_title(title if title else '热力图', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, ax


def plot_parameter_evolution(histories_params, param_names, title='', save_path=None):
    """
    绘制优化过程中参数演化图。

    每个参数单独一个子图，展示参数值随迭代次数的变化趋势。

    参数:
        histories_params: 参数历史记录，形状为 (n_iterations, n_params) 的二维数组，
                          或等长列表的列表。
        param_names: 参数名称列表，与列数对应。
        title: 图形总标题，默认为空。
        save_path: 保存路径，若提供则保存图形并关闭，默认 None。

    返回:
        fig, axes: 图形对象和坐标轴数组，便于进一步自定义。
    """
    histories_params = np.asarray(histories_params, dtype=np.float64)
    n_params = histories_params.shape[1] if histories_params.ndim == 2 else 1

    # 计算子图布局：每行最多3列
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # 统一为一维数组方便索引
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i in range(n_params):
        ax = axes[i]
        color = COLORS[i % len(COLORS)]
        param_name = param_names[i] if i < len(param_names) else f'参数{i + 1}'

        ax.plot(
            histories_params[:, i],
            color=color, linewidth=1.2, alpha=0.8
        )
        ax.set_title(param_name, fontsize=11, fontweight='bold')
        ax.set_xlabel('迭代次数', fontsize=10)
        ax.set_ylabel('参数值', fontsize=10)
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        title if title else '参数演化过程',
        fontsize=14, fontweight='bold', y=1.02
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, axes


def plot_flow_duration_curve(obs, sim, title='', save_path=None):
    """
    绘制流量历时曲线。

    将流量从大到小排序，绘制超越概率与流量的关系，
    用于对比实测与模拟的流量分布特征。

    参数:
        obs: 实测流量序列。
        sim: 模拟流量序列。
        title: 图形标题，默认为空。
        save_path: 保存路径，若提供则保存图形并关闭，默认 None。

    返回:
        fig, ax: 图形对象和坐标轴，便于进一步自定义。
    """
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)

    # 降序排列
    obs_sorted = np.sort(obs)[::-1]
    sim_sorted = np.sort(sim)[::-1]

    # 计算超越概率 (Weibull plotting position)
    n_obs = len(obs_sorted)
    n_sim = len(sim_sorted)
    exceedance_obs = np.arange(1, n_obs + 1) / (n_obs + 1) * 100
    exceedance_sim = np.arange(1, n_sim + 1) / (n_sim + 1) * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        exceedance_obs, obs_sorted,
        color='black', linewidth=1.8, label='实测流量',
        zorder=3
    )
    ax.plot(
        exceedance_sim, sim_sorted,
        color='#d62728', linewidth=1.5, linestyle='--', label='模拟流量',
        alpha=0.85, zorder=2
    )

    ax.set_xlabel('超越概率 (%)', fontsize=12)
    ax.set_ylabel('流量 ($m^3/s$)', fontsize=12)
    ax.set_title(title if title else '流量历时曲线', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, framealpha=0.9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')

    # x轴从0到100
    ax.set_xlim(0, 100)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    return fig, ax
