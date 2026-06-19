#!/usr/bin/env python3
"""
实验2：自适应PSO变体比较
======================
比较标准PSO、线性递减惯性权重PSO（LDWPSO）、收缩因子PSO（CFPSO）
和综合学习PSO（CLPSO）在GR4J模型率定中的性能差异。
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data, split_data
from src.hydro_model import gr4j, GR4J_BOUNDS, GR4J_PARAM_NAMES
from src.objective_functions import create_objective, nse, kge, rmse, pbias
from src.pso_standard import pso_optimize
from src.pso_adaptive import ldw_pso_optimize, cf_pso_optimize, clpso_optimize
from src.visualization import (
    plot_boxplot, plot_convergence, plot_hydrograph
)
import matplotlib.pyplot as plt


import multiprocessing


def run_single_run(args):
    """单次运行的辅助包装器，以支持多进程并行化"""
    alg_func, objective, bounds, n_particles, max_iter, seed = args
    return alg_func(
        objective_func=objective,
        bounds=bounds,
        n_particles=n_particles,
        max_iter=max_iter,
        seed=seed,
        verbose=False
    )


def main():
    """主函数：执行自适应PSO变体比较实验"""
    # ============================================================
    # 1. 数据准备
    # ============================================================
    print("=" * 60)
    print("实验2：自适应PSO变体比较")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / 'data' / '66193.csv'
    results_dir = project_root / 'results'
    os.makedirs(results_dir, exist_ok=True)

    data = load_data(str(data_path))
    cal_data, val_data = split_data(data, split_year=1990)
    print(f"率定期记录数: {len(cal_data.date)}, 验证期记录数: {len(val_data.date)}")

    # 创建目标函数
    objective = create_objective(
        cal_data.rainfall, cal_data.pet, cal_data.runoff_obs,
        metric='nse', warmup=365
    )

    # ============================================================
    # 2. 定义算法
    # ============================================================
    algorithms = {
        '标准PSO': pso_optimize,
        'LDWPSO': ldw_pso_optimize,
        'CFPSO': cf_pso_optimize,
        'CLPSO': clpso_optimize,
    }

    n_runs = 10
    n_particles = 50
    max_iter = 200

    # ============================================================
    # 3. 多次独立运行（并行加速）
    # ============================================================
    all_results = {}  # {算法名: [OptimizationResult, ...]}
    all_fitness = {}  # {算法名: [best_fitness, ...]}

    for alg_name, alg_func in algorithms.items():
        print(f"\n{'=' * 60}")
        print(f"运行算法: {alg_name} （{n_runs}次独立运行，并行加速）")
        print(f"{'=' * 60}")

        # 构造多进程任务参数
        args_list = [
            (alg_func, objective, GR4J_BOUNDS, n_particles, max_iter, run_idx)
            for run_idx in range(n_runs)
        ]

        # 启动多进程计算
        with multiprocessing.Pool(processes=4) as pool:
            results_list = pool.map(run_single_run, args_list)

        fitness_list = [r.best_fitness for r in results_list]
        for run_idx, res in enumerate(results_list):
            print(f"  第 {run_idx + 1}/{n_runs} 次运行: 最优适应度 = {res.best_fitness:.6f}, 耗时 = {res.wall_time:.2f}s")

        all_results[alg_name] = results_list
        all_fitness[alg_name] = fitness_list

    # ============================================================
    # 4. 统计结果
    # ============================================================
    print("\n" + "=" * 60)
    print("统计结果汇总（10次独立运行）")
    print("=" * 60)
    print(f"{'算法':<12} {'均值':<12} {'标准差':<12} {'最优':<12} {'最差':<12}")
    print("-" * 60)

    best_runs = {}  # 记录每个算法的最佳运行
    for alg_name in algorithms:
        fitness = np.array(all_fitness[alg_name])
        mean_f = np.mean(fitness)
        std_f = np.std(fitness)
        best_f = np.min(fitness)
        worst_f = np.max(fitness)
        best_idx = np.argmin(fitness)
        best_runs[alg_name] = all_results[alg_name][best_idx]

        print(f"{alg_name:<12} {mean_f:<12.6f} {std_f:<12.6f} {best_f:<12.6f} {worst_f:<12.6f}")

    # ============================================================
    # 5. 最佳运行的验证期评估
    # ============================================================
    print("\n" + "=" * 60)
    print("各算法最佳运行的验证期评估")
    print("=" * 60)

    warmup_val = min(365, len(val_data.date) // 4)
    val_metrics = {}

    for alg_name, best_result in best_runs.items():
        sim_val = gr4j(best_result.best_params, val_data.rainfall, val_data.pet)
        obs_val = val_data.runoff_obs[warmup_val:]
        sim_val_eval = sim_val[warmup_val:]

        metrics = {
            'NSE': nse(obs_val, sim_val_eval),
            'KGE': kge(obs_val, sim_val_eval),
            'RMSE': rmse(obs_val, sim_val_eval),
            'PBIAS': pbias(obs_val, sim_val_eval),
        }
        val_metrics[alg_name] = metrics

        print(f"\n{alg_name}:")
        print(f"  参数: {', '.join(f'{n}={v:.4f}' for n, v in zip(GR4J_PARAM_NAMES, best_result.best_params))}")
        for metric_name, metric_val in metrics.items():
            print(f"  {metric_name}: {metric_val:.4f}")

    # ============================================================
    # 6. 生成图形
    # ============================================================
    print("\n" + "-" * 60)
    print("生成图形...")
    print("-" * 60)

    # 图1：箱线图
    print("  生成适应度箱线图...")
    plot_boxplot(
        data_dict=all_fitness,
        ylabel='最优适应度值',
        title='自适应PSO变体比较 - 10次运行箱线图',
        save_path=str(results_dir / 'exp2_boxplot.png')
    )

    # 图2：收敛曲线（每个算法的最佳运行）
    print("  生成收敛曲线...")
    histories = [best_runs[name].convergence_history for name in algorithms]
    labels = list(algorithms.keys())
    plot_convergence(
        histories=histories,
        labels=labels,
        title='自适应PSO变体比较 - 最佳运行收敛曲线',
        save_path=str(results_dir / 'exp2_convergence.png')
    )

    # 图3：验证期水文过程线（2x2子图）
    print("  生成验证期水文过程线...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (alg_name, best_result) in enumerate(best_runs.items()):
        sim_val = gr4j(best_result.best_params, val_data.rainfall, val_data.pet)
        ax = axes[idx]

        ax.plot(val_data.date, val_data.runoff_obs, 'b-', label='实测', alpha=0.7, linewidth=0.8)
        ax.plot(val_data.date, sim_val, 'r-', label='模拟', alpha=0.7, linewidth=0.8)
        ax.set_title(f'{alg_name} (NSE={val_metrics[alg_name]["NSE"]:.4f})')
        ax.set_xlabel('日期')
        ax.set_ylabel('流量')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('各算法最佳运行 - 验证期水文过程线', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(results_dir / 'exp2_val_hydrographs.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n所有图形已保存至 results/ 目录")
    print("实验2完成！")


if __name__ == '__main__':
    main()
