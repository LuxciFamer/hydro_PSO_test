#!/usr/bin/env python3
"""
实验5：PSO超参数敏感性分析
=========================
分析粒子数量、惯性权重和学习因子对PSO优化性能的影响。
"""

import sys
import os
import numpy as np
from pathlib import Path

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data, split_data
from src.hydro_model import GR4J_BOUNDS
from src.objective_functions import create_objective
from src.pso_standard import pso_optimize
from src.visualization import plot_heatmap
import matplotlib.pyplot as plt


import multiprocessing


def run_pso_worker_exp5(args):
    """单次运行的辅助包装器，以支持多进程并行化"""
    objective, bounds, n_particles, max_iter, w, c1, c2, seed = args
    from src.pso_standard import pso_optimize
    return pso_optimize(
        objective_func=objective,
        bounds=bounds,
        n_particles=n_particles,
        max_iter=max_iter,
        w=w,
        c1=c1,
        c2=c2,
        seed=seed,
        verbose=False
    )


def main():
    """主函数：执行PSO超参数敏感性分析"""
    # ============================================================
    # 1. 数据准备
    # ============================================================
    print("=" * 60)
    print("实验5：PSO超参数敏感性分析")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / 'data' / '66193.csv'
    results_dir = project_root / 'results'
    os.makedirs(results_dir, exist_ok=True)

    data = load_data(str(data_path))
    cal_data, val_data = split_data(data, split_year=1990)
    print(f"率定期记录数: {len(cal_data.date)}, 验证期记录数: {len(val_data.date)}")

    objective = create_objective(
        cal_data.rainfall, cal_data.pet, cal_data.runoff_obs,
        metric='nse', warmup=365
    )

    # ============================================================
    # 分析1：粒子数量敏感性
    # ============================================================
    print("\n" + "=" * 60)
    print("分析1：粒子数量敏感性（总评估次数固定为10000，并行加速）")
    print("=" * 60)

    n_particles_list = [20, 50, 100, 200]
    total_evals = 6000
    n_runs_particle = 3

    # 构造任务参数列表
    args_list_1 = []
    for n_p in n_particles_list:
        max_iter = total_evals // n_p
        for run_idx in range(n_runs_particle):
            args_list_1.append((objective, GR4J_BOUNDS, n_p, max_iter, 0.729, 1.494, 1.494, run_idx))

    # 并行运行
    print(f"正在并行运行 {len(args_list_1)} 个优化任务...")
    with multiprocessing.Pool(processes=4) as pool:
        results_1 = pool.map(run_pso_worker_exp5, args_list_1)

    # 整理结果
    particle_results = {n_p: [] for n_p in n_particles_list}
    idx = 0
    for n_p in n_particles_list:
        for _ in range(n_runs_particle):
            particle_results[n_p].append(results_1[idx].best_fitness)
            idx += 1

    # 打印粒子数量敏感性结果
    print("\n" + "-" * 50)
    print("粒子数量敏感性结果:")
    print(f"{'粒子数':<10} {'迭代数':<10} {'均值':<12} {'标准差':<12}")
    print("-" * 44)
    for n_p in n_particles_list:
        max_iter = total_evals // n_p
        fitness = np.array(particle_results[n_p])
        print(f"{n_p:<10} {max_iter:<10} {np.mean(fitness):<12.6f} {np.std(fitness):<12.6f}")

    # ============================================================
    # 分析2：惯性权重敏感性
    # ============================================================
    print("\n" + "=" * 60)
    print("分析2：惯性权重敏感性（50粒子, 200迭代，并行加速）")
    print("=" * 60)

    w_list = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_runs_w = 3

    # 构造任务参数列表
    args_list_2 = []
    for w in w_list:
        for run_idx in range(n_runs_w):
            args_list_2.append((objective, GR4J_BOUNDS, 50, 120, w, 1.494, 1.494, run_idx))

    # 并行运行
    print(f"正在并行运行 {len(args_list_2)} 个优化任务...")
    with multiprocessing.Pool(processes=4) as pool:
        results_2 = pool.map(run_pso_worker_exp5, args_list_2)

    # 整理结果
    w_results = {w: [] for w in w_list}
    idx = 0
    for w in w_list:
        for _ in range(n_runs_w):
            w_results[w].append(results_2[idx].best_fitness)
            idx += 1

    # 打印惯性权重敏感性结果
    print("\n" + "-" * 50)
    print("惯性权重敏感性结果:")
    print(f"{'w值':<10} {'均值':<12} {'标准差':<12}")
    print("-" * 34)
    for w in w_list:
        fitness = np.array(w_results[w])
        print(f"{w:<10.1f} {np.mean(fitness):<12.6f} {np.std(fitness):<12.6f}")

    # ============================================================
    # 分析3：c1/c2平衡热力图
    # ============================================================
    print("\n" + "=" * 60)
    print("分析3：c1/c2学习因子平衡分析（并行加速）")
    print("=" * 60)

    c1_list = [0.5, 1.5, 2.5]
    c2_list = [0.5, 1.5, 2.5]
    n_runs_c = 3

    # 构造任务参数列表
    args_list_3 = []
    for c2 in c2_list:
        for c1 in c1_list:
            for run_idx in range(n_runs_c):
                args_list_3.append((objective, GR4J_BOUNDS, 50, 120, 0.729, c1, c2, run_idx))

    # 并行运行
    print(f"正在并行运行 {len(args_list_3)} 个优化任务...")
    with multiprocessing.Pool(processes=4) as pool:
        results_3 = pool.map(run_pso_worker_exp5, args_list_3)

    # 整理结果
    c_matrix = np.zeros((len(c2_list), len(c1_list)))
    idx = 0
    for i, c2 in enumerate(c2_list):
        for j, c1 in enumerate(c1_list):
            fitness_list = []
            for _ in range(n_runs_c):
                fitness_list.append(results_3[idx].best_fitness)
                idx += 1
            c_matrix[i, j] = np.mean(fitness_list)

    # 打印c1/c2热力图数据
    print("\n" + "-" * 50)
    print("c1/c2学习因子热力图数据（均值适应度）:")
    print(f"{'c2\\c1':<8}", end='')
    for c1 in c1_list:
        print(f"{c1:<10.1f}", end='')
    print()
    print("-" * (8 + 10 * len(c1_list)))
    for i, c2 in enumerate(c2_list):
        print(f"{c2:<8.1f}", end='')
        for j in range(len(c1_list)):
            print(f"{c_matrix[i, j]:<10.6f}", end='')
        print()

    # ============================================================
    # 最优超参数组合
    # ============================================================
    print("\n" + "=" * 60)
    print("最优超参数组合总结")
    print("=" * 60)

    # 粒子数量最优
    best_np = min(particle_results, key=lambda k: np.mean(particle_results[k]))
    print(f"\n最优粒子数量: {best_np} (均值适应度: {np.mean(particle_results[best_np]):.6f})")

    # 惯性权重最优
    best_w = min(w_results, key=lambda k: np.mean(w_results[k]))
    print(f"最优惯性权重: {best_w} (均值适应度: {np.mean(w_results[best_w]):.6f})")

    # c1/c2最优
    best_idx = np.unravel_index(np.argmin(c_matrix), c_matrix.shape)
    best_c2 = c2_list[best_idx[0]]
    best_c1 = c1_list[best_idx[1]]
    print(f"最优学习因子: c1={best_c1}, c2={best_c2} (均值适应度: {c_matrix[best_idx]:.6f})")

    # ============================================================
    # 生成图形
    # ============================================================
    print("\n" + "-" * 60)
    print("生成图形...")
    print("-" * 60)

    # 图1：粒子数量敏感性
    print("  生成粒子数量敏感性图...")
    fig, ax = plt.subplots(figsize=(10, 6))
    means = [np.mean(particle_results[n_p]) for n_p in n_particles_list]
    stds = [np.std(particle_results[n_p]) for n_p in n_particles_list]
    ax.errorbar(n_particles_list, means, yerr=stds, marker='o', capsize=5,
                linewidth=2, markersize=8, color='#2196F3', ecolor='#90CAF9')
    ax.set_xlabel('粒子数量', fontsize=12)
    ax.set_ylabel('均值最优适应度', fontsize=12)
    ax.set_title('粒子数量敏感性分析（总评估次数=10000）', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_particles_list)
    plt.tight_layout()
    plt.savefig(str(results_dir / 'exp5_particle_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图2：惯性权重敏感性
    print("  生成惯性权重敏感性图...")
    fig, ax = plt.subplots(figsize=(10, 6))
    means_w = [np.mean(w_results[w]) for w in w_list]
    stds_w = [np.std(w_results[w]) for w in w_list]
    ax.errorbar(w_list, means_w, yerr=stds_w, marker='s', capsize=5,
                linewidth=2, markersize=8, color='#FF5722', ecolor='#FFAB91')
    ax.set_xlabel('惯性权重 w', fontsize=12)
    ax.set_ylabel('均值最优适应度', fontsize=12)
    ax.set_title('惯性权重敏感性分析', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(w_list)
    plt.tight_layout()
    plt.savefig(str(results_dir / 'exp5_inertia_sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 图3：c1/c2热力图
    print("  生成c1/c2热力图...")
    plot_heatmap(
        data=c_matrix,
        xlabels=[str(c) for c in c1_list],
        ylabels=[str(c) for c in c2_list],
        title='学习因子c1/c2敏感性热力图（均值适应度）',
        save_path=str(results_dir / 'exp5_c1c2_heatmap.png')
    )

    print("\n所有图形已保存至 results/ 目录")
    print("实验5完成！")


if __name__ == '__main__':
    main()
