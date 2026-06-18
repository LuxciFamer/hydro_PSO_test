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
    print("分析1：粒子数量敏感性（总评估次数固定为10000）")
    print("=" * 60)

    n_particles_list = [10, 20, 30, 50, 80, 100, 150, 200]
    total_evals = 10000
    n_runs_particle = 5

    particle_results = {}  # {n_particles: [fitness values]}

    for n_p in n_particles_list:
        max_iter = total_evals // n_p
        print(f"\n  粒子数={n_p}, 迭代数={max_iter} (总评估={n_p * max_iter})")

        fitness_list = []
        for run_idx in range(n_runs_particle):
            result = pso_optimize(
                objective_func=objective,
                bounds=GR4J_BOUNDS,
                n_particles=n_p,
                max_iter=max_iter,
                seed=run_idx,
                verbose=False
            )
            fitness_list.append(result.best_fitness)
            print(f"    运行 {run_idx + 1}: 适应度={result.best_fitness:.6f}")

        particle_results[n_p] = fitness_list

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
    print("分析2：惯性权重敏感性（50粒子, 200迭代）")
    print("=" * 60)

    w_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    n_runs_w = 5

    w_results = {}  # {w: [fitness values]}

    for w in w_list:
        print(f"\n  惯性权重 w={w}")
        fitness_list = []
        for run_idx in range(n_runs_w):
            result = pso_optimize(
                objective_func=objective,
                bounds=GR4J_BOUNDS,
                n_particles=50,
                max_iter=200,
                w=w,
                c1=1.494,
                c2=1.494,
                seed=run_idx,
                verbose=False
            )
            fitness_list.append(result.best_fitness)
            print(f"    运行 {run_idx + 1}: 适应度={result.best_fitness:.6f}")

        w_results[w] = fitness_list

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
    print("分析3：c1/c2学习因子平衡分析")
    print("=" * 60)

    c1_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    c2_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    n_runs_c = 3

    c_matrix = np.zeros((len(c2_list), len(c1_list)))

    for i, c2 in enumerate(c2_list):
        for j, c1 in enumerate(c1_list):
            print(f"  c1={c1}, c2={c2}")
            fitness_list = []
            for run_idx in range(n_runs_c):
                result = pso_optimize(
                    objective_func=objective,
                    bounds=GR4J_BOUNDS,
                    n_particles=50,
                    max_iter=200,
                    w=0.729,
                    c1=c1,
                    c2=c2,
                    seed=run_idx,
                    verbose=False
                )
                fitness_list.append(result.best_fitness)

            mean_fitness = np.mean(fitness_list)
            c_matrix[i, j] = mean_fitness
            print(f"    均值适应度: {mean_fitness:.6f}")

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
