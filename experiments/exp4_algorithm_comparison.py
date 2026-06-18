#!/usr/bin/env python3
"""
实验4：跨算法比较（PSO vs DE vs SA）
===================================
比较粒子群优化（PSO）、差分进化（DE）和模拟退火（SA）
在GR4J模型率定中的性能，包括统计检验。
"""

import sys
import os
import numpy as np
from pathlib import Path
from scipy import stats

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_data, split_data
from src.hydro_model import gr4j, GR4J_BOUNDS, GR4J_PARAM_NAMES
from src.objective_functions import create_objective, nse, kge, rmse, pbias
from src.pso_standard import pso_optimize
from src.differential_evolution import de_optimize
from src.simulated_annealing import sa_optimize
from src.visualization import plot_boxplot, plot_convergence, plot_hydrograph
import matplotlib.pyplot as plt


def main():
    """主函数：执行跨算法比较实验"""
    # ============================================================
    # 1. 数据准备
    # ============================================================
    print("=" * 60)
    print("实验4：跨算法比较（PSO vs DE vs SA）")
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
    # 2. 定义算法（统一评估预算约10000次）
    # ============================================================
    n_runs = 10

    def run_pso(seed):
        """运行标准PSO: 50粒子 x 200迭代 = 10000次评估"""
        return pso_optimize(
            objective_func=objective,
            bounds=GR4J_BOUNDS,
            n_particles=50,
            max_iter=200,
            seed=seed,
            verbose=False
        )

    def run_de(seed):
        """运行差分进化: 50种群 x 200代 = 10000次评估"""
        return de_optimize(
            objective_func=objective,
            bounds=GR4J_BOUNDS,
            pop_size=50,
            max_iter=200,
            F=0.8,
            CR=0.9,
            seed=seed,
            verbose=False
        )

    def run_sa(seed):
        """运行模拟退火: 10000迭代 x 5重启"""
        return sa_optimize(
            objective_func=objective,
            bounds=GR4J_BOUNDS,
            max_iter=10000,
            T0=100.0,
            alpha=0.995,
            n_restarts=5,
            seed=seed,
            verbose=False
        )

    algorithms = {
        'PSO': run_pso,
        'DE': run_de,
        'SA': run_sa,
    }

    # ============================================================
    # 3. 多次独立运行
    # ============================================================
    all_results = {}
    all_fitness = {}

    for alg_name, alg_func in algorithms.items():
        print(f"\n{'=' * 60}")
        print(f"运行算法: {alg_name} （{n_runs}次独立运行）")
        print(f"{'=' * 60}")

        results_list = []
        fitness_list = []

        for run_idx in range(n_runs):
            print(f"  第 {run_idx + 1}/{n_runs} 次运行 (seed={run_idx})...")
            result = alg_func(seed=run_idx)
            results_list.append(result)
            fitness_list.append(result.best_fitness)
            print(f"    最优适应度: {result.best_fitness:.6f}, 耗时: {result.wall_time:.2f}秒")

        all_results[alg_name] = results_list
        all_fitness[alg_name] = fitness_list

    # ============================================================
    # 4. 统计结果
    # ============================================================
    print("\n" + "=" * 60)
    print("统计结果汇总（10次独立运行）")
    print("=" * 60)
    print(f"{'算法':<8} {'均值':<12} {'标准差':<12} {'最优':<12} {'最差':<12} {'平均耗时(s)':<12}")
    print("-" * 68)

    best_runs = {}
    median_runs = {}

    for alg_name in algorithms:
        fitness = np.array(all_fitness[alg_name])
        times = [r.wall_time for r in all_results[alg_name]]
        mean_f = np.mean(fitness)
        std_f = np.std(fitness)
        best_f = np.min(fitness)
        worst_f = np.max(fitness)
        mean_t = np.mean(times)

        best_idx = np.argmin(fitness)
        median_idx = np.argsort(fitness)[len(fitness) // 2]

        best_runs[alg_name] = all_results[alg_name][best_idx]
        median_runs[alg_name] = all_results[alg_name][median_idx]

        print(f"{alg_name:<8} {mean_f:<12.6f} {std_f:<12.6f} {best_f:<12.6f} {worst_f:<12.6f} {mean_t:<12.2f}")

    # ============================================================
    # 5. Wilcoxon秩和检验
    # ============================================================
    print("\n" + "=" * 60)
    print("Wilcoxon秩和检验（成对比较）")
    print("=" * 60)

    alg_names = list(algorithms.keys())
    print(f"{'比较':<15} {'统计量':<12} {'p值':<12} {'显著性(α=0.05)':<15}")
    print("-" * 54)

    for i in range(len(alg_names)):
        for j in range(i + 1, len(alg_names)):
            name_i, name_j = alg_names[i], alg_names[j]
            fitness_i = np.array(all_fitness[name_i])
            fitness_j = np.array(all_fitness[name_j])

            try:
                stat, p_value = stats.wilcoxon(fitness_i, fitness_j)
                sig = "显著 *" if p_value < 0.05 else "不显著"
                print(f"{name_i} vs {name_j:<8} {stat:<12.4f} {p_value:<12.6f} {sig}")
            except ValueError as e:
                print(f"{name_i} vs {name_j:<8} {'N/A':<12} {'N/A':<12} {str(e)}")

    # ============================================================
    # 6. 验证期评估（最佳运行）
    # ============================================================
    print("\n" + "=" * 60)
    print("各算法最佳运行 - 验证期评估")
    print("=" * 60)

    warmup_val = min(365, len(val_data.date) // 4)

    for alg_name, best_result in best_runs.items():
        sim_val = gr4j(best_result.best_params, val_data.rainfall, val_data.pet)
        obs_val = val_data.runoff_obs[warmup_val:]
        sim_val_eval = sim_val[warmup_val:]

        val_nse = nse(obs_val, sim_val_eval)
        val_kge = kge(obs_val, sim_val_eval)
        val_rmse = rmse(obs_val, sim_val_eval)
        val_pbias_val = pbias(obs_val, sim_val_eval)

        print(f"\n{alg_name}:")
        print(f"  参数: {', '.join(f'{n}={v:.4f}' for n, v in zip(GR4J_PARAM_NAMES, best_result.best_params))}")
        print(f"  NSE={val_nse:.4f}, KGE={val_kge:.4f}, RMSE={val_rmse:.4f}, PBIAS={val_pbias_val:.4f}%")

    # ============================================================
    # 7. 生成图形
    # ============================================================
    print("\n" + "-" * 60)
    print("生成图形...")
    print("-" * 60)

    # 图1：箱线图
    print("  生成适应度箱线图...")
    plot_boxplot(
        data_dict=all_fitness,
        ylabel='最优适应度值',
        title='跨算法比较 - 10次运行箱线图',
        save_path=str(results_dir / 'exp4_boxplot.png')
    )

    # 图2：收敛曲线（中位数运行）
    print("  生成收敛曲线（中位数运行）...")
    histories = [median_runs[name].convergence_history for name in algorithms]
    labels = list(algorithms.keys())
    plot_convergence(
        histories=histories,
        labels=labels,
        title='跨算法比较 - 中位数运行收敛曲线',
        save_path=str(results_dir / 'exp4_convergence.png')
    )

    # 图3：验证期水文过程线（3个子图）
    print("  生成验证期水文过程线...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    for idx, (alg_name, best_result) in enumerate(best_runs.items()):
        sim_val = gr4j(best_result.best_params, val_data.rainfall, val_data.pet)
        obs_val = val_data.runoff_obs[warmup_val:]
        sim_val_eval = sim_val[warmup_val:]
        val_nse_val = nse(obs_val, sim_val_eval)

        ax = axes[idx]
        ax.plot(val_data.date, val_data.runoff_obs, 'b-', label='实测', alpha=0.7, linewidth=0.8)
        ax.plot(val_data.date, sim_val, 'r-', label='模拟', alpha=0.7, linewidth=0.8)
        ax.set_title(f'{alg_name} - 验证期 (NSE={val_nse_val:.4f})')
        ax.set_xlabel('日期')
        ax.set_ylabel('流量')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('跨算法比较 - 最佳运行验证期水文过程线', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(results_dir / 'exp4_val_hydrographs.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n所有图形已保存至 results/ 目录")
    print("实验4完成！")


if __name__ == '__main__':
    main()
