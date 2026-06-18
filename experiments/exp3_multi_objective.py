#!/usr/bin/env python3
"""
实验3：多目标优化
===============
使用多目标粒子群优化算法（MOPSO）同时优化NSE和PBIAS，
分析Pareto前沿上的代表性解。
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
from src.pso_multi_objective import mopso_optimize
from src.visualization import plot_pareto_front, plot_hydrograph
import matplotlib.pyplot as plt


def main():
    """主函数：执行多目标优化实验"""
    # ============================================================
    # 1. 数据准备
    # ============================================================
    print("=" * 60)
    print("实验3：多目标优化（MOPSO）")
    print("=" * 60)

    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / 'data' / '66193.csv'
    results_dir = project_root / 'results'
    os.makedirs(results_dir, exist_ok=True)

    data = load_data(str(data_path))
    cal_data, val_data = split_data(data, split_year=1990)
    print(f"率定期记录数: {len(cal_data.date)}, 验证期记录数: {len(val_data.date)}")

    # ============================================================
    # 2. 构建多目标函数
    # ============================================================
    print("\n构建多目标函数...")

    # 目标1：最小化负NSE（即最大化NSE）
    obj_nse = create_objective(
        cal_data.rainfall, cal_data.pet, cal_data.runoff_obs,
        metric='nse', warmup=365
    )

    # 目标2：最小化|PBIAS|
    # 使用create_objective构建PBIAS目标，注意PBIAS越接近0越好
    # 需要自定义目标函数来最小化|PBIAS|
    warmup = 365

    def obj_pbias(params):
        """最小化绝对PBIAS"""
        sim = gr4j(params, cal_data.rainfall, cal_data.pet)
        obs = cal_data.runoff_obs[warmup:]
        sim_eval = sim[warmup:]
        return abs(pbias(obs, sim_eval))

    objective_funcs = [obj_nse, obj_pbias]

    # ============================================================
    # 3. 运行MOPSO
    # ============================================================
    print("\n" + "-" * 60)
    print("开始MOPSO优化...")
    print("-" * 60)

    result = mopso_optimize(
        objective_funcs=objective_funcs,
        bounds=GR4J_BOUNDS,
        n_particles=100,
        max_iter=200,
        archive_size=100,
        n_grids=10,
        seed=42,
        verbose=True
    )

    pareto_front = result.pareto_front  # shape: (n_solutions, n_objectives)
    pareto_set = result.pareto_set      # shape: (n_solutions, n_params)

    print(f"\nPareto前沿解的数量: {len(pareto_front)}")

    # ============================================================
    # 4. 选择代表性解
    # ============================================================
    print("\n" + "=" * 60)
    print("选择代表性解")
    print("=" * 60)

    # 解1：最佳NSE（目标1最小）
    idx_best_nse = np.argmin(pareto_front[:, 0])

    # 解2：最佳PBIAS（目标2最小，即|PBIAS|最小）
    idx_best_pbias = np.argmin(pareto_front[:, 1])

    # 解3：折中解（距理想点最近）
    # 理想点：两个目标都取最小值
    ideal = np.array([pareto_front[:, 0].min(), pareto_front[:, 1].min()])
    # 归一化距离
    ranges = pareto_front.max(axis=0) - pareto_front.min(axis=0)
    ranges[ranges == 0] = 1.0  # 避免除零
    normalized = (pareto_front - ideal) / ranges
    distances = np.sqrt(np.sum(normalized ** 2, axis=1))
    idx_compromise = np.argmin(distances)

    representative = {
        '最佳NSE': idx_best_nse,
        '最佳PBIAS': idx_best_pbias,
        '折中解': idx_compromise,
    }

    # ============================================================
    # 5. 评估代表性解
    # ============================================================
    warmup_val = min(365, len(val_data.date) // 4)
    rep_results = {}

    for rep_name, idx in representative.items():
        params = pareto_set[idx]
        obj_vals = pareto_front[idx]

        print(f"\n--- {rep_name} ---")
        print(f"  参数: {', '.join(f'{n}={v:.4f}' for n, v in zip(GR4J_PARAM_NAMES, params))}")
        print(f"  率定期目标值: 负NSE={obj_vals[0]:.4f}, |PBIAS|={obj_vals[1]:.4f}")

        # 率定期指标
        sim_cal = gr4j(params, cal_data.rainfall, cal_data.pet)
        obs_cal = cal_data.runoff_obs[warmup:]
        sim_cal_eval = sim_cal[warmup:]
        cal_nse_val = nse(obs_cal, sim_cal_eval)
        cal_pbias_val = pbias(obs_cal, sim_cal_eval)

        # 验证期指标
        sim_val = gr4j(params, val_data.rainfall, val_data.pet)
        obs_val = val_data.runoff_obs[warmup_val:]
        sim_val_eval = sim_val[warmup_val:]

        val_metrics = {
            'NSE': nse(obs_val, sim_val_eval),
            'KGE': kge(obs_val, sim_val_eval),
            'RMSE': rmse(obs_val, sim_val_eval),
            'PBIAS': pbias(obs_val, sim_val_eval),
        }

        rep_results[rep_name] = {
            'params': params,
            'cal_nse': cal_nse_val,
            'cal_pbias': cal_pbias_val,
            'val_metrics': val_metrics,
            'sim_val': sim_val,
        }

        print(f"  率定期: NSE={cal_nse_val:.4f}, PBIAS={cal_pbias_val:.4f}%")
        print(f"  验证期: NSE={val_metrics['NSE']:.4f}, KGE={val_metrics['KGE']:.4f}, "
              f"RMSE={val_metrics['RMSE']:.4f}, PBIAS={val_metrics['PBIAS']:.4f}%")

    # ============================================================
    # 6. 对比汇总表
    # ============================================================
    print("\n" + "=" * 60)
    print("三个代表性解对比汇总")
    print("=" * 60)
    print(f"{'解类型':<12} {'率定NSE':<10} {'率定PBIAS':<12} {'验证NSE':<10} {'验证KGE':<10} {'验证RMSE':<10} {'验证PBIAS':<12}")
    print("-" * 76)
    for rep_name, res in rep_results.items():
        vm = res['val_metrics']
        print(f"{rep_name:<12} {res['cal_nse']:<10.4f} {res['cal_pbias']:<12.4f} "
              f"{vm['NSE']:<10.4f} {vm['KGE']:<10.4f} {vm['RMSE']:<10.4f} {vm['PBIAS']:<12.4f}")

    # ============================================================
    # 7. 生成图形
    # ============================================================
    print("\n" + "-" * 60)
    print("生成图形...")
    print("-" * 60)

    # 图1：Pareto前沿
    print("  生成Pareto前沿图...")
    plot_pareto_front(
        pareto_front=pareto_front,
        obj_names=['负NSE（越小越好）', '|PBIAS| (%)（越小越好）'],
        title='MOPSO Pareto前沿（NSE vs PBIAS）',
        save_path=str(results_dir / 'exp3_pareto_front.png')
    )

    # 图2-4：各代表性解的验证期水文过程线
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))

    for idx, (rep_name, res) in enumerate(rep_results.items()):
        ax = axes[idx]
        vm = res['val_metrics']
        ax.plot(val_data.date, val_data.runoff_obs, 'b-', label='实测', alpha=0.7, linewidth=0.8)
        ax.plot(val_data.date, res['sim_val'], 'r-', label='模拟', alpha=0.7, linewidth=0.8)
        ax.set_title(f'{rep_name} - 验证期 (NSE={vm["NSE"]:.4f}, PBIAS={vm["PBIAS"]:.2f}%)')
        ax.set_xlabel('日期')
        ax.set_ylabel('流量')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.suptitle('多目标优化代表性解 - 验证期水文过程线', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(results_dir / 'exp3_representative_hydrographs.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n所有图形已保存至 results/ 目录")
    print("实验3完成！")


if __name__ == '__main__':
    main()
