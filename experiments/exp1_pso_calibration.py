#!/usr/bin/env python3
"""
实验1：基于标准PSO的GR4J模型率定
==============================
使用标准粒子群优化算法（PSO）对GR4J水文模型进行参数率定，
评估率定期和验证期的模拟效果。
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
from src.visualization import (
    plot_hydrograph, plot_convergence, plot_scatter, plot_flow_duration_curve
)


def main():
    """主函数：执行基本PSO率定实验"""
    # ============================================================
    # 1. 数据加载与分割
    # ============================================================
    print("=" * 60)
    print("实验1：基于标准PSO的GR4J模型率定")
    print("=" * 60)

    # 获取项目根目录
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / 'data' / '66193.csv'
    results_dir = project_root / 'results'
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n数据文件路径: {data_path}")
    data = load_data(str(data_path))
    print(f"数据加载完成，总记录数: {len(data.date)}")

    # 分割数据：率定期(1960-1989) / 验证期(1990-2000)
    cal_data, val_data = split_data(data, split_year=1990)
    print(f"率定期记录数: {len(cal_data.date)}")
    print(f"验证期记录数: {len(val_data.date)}")

    # ============================================================
    # 2. 构建目标函数并运行PSO优化
    # ============================================================
    print("\n" + "-" * 60)
    print("开始PSO优化...")
    print("-" * 60)

    # 创建目标函数（以NSE为优化指标，预热期365天）
    objective = create_objective(
        cal_data.rainfall, cal_data.pet, cal_data.runoff_obs,
        metric='nse', warmup=365
    )

    # 运行标准PSO
    result = pso_optimize(
        objective_func=objective,
        bounds=GR4J_BOUNDS,
        n_particles=50,
        max_iter=200,
        seed=42,
        verbose=True
    )

    # ============================================================
    # 3. 输出最优参数
    # ============================================================
    print("\n" + "=" * 60)
    print("优化结果")
    print("=" * 60)

    print("\n最优参数:")
    for name, value in zip(GR4J_PARAM_NAMES, result.best_params):
        print(f"  {name}: {value:.6f}")
    print(f"\n最优目标函数值: {result.best_fitness:.6f}")
    print(f"函数评估次数: {result.n_evaluations}")
    print(f"运行时间: {result.wall_time:.2f} 秒")

    # ============================================================
    # 4. 率定期评估
    # ============================================================
    print("\n" + "-" * 60)
    print("率定期模拟评估")
    print("-" * 60)

    sim_cal = gr4j(result.best_params, cal_data.rainfall, cal_data.pet)
    warmup = 365

    # 计算率定期指标（去除预热期）
    obs_cal = cal_data.runoff_obs[warmup:]
    sim_cal_eval = sim_cal[warmup:]

    cal_nse = nse(obs_cal, sim_cal_eval)
    cal_kge = kge(obs_cal, sim_cal_eval)
    cal_rmse = rmse(obs_cal, sim_cal_eval)
    cal_pbias = pbias(obs_cal, sim_cal_eval)

    print(f"  NSE:   {cal_nse:.4f}")
    print(f"  KGE:   {cal_kge:.4f}")
    print(f"  RMSE:  {cal_rmse:.4f}")
    print(f"  PBIAS: {cal_pbias:.4f}%")

    # ============================================================
    # 5. 验证期评估
    # ============================================================
    print("\n" + "-" * 60)
    print("验证期模拟评估")
    print("-" * 60)

    sim_val = gr4j(result.best_params, val_data.rainfall, val_data.pet)
    warmup_val = min(365, len(val_data.date) // 4)  # 验证期预热

    obs_val = val_data.runoff_obs[warmup_val:]
    sim_val_eval = sim_val[warmup_val:]

    val_nse = nse(obs_val, sim_val_eval)
    val_kge = kge(obs_val, sim_val_eval)
    val_rmse = rmse(obs_val, sim_val_eval)
    val_pbias = pbias(obs_val, sim_val_eval)

    print(f"  NSE:   {val_nse:.4f}")
    print(f"  KGE:   {val_kge:.4f}")
    print(f"  RMSE:  {val_rmse:.4f}")
    print(f"  PBIAS: {val_pbias:.4f}%")

    # ============================================================
    # 6. 综合结果汇总表
    # ============================================================
    print("\n" + "=" * 60)
    print("综合结果汇总")
    print("=" * 60)
    print(f"{'指标':<10} {'率定期':<15} {'验证期':<15}")
    print("-" * 40)
    print(f"{'NSE':<10} {cal_nse:<15.4f} {val_nse:<15.4f}")
    print(f"{'KGE':<10} {cal_kge:<15.4f} {val_kge:<15.4f}")
    print(f"{'RMSE':<10} {cal_rmse:<15.4f} {val_rmse:<15.4f}")
    print(f"{'PBIAS(%)':<10} {cal_pbias:<15.4f} {val_pbias:<15.4f}")

    # ============================================================
    # 7. 生成图形
    # ============================================================
    print("\n" + "-" * 60)
    print("生成图形...")
    print("-" * 60)

    # 图1：收敛曲线
    print("  生成收敛曲线...")
    plot_convergence(
        histories=[result.convergence_history],
        labels=['标准PSO'],
        title='GR4J模型率定 - PSO收敛曲线',
        save_path=str(results_dir / 'exp1_convergence.png')
    )

    # 图2：率定期水文过程线（最后3年：1987-1989）
    print("  生成率定期水文过程线...")
    # 找到最后3年的索引
    cal_dates = cal_data.date
    last_3yr_mask = cal_dates >= np.datetime64('1987-01-01') if hasattr(cal_dates[0], 'year') is False else None

    # 尝试基于日期筛选最后3年
    try:
        import pandas as pd
        cal_dates_pd = pd.to_datetime(cal_dates)
        last_3yr_idx = cal_dates_pd.year >= 1987
        plot_hydrograph(
            dates=cal_data.date[last_3yr_idx],
            rainfall=cal_data.rainfall[last_3yr_idx],
            obs=cal_data.runoff_obs[last_3yr_idx],
            sim=sim_cal[last_3yr_idx],
            title='率定期水文过程线 (1987-1989)',
            save_path=str(results_dir / 'exp1_hydrograph_cal.png')
        )
    except Exception:
        # 如果日期筛选失败，取最后3年的数据（约1095天）
        n_last = min(1095, len(cal_data.date))
        plot_hydrograph(
            dates=cal_data.date[-n_last:],
            rainfall=cal_data.rainfall[-n_last:],
            obs=cal_data.runoff_obs[-n_last:],
            sim=sim_cal[-n_last:],
            title='率定期水文过程线（最后3年）',
            save_path=str(results_dir / 'exp1_hydrograph_cal.png')
        )

    # 图3：验证期水文过程线（完整1990-2000）
    print("  生成验证期水文过程线...")
    plot_hydrograph(
        dates=val_data.date,
        rainfall=val_data.rainfall,
        obs=val_data.runoff_obs,
        sim=sim_val,
        title='验证期水文过程线 (1990-2000)',
        save_path=str(results_dir / 'exp1_hydrograph_val.png')
    )

    # 图4：验证期散点图
    print("  生成验证期散点图...")
    plot_scatter(
        obs=obs_val,
        sim=sim_val_eval,
        title='验证期实测 vs 模拟散点图',
        save_path=str(results_dir / 'exp1_scatter_val.png')
    )

    # 图5：验证期流量历时曲线
    print("  生成验证期流量历时曲线...")
    plot_flow_duration_curve(
        obs=obs_val,
        sim=sim_val_eval,
        title='验证期流量历时曲线',
        save_path=str(results_dir / 'exp1_fdc_val.png')
    )

    print("\n所有图形已保存至 results/ 目录")
    print("实验1完成！")


if __name__ == '__main__':
    main()
