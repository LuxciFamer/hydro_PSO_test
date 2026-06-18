"""
模拟退火算法 (Simulated Annealing, SA)

用于水文模型参数率定的全局优化算法。
采用自适应邻域搜索策略和多次重启机制，提高全局搜索能力。
"""

import time
import numpy as np

from src.objective_functions import OptimizationResult


def sa_optimize(
    objective_func,
    bounds,
    max_iter=10000,
    T0=100.0,
    alpha=0.995,
    n_restarts=5,
    seed=None,
    verbose=True,
) -> OptimizationResult:
    """
    模拟退火算法优化器（带自适应邻域和多次重启）。

    参数:
        objective_func: 目标函数，接受一维参数向量，返回标量适应度值（越小越好）。
        bounds: 参数边界，形状为 (n_params, 2) 的数组或列表，每行为 [下界, 上界]。
        max_iter: 每次重启的最大迭代次数，默认 10000。
        T0: 初始温度，默认 100.0。
        alpha: 温度衰减系数（每步乘以 alpha），默认 0.995。
        n_restarts: 重启次数，默认 5。
        seed: 随机数种子，用于结果复现，默认 None。
        verbose: 是否打印迭代信息，默认 True。

    返回:
        OptimizationResult: 包含最优参数、最优适应度、收敛历史、评价次数和运行时间。
    """
    # 记录开始时间
    start_time = time.time()

    # 初始化随机数生成器
    rng = np.random.RandomState(seed)

    # 解析参数边界
    bounds = np.array(bounds, dtype=np.float64)
    n_params = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    param_range = upper - lower  # 各维度参数范围

    # 全局最优记录
    global_best_params = None
    global_best_fitness = np.inf
    n_evaluations = 0

    # 收敛历史：每50次迭代记录一次当前全局最优
    convergence_history = []

    if verbose:
        print(f"模拟退火算法 (SA) 启动")
        print(f"  参数维度: {n_params}, 最大迭代: {max_iter}")
        print(f"  初始温度: {T0}, 衰减系数: {alpha}, 重启次数: {n_restarts}")
        print(f"  seed={seed}")
        print("-" * 60)

    # ========== 步骤1: 多次重启 ==========
    for restart in range(n_restarts):
        # ----- 1a. 随机初始化解 -----
        x_current = lower + rng.rand(n_params) * param_range
        f_current = objective_func(x_current)
        n_evaluations += 1

        # 本次重启的最优记录
        x_best_restart = x_current.copy()
        f_best_restart = f_current

        # 更新全局最优
        if f_current < global_best_fitness:
            global_best_fitness = f_current
            global_best_params = x_current.copy()

        # ----- 1b. 初始化温度 -----
        T = T0

        if verbose:
            print(f"  [重启 {restart + 1}/{n_restarts}] 初始适应度: {f_current:.6f}")

        # ----- 1c. 迭代搜索 -----
        for iteration in range(max_iter):
            # 生成邻域解：自适应邻域大小随温度降低而缩小
            # scale = (T / T0) * 0.5，温度越低搜索范围越小
            scale = (T / T0) * 0.5
            perturbation = scale * rng.randn(n_params) * param_range
            x_new = x_current + perturbation

            # 裁剪到合法参数范围
            x_new = np.clip(x_new, lower, upper)

            # 计算新解的适应度
            f_new = objective_func(x_new)
            n_evaluations += 1

            # Metropolis 准则
            delta = f_new - f_current
            if delta < 0 or rng.rand() < np.exp(-delta / T):
                # 接受新解
                x_current = x_new
                f_current = f_new

                # 更新本次重启最优
                if f_current < f_best_restart:
                    f_best_restart = f_current
                    x_best_restart = x_current.copy()

                # 更新全局最优
                if f_current < global_best_fitness:
                    global_best_fitness = f_current
                    global_best_params = x_current.copy()

            # 温度衰减
            T = T * alpha

            # ----- 1d. 记录收敛历史（每50次迭代记录一次） -----
            if (iteration + 1) % 50 == 0:
                convergence_history.append(global_best_fitness)

        if verbose:
            print(
                f"           本轮最优: {f_best_restart:.6f} | "
                f"全局最优: {global_best_fitness:.6f} | "
                f"累计评价: {n_evaluations}"
            )

    # ========== 步骤2: 整理结果 ==========
    wall_time = time.time() - start_time

    if verbose:
        print("-" * 60)
        print(f"模拟退火算法结束")
        print(f"  全局最优适应度: {global_best_fitness:.6f}")
        print(f"  总评价次数: {n_evaluations}")
        print(f"  运行时间: {wall_time:.2f} 秒")
        print(f"  最优参数: {global_best_params}")

    return OptimizationResult(
        best_params=global_best_params,
        best_fitness=global_best_fitness,
        convergence_history=convergence_history,
        n_evaluations=n_evaluations,
        wall_time=wall_time,
    )
