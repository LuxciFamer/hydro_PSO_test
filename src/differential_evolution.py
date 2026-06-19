"""
差分进化算法 (Differential Evolution, DE/rand/1/bin)

用于水文模型参数率定的全局优化算法。
采用经典的 DE/rand/1/bin 策略，包括随机变异、二项交叉和贪婪选择。
"""

import time
import numpy as np

from src.objective_functions import OptimizationResult


def de_optimize(
    objective_func,
    bounds,
    pop_size=50,
    max_iter=200,
    F=0.8,
    CR=0.9,
    seed=None,
    verbose=True,
) -> OptimizationResult:
    """
    差分进化算法优化器 (DE/rand/1/bin)。

    参数:
        objective_func: 目标函数，接受一维参数向量，返回标量适应度值（越小越好）。
        bounds: 参数边界，形状为 (n_params, 2) 的数组或列表，每行为 [下界, 上界]。
        pop_size: 种群大小，默认 50。
        max_iter: 最大迭代次数（代数），默认 200。
        F: 变异缩放因子，控制差分向量的放大程度，默认 0.8。
        CR: 交叉概率，控制试验向量从变异向量继承分量的概率，默认 0.9。
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

    # ========== 步骤1: 初始化种群 ==========
    # 在参数空间内均匀随机采样生成初始种群
    population = lower + rng.rand(pop_size, n_params) * (upper - lower)

    # ========== 步骤2: 评估所有个体的适应度 ==========
    fitness = np.array(objective_func(population))
    n_evaluations = pop_size

    # 记录当前全局最优
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_params = population[best_idx].copy()

    # 收敛历史：记录每一代的最优适应度
    convergence_history = [best_fitness]

    if verbose:
        print(f"差分进化算法 (DE/rand/1/bin) 启动")
        print(f"  参数维度: {n_params}, 种群大小: {pop_size}, 最大迭代: {max_iter}")
        print(f"  F={F}, CR={CR}, seed={seed}")
        print(f"  初始最优适应度: {best_fitness:.6f}")
        print("-" * 60)

    # ========== 步骤3: 迭代进化 ==========
    for gen in range(max_iter):
        trials = np.zeros((pop_size, n_params))
        for i in range(pop_size):
            # ----- 3a. 变异操作 (DE/rand/1) -----
            # 选择3个互不相同且不等于i的随机索引
            candidates = list(range(pop_size))
            candidates.remove(i)
            r1, r2, r3 = rng.choice(candidates, size=3, replace=False)

            # 生成变异向量: v = x_r1 + F * (x_r2 - x_r3)
            mutant = population[r1] + F * (population[r2] - population[r3])

            # ----- 3b. 交叉操作 (二项交叉) -----
            # 随机选择一个必须继承变异分量的维度
            j_rand = rng.randint(0, n_params)
            trial = np.copy(population[i])
            for d in range(n_params):
                if rng.rand() < CR or d == j_rand:
                    trial[d] = mutant[d]

            # ----- 3c. 边界处理 -----
            # 将试验向量裁剪到合法参数范围内
            trials[i] = np.clip(trial, lower, upper)

        # ----- 3d. 选择操作 (批量评价并选择) -----
        trial_fitness = np.array(objective_func(trials))
        n_evaluations += pop_size

        for i in range(pop_size):
            if trial_fitness[i] <= fitness[i]:
                population[i] = trials[i]
                fitness[i] = trial_fitness[i]

                # ----- 3e. 更新全局最优 -----
                if trial_fitness[i] < best_fitness:
                    best_fitness = trial_fitness[i]
                    best_params = trials[i].copy()

        # 记录本代最优适应度到收敛历史
        convergence_history.append(best_fitness)

        # 打印迭代信息
        if verbose and (gen + 1) % 10 == 0:
            print(
                f"  第 {gen + 1:>4d}/{max_iter} 代 | "
                f"最优适应度: {best_fitness:.6f} | "
                f"种群平均: {np.mean(fitness):.6f} | "
                f"评价次数: {n_evaluations}"
            )

    # ========== 步骤4: 整理结果 ==========
    wall_time = time.time() - start_time

    if verbose:
        print("-" * 60)
        print(f"差分进化算法结束")
        print(f"  最优适应度: {best_fitness:.6f}")
        print(f"  总评价次数: {n_evaluations}")
        print(f"  运行时间: {wall_time:.2f} 秒")
        print(f"  最优参数: {best_params}")

    return OptimizationResult(
        best_params=best_params,
        best_fitness=best_fitness,
        convergence_history=convergence_history,
        n_evaluations=n_evaluations,
        wall_time=wall_time,
    )
