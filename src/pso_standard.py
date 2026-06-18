"""
标准粒子群优化算法 (Standard Particle Swarm Optimization, PSO)

本模块实现了带惯性权重的标准PSO算法，用于水文模型参数率定。

算法概述：
    标准PSO算法由Kennedy和Eberhart于1995年提出，模拟鸟群觅食行为。
    每个粒子在搜索空间中飞行，通过跟踪个体历史最优位置（pbest）和
    全局最优位置（gbest）来更新自身的速度和位置。

    本实现采用Clerc和Kennedy (2002)推荐的惯性权重参数设置：
    - 惯性权重 w = 0.729
    - 学习因子 c1 = c2 = 1.494
    这组参数能够保证算法的收敛性。

速度更新公式：
    v_id = w * v_id + c1 * r1 * (pbest_id - x_id) + c2 * r2 * (gbest_d - x_id)

位置更新公式：
    x_id = x_id + v_id

边界处理策略：
    采用反射边界处理。当粒子越界时，将其反射回可行域内，
    并反转对应维度的速度分量，模拟弹性碰撞效果。

参考文献：
    [1] Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    [2] Clerc, M., & Kennedy, J. (2002). The particle swarm - explosion, 
        stability, and convergence in a multidimensional complex space.
"""

import time
import numpy as np
from src.objective_functions import OptimizationResult


def pso_optimize(
    objective_func,
    bounds,
    n_particles=50,
    max_iter=200,
    w=0.729,
    c1=1.494,
    c2=1.494,
    seed=None,
    verbose=True,
) -> OptimizationResult:
    """
    标准粒子群优化算法（带惯性权重）。

    使用固定惯性权重的经典PSO算法对目标函数进行最小化优化。
    适用于连续空间的单目标优化问题，如水文模型参数率定。

    参数：
        objective_func (callable): 目标函数，接受一维numpy数组（参数向量），
            返回标量适应度值。算法对该函数进行最小化。
        bounds (np.ndarray): 参数边界，形状为 (n_dims, 2)，每行为 [下界, 上界]。
        n_particles (int): 粒子群规模，默认50。较大的种群有助于全局搜索，
            但会增加每次迭代的计算量。
        max_iter (int): 最大迭代次数，默认200。
        w (float): 惯性权重，默认0.729。控制粒子保持原有运动方向的程度。
            较大的w有利于全局搜索，较小的w有利于局部搜索。
        c1 (float): 认知学习因子（个体学习因子），默认1.494。
            控制粒子向自身历史最优位置学习的程度。
        c2 (float): 社会学习因子（群体学习因子），默认1.494。
            控制粒子向全局最优位置学习的程度。
        seed (int or None): 随机数种子，用于结果可重复。默认None。
        verbose (bool): 是否打印优化过程信息，默认True。
            为True时每20次迭代打印一次进度。

    返回：
        OptimizationResult: 优化结果数据类，包含：
            - best_params (np.ndarray): 最优参数向量
            - best_fitness (float): 最优适应度值（最小值）
            - convergence_history (list): 每次迭代的最优适应度记录
            - n_evaluations (int): 目标函数总评价次数
            - wall_time (float): 优化总耗时（秒）

    示例：
        >>> import numpy as np
        >>> def sphere(x):
        ...     return np.sum(x**2)
        >>> bounds = np.array([[-5, 5], [-5, 5]])
        >>> result = pso_optimize(sphere, bounds, n_particles=30, max_iter=100, seed=42)
        >>> print(f"最优适应度: {result.best_fitness:.6f}")
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)

    n_dims = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # ---- 步骤1：初始化粒子位置和速度 ----
    # 在可行域内均匀随机初始化粒子位置
    positions = lower + rng.rand(n_particles, n_dims) * (upper - lower)
    # 速度初始化为零向量
    velocities = np.zeros((n_particles, n_dims))

    # ---- 步骤2：评价所有粒子，初始化pbest和gbest ----
    fitness = np.array([objective_func(positions[i]) for i in range(n_particles)])
    n_evaluations = n_particles

    # 个体最优
    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()

    # 全局最优
    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_history = []

    # ---- 步骤3：主循环 ----
    for iteration in range(max_iter):
        for i in range(n_particles):
            # 生成随机系数（每个维度独立）
            r1 = rng.rand(n_dims)
            r2 = rng.rand(n_dims)

            # 速度更新
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )

            # 位置更新
            positions[i] = positions[i] + velocities[i]

            # 反射边界处理
            for d in range(n_dims):
                while positions[i, d] < lower[d] or positions[i, d] > upper[d]:
                    if positions[i, d] < lower[d]:
                        positions[i, d] = 2.0 * lower[d] - positions[i, d]
                        velocities[i, d] = -velocities[i, d]
                    if positions[i, d] > upper[d]:
                        positions[i, d] = 2.0 * upper[d] - positions[i, d]
                        velocities[i, d] = -velocities[i, d]

            # 评价适应度
            fitness[i] = objective_func(positions[i])
            n_evaluations += 1

            # 更新个体最优
            if fitness[i] < pbest_fitness[i]:
                pbest_fitness[i] = fitness[i]
                pbest_positions[i] = positions[i].copy()

        # 更新全局最优
        current_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[current_best_idx]
            gbest_position = pbest_positions[current_best_idx].copy()

        # 记录收敛历史
        convergence_history.append(gbest_fitness)

        # 打印进度
        if verbose and (iteration + 1) % 20 == 0:
            print(
                f"  [标准PSO] 迭代 {iteration + 1}/{max_iter}, "
                f"最优适应度 = {gbest_fitness:.6f}"
            )

    wall_time = time.time() - start_time

    if verbose:
        print(
            f"  [标准PSO] 优化完成: 最优适应度 = {gbest_fitness:.6f}, "
            f"耗时 = {wall_time:.2f}s, 评价次数 = {n_evaluations}"
        )

    return OptimizationResult(
        best_params=gbest_position,
        best_fitness=gbest_fitness,
        convergence_history=convergence_history,
        n_evaluations=n_evaluations,
        wall_time=wall_time,
    )
