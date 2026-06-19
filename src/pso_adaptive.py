"""
自适应粒子群优化算法变体 (Adaptive PSO Variants)

本模块实现了三种自适应PSO算法变体，用于水文模型参数率定：

1. 线性递减权重PSO (LDW-PSO)：
   惯性权重从w_max线性递减到w_min，在搜索前期保持较强的全局探索能力，
   后期增强局部开发能力，实现全局搜索与局部搜索的平衡。
   参考：Shi, Y., & Eberhart, R. (1998).

2. 收缩因子PSO (CF-PSO)：
   使用Clerc收缩因子代替惯性权重，从理论上保证算法收敛。
   收缩因子由学习因子之和计算得出，无需额外的速度限制策略。
   参考：Clerc, M., & Kennedy, J. (2002).

3. 综合学习PSO (CLPSO)：
   每个粒子的每个维度可以从不同粒子的pbest学习，而非统一向gbest学习。
   通过学习概率Pc控制学习策略，有效防止早熟收敛，适合多模态优化问题。
   参考：Liang, J.J., et al. (2006). Comprehensive learning particle swarm
   optimizer for global optimization of multimodal functions.
"""

import time
from math import sqrt, exp
import numpy as np
from src.objective_functions import OptimizationResult


def _reflect_boundary(positions, velocities, lower, upper):
    """
    反射边界处理（安全且向量化，无循环，O(1)时间复杂度）。
    当粒子超出边界时，反射其位置并反转速度。如果反射后仍超出边界，则进行截断并清零速度。
    """
    lower_b = np.broadcast_to(lower, positions.shape)
    upper_b = np.broadcast_to(upper, positions.shape)

    under_mask = positions < lower_b
    if np.any(under_mask):
        positions[under_mask] = 2.0 * lower_b[under_mask] - positions[under_mask]
        velocities[under_mask] = -velocities[under_mask]
        still_under = positions < lower_b
        if np.any(still_under):
            positions[still_under] = lower_b[still_under]
            velocities[still_under] = 0.0

    over_mask = positions > upper_b
    if np.any(over_mask):
        positions[over_mask] = 2.0 * upper_b[over_mask] - positions[over_mask]
        velocities[over_mask] = -velocities[over_mask]
        still_over = positions > upper_b
        if np.any(still_over):
            positions[still_over] = upper_b[still_over]
            velocities[still_over] = 0.0

    return positions, velocities


# ==============================================================================
#  线性递减权重PSO (Linear Decreasing Weight PSO)
# ==============================================================================


def ldw_pso_optimize(
    objective_func,
    bounds,
    n_particles=50,
    max_iter=200,
    w_max=0.9,
    w_min=0.4,
    c1=2.0,
    c2=2.0,
    seed=None,
    verbose=True,
) -> OptimizationResult:
    """
    线性递减权重粒子群优化算法 (LDW-PSO)（已进行向量化和安全性优化）。
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)

    n_dims = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # 初始化粒子位置和速度
    positions = lower + rng.rand(n_particles, n_dims) * (upper - lower)
    velocities = np.zeros((n_particles, n_dims))

    # 评价所有粒子 (批量)
    fitness = np.array(objective_func(positions))
    n_evaluations = n_particles

    # 初始化pbest和gbest
    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()

    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_history = []
    
    # 速度限制
    max_vel = 0.5 * (upper - lower)

    # 主循环
    for iteration in range(max_iter):
        # 线性递减惯性权重
        w = w_max - (w_max - w_min) * iteration / max_iter

        r1 = rng.rand(n_particles, n_dims)
        r2 = rng.rand(n_particles, n_dims)

        # 速度更新
        velocities = (
            w * velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position[np.newaxis, :] - positions)
        )
        velocities = np.clip(velocities, -max_vel[np.newaxis, :], max_vel[np.newaxis, :])

        # 位置更新
        positions = positions + velocities

        # 反射边界处理
        positions, velocities = _reflect_boundary(positions, velocities, lower, upper)

        # 评价适应度 (批量)
        fitness = np.array(objective_func(positions))
        n_evaluations += n_particles

        # 更新个体最优
        better_mask = fitness < pbest_fitness
        if np.any(better_mask):
            pbest_fitness[better_mask] = fitness[better_mask]
            pbest_positions[better_mask] = positions[better_mask].copy()

        # 更新全局最优
        current_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[current_best_idx]
            gbest_position = pbest_positions[current_best_idx].copy()

        convergence_history.append(gbest_fitness)

        if verbose and (iteration + 1) % 20 == 0:
            print(
                f"  [LDW-PSO] 迭代 {iteration + 1}/{max_iter}, "
                f"w = {w:.4f}, 最优适应度 = {gbest_fitness:.6f}"
            )

    wall_time = time.time() - start_time

    if verbose:
        print(
            f"  [LDW-PSO] 优化完成: 最优适应度 = {gbest_fitness:.6f}, "
            f"耗时 = {wall_time:.2f}s, 评价次数 = {n_evaluations}"
        )

    return OptimizationResult(
        best_params=gbest_position,
        best_fitness=gbest_fitness,
        convergence_history=convergence_history,
        n_evaluations=n_evaluations,
        wall_time=wall_time,
    )


# ==============================================================================
#  收缩因子PSO (Constriction Factor PSO)
# ==============================================================================


def cf_pso_optimize(
    objective_func,
    bounds,
    n_particles=50,
    max_iter=200,
    c1=2.05,
    c2=2.05,
    seed=None,
    verbose=True,
) -> OptimizationResult:
    """
    收缩因子粒子群优化算法 (CF-PSO)（已进行向量化和安全性优化）。
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)

    # 计算收缩因子
    phi = c1 + c2
    if phi <= 4.0:
        raise ValueError(
            f"收缩因子PSO要求 c1 + c2 > 4，当前值为 {phi:.4f}。"
            f"请增大c1或c2的值。"
        )
    chi = 2.0 / abs(2.0 - phi - sqrt(phi ** 2 - 4.0 * phi))

    n_dims = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # 初始化
    positions = lower + rng.rand(n_particles, n_dims) * (upper - lower)
    velocities = np.zeros((n_particles, n_dims))

    fitness = np.array(objective_func(positions))
    n_evaluations = n_particles

    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()

    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_history = []
    
    # 速度限制
    max_vel = 0.5 * (upper - lower)

    if verbose:
        print(f"  [CF-PSO] 收缩因子 chi = {chi:.6f}, phi = {phi:.4f}")

    # 主循环
    for iteration in range(max_iter):
        r1 = rng.rand(n_particles, n_dims)
        r2 = rng.rand(n_particles, n_dims)

        # 收缩因子速度更新
        velocities = chi * (
            velocities
            + c1 * r1 * (pbest_positions - positions)
            + c2 * r2 * (gbest_position[np.newaxis, :] - positions)
        )
        velocities = np.clip(velocities, -max_vel[np.newaxis, :], max_vel[np.newaxis, :])

        positions = positions + velocities

        # 反射边界处理
        positions, velocities = _reflect_boundary(positions, velocities, lower, upper)

        fitness = np.array(objective_func(positions))
        n_evaluations += n_particles

        # 更新个体最优
        better_mask = fitness < pbest_fitness
        if np.any(better_mask):
            pbest_fitness[better_mask] = fitness[better_mask]
            pbest_positions[better_mask] = positions[better_mask].copy()

        current_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[current_best_idx]
            gbest_position = pbest_positions[current_best_idx].copy()

        convergence_history.append(gbest_fitness)

        if verbose and (iteration + 1) % 20 == 0:
            print(
                f"  [CF-PSO] 迭代 {iteration + 1}/{max_iter}, "
                f"最优适应度 = {gbest_fitness:.6f}"
            )

    wall_time = time.time() - start_time

    if verbose:
        print(
            f"  [CF-PSO] 优化完成: 最优适应度 = {gbest_fitness:.6f}, "
            f"耗时 = {wall_time:.2f}s, 评价次数 = {n_evaluations}"
        )

    return OptimizationResult(
        best_params=gbest_position,
        best_fitness=gbest_fitness,
        convergence_history=convergence_history,
        n_evaluations=n_evaluations,
        wall_time=wall_time,
    )


# ==============================================================================
#  综合学习PSO (Comprehensive Learning PSO, CLPSO)
# ==============================================================================


def clpso_optimize(
    objective_func,
    bounds,
    n_particles=50,
    max_iter=200,
    seed=None,
    verbose=True,
) -> OptimizationResult:
    """
    综合学习粒子群优化算法 (CLPSO)（已进行向量化和安全性优化）。
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)

    n_dims = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]
    c = 1.494  # 加速系数
    refreshing_gap = 7  # 刷新间隔

    # 惯性权重线性递减参数
    w_max = 0.9
    w_min = 0.4

    # 初始化
    positions = lower + rng.rand(n_particles, n_dims) * (upper - lower)
    velocities = np.zeros((n_particles, n_dims))

    fitness = np.array(objective_func(positions))
    n_evaluations = n_particles

    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()

    # 计算每个粒子的学习概率 Pc
    if n_particles > 1:
        pc = np.array(
            [
                0.05
                + 0.45
                * (exp(10.0 * i / (n_particles - 1)) - 1.0)
                / (exp(10.0) - 1.0)
                for i in range(n_particles)
            ]
        )
    else:
        pc = np.array([0.5])

    # 为每个粒子生成范例向量索引
    exemplar = np.zeros((n_particles, n_dims), dtype=int)
    no_improve_count = np.zeros(n_particles, dtype=int)

    def _generate_exemplar(particle_idx):
        ex = np.full(n_dims, particle_idx, dtype=int)
        any_other = False

        for d in range(n_dims):
            if rng.rand() < pc[particle_idx]:
                candidates = rng.choice(n_particles, size=2, replace=False)
                winner = (
                    candidates[0]
                    if pbest_fitness[candidates[0]] < pbest_fitness[candidates[1]]
                    else candidates[1]
                )
                ex[d] = winner
                if winner != particle_idx:
                    any_other = True

        # 确保至少有一个维度从其他粒子学习
        if not any_other:
            rand_dim = rng.randint(n_dims)
            candidates = rng.choice(n_particles, size=2, replace=False)
            winner = (
                candidates[0]
                if pbest_fitness[candidates[0]] < pbest_fitness[candidates[1]]
                else candidates[1]
            )
            ex[rand_dim] = winner

        return ex

    # 初始化所有粒子的范例向量
    for i in range(n_particles):
        exemplar[i] = _generate_exemplar(i)

    # 全局最优
    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_history = []
    
    # 速度限制
    max_vel = 0.5 * (upper - lower)

    # 主循环
    for iteration in range(max_iter):
        w = w_max - (w_max - w_min) * iteration / max_iter

        # 构建范例向量fi (使用NumPy高级索引)
        fi = pbest_positions[exemplar, np.arange(n_dims)]

        # 速度更新
        r = rng.rand(n_particles, n_dims)
        velocities = w * velocities + c * r * (fi - positions)
        velocities = np.clip(velocities, -max_vel[np.newaxis, :], max_vel[np.newaxis, :])

        # 位置更新
        positions = positions + velocities

        # 反射边界处理
        positions, velocities = _reflect_boundary(positions, velocities, lower, upper)

        # 评价适应度
        fitness = np.array(objective_func(positions))
        n_evaluations += n_particles

        # 更新个体最优
        better_mask = fitness < pbest_fitness
        no_improve_count += 1
        if np.any(better_mask):
            pbest_fitness[better_mask] = fitness[better_mask]
            pbest_positions[better_mask] = positions[better_mask].copy()
            no_improve_count[better_mask] = 0

        # 刷新间隔：如果连续多代未改进，重新生成范例向量
        for i in range(n_particles):
            if no_improve_count[i] >= refreshing_gap:
                exemplar[i] = _generate_exemplar(i)
                no_improve_count[i] = 0

        # 更新全局最优
        current_best_idx = np.argmin(pbest_fitness)
        if pbest_fitness[current_best_idx] < gbest_fitness:
            gbest_fitness = pbest_fitness[current_best_idx]
            gbest_position = pbest_positions[current_best_idx].copy()

        convergence_history.append(gbest_fitness)

        if verbose and (iteration + 1) % 20 == 0:
            print(
                f"  [CLPSO] 迭代 {iteration + 1}/{max_iter}, "
                f"w = {w:.4f}, 最优适应度 = {gbest_fitness:.6f}"
            )

    wall_time = time.time() - start_time

    if verbose:
        print(
            f"  [CLPSO] 优化完成: 最优适应度 = {gbest_fitness:.6f}, "
            f"耗时 = {wall_time:.2f}s, 评价次数 = {n_evaluations}"
        )

    return OptimizationResult(
        best_params=gbest_position,
        best_fitness=gbest_fitness,
        convergence_history=convergence_history,
        n_evaluations=n_evaluations,
        wall_time=wall_time,
    )
