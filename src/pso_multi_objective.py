"""
多目标粒子群优化算法 (Multi-Objective PSO, MOPSO)

本模块实现了基于Coello等人(2004)提出的MOPSO算法，用于水文模型多目标参数率定。

算法概述：
    MOPSO将PSO扩展到多目标优化场景。与单目标PSO不同，多目标优化中不存在
    单一最优解，而是一组Pareto最优解（非支配解集）。MOPSO通过外部存档
    （External Archive）维护Pareto前沿的近似，并使用自适应网格机制
    （Adaptive Grid）来保持解的多样性。

核心机制：
    1. 外部存档（External Archive）：
       存储迭代过程中发现的所有非支配解。当存档大小超过限制时，
       使用自适应网格中最拥挤区域的解进行裁剪。

    2. 自适应网格（Adaptive Grid）：
       将目标空间划分为超立方体网格。每个存档解被分配到一个网格单元。
       选择领导者时优先从人口密度较低的网格单元中选择（轮盘赌选择，
       概率与网格密度成反比），从而促进Pareto前沿的均匀分布。

    3. 支配关系（Dominance）：
       解a支配解b，当且仅当a在所有目标上不差于b，且至少有一个目标严格优于b。

    4. pbest更新策略：
       - 如果新位置支配旧pbest，则更新
       - 如果旧pbest支配新位置，则保持不变
       - 如果二者互不支配，则随机选择

参考文献：
    Coello, C.A.C., Pulido, G.T., & Lechuga, M.S. (2004). Handling multiple
    objectives with particle swarm optimization. IEEE Transactions on
    Evolutionary Computation, 8(3), 256-279.
"""

import time
import numpy as np
from src.objective_functions import MOPSOResult


def _dominates(obj_a, obj_b):
    """
    判断解a是否支配解b（帕累托支配关系）。

    支配关系定义：
        解a支配解b，当且仅当：
        1. 对所有目标i，a[i] <= b[i]（a在所有目标上不差于b）
        2. 存在至少一个目标j，a[j] < b[j]（a在至少一个目标上严格优于b）

    本算法假设所有目标均为最小化。

    参数：
        obj_a (np.ndarray): 解a的目标函数值向量。
        obj_b (np.ndarray): 解b的目标函数值向量。

    返回：
        bool: 如果a支配b则返回True，否则返回False。
    """
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


def _compute_grid_indices(archive_objectives, grid_bounds, n_grids):
    """
    计算存档中每个解在自适应网格中的网格索引。

    将目标空间均匀划分为n_grids^n_objectives个超立方体网格，
    每个存档解被分配到对应的网格单元中。

    参数：
        archive_objectives (np.ndarray): 存档中所有解的目标值矩阵，
            形状 (archive_size, n_objectives)。
        grid_bounds (np.ndarray): 网格边界，形状 (n_objectives, 2)，
            每行为 [最小值, 最大值]。
        n_grids (int): 每个目标维度的网格划分数。

    返回：
        list: 每个存档解的网格索引元组列表。
    """
    n_objectives = archive_objectives.shape[1]
    grid_indices = []

    for i in range(len(archive_objectives)):
        idx = []
        for j in range(n_objectives):
            range_j = grid_bounds[j, 1] - grid_bounds[j, 0]
            if range_j < 1e-12:
                idx.append(0)
            else:
                # 将目标值映射到网格索引
                cell = int(
                    (archive_objectives[i, j] - grid_bounds[j, 0])
                    / range_j
                    * n_grids
                )
                cell = min(cell, n_grids - 1)
                cell = max(cell, 0)
                idx.append(cell)
        grid_indices.append(tuple(idx))

    return grid_indices


def _adaptive_grid_selection(archive_objectives, n_grids, rng):
    """
    基于自适应网格的领导者选择。

    使用轮盘赌方法从存档中选择一个领导者，选择概率与其所在网格单元的
    人口密度成反比。这种策略鼓励粒子向Pareto前沿中较稀疏的区域移动，
    从而促进解的多样性和均匀分布。

    选择概率计算：
        P(grid_cell) = (10 / count(grid_cell)) / sum(10 / count(all_cells))
        其中count(grid_cell)为该网格单元中解的数量。

    参数：
        archive_objectives (np.ndarray): 存档目标值矩阵，
            形状 (archive_size, n_objectives)。
        n_grids (int): 每个目标维度的网格划分数。
        rng (np.random.RandomState): 随机数生成器。

    返回：
        int: 被选中领导者在存档中的索引。
    """
    n_archive = len(archive_objectives)
    if n_archive == 1:
        return 0

    n_objectives = archive_objectives.shape[1]

    # 计算网格边界（带一定的边距）
    grid_bounds = np.zeros((n_objectives, 2))
    for j in range(n_objectives):
        obj_min = np.min(archive_objectives[:, j])
        obj_max = np.max(archive_objectives[:, j])
        margin = (obj_max - obj_min) * 0.1 if obj_max > obj_min else 0.5
        grid_bounds[j, 0] = obj_min - margin
        grid_bounds[j, 1] = obj_max + margin

    # 计算网格索引
    grid_indices = _compute_grid_indices(archive_objectives, grid_bounds, n_grids)

    # 统计每个网格单元的密度
    grid_count = {}
    for gi in grid_indices:
        grid_count[gi] = grid_count.get(gi, 0) + 1

    # 计算选择概率（与密度成反比）
    probs = np.array([10.0 / grid_count[gi] for gi in grid_indices])
    probs = probs / probs.sum()

    # 轮盘赌选择
    selected_idx = rng.choice(n_archive, p=probs)
    return selected_idx


def _update_archive(
    archive_positions,
    archive_objectives,
    new_positions,
    new_objectives,
    archive_size,
    n_grids,
    rng,
):
    """
    更新外部存档。

    将新解加入存档，移除被支配的解，当存档超过大小限制时
    从最拥挤的网格区域中移除解。

    更新步骤：
        1. 将现有存档和新解合并
        2. 移除所有被支配的解，仅保留非支配解
        3. 如果存档大小超过限制，使用自适应网格裁剪策略：
           - 找到密度最高的网格单元
           - 从中随机移除一个解
           - 重复直到存档大小满足限制

    参数：
        archive_positions (np.ndarray): 当前存档的参数矩阵。
        archive_objectives (np.ndarray): 当前存档的目标值矩阵。
        new_positions (np.ndarray): 新解的参数矩阵。
        new_objectives (np.ndarray): 新解的目标值矩阵。
        archive_size (int): 存档大小上限。
        n_grids (int): 网格划分数。
        rng (np.random.RandomState): 随机数生成器。

    返回：
        tuple: (更新后的存档参数矩阵, 更新后的存档目标值矩阵)
    """
    # 合并现有存档和新解
    if len(archive_positions) > 0:
        combined_positions = np.vstack([archive_positions, new_positions])
        combined_objectives = np.vstack([archive_objectives, new_objectives])
    else:
        combined_positions = new_positions.copy()
        combined_objectives = new_objectives.copy()

    n_combined = len(combined_positions)

    # 提取非支配解
    is_dominated = np.zeros(n_combined, dtype=bool)
    for i in range(n_combined):
        if is_dominated[i]:
            continue
        for j in range(n_combined):
            if i == j or is_dominated[j]:
                continue
            if _dominates(combined_objectives[j], combined_objectives[i]):
                is_dominated[i] = True
                break

    non_dominated_mask = ~is_dominated
    archive_positions = combined_positions[non_dominated_mask].copy()
    archive_objectives = combined_objectives[non_dominated_mask].copy()

    # 如果存档超过大小限制，裁剪最拥挤区域的解
    while len(archive_positions) > archive_size:
        n_objectives = archive_objectives.shape[1]
        grid_bounds = np.zeros((n_objectives, 2))
        for j in range(n_objectives):
            obj_min = np.min(archive_objectives[:, j])
            obj_max = np.max(archive_objectives[:, j])
            margin = (obj_max - obj_min) * 0.1 if obj_max > obj_min else 0.5
            grid_bounds[j, 0] = obj_min - margin
            grid_bounds[j, 1] = obj_max + margin

        grid_indices = _compute_grid_indices(
            archive_objectives, grid_bounds, n_grids
        )

        # 找到最拥挤的网格单元
        grid_count = {}
        for gi in grid_indices:
            grid_count[gi] = grid_count.get(gi, 0) + 1

        max_count = max(grid_count.values())
        most_crowded_grids = [
            gi for gi, cnt in grid_count.items() if cnt == max_count
        ]
        target_grid = most_crowded_grids[rng.randint(len(most_crowded_grids))]

        # 从最拥挤网格中随机移除一个解
        candidates = [
            i for i, gi in enumerate(grid_indices) if gi == target_grid
        ]
        remove_idx = candidates[rng.randint(len(candidates))]

        archive_positions = np.delete(archive_positions, remove_idx, axis=0)
        archive_objectives = np.delete(archive_objectives, remove_idx, axis=0)

    return archive_positions, archive_objectives


def _reflect_boundary(positions, velocities, lower, upper, n_dims):
    """
    反射边界处理。

    当粒子越界时反射回可行域，并反转对应维度的速度分量。

    参数：
        positions (np.ndarray): 粒子位置向量，形状 (n_dims,)。
        velocities (np.ndarray): 粒子速度向量，形状 (n_dims,)。
        lower (np.ndarray): 各维度下界。
        upper (np.ndarray): 各维度上界。
        n_dims (int): 维度数。

    返回：
        tuple: (修正后的位置, 修正后的速度)
    """
    for d in range(n_dims):
        while positions[d] < lower[d] or positions[d] > upper[d]:
            if positions[d] < lower[d]:
                positions[d] = 2.0 * lower[d] - positions[d]
                velocities[d] = -velocities[d]
            if positions[d] > upper[d]:
                positions[d] = 2.0 * upper[d] - positions[d]
                velocities[d] = -velocities[d]
    return positions, velocities


def mopso_optimize(
    objective_funcs,
    bounds,
    n_particles=100,
    max_iter=200,
    archive_size=100,
    n_grids=10,
    seed=None,
    verbose=True,
) -> MOPSOResult:
    """
    多目标粒子群优化算法 (MOPSO)。

    基于Coello等人(2004)的MOPSO算法，使用外部存档维护Pareto前沿近似，
    并通过自适应网格机制保持解的多样性。适用于水文模型的多目标参数率定，
    例如同时优化NSE（纳什效率系数）和水量平衡误差。

    算法流程：
        1. 初始化粒子群，评价所有目标函数
        2. 用非支配解初始化外部存档
        3. 迭代优化：
           a. 通过自适应网格从存档中选择领导者（偏好稀疏区域）
           b. 更新速度：v = w*v + c1*r1*(pbest-x) + c2*r2*(leader-x)
              惯性权重w从0.5线性递减到0.1
           c. 更新位置，应用反射边界处理
           d. 评价所有粒子的目标函数值
           e. 更新pbest：新解支配旧解则更新，互不支配则随机选择
           f. 更新外部存档：加入非支配解，超出限制时裁剪最拥挤区域
           g. 记录存档大小到收敛历史

    参数：
        objective_funcs (list of callable): 目标函数列表，每个函数接受
            一维numpy数组（参数向量），返回标量值。所有目标均为最小化。
        bounds (np.ndarray): 参数边界，形状 (n_dims, 2)，每行 [下界, 上界]。
        n_particles (int): 粒子群规模，默认100。多目标优化通常需要更多粒子。
        max_iter (int): 最大迭代次数，默认200。
        archive_size (int): 外部存档最大容量，默认100。
            较大的存档能更好地近似Pareto前沿，但增加计算开销。
        n_grids (int): 每个目标维度的网格划分数，默认10。
            控制自适应网格的分辨率。
        seed (int or None): 随机数种子，默认None。
        verbose (bool): 是否打印优化过程信息，默认True。

    返回：
        MOPSOResult: 多目标优化结果数据类，包含：
            - pareto_front (np.ndarray): Pareto前沿目标值矩阵，
              形状 (n_solutions, n_objectives)
            - pareto_set (np.ndarray): Pareto最优解参数矩阵，
              形状 (n_solutions, n_dims)
            - convergence_history (list): 每次迭代的存档大小记录

    示例：
        >>> import numpy as np
        >>> def f1(x): return np.sum(x**2)
        >>> def f2(x): return np.sum((x - 2)**2)
        >>> bounds = np.array([[-5, 5], [-5, 5]])
        >>> result = mopso_optimize([f1, f2], bounds, n_particles=50, max_iter=100)
        >>> print(f"Pareto前沿大小: {len(result.pareto_front)}")
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)

    n_dims = bounds.shape[0]
    n_objectives = len(objective_funcs)
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # PSO参数
    c1 = 1.0
    c2 = 2.0
    w_max = 0.5
    w_min = 0.1

    # ---- 步骤1：初始化粒子群 ----
    positions = lower + rng.rand(n_particles, n_dims) * (upper - lower)
    velocities = np.zeros((n_particles, n_dims))

    # 评价所有目标
    objectives = np.zeros((n_particles, n_objectives))
    for i in range(n_particles):
        for j, func in enumerate(objective_funcs):
            objectives[i, j] = func(positions[i])

    # 初始化pbest
    pbest_positions = positions.copy()
    pbest_objectives = objectives.copy()

    # ---- 步骤2：初始化外部存档 ----
    archive_positions, archive_objectives = _update_archive(
        np.empty((0, n_dims)),
        np.empty((0, n_objectives)),
        positions,
        objectives,
        archive_size,
        n_grids,
        rng,
    )

    convergence_history = []

    if verbose:
        print(
            f"  [MOPSO] 初始化完成: {n_particles}个粒子, "
            f"{n_objectives}个目标, 初始存档大小 = {len(archive_positions)}"
        )

    # ---- 步骤3：主循环 ----
    for iteration in range(max_iter):
        # 线性递减惯性权重
        w = w_max - (w_max - w_min) * iteration / max_iter

        for i in range(n_particles):
            # 从存档中选择领导者（自适应网格选择）
            if len(archive_positions) > 0:
                leader_idx = _adaptive_grid_selection(
                    archive_objectives, n_grids, rng
                )
                leader = archive_positions[leader_idx]
            else:
                # 存档为空时（不应发生），使用自身pbest
                leader = pbest_positions[i]

            r1 = rng.rand(n_dims)
            r2 = rng.rand(n_dims)

            # 速度更新
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (leader - positions[i])
            )

            # 位置更新
            positions[i] = positions[i] + velocities[i]

            # 反射边界处理
            positions[i], velocities[i] = _reflect_boundary(
                positions[i], velocities[i], lower, upper, n_dims
            )

        # 评价所有粒子的目标函数
        for i in range(n_particles):
            for j, func in enumerate(objective_funcs):
                objectives[i, j] = func(positions[i])

        # 更新pbest
        for i in range(n_particles):
            if _dominates(objectives[i], pbest_objectives[i]):
                # 新解支配旧pbest，更新
                pbest_positions[i] = positions[i].copy()
                pbest_objectives[i] = objectives[i].copy()
            elif not _dominates(pbest_objectives[i], objectives[i]):
                # 互不支配，随机选择
                if rng.rand() < 0.5:
                    pbest_positions[i] = positions[i].copy()
                    pbest_objectives[i] = objectives[i].copy()
            # 如果旧pbest支配新解，保持不变

        # 更新外部存档
        archive_positions, archive_objectives = _update_archive(
            archive_positions,
            archive_objectives,
            positions,
            objectives,
            archive_size,
            n_grids,
            rng,
        )

        # 记录存档大小
        convergence_history.append(len(archive_positions))

        if verbose and (iteration + 1) % 20 == 0:
            print(
                f"  [MOPSO] 迭代 {iteration + 1}/{max_iter}, "
                f"存档大小 = {len(archive_positions)}, "
                f"w = {w:.4f}"
            )

    wall_time = time.time() - start_time

    if verbose:
        print(
            f"  [MOPSO] 优化完成: 存档大小 = {len(archive_positions)}, "
            f"耗时 = {wall_time:.2f}s"
        )

    return MOPSOResult(
        pareto_front=archive_objectives.copy(),
        pareto_set=archive_positions.copy(),
        convergence_history=convergence_history,
    )
