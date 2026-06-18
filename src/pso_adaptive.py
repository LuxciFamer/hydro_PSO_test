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

所有算法均采用反射边界处理策略，并支持随机种子以保证可重复性。
"""

import time
from math import sqrt, exp

import numpy as np
from src.objective_functions import OptimizationResult


def _reflect_boundary(positions, velocities, lower, upper, n_dims):
    """
    反射边界处理（向量化辅助函数）。

    当粒子位置超出边界时，将其反射回可行域内，并反转对应维度的速度分量。
    该策略模拟弹性碰撞，既保持粒子在可行域内，又保留其运动趋势信息。

    参数：
        positions (np.ndarray): 单个粒子的位置向量，形状 (n_dims,)。
        velocities (np.ndarray): 单个粒子的速度向量，形状 (n_dims,)。
        lower (np.ndarray): 各维度下界，形状 (n_dims,)。
        upper (np.ndarray): 各维度上界，形状 (n_dims,)。
        n_dims (int): 搜索空间维度数。

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
    线性递减权重粒子群优化算法 (LDW-PSO)。

    惯性权重随迭代次数线性递减：
        w(t) = w_max - (w_max - w_min) * t / max_iter

    在搜索初期，较大的惯性权重使粒子具有较强的全局探索能力；
    在搜索后期，较小的惯性权重使粒子更专注于当前最优区域的局部开发。
    这种策略有效平衡了全局搜索和局部搜索之间的矛盾。

    与标准PSO的唯一区别在于惯性权重的动态调整策略。

    参数：
        objective_func (callable): 目标函数，接受一维numpy数组（参数向量），
            返回标量适应度值。算法对该函数进行最小化。
        bounds (np.ndarray): 参数边界，形状为 (n_dims, 2)，每行为 [下界, 上界]。
        n_particles (int): 粒子群规模，默认50。
        max_iter (int): 最大迭代次数，默认200。
        w_max (float): 惯性权重最大值（初始值），默认0.9。
        w_min (float): 惯性权重最小值（终止值），默认0.4。
        c1 (float): 认知学习因子，默认2.0。
        c2 (float): 社会学习因子，默认2.0。
        seed (int or None): 随机数种子，默认None。
        verbose (bool): 是否打印优化过程信息，默认True。

    返回：
        OptimizationResult: 优化结果数据类，包含：
            - best_params: 最优参数向量
            - best_fitness: 最优适应度值
            - convergence_history: 收敛历史
            - n_evaluations: 目标函数评价次数
            - wall_time: 优化耗时（秒）
    """
    start_time = time.time()
    rng = np.random.RandomState(seed)

    n_dims = bounds.shape[0]
    lower = bounds[:, 0]
    upper = bounds[:, 1]

    # 初始化粒子位置和速度
    positions = lower + rng.rand(n_particles, n_dims) * (upper - lower)
    velocities = np.zeros((n_particles, n_dims))

    # 评价所有粒子
    fitness = np.array([objective_func(positions[i]) for i in range(n_particles)])
    n_evaluations = n_particles

    # 初始化pbest和gbest
    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()

    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_history = []

    # 主循环
    for iteration in range(max_iter):
        # 线性递减惯性权重
        w = w_max - (w_max - w_min) * iteration / max_iter

        for i in range(n_particles):
            r1 = rng.rand(n_dims)
            r2 = rng.rand(n_dims)

            # 速度更新（使用当前迭代的惯性权重）
            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )

            # 位置更新
            positions[i] = positions[i] + velocities[i]

            # 反射边界处理
            positions[i], velocities[i] = _reflect_boundary(
                positions[i], velocities[i], lower, upper, n_dims
            )

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
    收缩因子粒子群优化算法 (CF-PSO)。

    采用Clerc收缩因子代替传统惯性权重，从数学上保证算法收敛。

    收缩因子计算：
        phi = c1 + c2  (必须 > 4)
        chi = 2.0 / |2.0 - phi - sqrt(phi^2 - 4*phi)|

    速度更新公式：
        v_id = chi * (v_id + c1 * r1 * (pbest_id - x_id) + c2 * r2 * (gbest_d - x_id))

    注意：收缩因子chi作用于整个速度更新项（包括原始速度和学习项），
    而非仅作用于原始速度分量。这与惯性权重方法在数学形式上有本质区别。

    当c1 = c2 = 2.05时：
        phi = 4.1
        chi ≈ 0.7298

    优势：
        - 无需额外的速度限制（Vmax）策略
        - 理论上保证收敛
        - 参数设置简单，仅需调节c1和c2

    参数：
        objective_func (callable): 目标函数，接受一维numpy数组，返回标量适应度值。
        bounds (np.ndarray): 参数边界，形状 (n_dims, 2)。
        n_particles (int): 粒子群规模，默认50。
        max_iter (int): 最大迭代次数，默认200。
        c1 (float): 认知学习因子，默认2.05。c1 + c2必须大于4。
        c2 (float): 社会学习因子，默认2.05。
        seed (int or None): 随机数种子，默认None。
        verbose (bool): 是否打印优化过程信息，默认True。

    返回：
        OptimizationResult: 优化结果数据类。

    异常：
        ValueError: 当 c1 + c2 <= 4 时抛出，因为收缩因子要求 phi > 4。
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

    fitness = np.array([objective_func(positions[i]) for i in range(n_particles)])
    n_evaluations = n_particles

    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()

    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_history = []

    if verbose:
        print(f"  [CF-PSO] 收缩因子 chi = {chi:.6f}, phi = {phi:.4f}")

    # 主循环
    for iteration in range(max_iter):
        for i in range(n_particles):
            r1 = rng.rand(n_dims)
            r2 = rng.rand(n_dims)

            # 收缩因子速度更新：chi作用于整个更新项
            velocities[i] = chi * (
                velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )

            positions[i] = positions[i] + velocities[i]

            # 反射边界处理
            positions[i], velocities[i] = _reflect_boundary(
                positions[i], velocities[i], lower, upper, n_dims
            )

            fitness[i] = objective_func(positions[i])
            n_evaluations += 1

            if fitness[i] < pbest_fitness[i]:
                pbest_fitness[i] = fitness[i]
                pbest_positions[i] = positions[i].copy()

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
    综合学习粒子群优化算法 (CLPSO)。

    由Liang等人于2006年提出，是一种专为多模态函数优化设计的PSO变体。
    与标准PSO不同，CLPSO中每个粒子的每个维度可以从不同粒子的pbest学习，
    而非所有维度都统一向gbest学习，从而有效避免早熟收敛。

    核心机制：
        1. 学习概率 Pc_i：
           Pc_i = 0.05 + 0.45 * (exp(10*(i-1)/(N-1)) - 1) / (exp(10) - 1)
           其中i为粒子索引（从1开始），N为粒子总数。
           Pc按粒子索引指数递增：排名靠前的粒子倾向于从自身学习（探索），
           排名靠后的粒子倾向于从他人学习（开发）。

        2. 范例向量构建：
           对于粒子i的每个维度d：
           - 以概率Pc_i：通过锦标赛选择（大小为2）从另一个粒子的pbest[d]学习
           - 以概率1-Pc_i：从自身的pbest[d]学习
           这构建了一个融合多个粒子优势信息的范例向量fi。

        3. 刷新间隔（Refreshing Gap）：
           如果粒子连续7代没有改进，重新生成其范例向量，
           避免粒子在无效方向上持续搜索。

        4. 惯性权重：
           从0.9线性递减到0.4，与LDW-PSO相同。

    速度更新公式：
        v_id = w * v_id + c * rand() * (fi_d - x_id)
        其中c = 1.494，fi_d为范例向量的第d个分量。

    优势：
        - 有效防止早熟收敛
        - 特别适合多模态优化问题
        - 保持了种群多样性
        - 不依赖gbest，减少了被局部最优吸引的风险

    参数：
        objective_func (callable): 目标函数。
        bounds (np.ndarray): 参数边界，形状 (n_dims, 2)。
        n_particles (int): 粒子群规模，默认50。
        max_iter (int): 最大迭代次数，默认200。
        seed (int or None): 随机数种子，默认None。
        verbose (bool): 是否打印优化过程信息，默认True。

    返回：
        OptimizationResult: 优化结果数据类。

    参考文献：
        Liang, J.J., Qin, A.K., Suganthan, P.N., & Baskar, S. (2006).
        Comprehensive learning particle swarm optimizer for global optimization
        of multimodal functions. IEEE Transactions on Evolutionary Computation,
        10(3), 281-295.
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

    fitness = np.array([objective_func(positions[i]) for i in range(n_particles)])
    n_evaluations = n_particles

    pbest_positions = positions.copy()
    pbest_fitness = fitness.copy()

    # 计算每个粒子的学习概率 Pc
    # Pc_i = 0.05 + 0.45 * (exp(10*(i-1)/(N-1)) - 1) / (exp(10) - 1)
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
    # exemplar[i][d] 表示粒子i在维度d上学习的目标粒子索引
    exemplar = np.zeros((n_particles, n_dims), dtype=int)
    # 连续未改进计数器
    no_improve_count = np.zeros(n_particles, dtype=int)

    def _generate_exemplar(particle_idx):
        """
        为指定粒子生成范例向量。

        对于每个维度，以概率Pc从锦标赛选择的另一个粒子的pbest学习，
        否则从自身pbest学习。确保至少有一个维度从其他粒子学习。

        参数：
            particle_idx (int): 粒子索引。

        返回：
            np.ndarray: 范例索引向量，形状 (n_dims,)。
        """
        ex = np.full(n_dims, particle_idx, dtype=int)
        any_other = False

        for d in range(n_dims):
            if rng.rand() < pc[particle_idx]:
                # 锦标赛选择（大小为2）
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

    # 全局最优（用于记录和返回）
    gbest_idx = np.argmin(pbest_fitness)
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_fitness = pbest_fitness[gbest_idx]

    convergence_history = []

    # 主循环
    for iteration in range(max_iter):
        # 线性递减惯性权重
        w = w_max - (w_max - w_min) * iteration / max_iter

        for i in range(n_particles):
            # 构建范例向量fi
            fi = np.array(
                [pbest_positions[exemplar[i, d], d] for d in range(n_dims)]
            )

            # 速度更新
            r = rng.rand(n_dims)
            velocities[i] = w * velocities[i] + c * r * (fi - positions[i])

            # 位置更新
            positions[i] = positions[i] + velocities[i]

            # 反射边界处理
            positions[i], velocities[i] = _reflect_boundary(
                positions[i], velocities[i], lower, upper, n_dims
            )

            # 评价适应度
            fitness[i] = objective_func(positions[i])
            n_evaluations += 1

            # 更新个体最优
            if fitness[i] < pbest_fitness[i]:
                pbest_fitness[i] = fitness[i]
                pbest_positions[i] = positions[i].copy()
                no_improve_count[i] = 0
            else:
                no_improve_count[i] += 1

            # 刷新间隔：如果连续多代未改进，重新生成范例向量
            if no_improve_count[i] >= refreshing_gap:
                exemplar[i] = _generate_exemplar(i)
                no_improve_count[i] = 0

        # 更新全局最优（仅用于记录）
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
