"""
GR4J概念性水文模型

实现基于Perrin et al. (2003)的GR4J日尺度降雨-径流模型。
模型包含4个参数：
    X1: 产流水库容量 (mm)
    X2: 地下水交换系数 (mm/day)
    X3: 汇流水库容量 (mm)
    X4: 单位线汇流时间 (day)

参考文献:
    Perrin, C., Michel, C., & Andréassian, V. (2003).
    Improvement of a parsimonious model for streamflow simulation.
    Journal of Hydrology, 279(1-4), 275-289.
"""

import numpy as np

# ============================================================
# 模型参数边界与名称
# ============================================================

# 参数搜索空间的上下界: [X1, X2, X3, X4]
GR4J_BOUNDS = np.array([
    [100.0, 1200.0],   # X1: 产流水库容量 (mm)
    [-5.0,    3.0],    # X2: 地下水交换系数 (mm/day)
    [ 20.0, 300.0],    # X3: 汇流水库容量 (mm)
    [  1.1,   5.0],    # X4: 单位线汇流时间 (day)
])

GR4J_PARAM_NAMES = ['X1', 'X2', 'X3', 'X4']


# ============================================================
# 单位线(Unit Hydrograph)计算
# ============================================================

def _compute_SH1(t: float, X4: float | np.ndarray) -> float | np.ndarray:
    """计算SH1累积分布函数值

    SH1用于构造UH1单位线（快速响应分量）

    Args:
        t: 时间 (day)
        X4: 单位线汇流时间参数 (标量或形状为 (N,) 的数组)

    Returns:
        SH1(t)的值，范围[0, 1]
    """
    if isinstance(X4, np.ndarray):
        if t <= 0.0:
            return np.zeros_like(X4)
        res = np.ones_like(X4)
        mask = t < X4
        res[mask] = (t / X4[mask]) ** 2.5
        return res
    else:
        if t <= 0.0:
            return 0.0
        elif t < X4:
            return (t / X4) ** 2.5
        else:
            return 1.0


def _compute_SH2(t: float, X4: float | np.ndarray) -> float | np.ndarray:
    """计算SH2累积分布函数值

    SH2用于构造UH2单位线（慢速响应分量）

    Args:
        t: 时间 (day)
        X4: 单位线汇流时间参数 (标量或形状为 (N,) 的数组)

    Returns:
        SH2(t)的值，范围[0, 1]
    """
    if isinstance(X4, np.ndarray):
        if t <= 0.0:
            return np.zeros_like(X4)
        res = np.ones_like(X4)
        # Case 1: t < X4
        mask1 = t < X4
        res[mask1] = 0.5 * (t / X4[mask1]) ** 2.5
        # Case 2: X4 <= t < 2*X4
        mask2 = (X4 <= t) & (t < 2.0 * X4)
        res[mask2] = 1.0 - 0.5 * (2.0 - t / X4[mask2]) ** 2.5
        return res
    else:
        if t <= 0.0:
            return 0.0
        elif t < X4:
            return 0.5 * (t / X4) ** 2.5
        elif t < 2.0 * X4:
            return 1.0 - 0.5 * (2.0 - t / X4) ** 2.5
        else:
            return 1.0


def _build_unit_hydrographs(X4: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """预计算UH1和UH2的单位线权重

    UH1长度为ceil(X4)，UH2长度为ceil(2*X4)
    当输入为数组时，统一使用最大可能长度（UH1为5，UH2为10），不足部分补0。

    Args:
        X4: 单位线汇流时间参数 (标量或形状为 (N,) 的数组)

    Returns:
        tuple: (UH1权重数组, UH2权重数组)
    """
    if isinstance(X4, np.ndarray):
        N = len(X4)
        n1 = 5  # max ceil(X4) since X4 <= 5.0
        n2 = 10 # max ceil(2*X4) since X4 <= 5.0
        UH1 = np.zeros((N, n1))
        UH2 = np.zeros((N, n2))
        for j in range(n1):
            UH1[:, j] = _compute_SH1(j + 1, X4) - _compute_SH1(j, X4)
        for j in range(n2):
            UH2[:, j] = _compute_SH2(j + 1, X4) - _compute_SH2(j, X4)
        return UH1, UH2
    else:
        n1 = int(np.ceil(X4))
        n2 = int(np.ceil(2.0 * X4))

        # UH1权重: UH1[j] = SH1(j+1) - SH1(j)
        UH1 = np.zeros(n1)
        for j in range(n1):
            UH1[j] = _compute_SH1(j + 1, X4) - _compute_SH1(j, X4)

        # UH2权重: UH2[j] = SH2(j+1) - SH2(j)
        UH2 = np.zeros(n2)
        for j in range(n2):
            UH2[j] = _compute_SH2(j + 1, X4) - _compute_SH2(j, X4)

        return UH1, UH2


# ============================================================
# GR4J模型主函数
# ============================================================

def gr4j(params: np.ndarray, rainfall: np.ndarray, pet: np.ndarray) -> np.ndarray:
    """GR4J日尺度概念性降雨-径流模型

    支持标量参数（单体模拟）和二维数组参数（批处理/粒子群并行模拟）。

    Args:
        params: 模型参数数组 [X1, X2, X3, X4]（形状 (4,) 或 (N, 4)）
        rainfall: 日降雨序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)

    Returns:
        np.ndarray: 模拟径流序列 (mm/day)。
                    如果输入是单组参数，返回形状为 (n_steps,)；
                    如果输入是 N 组参数，返回形状为 (n_steps, N)。
    """
    params = np.asarray(params)
    is_batch = params.ndim == 2

    if is_batch:
        X1 = params[:, 0]  # 产流水库容量 (mm)
        X2 = params[:, 1]  # 地下水交换系数 (mm/day)
        X3 = params[:, 2]  # 汇流水库容量 (mm)
        X4 = params[:, 3]  # 单位线汇流时间 (day)
        N = len(params)

        n_steps = len(rainfall)
        Q = np.zeros((n_steps, N), dtype=np.float64)

        # 初始化状态变量 (所有粒子并行)
        S = X1 * 0.5   # 产流水库初始蓄量
        R = X3 * 0.5   # 汇流水库初始蓄量

        # 预计算单位线权重
        UH1_ord, UH2_ord = _build_unit_hydrographs(X4)  # 形状 (N, 5), (N, 10)
        n1 = 5
        n2 = 10

        # 初始化单位线卷积状态数组
        Q9_UH = np.zeros((N, n1), dtype=np.float64)
        Q1_UH = np.zeros((N, n2), dtype=np.float64)

        # 逐日模拟
        for t in range(n_steps):
            P = rainfall[t]
            E = pet[t]

            # 步骤1: 净降雨/净蒸发
            if P >= E:
                Pn = P - E
                En = 0.0
            else:
                Pn = 0.0
                En = E - P

            # 步骤2: 产流水库更新
            Ps = np.zeros(N, dtype=np.float64)
            if Pn > 0.0:
                s_ratio = S / X1
                tanh_pn = np.tanh(Pn / X1)
                Ps = X1 * (1.0 - s_ratio ** 2) * tanh_pn / (1.0 + s_ratio * tanh_pn)
                Ps = np.maximum(0.0, Ps)
                S = S + Ps

            if En > 0.0:
                s_ratio = S / X1
                tanh_en = np.tanh(En / X1)
                Es = S * (2.0 - s_ratio) * tanh_en / (1.0 + (1.0 - s_ratio) * tanh_en)
                Es = np.maximum(0.0, np.minimum(Es, S))
                S = S - Es

            S = np.maximum(0.0, np.minimum(S, X1))

            # 步骤3: 渗漏 (Percolation)
            s_ratio_4 = (4.0 / 9.0 * S / X1) ** 4
            Perc = S * (1.0 - (1.0 + s_ratio_4) ** (-0.25))
            Perc = np.maximum(0.0, np.minimum(Perc, S))
            S = S - Perc

            # 步骤4: 有效降雨分配
            Pr = Perc + (Pn - Ps)
            Q9_input = 0.9 * Pr
            Q1_input = 0.1 * Pr

            # 步骤5: 单位线卷积
            Q9 = Q9_UH[:, 0].copy()
            Q9_UH[:, :-1] = Q9_UH[:, 1:]
            Q9_UH[:, -1] = 0.0
            Q9_UH += Q9_input[:, np.newaxis] * UH1_ord

            Q1 = Q1_UH[:, 0].copy()
            Q1_UH[:, :-1] = Q1_UH[:, 1:]
            Q1_UH[:, -1] = 0.0
            Q1_UH += Q1_input[:, np.newaxis] * UH2_ord

            # 步骤6: 汇流水库与地下水交换
            r_ratio = np.maximum(0.0, R / X3)
            F = X2 * r_ratio ** 3.5
            R = np.maximum(0.0, R + Q9 + F)

            # 汇流水库出流
            r_ratio_4 = (R / X3) ** 4
            Qr = R * (1.0 - (1.0 + r_ratio_4) ** (-0.25))
            Qr = np.maximum(0.0, np.minimum(Qr, R))
            R = R - Qr

            # 直接径流
            Qd = np.maximum(0.0, Q1 + F)

            # 总径流
            Q[t] = Qr + Qd

        return Q

    else:
        from math import tanh
        # 单组参数的原始标量逻辑 (已用 pure Python + math.tanh 优化以消除 NumPy 标量循环开销)
        X1 = float(params[0])
        X2 = float(params[1])
        X3 = float(params[2])
        X4 = float(params[3])

        n_steps = len(rainfall)
        Q = np.zeros(n_steps, dtype=np.float64)

        S = X1 * 0.5
        R = X3 * 0.5

        UH1_ord, UH2_ord = _build_unit_hydrographs(X4)
        UH1_ord_list = list(UH1_ord)
        UH2_ord_list = list(UH2_ord)
        n1 = len(UH1_ord_list)
        n2 = len(UH2_ord_list)

        Q9_UH = [0.0] * n1
        Q1_UH = [0.0] * n2

        rainfall_list = list(rainfall)
        pet_list = list(pet)

        for t in range(n_steps):
            P = rainfall_list[t]
            E = pet_list[t]

            if P >= E:
                Pn = P - E
                En = 0.0
            else:
                Pn = 0.0
                En = E - P

            Ps = 0.0
            if Pn > 0.0:
                s_ratio = S / X1
                tanh_pn = tanh(Pn / X1)
                Ps = X1 * (1.0 - s_ratio ** 2) * tanh_pn / (1.0 + s_ratio * tanh_pn)
                Ps = max(0.0, Ps)
                S = S + Ps

            if En > 0.0:
                s_ratio = S / X1
                tanh_en = tanh(En / X1)
                Es = S * (2.0 - s_ratio) * tanh_en / (1.0 + (1.0 - s_ratio) * tanh_en)
                Es = max(0.0, min(Es, S))
                S = S - Es

            S = max(0.0, min(S, X1))

            s_ratio_4 = (4.0 / 9.0 * S / X1) ** 4
            Perc = S * (1.0 - (1.0 + s_ratio_4) ** (-0.25))
            Perc = max(0.0, min(Perc, S))
            S = S - Perc

            if Pn > 0.0:
                Pr = Perc + (Pn - Ps)
            else:
                Pr = Perc

            Q9_input = 0.9 * Pr
            Q1_input = 0.1 * Pr

            Q9 = Q9_UH[0]
            for i in range(n1 - 1):
                Q9_UH[i] = Q9_UH[i + 1]
            Q9_UH[-1] = 0.0
            for i in range(n1):
                Q9_UH[i] += Q9_input * UH1_ord_list[i]

            Q1 = Q1_UH[0]
            for i in range(n2 - 1):
                Q1_UH[i] = Q1_UH[i + 1]
            Q1_UH[-1] = 0.0
            for i in range(n2):
                Q1_UH[i] += Q1_input * UH2_ord_list[i]

            r_ratio = max(0.0, R / X3)
            F = X2 * r_ratio ** 3.5
            R = max(0.0, R + Q9 + F)

            r_ratio_4 = (R / X3) ** 4
            Qr = R * (1.0 - (1.0 + r_ratio_4) ** (-0.25))
            Qr = max(0.0, min(Qr, R))
            R = R - Qr

            Qd = max(0.0, Q1 + F)

            Q[t] = Qr + Qd

        return Q
