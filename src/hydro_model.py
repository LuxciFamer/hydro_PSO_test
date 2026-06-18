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

def _compute_SH1(t: float, X4: float) -> float:
    """计算SH1累积分布函数值

    SH1用于构造UH1单位线（快速响应分量）

    Args:
        t: 时间 (day)
        X4: 单位线汇流时间参数

    Returns:
        SH1(t)的值，范围[0, 1]
    """
    if t <= 0.0:
        return 0.0
    elif t < X4:
        return (t / X4) ** 2.5
    else:
        return 1.0


def _compute_SH2(t: float, X4: float) -> float:
    """计算SH2累积分布函数值

    SH2用于构造UH2单位线（慢速响应分量）

    Args:
        t: 时间 (day)
        X4: 单位线汇流时间参数

    Returns:
        SH2(t)的值，范围[0, 1]
    """
    if t <= 0.0:
        return 0.0
    elif t < X4:
        return 0.5 * (t / X4) ** 2.5
    elif t < 2.0 * X4:
        return 1.0 - 0.5 * (2.0 - t / X4) ** 2.5
    else:
        return 1.0


def _build_unit_hydrographs(X4: float) -> tuple[np.ndarray, np.ndarray]:
    """预计算UH1和UH2的单位线权重

    UH1长度为ceil(X4)，UH2长度为ceil(2*X4)

    Args:
        X4: 单位线汇流时间参数

    Returns:
        tuple: (UH1权重数组, UH2权重数组)
    """
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

    模型结构包括：
        - 产流水库（拦截净雨并产生有效降雨）
        - 两条汇流路径（UH1快速分量90% + UH2慢速分量10%）
        - 汇流水库（非线性汇流）
        - 地下水交换（可正可负）

    Args:
        params: 模型参数数组 [X1, X2, X3, X4]
            X1 - 产流水库容量 (mm), 范围[100, 1200]
            X2 - 地下水交换系数 (mm/day), 范围[-5, 3]
            X3 - 汇流水库容量 (mm), 范围[20, 300]
            X4 - 单位线汇流时间 (day), 范围[1.1, 5]
        rainfall: 日降雨序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)

    Returns:
        np.ndarray: 模拟径流序列 (mm/day)，长度与rainfall相同。
                    前365天为模型预热期，结果仍保留，但优化时可选择忽略。
    """
    # 提取参数
    X1 = float(params[0])  # 产流水库容量 (mm)
    X2 = float(params[1])  # 地下水交换系数 (mm/day)
    X3 = float(params[2])  # 汇流水库容量 (mm)
    X4 = float(params[3])  # 单位线汇流时间 (day)

    n_steps = len(rainfall)

    # 初始化输出数组
    Q = np.zeros(n_steps, dtype=np.float64)

    # 初始化状态变量
    S = X1 * 0.5   # 产流水库初始蓄量
    R = X3 * 0.5   # 汇流水库初始蓄量

    # 预计算单位线权重
    UH1_ord, UH2_ord = _build_unit_hydrographs(X4)
    n1 = len(UH1_ord)
    n2 = len(UH2_ord)

    # 初始化单位线卷积状态数组
    Q9_UH = np.zeros(n1, dtype=np.float64)
    Q1_UH = np.zeros(n2, dtype=np.float64)

    # --------------------------------------------------------
    # 逐日模拟
    # --------------------------------------------------------
    for t in range(n_steps):
        P = rainfall[t]
        E = pet[t]

        # ====================================================
        # 步骤1: 净降雨/净蒸发
        # ====================================================
        if P >= E:
            Pn = P - E   # 净降雨 (mm)
            En = 0.0
        else:
            Pn = 0.0
            En = E - P   # 净蒸发 (mm)

        # ====================================================
        # 步骤2: 产流水库更新
        # ====================================================
        Ps = 0.0  # 产流水库吸收的降雨

        if Pn > 0.0:
            # 产流水库吸收净降雨
            # 防止除零：当X1极小时使用安全计算
            s_ratio = S / X1
            tanh_pn = np.tanh(Pn / X1)
            Ps = X1 * (1.0 - s_ratio ** 2) * tanh_pn / \
                (1.0 + s_ratio * tanh_pn)
            # 确保Ps非负
            Ps = max(0.0, Ps)
            S = S + Ps

        if En > 0.0:
            # 产流水库蒸发损失
            s_ratio = S / X1
            tanh_en = np.tanh(En / X1)
            Es = S * (2.0 - s_ratio) * tanh_en / \
                (1.0 + (1.0 - s_ratio) * tanh_en)
            # 确保Es非负且不超过当前蓄量
            Es = max(0.0, min(Es, S))
            S = S - Es

        # 钳制产流水库蓄量到合理范围
        S = max(0.0, min(S, X1))

        # ====================================================
        # 步骤3: 渗漏 (Percolation)
        # ====================================================
        s_ratio_4 = (4.0 / 9.0 * S / X1) ** 4
        Perc = S * (1.0 - (1.0 + s_ratio_4) ** (-0.25))
        Perc = max(0.0, min(Perc, S))  # 确保渗漏量不超过蓄量
        S = S - Perc

        # ====================================================
        # 步骤4: 有效降雨分配
        # ====================================================
        if Pn > 0.0:
            Pr = Perc + (Pn - Ps)  # 总有效降雨
        else:
            Pr = Perc

        # 分配到两条汇流路径
        Q9_input = 0.9 * Pr   # 快速响应分量 (90%)
        Q1_input = 0.1 * Pr   # 慢速响应分量 (10%)

        # ====================================================
        # 步骤5: 单位线卷积
        # ====================================================
        # UH1卷积（快速分量）
        # 移位：丢弃第一个元素，追加0
        Q9 = Q9_UH[0]
        Q9_UH[:-1] = Q9_UH[1:]
        Q9_UH[-1] = 0.0
        # 添加新输入的贡献
        Q9_UH += Q9_input * UH1_ord

        # UH2卷积（慢速分量）
        Q1 = Q1_UH[0]
        Q1_UH[:-1] = Q1_UH[1:]
        Q1_UH[-1] = 0.0
        Q1_UH += Q1_input * UH2_ord

        # ====================================================
        # 步骤6: 汇流水库与地下水交换
        # ====================================================
        # 地下水交换量
        r_ratio = R / X3
        # 防止负数的幂运算问题：确保R/X3 >= 0
        r_ratio = max(0.0, r_ratio)
        F = X2 * r_ratio ** 3.5

        # 更新汇流水库蓄量
        R = max(0.0, R + Q9 + F)

        # 汇流水库出流（非线性）
        r_ratio_4 = (R / X3) ** 4
        Qr = R * (1.0 - (1.0 + r_ratio_4) ** (-0.25))
        Qr = max(0.0, min(Qr, R))  # 确保出流量合理
        R = R - Qr

        # 直接径流（慢速分量 + 地下水交换）
        Qd = max(0.0, Q1 + F)

        # 总径流
        Q[t] = Qr + Qd

    return Q
