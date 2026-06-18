"""
目标函数与优化结果模块

功能：
    - 多种水文模型评价指标（NSE, KGE, RMSE, PBIAS, LogNSE）
    - 优化结果数据结构（单目标/多目标）
    - 目标函数工厂（将模型模拟与评价指标组合为可直接调用的目标函数）
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable

from .hydro_model import gr4j


# ============================================================
# 水文模型评价指标
# ============================================================

def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """Nash-Sutcliffe效率系数 (NSE)

    NSE = 1 - sum((obs - sim)^2) / sum((obs - mean(obs))^2)
    完美拟合时NSE=1，值越大越好，范围(-inf, 1]

    Args:
        obs: 实测径流序列
        sim: 模拟径流序列

    Returns:
        float: NSE值
    """
    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    if denominator == 0.0:
        return -np.inf
    return 1.0 - numerator / denominator


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    """Kling-Gupta效率系数 (KGE)

    KGE = 1 - sqrt((r-1)^2 + (beta-1)^2 + (gamma-1)^2)
    其中:
        r     = 相关系数 (Pearson)
        beta  = mean(sim) / mean(obs)          均值比
        gamma = (std(sim)/mean(sim)) / (std(obs)/mean(obs))  变异系数比

    完美拟合时KGE=1，值越大越好

    Args:
        obs: 实测径流序列
        sim: 模拟径流序列

    Returns:
        float: KGE值
    """
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    std_obs = np.std(obs)
    std_sim = np.std(sim)

    # 防止除零
    if mean_obs == 0.0 or std_obs == 0.0:
        return -np.inf

    # 相关系数
    if std_sim == 0.0:
        r = 0.0
    else:
        r = np.corrcoef(obs, sim)[0, 1]
        # 处理可能的NaN
        if np.isnan(r):
            r = 0.0

    # 均值比
    beta = mean_sim / mean_obs

    # 变异系数比
    cv_sim = std_sim / mean_sim if mean_sim != 0.0 else 0.0
    cv_obs = std_obs / mean_obs
    gamma = cv_sim / cv_obs if cv_obs != 0.0 else 0.0

    kge_val = 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)
    return float(kge_val)


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    """均方根误差 (RMSE)

    RMSE = sqrt(mean((obs - sim)^2))
    完美拟合时RMSE=0，值越小越好

    Args:
        obs: 实测径流序列
        sim: 模拟径流序列

    Returns:
        float: RMSE值
    """
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    """百分比偏差 (PBIAS)

    PBIAS = 100 * sum(sim - obs) / sum(obs)
    正值表示模拟偏高，负值表示模拟偏低，0为完美

    Args:
        obs: 实测径流序列
        sim: 模拟径流序列

    Returns:
        float: PBIAS值 (%)
    """
    sum_obs = np.sum(obs)
    if sum_obs == 0.0:
        return np.inf
    return float(100.0 * np.sum(sim - obs) / sum_obs)


def log_nse(obs: np.ndarray, sim: np.ndarray) -> float:
    """对数流量的Nash-Sutcliffe效率系数 (LogNSE)

    对实测和模拟流量取对数后计算NSE，侧重评价低流量段的拟合效果。
    添加小常数epsilon=0.01以避免log(0)的问题。

    Args:
        obs: 实测径流序列
        sim: 模拟径流序列

    Returns:
        float: LogNSE值
    """
    epsilon = 0.01  # 防止log(0)的小常数
    log_obs = np.log(obs + epsilon)
    log_sim = np.log(sim + epsilon)
    return nse(log_obs, log_sim)


# ============================================================
# 优化结果数据结构
# ============================================================

@dataclass
class OptimizationResult:
    """单目标优化结果

    Attributes:
        best_params: 最优参数组合
        best_fitness: 最优适应度值
        convergence_history: 收敛历史记录（每代最优值列表）
        n_evaluations: 目标函数总评价次数
        wall_time: 实际运行时间 (秒)
    """
    best_params: np.ndarray
    best_fitness: float
    convergence_history: list = field(default_factory=list)
    n_evaluations: int = 0
    wall_time: float = 0.0


@dataclass
class MOPSOResult:
    """多目标粒子群优化结果

    Attributes:
        pareto_front: Pareto前沿目标值矩阵 (n_solutions × n_objectives)
        pareto_set: Pareto最优参数集矩阵 (n_solutions × n_params)
        convergence_history: 收敛历史记录
    """
    pareto_front: np.ndarray
    pareto_set: np.ndarray
    convergence_history: list = field(default_factory=list)


# ============================================================
# 目标函数工厂
# ============================================================

def create_objective(
    rainfall: np.ndarray,
    pet: np.ndarray,
    obs_runoff: np.ndarray,
    metric: str = 'nse',
    warmup: int = 365,
) -> Callable[[np.ndarray], float]:
    """创建目标函数（用于优化器最小化）

    将GR4J模型模拟与指定评价指标组合，返回一个可直接传入优化器的目标函数。
    返回的目标函数接受参数向量，返回待最小化的标量值。

    对于NSE和KGE等"越大越好"的指标，返回其负值以适配最小化优化器。
    对于RMSE等"越小越好"的指标，直接返回原值。

    Args:
        rainfall: 日降雨序列 (mm/day)
        pet: 日潜在蒸散发序列 (mm/day)
        obs_runoff: 实测日径流序列 (mm/day)
        metric: 评价指标名称，可选 'nse', 'kge', 'rmse', 'pbias', 'log_nse'
        warmup: 预热期天数，预热期内的数据不参与评价指标计算

    Returns:
        Callable: 目标函数 f(params) -> float (待最小化的值)

    Raises:
        ValueError: 当metric参数不在支持的指标列表中时
    """
    # 支持的指标映射
    metric_functions = {
        'nse': nse,
        'kge': kge,
        'rmse': rmse,
        'pbias': pbias,
        'log_nse': log_nse,
    }

    # 需要取负值的指标（这些指标越大越好，最小化时取负）
    negate_metrics = {'nse', 'kge', 'log_nse'}

    metric = metric.lower()
    if metric not in metric_functions:
        raise ValueError(
            f"不支持的评价指标: '{metric}'。"
            f"可选指标: {list(metric_functions.keys())}"
        )

    metric_func = metric_functions[metric]
    negate = metric in negate_metrics

    def objective(params: np.ndarray) -> float:
        """目标函数: 运行GR4J模型并计算评价指标

        Args:
            params: 模型参数向量 [X1, X2, X3, X4]

        Returns:
            float: 待最小化的目标值
        """
        # 运行GR4J模型
        sim_runoff = gr4j(params, rainfall, pet)

        # 跳过预热期，仅用预热期后的数据计算指标
        obs_eval = obs_runoff[warmup:]
        sim_eval = sim_runoff[warmup:]

        # 计算评价指标
        value = metric_func(obs_eval, sim_eval)

        # 对"越大越好"的指标取负值，使其适配最小化优化
        if negate:
            return -value
        else:
            return value

    return objective
