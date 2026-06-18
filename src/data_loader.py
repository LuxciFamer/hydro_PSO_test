"""
数据加载与预处理模块

功能：
    - 从CSV文件加载水文数据（降雨、蒸发、径流）
    - 数据分割（率定期/验证期）
    - 基本统计信息输出
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class HydroData:
    """水文数据容器

    Attributes:
        date: 日期数组 (numpy datetime64)
        rainfall: 降雨量数组 (mm/day, float64)
        pet: 潜在蒸散发数组 (mm/day, float64)
        runoff_obs: 实测径流数组 (mm/day, float64)
    """
    date: np.ndarray       # datetime64
    rainfall: np.ndarray   # float64, mm/day
    pet: np.ndarray        # float64, mm/day
    runoff_obs: np.ndarray  # float64, mm/day


def load_data(filepath: str) -> HydroData:
    """从CSV文件加载水文数据

    CSV文件应包含以下列：date, rainfall, pet, runoff_obs
    日期格式要求：'%Y/%m/%d'

    Args:
        filepath: CSV文件路径

    Returns:
        HydroData: 包含日期、降雨、蒸发、径流的数据容器
    """
    # 读取CSV文件，解析日期列
    df = pd.read_csv(filepath, parse_dates=['date'], date_format='%Y/%m/%d')

    # 提取各列为numpy数组
    date = df['date'].values.astype('datetime64[D]')
    rainfall = df['rainfall'].values.astype(np.float64)
    pet = df['pet'].values.astype(np.float64)
    runoff_obs = df['runoff'].values.astype(np.float64)

    # 创建数据对象
    data = HydroData(date=date, rainfall=rainfall, pet=pet, runoff_obs=runoff_obs)

    # 打印摘要信息
    print("=" * 60)
    print("水文数据加载完成")
    print("=" * 60)
    print(f"  时间范围: {date[0]} ~ {date[-1]}")
    print(f"  数据长度: {len(date)} 天")
    print("-" * 60)
    print(f"  降雨量 (mm/day): 均值={rainfall.mean():.2f}, "
          f"最大={rainfall.max():.2f}, 总量={rainfall.sum():.1f}")
    print(f"  蒸散发 (mm/day): 均值={pet.mean():.2f}, "
          f"最大={pet.max():.2f}, 总量={pet.sum():.1f}")
    print(f"  实测径流 (mm/day): 均值={runoff_obs.mean():.2f}, "
          f"最大={runoff_obs.max():.2f}, 总量={runoff_obs.sum():.1f}")
    print("=" * 60)

    return data


def split_data(data: HydroData, split_year: int = 1990) -> tuple[HydroData, HydroData]:
    """按年份分割数据为率定期和验证期

    Args:
        data: 完整的水文数据
        split_year: 分割年份，该年及之前为率定期，之后为验证期

    Returns:
        tuple: (率定期数据, 验证期数据)
    """
    # 将分割年份转换为datetime64进行比较
    split_date = np.datetime64(f'{split_year}-01-01')

    # 生成率定期和验证期的布尔掩码
    cal_mask = data.date < split_date
    val_mask = data.date >= split_date

    # 分割数据
    cal_data = HydroData(
        date=data.date[cal_mask],
        rainfall=data.rainfall[cal_mask],
        pet=data.pet[cal_mask],
        runoff_obs=data.runoff_obs[cal_mask],
    )

    val_data = HydroData(
        date=data.date[val_mask],
        rainfall=data.rainfall[val_mask],
        pet=data.pet[val_mask],
        runoff_obs=data.runoff_obs[val_mask],
    )

    print(f"数据分割完成 (分割年份: {split_year})")
    print(f"  率定期: {cal_data.date[0]} ~ {cal_data.date[-1]} ({len(cal_data.date)} 天)")
    print(f"  验证期: {val_data.date[0]} ~ {val_data.date[-1]} ({len(val_data.date)} 天)")

    return cal_data, val_data
