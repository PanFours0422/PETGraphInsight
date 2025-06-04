import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from typing import Tuple, List
import argparse


def generate_detector_data(num_detectors: int) -> pd.DataFrame:
    """
    生成PET探测器位置数据（在圆环基础上 XYZ 全部随机排布）
    """
    radius = 100  # 探测器平均半径（mm）
    z_range = (-40, 40)  # Z轴方向随机范围（mm）

    # 生成随机角度和半径（XY平面上圆环结构）
    theta = np.random.uniform(0, 2 * np.pi, num_detectors)
    r = radius * (0.95 + 0.05 * np.random.random(num_detectors))

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(z_range[0], z_range[1], num_detectors)

    # 添加小幅随机扰动（模拟安装误差）
    x += np.random.normal(0, 0.2, num_detectors)
    y += np.random.normal(0, 0.2, num_detectors)
    z += np.random.normal(0, 0.5, num_detectors)

    # 生成其他探测器参数
    crystal_sizes = np.random.uniform(2, 4, (num_detectors, 3))  # 晶体尺寸（mm）
    ring_indices = np.zeros(num_detectors, dtype=int)  # 不再分层，可置零或用 kmeans 聚类赋值
    compute_capabilities = np.random.uniform(0.5, 1.0, num_detectors)  # 计算能力

    # 创建DataFrame
    detector_data = pd.DataFrame({
        'detector_id': range(num_detectors),
        'x': x,
        'y': y,
        'z': z,
        'crystal_size_x': crystal_sizes[:, 0],
        'crystal_size_y': crystal_sizes[:, 1],
        'crystal_size_z': crystal_sizes[:, 2],
        'ring_index': ring_indices,
        'compute_capability': compute_capabilities
    })

    return detector_data


def is_valid_event(pos_i: np.ndarray, pos_j: np.ndarray, energy_i: float, energy_j: float, dt: float) -> bool:
    # 条件 1: 探测器不能相同（由主函数保证）
    # 条件 2: 能量接近511 keV
    if not (480 <= energy_i <= 540 and 480 <= energy_j <= 540):
        return False
    # 条件 3: 时间差小于2 ns
    if abs(dt) > 2:
        return False
    # 条件 4: 发射方向大致对冲，夹角接近180度
    vec = pos_j - pos_i
    dot_product = np.dot(pos_i, vec)
    angle_cos = dot_product / (np.linalg.norm(pos_i) * np.linalg.norm(vec))
    angle = np.arccos(np.clip(angle_cos, -1.0, 1.0)) * 180 / np.pi
    if angle < 150:
        return False
    return True

def generate_lm_data(num_events: int, detector_data: pd.DataFrame) -> pd.DataFrame:
    num_detectors = len(detector_data)
    valid_events = []
    attempts = 0
    max_attempts = num_events * 10  # 防止死循环

    while len(valid_events) < num_events and attempts < max_attempts:
        attempts += 1
        i, j = np.random.choice(num_detectors, 2, replace=False)
        energy_i, energy_j = np.random.uniform(495, 535, 2)
        timestamp_i, timestamp_j = np.random.uniform(0, 50, 2)

        pos_i = detector_data.iloc[i][['x', 'y', 'z']].values
        pos_j = detector_data.iloc[j][['x', 'y', 'z']].values

        if is_valid_event(pos_i, pos_j, energy_i, energy_j, timestamp_i - timestamp_j):
            valid_events.append({
                'event_id': len(valid_events),
                'detector_i': i,
                'detector_j': j,
                'pos_i_x': pos_i[0],
                'pos_i_y': pos_i[1],
                'pos_i_z': pos_i[2],
                'pos_j_x': pos_j[0],
                'pos_j_y': pos_j[1],
                'pos_j_z': pos_j[2],
                'energy_i': energy_i,
                'energy_j': energy_j,
                'timestamp_i': timestamp_i,
                'timestamp_j': timestamp_j
            })

    lm_data = pd.DataFrame(valid_events)
    return lm_data

def save_data(detector_data: pd.DataFrame, lm_data: pd.DataFrame) -> None:
    os.makedirs('raw', exist_ok=True)
    detector_data.to_csv('raw/detector.csv', index=False)
    lm_data.to_csv('raw/lm_data.csv', index=False)

def main():
    parser = argparse.ArgumentParser(description='生成PET系统模拟数据')
    parser.add_argument('--num_detectors', type=int, default=100,
                        help='探测器数量')
    parser.add_argument('--num_events', type=int, default=3000,
                        help='有效事件数量')
    args = parser.parse_args()

    detector_data = generate_detector_data(args.num_detectors)
    lm_data = generate_lm_data(args.num_events, detector_data)

    save_data(detector_data, lm_data)

    print(f"探测器数据保存在: data/raw/detector.csv")
    print(f"事件数据保存在: data/raw/lm_data.csv（共{len(lm_data)}条有效事件）")

if __name__ == "__main__":
    main()
