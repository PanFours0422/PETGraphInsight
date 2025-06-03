import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Tuple
import argparse

# 获取当前脚本目录（例如 graph/）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载探测器数据和事件数据

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (探测器数据, 事件数据)
    """
    raw_data_dir = os.path.join(BASE_DIR, '..', 'data', 'raw')
    detector_data = pd.read_csv(os.path.join(raw_data_dir, 'detector.csv'))
    lm_data = pd.read_csv(os.path.join(raw_data_dir, 'lm_data.csv'))
    return detector_data, lm_data


def build_node_features(detector_data: pd.DataFrame) -> torch.Tensor:
    position_features = detector_data[['x', 'y', 'z']].values
    crystal_features = detector_data[['crystal_size_x', 'crystal_size_y', 'crystal_size_z']].values
    ring_index = detector_data['ring_index'].values.reshape(-1, 1)
    compute_capability = detector_data['compute_capability'].values.reshape(-1, 1)

    node_features = np.concatenate([
        position_features,
        crystal_features,
        ring_index,
        compute_capability
    ], axis=1)

    return torch.FloatTensor(node_features)


def build_edge_index_and_features(lm_data: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.tensor([
        lm_data['detector_i'].values,
        lm_data['detector_j'].values
    ], dtype=torch.long)

    edge_features = torch.tensor([
        lm_data['energy_i'].values,
        lm_data['energy_j'].values,
        lm_data['timestamp_i'].values,
        lm_data['timestamp_j'].values
    ], dtype=torch.float).t()

    return edge_index, edge_features


def build_graph(detector_data: pd.DataFrame, lm_data: pd.DataFrame) -> Data:
    x = build_node_features(detector_data)
    edge_index, edge_attr = build_edge_index_and_features(lm_data)

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(detector_data)
    )
    return graph


def save_graph(graph: Data, output_dir: str) -> None:
    output_path = os.path.join(BASE_DIR, '..', output_dir)
    os.makedirs(output_path, exist_ok=True)
    torch.save(graph, os.path.join(output_path, 'pet_graph.pt'))


def main():
    parser = argparse.ArgumentParser(description='构建PET系统图结构')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='输出目录（相对于项目根目录）')
    args = parser.parse_args()

    print("加载数据...")
    detector_data, lm_data = load_data()

    print("构建图结构...")
    graph = build_graph(detector_data, lm_data)

    print("保存图数据...")
    save_graph(graph, args.output_dir)

    print("图结构构建完成！")
    print(f"节点数量: {graph.num_nodes}")
    print(f"边数量: {graph.edge_index.shape[1]}")
    print(f"节点特征维度: {graph.x.shape[1]}")
    print(f"边特征维度: {graph.edge_attr.shape[1]}")
    print(f"图数据保存在: {os.path.join(args.output_dir, 'pet_graph.pt')}")


if __name__ == "__main__":
    main()
