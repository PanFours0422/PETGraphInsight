import warnings
warnings.filterwarnings("ignore")

import os
import torch
import numpy as np
from torch_geometric.data import Data, Batch
from torch_geometric.utils import k_hop_subgraph
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm
from torch.serialization import add_safe_globals
import random

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 添加PyTorch Geometric相关的安全全局变量
add_safe_globals(['torch_geometric.data.data.DataEdgeAttr'])

def load_graph(graph_path: str) -> Data:
    if not os.path.isabs(graph_path):
        graph_path = os.path.join(PROJECT_ROOT, graph_path)
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"图数据文件不存在: {graph_path}")
    try:
        return torch.load(graph_path, weights_only=False)
    except Exception as e:
        print(f"使用weights_only=False加载失败: {str(e)}")
        print("尝试使用weights_only=True加载...")
        return torch.load(graph_path, weights_only=True)

def extract_subgraph(graph: Data, center_nodes: torch.Tensor, num_hops: int = 2, max_nodes: int = 50) -> Data:
    subset, edge_index, mapping, edge_mask = k_hop_subgraph(
        center_nodes, num_hops, graph.edge_index, relabel_nodes=True, num_nodes=graph.num_nodes
    )
    if len(subset) > max_nodes:
        center_mask = torch.isin(torch.arange(len(subset)), mapping)
        other_nodes = torch.where(~center_mask)[0]
        num_other = max_nodes - len(center_nodes)
        selected_other = other_nodes[torch.randperm(len(other_nodes))[:num_other]]
        selected_nodes = torch.cat([torch.where(center_mask)[0], selected_other])
        subset = subset[selected_nodes]
        new_edge_index = []
        new_edge_mask = []
        new_edge_attr = []
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            if src in selected_nodes and dst in selected_nodes:
                new_edge_index.append([src, dst])
                new_edge_mask.append(edge_mask[i])
                if graph.edge_attr is not None:
                    new_edge_attr.append(graph.edge_attr[i])
        if len(new_edge_index) > 0:
            edge_index = torch.tensor(new_edge_index, dtype=torch.long).t()
            edge_mask = torch.tensor(new_edge_mask, dtype=torch.bool)
            edge_attr = torch.stack(new_edge_attr) if graph.edge_attr is not None else None
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_mask = torch.zeros(0, dtype=torch.bool)
            edge_attr = None
        new_mapping = torch.zeros(len(subset), dtype=torch.long)
        for i, node in enumerate(selected_nodes):
            new_mapping[i] = node
        if edge_index.shape[1] > 0:
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[:, i]
                edge_index[0, i] = torch.where(new_mapping == src)[0][0]
                edge_index[1, i] = torch.where(new_mapping == dst)[0][0]
        mapping = new_mapping
    else:
        edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr is not None else None
    x = graph.x[subset]
    subgraph = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=len(subset),
        center_nodes=torch.arange(len(center_nodes)),
        original_nodes=subset
    )
    return subgraph

def generate_labels(graph: Data, subgraph: Data, center_nodes: torch.Tensor) -> torch.Tensor:
    """
    耦合区域判断逻辑融合：以探测器距离为主，能量/时间为辅
    """
    node_features = graph.x
    pos1 = node_features[center_nodes[0], :3]  # 前3维假设是位置
    pos2 = node_features[center_nodes[1], :3]
    distance = torch.norm(pos1 - pos2)

    # 设定距离阈值（单位根据x的单位设定，假设是毫米）
    distance_threshold = 100.0  # 小于此为低耦合区
    if distance < distance_threshold:
        return torch.tensor(0)  # 距离太近，低耦合

    # 进一步考虑能量和时间差影响
    edge_idx = torch.where(
        (graph.edge_index[0] == center_nodes[0]) &
        (graph.edge_index[1] == center_nodes[1])
    )[0]
    if len(edge_idx) == 0:
        return torch.tensor(0)

    edge_features = graph.edge_attr[edge_idx]
    energy_diff = torch.abs(edge_features[0, 0] - edge_features[0, 1])
    time_diff = torch.abs(edge_features[0, 2] - edge_features[0, 3])

    energy_threshold = 10.0
    time_threshold = 10.0

    is_high = (energy_diff < energy_threshold) & (time_diff < time_threshold)
    return torch.tensor(1 if is_high else 0)

def build_training_samples(graph: Data, num_samples: int = 1000, num_hops: int = 2, max_nodes: int = 50, min_samples_per_class: int = 2) -> Tuple[List[Data], torch.Tensor]:
    subgraphs = []
    labels = []
    unique_edges = torch.unique(graph.edge_index, dim=1)
    print("确保每个类别都有足够的样本...")
    class_samples = {0: [], 1: []}
    for i in range(len(unique_edges[0])):
        center_nodes = unique_edges[:, i]
        label = generate_labels(graph, None, center_nodes)
        class_samples[label.item()].append((center_nodes, i))
    for label, samples in class_samples.items():
        if len(samples) < min_samples_per_class:
            print(f"警告：类别 {label} 的样本数量不足（{len(samples)} < {min_samples_per_class}）")
            while len(samples) < min_samples_per_class:
                samples.append(samples[0])
    selected_samples = []
    samples_per_class = num_samples // 2
    for label, samples in class_samples.items():
        if len(samples) > samples_per_class:
            indices = torch.randperm(len(samples))[:samples_per_class]
            selected_samples.extend([samples[i] for i in indices])
        else:
            selected_samples.extend(samples)
    random.shuffle(selected_samples)
    print("构建训练样本...")
    for center_nodes, edge_idx in tqdm(selected_samples):
        subgraph = extract_subgraph(graph, center_nodes, num_hops, max_nodes)
        label = generate_labels(graph, subgraph, center_nodes)
        subgraphs.append(subgraph)
        labels.append(label)
    return subgraphs, torch.stack(labels)

def save_samples(subgraphs: List[Data], labels: torch.Tensor, output_dir: str = 'data/processed') -> None:
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)
    subgraphs_path = os.path.join(output_dir, 'subgraphs.pt')
    torch.save(subgraphs, subgraphs_path)
    labels_path = os.path.join(output_dir, 'labels.pt')
    torch.save(labels, labels_path)
    info = {
        'num_samples': len(subgraphs),
        'num_classes': 2,
        'node_feature_dim': subgraphs[0].x.shape[1],
        'edge_feature_dim': subgraphs[0].edge_attr.shape[1] if subgraphs[0].edge_attr is not None else 0
    }
    info_path = os.path.join(output_dir, 'dataset_info.pt')
    torch.save(info, info_path)
    print(f"数据已保存到: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='构建PET系统训练样本')
    parser.add_argument('--graph_path', type=str, default='data/processed/pet_graph.pt', help='图数据文件路径')
    parser.add_argument('--output_dir', type=str, default='data/processed', help='输出目录')
    parser.add_argument('--num_samples', type=int, default=1000, help='样本数量')
    parser.add_argument('--num_hops', type=int, default=2, help='邻居跳数')
    parser.add_argument('--max_nodes', type=int, default=50, help='最大节点数量')
    args = parser.parse_args()
    print("加载图数据...")
    graph = load_graph(args.graph_path)
    print("构建训练样本...")
    subgraphs, labels = build_training_samples(
        graph,
        args.num_samples,
        args.num_hops,
        args.max_nodes
    )
    print("保存训练样本...")
    save_samples(subgraphs, labels, args.output_dir)
    print("训练样本构建完成！")
    print(f"样本数量: {len(subgraphs)}")
    print(f"正样本比例: {labels.float().mean():.2%}")
    print(f"节点特征维度: {subgraphs[0].x.shape[1]}")
    print(f"边特征维度: {subgraphs[0].edge_attr.shape[1] if subgraphs[0].edge_attr is not None else 0}")

if __name__ == "__main__":
    main()
