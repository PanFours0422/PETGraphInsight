import warnings
warnings.filterwarnings("ignore")

import torch
from torch_geometric.data import Data, Dataset, DataLoader
from typing import List, Tuple, Optional
import numpy as np
from sklearn.model_selection import train_test_split
import os
from torch.serialization import add_safe_globals

# 添加PyTorch Geometric相关的安全全局变量
add_safe_globals(['torch_geometric.data.data.DataEdgeAttr'])

class PETGraphDataset(Dataset):
    """
    PET系统图数据集
    """
    def __init__(self, 
                 subgraphs: List[Data],
                 labels: torch.Tensor,
                 transform: Optional[callable] = None):
        """
        初始化数据集
        
        Args:
            subgraphs: 子图列表
            labels: 标签张量
            transform: 数据转换函数
        """
        super(PETGraphDataset, self).__init__(transform)
        self.subgraphs = subgraphs
        self.labels = labels
        
    def len(self) -> int:
        """返回数据集大小"""
        return len(self.subgraphs)
    
    def get(self, idx: int) -> Data:
        """获取指定索引的图数据"""
        data = self.subgraphs[idx]
        data.y = self.labels[idx]
        return data

def create_data_loaders(subgraphs: List[Data],
                       labels: torch.Tensor,
                       batch_size: int = 32,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       test_ratio: float = 0.15,
                       num_workers: int = 4,
                       shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练、验证和测试数据加载器
    
    Args:
        subgraphs: 子图列表
        labels: 标签张量
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        num_workers: 数据加载线程数
        shuffle: 是否打乱数据
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (训练集加载器, 验证集加载器, 测试集加载器)
    """
    # 确保比例之和为1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "数据集比例之和必须为1"
    
    # 划分数据集
    indices = np.arange(len(subgraphs))
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_ratio,
        random_state=42,
        stratify=labels.numpy()
    )
    
    # 从剩余数据中划分验证集和测试集
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=val_test_ratio,
        random_state=42,
        stratify=labels[temp_indices].numpy()
    )
    
    # 创建数据集
    train_dataset = PETGraphDataset(
        [subgraphs[i] for i in train_indices],
        labels[train_indices]
    )
    val_dataset = PETGraphDataset(
        [subgraphs[i] for i in val_indices],
        labels[val_indices]
    )
    test_dataset = PETGraphDataset(
        [subgraphs[i] for i in test_indices],
        labels[test_indices]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def load_and_prepare_data(data_dir: str,
                         batch_size: int = 32,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    加载并准备数据
    
    Args:
        data_dir: 数据目录
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, dict]: (训练集加载器, 验证集加载器, 测试集加载器, 数据集信息)
    """
    # 加载数据
    try:
        # 首先尝试使用weights_only=False加载
        subgraphs = torch.load(os.path.join(data_dir, 'subgraphs.pt'), weights_only=False)
        labels = torch.load(os.path.join(data_dir, 'labels.pt'), weights_only=False)
        dataset_info = torch.load(os.path.join(data_dir, 'dataset_info.pt'), weights_only=False)
    except Exception as e:
        print(f"使用weights_only=False加载失败: {str(e)}")
        print("尝试使用weights_only=True加载...")
        # 如果失败，使用weights_only=True加载
        subgraphs = torch.load(os.path.join(data_dir, 'subgraphs.pt'), weights_only=True)
        labels = torch.load(os.path.join(data_dir, 'labels.pt'), weights_only=True)
        dataset_info = torch.load(os.path.join(data_dir, 'dataset_info.pt'), weights_only=True)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        subgraphs,
        labels,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio
    )
    
    return train_loader, val_loader, test_loader, dataset_info

def print_dataset_info(train_loader: DataLoader,
                      val_loader: DataLoader,
                      test_loader: DataLoader,
                      dataset_info: dict) -> None:
    """
    打印数据集信息
    
    Args:
        train_loader: 训练集加载器
        val_loader: 验证集加载器
        test_loader: 测试集加载器
        dataset_info: 数据集信息
    """
    print("\n数据集信息:")
    print(f"总样本数: {dataset_info['num_samples']}")
    print(f"类别数: {dataset_info['num_classes']}")
    print(f"节点特征维度: {dataset_info['node_feature_dim']}")
    print(f"边特征维度: {dataset_info['edge_feature_dim']}")
    
    print("\n数据集划分:")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 计算每个数据集的标签分布
    train_labels = torch.cat([batch.y for batch in train_loader])
    val_labels = torch.cat([batch.y for batch in val_loader])
    test_labels = torch.cat([batch.y for batch in test_loader])
    
    print("\n标签分布:")
    print(f"训练集 - 正样本比例: {train_labels.float().mean():.2%}")
    print(f"验证集 - 正样本比例: {val_labels.float().mean():.2%}")
    print(f"测试集 - 正样本比例: {test_labels.float().mean():.2%}") 