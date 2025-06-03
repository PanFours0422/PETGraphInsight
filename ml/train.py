import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv
import torch.nn as nn
import os
from tqdm import tqdm

class GNNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=2):
        super(GNNModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # 全局池化
        x = torch_geometric.nn.global_mean_pool(x, batch)
        
        # 分类层
        x = self.classifier(x)
        return x

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == batch.y).sum())
        total += len(batch.y)
    
    return total_loss / len(train_loader), correct / total

def evaluate_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total += len(batch.y)
    
    return correct / total

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    subgraphs = torch.load(os.path.join(data_dir, "subgraphs.pt"), weights_only=False)
    labels = torch.load(os.path.join(data_dir, "labels.pt"), weights_only=False)
    dataset_info = torch.load(os.path.join(data_dir, "dataset_info.pt"), weights_only=False)
    
    # 将标签添加到每个图数据中
    for i, graph in enumerate(subgraphs):
        graph.y = torch.tensor([labels[i]], dtype=torch.long)
    
    # 创建数据加载器
    train_loader = DataLoader(subgraphs, batch_size=32, shuffle=True)
    
    # 打印数据集信息
    print(f"数据集大小: {len(subgraphs)}")
    print(f"节点特征维度: {dataset_info['node_feature_dim']}")
    print(f"边特征维度: {dataset_info['edge_feature_dim']}")
    print(f"类别数: {dataset_info['num_classes']}")
    
    # 初始化模型
    model = GNNModel(
        in_channels=dataset_info['node_feature_dim'],
        hidden_channels=64,
        out_channels=2
    ).to(device)
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 100
    best_acc = 0
    
    # 创建模型保存目录
    model_save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    os.makedirs(model_save_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, device)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        # 保存最佳模型
        if train_acc > best_acc:
            best_acc = train_acc
            model_save_path = os.path.join(model_save_dir, 'best_model.pt')
            torch.save(model.state_dict(), model_save_path)
            print(f'New best model saved with accuracy: {best_acc:.4f}')

if __name__ == "__main__":
    main()
