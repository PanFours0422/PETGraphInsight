import warnings

warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool


# 定义模型
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
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x


# 训练过程
def train_model(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader, desc="Training"):
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

    return total_loss / len(loader), correct / total


# 验证或评估过程
def evaluate_model(model, loader, device):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            pred = out.argmax(dim=1)
            preds.append(pred.cpu())
            labels.append(batch.y.cpu())

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)

    return acc, precision, recall, f1


# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_dir, "data", "processed")
    subgraphs = torch.load(os.path.join(data_dir, "subgraphs.pt"), weights_only=False)
    labels = torch.load(os.path.join(data_dir, "labels.pt"), weights_only=False)
    dataset_info = torch.load(os.path.join(data_dir, "dataset_info.pt"), weights_only=False)

    # 添加标签
    for i, graph in enumerate(subgraphs):
        graph.y = torch.tensor([labels[i]], dtype=torch.long)

    # 划分训练/验证集（80/20）
    num_train = int(len(subgraphs) * 0.8)
    train_dataset = subgraphs[:num_train]
    val_dataset = subgraphs[num_train:]

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}")
    print(f"节点特征维度: {dataset_info['node_feature_dim']}")
    print(f"边特征维度: {dataset_info['edge_feature_dim']}")
    print(f"类别数: {dataset_info['num_classes']}")

    # 初始化模型
    model = GNNModel(
        in_channels=dataset_info['node_feature_dim'],
        hidden_channels=64,
        out_channels=dataset_info['num_classes']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model_save_dir = os.path.join(base_dir, "models")
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'best_model.pt')

    num_epochs = 100
    best_val_acc = 0

    train_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        train_loss, train_acc = train_model(model, train_loader, optimizer, device)
        val_acc, _, _, _ = evaluate_model(model, val_loader, device)

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved with Val Acc: {best_val_acc:.4f}")

    # 绘图：训练损失与验证准确率
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.title("Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 最佳模型评估
    model.load_state_dict(torch.load(model_save_path))
    acc, precision, recall, f1 = evaluate_model(model, val_loader, device)
    print("\n📊 最佳模型评估指标：")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")


if __name__ == "__main__":
    main()
