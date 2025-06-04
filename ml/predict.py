import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv
import torch.nn as nn
import os
import numpy as np
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

def load_model(model_path, in_channels, hidden_channels=64, out_channels=2):
    """加载训练好的模型"""
    model = GNNModel(in_channels, hidden_channels, out_channels)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_coupling(model, subgraphs, device):
    """对子图进行耦合度预测"""
    predictions = []
    probabilities = []
    
    # 创建数据加载器
    loader = DataLoader(subgraphs, batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            probs = F.softmax(out, dim=1)
            pred = out.argmax(dim=1)
            
            predictions.extend(pred.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)


from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 指定全局字体为黑体（SimHei），解决中文乱码
rcParams['font.sans-serif'] = ['SimHei']  # 黑体
rcParams['axes.unicode_minus'] = False    # 解决负号 '-' 显示为方块的问题

def analyze_predictions(predictions, probabilities, data_dir, results_dir):
    """对比预测结果与真实标签，并生成分析报告与可视化图表"""
    labels_path = os.path.join(data_dir, "labels.pt")
    true_labels = torch.load(labels_path).numpy()

    # 分类报告
    print("\n分类报告:")
    print(classification_report(true_labels, predictions, target_names=["低耦合", "高耦合"]))

    # 混淆矩阵
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["低耦合", "高耦合"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("混淆矩阵")
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # ROC 曲线
    fpr, tpr, _ = roc_curve(true_labels, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC 曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title('ROC 曲线')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(results_dir, "roc_curve.png"))
    plt.close()

    # 正确/错误分类柱状图
    correct = predictions == true_labels
    correct_count = np.sum(correct)
    incorrect_count = len(predictions) - correct_count

    plt.figure()
    plt.bar(['预测正确', '预测错误'], [correct_count, incorrect_count], color=['green', 'red'])
    plt.title('预测结果统计')
    plt.ylabel('样本数量')
    plt.savefig(os.path.join(results_dir, 'prediction_accuracy_bar.png'))
    plt.close()

    # 写入详细比对信息
    with open(os.path.join(results_dir, 'prediction_details.txt'), 'a', encoding='utf-8') as f:
        f.write("\n预测 vs 真实比对:\n")
        f.write("-" * 50 + "\n")
        for i, (pred, prob, true_label) in enumerate(zip(predictions, probabilities, true_labels)):
            f.write(f"样本 {i + 1}:\n")
            f.write(f"预测类别: {'高耦合' if pred == 1 else '低耦合'}\n")
            f.write(f"真实类别: {'高耦合' if true_label == 1 else '低耦合'}\n")
            f.write(f"预测是否正确: {'是' if pred == true_label else '否'}\n")
            f.write(f"低耦合概率: {prob[0]:.4f}\n")
            f.write(f"高耦合概率: {prob[1]:.4f}\n")
            f.write("-" * 30 + "\n")


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")
    test_subgraphs = torch.load(os.path.join(data_dir, "subgraphs.pt"), weights_only=False)
    dataset_info = torch.load(os.path.join(data_dir, "dataset_info.pt"), weights_only=False)
    
    # 加载模型
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
    model_path = os.path.join(model_dir, "best_model.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}")
    
    model = load_model(
        model_path,
        in_channels=dataset_info['node_feature_dim'],
        hidden_channels=64,
        out_channels=2
    ).to(device)
    
    # 进行预测
    predictions, probabilities = predict_coupling(model, test_subgraphs, device)
    
    # 保存预测结果
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存预测结果
    results = {
        'predictions': predictions,  # 预测的类别（0或1）
        'probabilities': probabilities,  # 预测的概率分布
        'coupling_level': ['低耦合' if p == 0 else '高耦合' for p in predictions]  # 耦合度描述
    }
    
    # 保存为numpy文件
    np.save(os.path.join(results_dir, 'prediction_results.npy'), results)
    
    # 打印预测统计信息
    print("\n预测结果统计:")
    print(f"总样本数: {len(predictions)}")
    print(f"低耦合区域数量: {np.sum(predictions == 0)}")
    print(f"高耦合区域数量: {np.sum(predictions == 1)}")
    print(f"平均高耦合概率: {np.mean(probabilities[:, 1]):.4f}")
    
    # 保存详细的预测结果到文本文件
    with open(os.path.join(results_dir, 'prediction_details.txt'), 'w', encoding='utf-8') as f:
        f.write("预测结果详细信息:\n")
        f.write("-" * 50 + "\n")
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            f.write(f"样本 {i+1}:\n")
            f.write(f"预测类别: {'高耦合' if pred == 1 else '低耦合'}\n")
            f.write(f"低耦合概率: {prob[0]:.4f}\n")
            f.write(f"高耦合概率: {prob[1]:.4f}\n")
            f.write("-" * 30 + "\n")

        # 分析预测结果
    analyze_predictions(predictions, probabilities, data_dir, results_dir)

if __name__ == "__main__":
    main() 