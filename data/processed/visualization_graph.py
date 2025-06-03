import os
import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# 图数据路径
GRAPH_PATH = 'pet_graph.pt'
SUBGRAPHS_PATH = 'subgraphs.pt'

# 加载图和子图数据
graph = torch.load(GRAPH_PATH, weights_only= False)
subgraphs = torch.load(SUBGRAPHS_PATH, weights_only= False)

# 转换为networkx图
def visualize_graph(pyG_data, title, ax, node_size=20, edge_color='gray'):
    G = to_networkx(pyG_data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)  # 位置可根据坐标信息改为pos=坐标
    nx.draw(G, pos, ax=ax, node_size=node_size, edge_color=edge_color)
    ax.set_title(title)
    ax.axis('off')

# 创建画布
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# 可视化整个图（大图）
visualize_graph(graph, "Full PET Graph", axes[0, 0])

# 可视化前5个子图
for i in range(5):
    row = (i + 1) // 3
    col = (i + 1) % 3
    visualize_graph(subgraphs[i], f"Subgraph {i+1}", axes[row, col])

plt.tight_layout()
plt.show()
