import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# 子图路径
SUBGRAPHS_PATH = 'subgraphs.pt'

# 加载子图
subgraphs = torch.load(SUBGRAPHS_PATH, weights_only=False)

# 可视化函数
def visualize_graph(pyG_data, title, ax, node_size=20, edge_color='gray'):
    G = to_networkx(pyG_data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, node_size=node_size, edge_color=edge_color)
    ax.set_title(title)
    ax.axis('off')

# 创建画布
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

# 可视化前5个子图
for i in range(5):
    row = i // 3
    col = i % 3
    visualize_graph(subgraphs[i], f"Subgraph {i+1}", axes[row, col])

# 最后一个子图位置空着
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
