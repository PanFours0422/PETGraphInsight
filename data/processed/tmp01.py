import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

# 图数据路径
GRAPH_PATH = 'pet_graph.pt'

# 加载大图数据
graph = torch.load(GRAPH_PATH, weights_only=False)

# 可视化函数
def visualize_graph(pyG_data, title, ax, node_size=20, edge_color='gray'):
    G = to_networkx(pyG_data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, ax=ax, node_size=node_size, edge_color=edge_color)
    ax.set_title(title)
    ax.axis('off')

# 画布
fig, ax = plt.subplots(figsize=(10, 8))
visualize_graph(graph, "Full PET Graph", ax)

plt.tight_layout()
plt.show()
