import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取 CSV 文件
df = pd.read_csv("lm_data.csv")  # 将文件名替换为你的实际文件名

# 提取坐标数据
pos_i = df[["pos_i_x", "pos_i_y", "pos_i_z"]].values
pos_j = df[["pos_j_x", "pos_j_y", "pos_j_z"]].values

# 计算中点坐标
midpoints = (pos_i + pos_j) / 2

# 创建3D图形
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制 pos_i 点（蓝色）
ax.scatter(pos_i[:, 0], pos_i[:, 1], pos_i[:, 2], c='blue', label='pos_i', s=10)

# 绘制 pos_j 点（红色）
ax.scatter(pos_j[:, 0], pos_j[:, 1], pos_j[:, 2], c='red', label='pos_j', s=10)

# 绘制中点（绿色）
ax.scatter(midpoints[:, 0], midpoints[:, 1], midpoints[:, 2], c='green', label='midpoint', s=10)

# 添加图例和标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title("3D Scatter Plot of pos_i, pos_j and Midpoints")

plt.show()
