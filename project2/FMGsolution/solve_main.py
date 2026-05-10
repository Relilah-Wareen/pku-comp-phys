import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix

def idx_2d(n, i, j):
    return i + (n + 1) * j

N = 8
n = N
size = (n+1)*(n+1)
A_curr = lil_matrix((size, size))
h = 2.0 / n
h2 = h**2

for i in range(n+1):
    for j in range(n+1):
        idx = idx_2d(n, i, j)
        if i == 0 or i == n or j == 0 or j == n:
            A_curr[idx, idx] = 1
        else:
            A_curr[idx, idx] = -4.0 / h2
            A_curr[idx, idx_2d(n, i-1, j)] = 1.0 / h2
            A_curr[idx, idx_2d(n, i+1, j)] = 1.0 / h2
            A_curr[idx, idx_2d(n, i, j+1)] = 1.0 / h2
            A_curr[idx, idx_2d(n, i, j-1)] = 1.0 / h2

A_curr = A_curr.tocsr()

# ---------- 按几何位置输出 CSV ----------
# 转为稠密矩阵
dense = A_curr.toarray()

# 生成坐标标签列表，顺序与线性索引 idx = i + (n+1)*j 一致
labels = []
for j in range(n+1):
    for i in range(n+1):
        labels.append(f"({i},{j})")

# 创建 DataFrame
df = pd.DataFrame(dense, index=labels, columns=labels)

# 保存为 CSV
df.to_csv("matrix_geometric.csv")
print("已保存为 matrix_geometric.csv")