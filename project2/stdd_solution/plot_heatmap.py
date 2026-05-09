import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("solution.csv")
# 获取坐标和值
x = data['x'].values
y = data['y'].values
u = data['u'].values

# 确定网格尺寸
n = int(np.sqrt(len(x)))  # 假设方形网格
X = x.reshape(n, n)
Y = y.reshape(n, n)
U = u.reshape(n, n)

plt.figure(figsize=(8, 6))
# 可改用 pcolormesh，这里用 contourf 或 imshow
plt.contourf(X, Y, U, levels=50, cmap='hot')
plt.colorbar(label='u')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Solution of 2D Poisson equation (FMG)')
plt.axis('equal')
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.show()