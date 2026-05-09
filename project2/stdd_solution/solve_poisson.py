import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import pandas as pd

# ---------- 参数 ----------
N = 1024                  # 每边网格点数（含边界）
L = 1.0                  # 区域半宽 [-L, L]
x = np.linspace(-L, L, N)
y = np.linspace(-L, L, N)
dx = x[1] - x[0]
dy = y[1] - y[0]

# ---------- 源项 v(x,y) ----------
def source(xx, yy):
    # Θ(0.25 - sqrt(x²+y²))
    return np.where(np.sqrt(xx**2 + yy**2) < 0.25, 1.0, 0.0)

# 在网格上计算 v
X, Y = np.meshgrid(x, y)
V = source(X, Y)

# ---------- 构建离散拉普拉斯算子 ----------
# 未知量为内部点 (i=1..N-2, j=1..N-2)，共 n_unknowns = (N-2)*(N-2)
n_inner = N - 2
n_unk = n_inner * n_inner

# 一维二阶导数矩阵 (N-2 x N-2) 使用固定 dx
main_diag = -2.0 / dx**2 * np.ones(n_inner)
off_diag = 1.0 / dx**2 * np.ones(n_inner - 1)
Dxx = sparse.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n_inner, n_inner))

# 二维拉普拉斯 = I ⊗ Dxx + Dyy ⊗ I, 这里 dy=dx 所以 Dyy = Dxx
I = sparse.eye(n_inner)
A = sparse.kron(I, Dxx) + sparse.kron(Dxx, I)   # 形状 (n_unk, n_unk)

# ---------- 构建右边向量 ----------
# 内部点按行优先排列，索引 (j,i) -> k = j*n_inner + i
b = np.zeros(n_unk)
for j in range(n_inner):          # y方向内部点索引
    for i in range(n_inner):      # x方向内部点索引
        k = j * n_inner + i
        # 网格点真实坐标
        xi = x[i+1]
        yj = y[j+1]
        b[k] = V[j+1, i+1]        # 源项已映射

# 齐次边界条件已隐含（边界点不进入未知量，且五点模板不会用到边界外的值，因为拉普拉斯算子在边界处仅牵涉内部点）

# ---------- 求解线性方程组 ----------
u_inner = spsolve(A, b)

# ---------- 重建完整场（边界值为0） ----------
U = np.zeros((N, N))
U[1:-1, 1:-1] = u_inner.reshape(n_inner, n_inner)

# ---------- 输出CSV ----------
# 按网格顺序（y行优先或x优先？绘图代码用 pd.read_csv 读取三列并 reshape(n,n)，
# 需要保持与 reshape 一致的顺序。meshgrid 产生的 X,Y 默认是 'xy' 索引，
# 即 X[j,i], Y[j,i] 对应 y[j], x[i]。我们输出的顺序应与 reshape 兼容。
# 采用按列展开（C order）：行优先，y 作为第一个索引，x 为第二个索引。
x_flat = X.ravel()   # 按行（C order）展开
y_flat = Y.ravel()
u_flat = U.ravel()

df = pd.DataFrame({'x': x_flat, 'y': y_flat, 'u': u_flat})
df.to_csv('solution.csv', index=False)

print("求解完成，结果已保存至 solution.csv")