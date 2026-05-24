"""
FMG 求解器：二维泊松方程  Δu = v(x,y)  定义域 [-1,1]×[-1,1]，Dirichlet 边界条件。
源项 v 为圆形顶帽函数：r < 0.25 时 v=1，否则 v=0。
运行后输出 CSV 和热力图。
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import time


# ============================================================
#  工具函数
# ============================================================

def idx_2d(n, i, j):
    """(n+1)×(n+1) 网格上点 (i,j) 的一维索引（列优先）。"""
    return i + (n + 1) * j


# ============================================================
#  源项
# ============================================================

def source(x, y):
    """圆形顶帽函数：r < 0.25 时值为 1，否则为 0。"""
    return np.where(np.sqrt(x**2 + y**2) < 0.25, 1.0, 0.0)


# ============================================================
#  多层网格算子构建
# ============================================================

def build_matrix(n, L=1.0):
    """
    构建 n 区间（每边 n+1 点）网格的 CSR 矩阵。
    内部点使用五点差分模板，边界点为恒等行（Dirichlet）。
    直接拼接 CSR 数组，避免中间大矩阵。
    """
    N = n + 1
    total = N * N
    h = 2.0 * L / n
    h2 = h * h

    n_interior = (n - 1) * (n - 1)
    n_boundary = total - n_interior
    cap = 5 * n_interior + n_boundary

    data = np.empty(cap)
    indices = np.empty(cap, dtype=np.int32)
    indptr = np.empty(total + 1, dtype=np.int32)

    ptr = 0
    for k in range(total):
        indptr[k] = ptr
        i = k % N           # x 坐标（列优先：k = i + N*j）
        j = k // N           # y 坐标
        if i == 0 or i == n or j == 0 or j == n:
            data[ptr] = 1.0
            indices[ptr] = k
            ptr += 1
        else:
            data[ptr] = 4.0 / h2
            indices[ptr] = k
            ptr += 1
            data[ptr] = -1.0 / h2
            indices[ptr] = idx_2d(n, i - 1, j)   # 左
            ptr += 1
            data[ptr] = -1.0 / h2
            indices[ptr] = idx_2d(n, i + 1, j)   # 右
            ptr += 1
            data[ptr] = -1.0 / h2
            indices[ptr] = idx_2d(n, i, j - 1)   # 下
            ptr += 1
            data[ptr] = -1.0 / h2
            indices[ptr] = idx_2d(n, i, j + 1)   # 上
            ptr += 1
    indptr[total] = ptr

    return csr_matrix((data[:ptr], indices[:ptr], indptr), shape=(total, total))


def build_rhs(n, L=1.0):
    """
    构建最细网格的右端向量。
    内部点为源项值，边界点为 0。
    """
    N = n + 1
    xs = np.linspace(-L, L, N)
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    V = source(X, Y)
    V[0, :] = V[n, :] = 0.0
    V[:, 0] = V[:, n] = 0.0
    return V.ravel(order='F')


# ============================================================
#  松弛 — 纯 Gauss–Seidel
# ============================================================

def relax(A, u, f, n_sweeps=1):
    """
    基于 CSR 矩阵 A 的 Gauss–Seidel 扫描，原地更新 u。
    """
    for _ in range(n_sweeps):
        for k in range(A.shape[0]):
            row_start = A.indptr[k]
            row_end = A.indptr[k + 1]
            sigma = 0.0
            diag = 0.0
            for p in range(row_start, row_end):
                col = A.indices[p]
                val = A.data[p]
                if col == k:
                    diag = val
                else:
                    sigma += val * u[col]
            u[k] = (f[k] - sigma) / diag


# ============================================================
#  限制算子 —（Injection）
# ============================================================

def restrict(v, n_fine):
    """直接注入：reshape 到二维后两轴均隔点采样。"""
    N_f = n_fine + 1
    n_c = n_fine // 2
    N_c = n_c + 1
    v_2d = v.reshape(N_f, N_f, order='F')
    v_c = (v_2d[0::2, 0::2]).ravel(order='F')
    return v_c


# ============================================================
#  延拓算子 —（Linear interpolation）
# ============================================================

def prolong(v_c, n_coarse):
    """
    从粗网格 (n_coarse) 到细网格 (2*n_coarse) 的双线性插值。
    通过二维 reshape 向量化。
    """
    N_c = n_coarse + 1
    n_f = 2 * n_coarse
    N_f = n_f + 1

    c2d = v_c.reshape(N_c, N_c, order='F')
    f2d = np.zeros((N_f, N_f))

    # 偶-偶：直接复制
    f2d[0::2, 0::2] = c2d

    # 奇-偶：水平方向线性插值
    f2d[1::2, 0::2] = 0.5 * (c2d[:-1, :] + c2d[1:, :])

    # 偶-奇：竖直方向线性插值
    f2d[0::2, 1::2] = 0.5 * (c2d[:, :-1] + c2d[:, 1:])

    # 奇-奇：对角方向双线性插值
    f2d[1::2, 1::2] = 0.25 * (c2d[:-1, :-1] + c2d[1:, :-1] +
                               c2d[:-1, 1:] + c2d[1:, 1:])

    return f2d.ravel(order='F')


# ============================================================
#  V-cycle
# ============================================================

def v_cycle(n, v, f, A_dict, nu1=3, nu2=3):
    """在网格层级 n 上执行一次 V-cycle。"""
    A = A_dict[n]
    relax(A, v, f, nu1)

    if n <= 2:
        relax(A, v, f, nu2)
        return v

    # 残差 → 限制到粗网格
    r = f - A.dot(v)
    r_c = restrict(r, n)

    # 粗网格修正
    n_coarse = n // 2
    e_c = np.zeros_like(r_c)
    e_c = v_cycle(n_coarse, e_c, r_c, A_dict, nu1, nu2)

    # 延拓修正量并叠加
    v += prolong(e_c, n_coarse)

    relax(A, v, f, nu2)
    return v


# ============================================================
#  FMG 递归
# ============================================================

def fmg(n, f, A_dict, nu1=3, nu2=3):
    """第 n 层的 FMG：从粗网格解延拓得到初值，再做 V-cycle。"""
    v = np.zeros_like(f)
    if n <= 2:
        return v
    f_c = restrict(f, n)
    v_c = fmg(n // 2, f_c, A_dict, nu1, nu2)
    v = prolong(v_c, n // 2)
    v = v_cycle(n, v, f, A_dict, nu1, nu2)
    return v


# ============================================================
#  FMG 外层迭代
# ============================================================

def solve_fmg(n_finest, L=1.0, nu1=3, nu2=3, tol=1e-8, max_iter=100):
    """
    完整 FMG 求解器。
    外层循环：每次 FMG 求解残差方程的修正量，叠加至解向量。
    """
    print("构建矩阵层级 ...", flush=True)
    A_dict = {}
    n = n_finest
    while n >= 2:
        A_dict[n] = build_matrix(n, L)
        print(f"  A[{n}]  尺寸 = {A_dict[n].shape}", flush=True)
        n //= 2

    print("构建右端向量 ...", flush=True)
    F_orig = build_rhs(n_finest, L)

    x = np.zeros_like(F_orig)
    f = F_orig.copy()

    print("开始 FMG 外层迭代 ...", flush=True)
    t0 = time.time()
    for iteration in range(max_iter):
        x += fmg(n_finest, f, A_dict, nu1, nu2)

        r = F_orig - A_dict[n_finest].dot(x)
        res_norm = np.max(np.abs(r))
        print(f"  iter {iteration + 1:3d}  残差 = {res_norm:.3e}", flush=True)

        if res_norm < tol:
            print(f"收敛，共 {iteration + 1} 次 FMG 迭代，"
                  f"用时 = {time.time() - t0:.2f} s", flush=True)
            break

        f = r.copy()
    else:
        print(f"警告：{max_iter} 次迭代后未收敛，"
              f"用时 = {time.time() - t0:.2f} s", flush=True)

    return x, A_dict[n_finest]


# ============================================================
#  后处理：转二维场，输出 CSV 和热力图
# ============================================================

def save_and_plot(u_flat, n, L=1.0, out_csv="solution_fmg.csv",
                  out_png="heatmap_fmg.png"):
    """将一维解向量转为二维场，输出 CSV 和热力图。"""
    N = n + 1
    u_2d = u_flat.reshape(N, N, order='F')

    xs = np.linspace(-L, L, N)
    X, Y = np.meshgrid(xs, xs, indexing='ij')

    df = pd.DataFrame({
        'x': X.ravel(order='C'),
        'y': Y.ravel(order='C'),
        'u': u_2d.ravel(order='C'),
    })
    df.to_csv(out_csv, index=False)
    print(f"CSV 已保存 → {out_csv}", flush=True)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.contourf(X.T, Y.T, u_2d.T, levels=50, cmap='hot')
    plt.colorbar(label='u')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('FMG 求解二维泊松方程')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"热力图已保存 → {out_png}", flush=True)


# ============================================================
#  主程序
# ============================================================

if __name__ == "__main__":
    N_INTERVALS = 1024
    u, _ = solve_fmg(N_INTERVALS, L=1.0, nu1=3, nu2=3, tol=1e-8)
    save_and_plot(u, N_INTERVALS, L=1.0)
