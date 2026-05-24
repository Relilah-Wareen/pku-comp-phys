# hydro.ipynb 代码详细解释

## 概述

这份代码实现了一个**一维 Euler 方程（可压缩流体力学）的有限体积法（Finite Volume Method）求解器**。Euler 方程描述了无粘性可压缩流体的运动，是计算流体力学中最基础的模型之一。

代码使用 GoTUnov 型格式，结合 HLL（Harten-Lax-van Leer）近似 Riemann 求解器来计算数值通量。

---

## 1. 导入和全局变量定义

```python
from pylab import *
idx_rho = 0   # 密度 (density)
idx_ene = 1   # 能量 (energy)
idx_mom = 2   # 动量 (momentum)
idx_spe = 5   # 化学组分 (chemical species) — 未使用
```

定义了守恒变量的索引常量：
- `idx_rho = 0`：密度 $\rho$
- `idx_ene = 1`：总能量 $E$
- `idx_mom = 2`：动量 $p = \rho v$

`n_int = 3` 意味着每个单元格存储 3 个守恒变量（密度、能量、动量），即求解一维 Euler 方程组。`idx_spe = 5` 是为多组分扩展预留的，当前代码未使用。

---

## 2. `minmod` 函数（不完整）

```python
def minmod( a, b ):
    if a * b < 0:
    # 函数体不完整，res 未定义
    return res
```

### 设计意图

minmod 是一种**斜率限制器（slope limiter）**，用于 MUSCL 重构中抑制数值振荡。它的数学定义是：

$$\text{minmod}(a, b) = \begin{cases} 0 & \text{if } ab \le 0 \\ \text{sign}(a) \cdot \min(|a|, |b|) & \text{if } ab > 0 \end{cases}$$

### 行为解释：
- 当 $a$ 和 $b$ **异号**（`a * b < 0`）时，说明梯度方向不一致，取 0，从而在间断附近退化为常数重构，避免引入虚假振荡（满足 TVD 性质）。
- 当 $a$ 和 $b$ **同号**（`a * b >= 0`）时，取绝对值较小的那个，实现最保守的梯度限制。

### 当前问题
函数体不完整：`if a * b < 0:` 下的分支没有写，`res` 也未定义。这是一个半成品，不能直接运行。

**修正版本应为：**
```python
def minmod(a, b):
    if a * b <= 0:
        res = 0.0
    else:
        res = copysign(min(abs(a), abs(b)), a)
    return res
```

---

## 3. `minmod_arr` 函数

```python
def minmod_arr( a, b ):
    res = zeros_like( a )
    for i in range( len( res ) ):
        res[ i ] = minmod( a[i], b[i] )
    return res
```

### 行为解释

这是 `minmod` 的**数组版本**，对两个数组的元素逐对调用标量 `minmod`。`a` 和 `b` 通常代表左、右两侧的梯度估计，返回值是受限后的斜率数组。

由于依赖有 bug 的 `minmod`，目前也不能正常工作。

---

## 4. `hydro` 类（核心求解器）

### 构造函数 `__init__`

```python
class hydro:
    def __init__(self, n_cell, dx, n_int=3, n_gh=1, gamma=1.667, CFL=0.5):
        self.u = zeros((n_cell,            n_int))   # 守恒变量
        self.w = zeros((n_cell + 2 * n_gh, n_int))   # 原变量（含 ghost cell）
        self.f = zeros((n_cell + 1,        n_int))   # 界面通量

        self.n_gh   = n_gh        # ghost cell 数量
        self.n_int  = n_int       # 变量数目（3）
        self.n_cell = n_cell      # 网格数
        self.dx     = dx          # 网格间距
        self.CFL    = CFL         # CFL 数
        self.gamma  = gamma       # 绝热指数（默认 1.667 ≈ 5/3，对应单原子气体）
        self.gam1   = gamma - 1
```

### 关键参数解释

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `n_cell` | 计算单元数 | 256 |
| `dx` | 网格间距 | 0.1 |
| `n_int` | 方程数（守恒变量数） | 3 |
| `n_gh` | ghost cell 层数 | 1 |
| `gamma` | 绝热指数 $\gamma = c_p / c_v$ | 1.667 |
| `CFL` | CFL（Courant-Friedrichs-Lewy）数 | 0.5 |

### 数据结构解释

- **`self.u`**（shape: `(n_cell, 3)`）：**守恒变量**（conservative variables）数组。每行是一个单元格的 $[\rho, E, p]$。这是时间推进的实际变量。

- **`self.w`**（shape: `(n_cell+2*ng, 3)`）：**原变量**（primitive variables）数组。比 `u` 多出两端的 ghost cell。每行是 $[\rho, P, v]$（密度、压强、速度）。在计算 Riemann 问题时需要原变量。

- **`self.f`**（shape: `(n_cell+1, 3)`）：**数值通量**数组，定义在 $n+1$ 个单元界面上。`f[i]` 是界面 $i-1/2$（即单元 $i-1$ 和 $i$ 之间）的通量。

    ```
    单元:     |   0   |   1   |   2   |  ...  | n-1   |
    界面:   0       1       2       3       n-1      n
    通量: f[0]   f[1]   f[2]   f[3]    ...     f[n]
    ```

---

### `init` 方法 — 初始化 Riemann 问题

```python
def init(self, rho_l, pre_l, vel_l, rho_r, pre_r, vel_r):
    i_half = len(self.u) // 2
    ene_l = pre_l / (self.gamma - 1) + vel_l**2 * rho_l / 2
    ene_r = pre_r / (self.gamma - 1) + vel_r**2 * rho_r / 2
    self.u[:i_half, idx_rho] = rho_l
    self.u[:i_half, idx_ene] = ene_l
    self.u[:i_half, idx_mom] = rho_l * vel_l
    self.u[i_half:, idx_rho] = rho_r
    self.u[i_half:, idx_ene] = ene_r
    self.u[i_half:, idx_mom] = rho_r * vel_r
```

### 行为解释

设置一个经典的 **Riemann 问题（激波管问题，Sod's shock tube）**初值：

- 计算域被等分为两半（`i_half = n_cell // 2`）
- **左半区域**：密度 `rho_l`、压强 `pre_l`、速度 `vel_l`
- **右半区域**：密度 `rho_r`、压强 `pre_r`、速度 `vel_r`

总能量 $E$ 由理想气体状态方程给出：
$$E = \frac{P}{\gamma - 1} + \frac{1}{2} \rho v^2$$

其中第一项是**内能**（由压强和绝热指数导出），第二项是**动能**。

**Sod 激波管经典参数**（`init(1, 1, 0, 0.125, 0.1, 0)`）：
- 左：$(\rho, P, v) = (1, 1, 0)$ — 高压高密静止气体
- 右：$(\rho, P, v) = (0.125, 0.1, 0)$ — 低压低密静止气体
- 隔膜移除后会产生：稀疏波向左传播、接触间断和激波向右传播

---

### `cons2prim` 方法 — 守恒变量 → 原变量

```python
def cons2prim(self):
    u_T = self.u.T     # 转置方便按行取变量
    rho = u_T[idx_rho]
    mom = u_T[idx_mom]
    ene = u_T[idx_ene]

    pre = (ene - mom**2 / (2 * rho)) * (self.gamma - 1)
    vel = mom / rho

    self.w[self.n_gh:-self.n_gh, idx_rho] = rho
    self.w[self.n_gh:-self.n_gh, idx_ene] = pre
    self.w[self.n_gh:-self.n_gh, idx_mom] = vel

    cs = sqrt(self.gamma * pre / rho)
    self.dt = min(self.CFL * self.dx / (abs(vel) + cs))
```

### 行为解释

1. **变量反演**：从守恒变量 $(\rho, \rho v, E)$ 推导原变量 $(\rho, P, v)$：
   $$\rho = \rho$$
   $$v = \frac{\rho v}{\rho} = \frac{p}{\rho}$$
   $$P = (\gamma - 1)\left(E - \frac{(\rho v)^2}{2\rho}\right) = (\gamma - 1)\left(E - \frac{1}{2}\rho v^2\right)$$

   这利用了理想气体状态方程 $P = (\gamma-1)\rho e$，其中 $e = \frac{E}{\rho} - \frac{1}{2}v^2$ 是比内能。

2. **填充 ghost cell 内部**：`self.w[self.n_gh:-self.n_gh]` 切片将原变量写入内部单元，ghost cell 将在 `set_boundary` 中填充。

3. **计算时间步长 $\Delta t$**：由 CFL 条件决定：
   $$\Delta t = \text{CFL} \cdot \frac{\Delta x}{\max(|v| + c_s)}$$
   其中 $c_s = \sqrt{\gamma P / \rho}$ 是**声速**。`|v| + c_s` 是信号传播的最大速度（流体速度 + 声速）。取所有单元的最小值保证全局稳定。

---

### `set_boundary` 方法 — 边界条件

```python
def set_boundary(self):
    for i in range(self.n_gh):
        self.u[i]      = self.u[ self.n_gh     ]
        self.u[-i - 1] = self.u[-self.n_gh - 1 ]
```

### 行为解释

采用**零阶外推**（Neumann / outflow 边界条件）：
- **左边界 ghost cell**（`u[0]`）：复制第一个内部单元（`u[n_gh]`）的值
- **右边界 ghost cell**（`u[-1]`）：复制最后一个内部单元（`u[-n_gh-1]`）的值

这意味着边界上没有梯度，波可以自由地传出计算域而不产生非物理反射。

---

### `reconstruct` 方法 — 界面重构

```python
def reconstruct(self):
    self.wl = zeros(self.n_cell + 1)
    self.wr = zeros(self.n_cell + 1)

    self.wl = self.w[self.n_gh - 1 : -self.n_gh]
    if self.n_gh == 1:
        self.wr = self.w[ self.n_gh : ]
    else:
        self.wr = self.w[ self.n_gh : 1 - self.n_gh ]
    return
```

### 行为解释

采用**分片常数重构（piecewise constant）**，即一阶精度：

- `self.wl[i]`：界面 $i-1/2$ 的**左侧**状态 — 取自单元 $i-1$ 的原变量
- `self.wr[i]`：界面 $i-1/2$ 的**右侧**状态 — 取自单元 $i$ 的原变量

```
界面 i-1/2:
  wl[i] ← 单元 i-1 的值
  wr[i] ← 单元 i 的值
```

**关于两个 `reconstruct` 方法**：代码中有两个同名方法。Python 中后定义的会覆盖前者。第一个（不完整的）试图实现带 minmod 限制器的 MUSCL 线性重构（二阶精度），第二个是降级的常数重构（一阶精度）。老师最终选择了一阶方案，因为二阶重构代码未完成。

---

### `calc_flux_single` 方法 — 单界面 HLL 通量

```python
def calc_flux_single(self, w_l, w_r):
```

这是代码的**核心** — 计算一个界面的数值通量，使用 **HLL（Harten-Lax-van Leer）近似 Riemann 求解器**。

### HLL 求解器原理

Riemann 问题的精确解结构复杂（激波、接触间断、稀疏波），计算成本高。HLL 是一种**两波近似**：
- 只考虑最左波速 $S_L$ 和最右波速 $S_R$（忽略中间的接触间断）
- 将解空间分为三个区域

$$
F_{\text{HLL}} = \begin{cases}
F_L & \text{if } S_L > 0 \quad \text{（所有波向右，流动是超音速右行）} \\[4pt]
F_R & \text{if } S_R < 0 \quad \text{（所有波向左，流动是超音速左行）} \\[4pt]
\dfrac{S_R F_L - S_L F_R + S_L S_R (U_R - U_L)}{S_R - S_L} & \text{otherwise（亚音速情况，中间状态）}
\end{cases}
$$

其中 $F_L, F_R$ 是物理通量，$U_L, U_R$ 是守恒变量。

### 代码逐步分析

**步骤 1：提取原变量**
```python
rho_l = w_l[idx_rho]; pre_l = w_l[idx_ene]; vel_l = w_l[idx_mom]
rho_r = w_r[idx_rho]; pre_r = w_r[idx_ene]; vel_r = w_r[idx_mom]
```

**步骤 2：计算焓（hatalpy）**
```python
h_l = pre_l // (self.gamma - 1) * self.gamma
h_r = pre_r // (self.gamma - 1) * self.gamma
```

> ⚠ **注意**：`//` 是 Python 的**整数除法**。对于物理计算这**几乎肯定是 bug**，应该用 `/`。焓的定义是：
> $$H = \frac{\gamma}{\gamma-1} \frac{P}{\rho}$$

但代码计算的是 $\frac{\gamma}{\gamma-1} \cdot P$，缺少除以 $\rho$。不过在当前代码中 `h_l` 和 `h_r` 计算后并未被使用，所以不影响结果。

**步骤 3：计算物理通量**
```python
f_rho_l = rho_l * vel_l                           # 质量通量
f_mom_l = rho_l * vel_l**2 + pre_l                # 动量通量
f_ene_l = (rho_l * vel_l**2 / 2 + h_l) * vel_l    # 能量通量
```

物理通量来自 Euler 方程的通量函数：
$$\mathbf{F} = \begin{pmatrix} \rho v \\ \rho v^2 + P \\ (E + P) v \end{pmatrix}$$

但代码中能量通量用的是 $(\frac{1}{2}\rho v^2 + H)v$，其中 $H$ 的计算有上述整数除法问题。标准的能量通量应该是 $(E+P)v$。

**步骤 4：构造通量向量**
```python
f_l = zeros(self.n_int)
f_r = zeros(self.n_int)
f_l[idx_rho] = f_rho_l; f_l[idx_mom] = f_mom_l; f_l[idx_ene] = f_ene_l
f_r[idx_rho] = f_rho_r; f_r[idx_mom] = f_mom_r; f_r[idx_ene] = f_ene_r
```

**步骤 5：将原变量转换回守恒变量**
```python
u_l = copy(w_l)
u_l[idx_mom] *= w_l[idx_rho]                                   # v → ρv
u_l[idx_ene] = w_l[idx_ene]/(self.gamma-1) + w_l[idx_rho]*w_l[idx_mom]**2/2  # P → E
u_r = copy(w_r)   # 对右侧同样操作
u_r[idx_mom] *= u_r[idx_rho]
u_r[idx_ene] = w_r[idx_ene]/(self.gamma-1) + w_r[idx_rho]*w_r[idx_mom]**2/2
```

这里有一个细微的 bug：`u_r[idx_mom] *= u_r[idx_rho]` 中，因为 `u_r` 是 `w_r` 的副本，`idx_mom` 位置存储的是速度 $v$。乘以 `u_r[idx_rho]`（即 $\rho_r$）得到动量 $\rho v$。之后 `u_r[idx_rho]` 已经是 $\rho$（拷贝自 `w_r`），所以正确。

但代码第二行引用 `u_r[idx_rho]` 和 `u_r[idx_mom]` 时，`u_r[idx_mom]` 已经在这行之前被修改为 $\rho v$。如果第一行写在第二行**前面**，那第二行使用的就是正确的 $\rho v$。当前代码写法的顺序是正确的（虽然看起来有点混乱）。

**步骤 6：估计波速**
```python
s_l = vel_l - sqrt(self.gamma * pre_l / rho_l)   # v_L - c_L
s_r = vel_r - sqrt(self.gamma * pre_r / rho_r)   # v_R - c_R
```

> ⚠ **这是代码中另一个重要问题**：HLL 标准做法是用 **Roe 平均**或至少取更保守的波速估计：
> $$S_L = \min(v_L - c_L, \tilde{v} - \tilde{c})$$
> $$S_R = \max(v_R + c_R, \tilde{v} + \tilde{c})$$
>
> 当前代码只用了局域声速作为波速下界，且 `s_r = v_R - c_R` 应该是 `v_R + c_R`（右波向右传）。这是一个实现错误。

**步骤 7：HLL 通量选择**
```python
if s_l > 0:
    f = f_l          # 超音速右行 → 完全取左侧通量
elif s_r < 0:
    f = f_r          # 超音速左行 → 完全取右侧通量
else:
    f = s_r * f_l - s_l * f_r + s_l * s_r * (u_r - u_l)
    f /= s_r - s_l   # 亚音速 → 中间状态的 HLL 通量
```

**当前 bug**：函数最后是 `return` 而不是 `return f` — **没有返回值**。所以调用者收到的将是 `None`。

---

### `calc_flux` 方法 — 遍历所有界面

```python
def calc_flux(self):
    for i in range(len(self.f)):
        w_l = self.wl[i]
        w_r = self.wr[i]
        self.f[i] = self.calc_flux_single(w_l, w_r)
```

### 行为解释

对每个界面 $i-1/2$（$i = 0, 1, \dots, n_\text{cell}$），调用 `calc_flux_single` 计算通量并存入 `self.f[i]`。

`self.f[i]` 是界面 $i-\frac{1}{2}$ 的通量，即单元 $i-1$ 和 $i$ 之间的界面。

---

### `intg_flux` 方法 — 时间积分

```python
def intg_flux(self):
    self.u += self.dt / self.dx * (self.f[:-1] - self.f[1:])
```

### 行为解释

这是有限体积法的**最终更新步骤** — 对守恒变量做 Euler 显式时间推进（一阶前向 Euler）：

$$U_i^{n+1} = U_i^n + \frac{\Delta t}{\Delta x} \left( F_{i-1/2} - F_{i+1/2} \right)$$

- `self.f[:-1]`：$F_{i-1/2}$（左界面通量，索引 $0, 1, \dots, n-1$）
- `self.f[1:]`：$F_{i+1/2}$（右界面通量，索引 $1, 2, \dots, n$）
- `self.dt / self.dx`：$\Delta t / \Delta x$

守恒律的物理含义：单元内守恒量的变化 = 净通量流入（左界面流入 $-$ 右界面流出）。

---

## 5. 主程序

```python
dx = 0.1
n_cell = 256
hyd = hydro(n_cell, dx)
hyd.init(1, 2, 0, 0.1, 0.2, 0)

for step in range(1):
    hyd.cons2prim()       # 守恒变量 → 原变量 + 计算 dt
    hyd.set_boundary()    # 填充 ghost cell
    hyd.reconstruct()     # 界面左右状态重构
    hyd.calc_flux()       # 计算数值通量 (HLL)
    hyd.intg_flux()       # 时间推进

plot(hyd.u[:, idx_mom])
print(hyd.f.shape)
```

### 执行流程

$$
\boxed{U^n} \xrightarrow{\text{cons2prim}} \boxed{W^n} \xrightarrow{\text{set\\_boundary}} \boxed{\text{ghost cells 填充}}
\xrightarrow{\text{reconstruct}} \boxed{W_L, W_R \text{ at interfaces}}
\xrightarrow{\text{calc\\_flux}} \boxed{F_{i-1/2} (\text{HLL})}
\xrightarrow{\text{intg\\_flux}} \boxed{U^{n+1}}
$$

每个时间步的完整循环就是：原变量转换 → 边界条件 → 界面重构 → 通量计算 → 时间推进。

目前 `range(1)` 只运行了 1 步。要看到 Riemann 问题的传播，需要多步（如 100-200 步）。

---

## 6. 已知 bug 汇总

| 位置 | 问题 | 影响 |
|------|------|------|
| `minmod` | 函数体不完整，`res` 未定义 | 如有调用则报错 |
| `calc_flux_single` L100-101 | `//` 整数除法计算焓 | 数值精度错误（未被使用，当前无影响） |
| `calc_flux_single` L128-129 | 波速 `s_r = v_R - c_R` 应为 `v_R + c_R` | Riemann 求解器可能选错通量分支 |
| `calc_flux_single` 结尾 | `return` 无返回值，应为 `return f` | `self.f` 全为 `None`，结果无意义 |
| `reconstruct` | 第一个方法体不完整，被第二个覆盖 | 无法使用二阶 MUSCL 重构 |

---

## 7. 与讲义（chap03.pdf）的对应关系

- 讲义中介绍了**双曲型守恒律和 Euler 方程组** → 代码求解的就是一维 Euler 方程
- 讲义讨论了 **Riemann 问题和 Godunov 方法** → 代码使用 HLL 近似 Riemann 求解器（Godunov 型格式的一种）
- 讲义涉及**有限体积法离散和通量计算** → 代码的 `calc_flux_single` + `intg_flux` 就是 FV 法的核心
- 讲义中**重构和限制器**的内容 → 代码的 `reconstruct` 和 `minmod` 对应这部分
- 讲义中**时间步长的 CFL 条件** → 代码的 `cons2prim` 中计算 `self.dt` 即基于 CFL 条件

总体而言，这份代码是讲义的编程实践 — 将 Euler 方程的数学理论翻译为可执行的数值求解器。
