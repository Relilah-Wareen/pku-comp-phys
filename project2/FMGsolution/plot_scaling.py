"""
Plot FMG runtime scaling.
Reads scaling.csv, performs linear regression on log(time) vs log(N_points),
verifies O(N²) complexity.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- read data ----
df = pd.read_csv("scaling.csv")
N_pts = df["N_points"].values.astype(float)
t_s   = df["time_s"].values.astype(float)

log_N = np.log(N_pts)
log_t = np.log(t_s)

# ---- linear fit:  log(t) = b * log(N) + log(a)  →  t = a * N^b ----
from numpy.polynomial.polynomial import polyfit
coeffs = polyfit(log_N, log_t, 1)          # intercept + slope
log_a, b = coeffs[0], coeffs[1]
a = np.exp(log_a)

# fitted curve
N_fine = np.logspace(np.log10(N_pts[0]), np.log10(N_pts[-1]), 100)
t_fit = a * N_fine ** b

# ---- figure ----
fig, ax = plt.subplots(1, 1, figsize=(8, 5))

ax.loglog(N_pts, t_s, "o-", markersize=8, label="FMG measured")
ax.loglog(N_fine, t_fit, "--", linewidth=1.5,
          label=f"Fit:  t = {a:.2e} · N^{{{b:.4f}}}")

# ideal O(N²) reference line (anchored to first data point)
t_ref = t_s[0] * (N_fine / N_pts[0])
ax.loglog(N_fine, t_ref, ":", linewidth=1.2, color="gray",
          label=r"O($N^2$) reference")

ax.set_xlabel("N (total grid points)")
ax.set_ylabel("Wall time  [s]")
ax.set_title(f"FMG solver scaling  (b = {b:.4f} ≈ 1 ⇒ O(N²) with N=(n+1)²)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("scaling.png", dpi=150)
plt.close()

# ---- print results ----
print(f"Fit:  log(t) = {log_a:.4f} + {b:.4f} · log(N)")
print(f"      t = {a:.4e} · N^{b:.4f}")
print(f"Expected:  b ≈ 1.0  (O(N²) complexity)")
print(f"Measured:  b = {b:.4f}")
print()
print("Data:")
print(df.to_string(index=False))
print()
print("Saved → scaling.png")
