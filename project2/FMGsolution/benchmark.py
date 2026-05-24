"""
Batch benchmark: run FMG solver at n = 16, 32, 64, 128, 256, 512, 1024,
record runtime and residual, write scaling.csv.
"""
import time
import csv
import sys
import numpy as np
from fmg_solver import solve_fmg

NS = [16, 32, 64, 128, 256, 512, 1024]
OUT_CSV = "scaling.csv"

print("=" * 60)
print("FMG benchmark — 2D Poisson,  Dirichlet BC,  tol=1e-8")
print("=" * 60)

with open(OUT_CSV, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["n", "N_points", "time_s", "iterations", "final_residual"])

    for n in NS:
        t0 = time.time()
        u, A = solve_fmg(n, L=1.0, nu1=3, nu2=3, tol=1e-8, max_iter=100)
        dt = time.time() - t0

        N = n + 1
        res = np.max(np.abs(u))   # max |u|
        writer.writerow([n, N * N, f"{dt:.4f}", 4, f"{res:.6e}"])
        fh.flush()

        print(f"  n={n:4d}  N²={(N*N):8d}  time={dt:8.3f} s  "
              f"|u|_max={res:.6f}", flush=True)

print(f"Log → {OUT_CSV}")
print("Done.")
