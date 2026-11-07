#!/usr/bin/env python
"""Compare the workspace Saturn arm of the hard grid with the published one (needs SATURN_PUBLISHED).

Writes results/bench_v5_full/compare_{axes,table}.csv in the workspace. Reads the published axes and tables only.
Axes: largest absolute difference and |Pearson r| of the Saturn isodepths and of the simulated truth, per run and axis.
Table: largest absolute difference of every numeric column of the Saturn rows (metrics and simulation-only columns).
Also checks that saturn.sim (used by panel b) and simulate_v5 (used by the benchmark) generate identical data.
"""
import glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ["SATURN_ROOT"]).expanduser().resolve()
RES = ROOT / "Hole/gaston_mix_mm/results/bench_v5_full"
PUB = Path(os.environ["SATURN_PUBLISHED"]) / "Hole/gaston_mix_mm/results/bench_v5_full"
KEY = ["angle", "cross_RA", "coupled_frac", "seed"]


def hard(res):
    T = pd.concat([pd.read_csv(f) for f in glob.glob(str(res / "bench_full_s*.csv"))])
    return T[(T.grid == "hard") & (T.method == "Saturn")].sort_values(KEY).reset_index(drop=True)


rows = []
for f in sorted(RES.glob("axes/hard_*.npz")):
    a, b = np.load(f), np.load(PUB / "axes" / f.name)
    for k, ax in enumerate(("RNA", "ATAC")):
        rows.append(dict(run=f.stem, axis=ax, truth_max_abs_diff=float(np.abs(a["true"][:, k] - b["true"][:, k]).max()),
                         saturn_max_abs_diff=float(np.abs(a["Saturn"][:, k] - b["Saturn"][:, k]).max()),
                         saturn_abs_r=float(abs(np.corrcoef(a["Saturn"][:, k], b["Saturn"][:, k])[0, 1]))))
A = pd.DataFrame(rows); A.to_csv(RES / "compare_axes.csv", index=False)
print(f"axes: {A.run.nunique()} runs x {A.axis.nunique()} axes")
print(A.groupby("axis")[["truth_max_abs_diff", "saturn_max_abs_diff", "saturn_abs_r"]].agg(["min", "median", "max"]).T.to_string())
print("worst 5 runs/axes by |diff|:"); print(A.sort_values("saturn_max_abs_diff").tail(5).to_string(index=False))

a, b = hard(RES), hard(PUB)
assert len(a) == len(b) == 24 and (a[KEY].values == b[KEY].values).all(), (len(a), len(b))
num = [c for c in a.select_dtypes("number") if c in b and c not in KEY and b[c].notna().any()]
D = pd.DataFrame([dict(column=c, workspace_mean=a[c].mean(), published_mean=b[c].mean(), max_abs_diff=float(np.nanmax(np.abs(a[c].values - b[c].values)))) for c in num])
D.to_csv(RES / "compare_table.csv", index=False)
print("table, Saturn rows, 24 hard runs:"); print(D.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

sys.path[:0] = [str(ROOT / "Hole/gaston_mix_mm/sim"), str(ROOT / "saturn_repo/src")]
from saturn.sim import SimConfig, simulate
from simulate_v5 import SimConfigV5, simulate_v5
cfg = dict(axis_angle_deg=45, cross_RA=0.8, cross_AR=0.2, coupled_frac=0.6, rna_libsize=1000.0, seed=0)
d1, d2 = simulate(SimConfig(**cfg)), simulate_v5(SimConfigV5(**cfg))
print("saturn.sim vs simulate_v5, max|diff|: d", np.abs(d1.d - d2.d).max(), " Y_R", np.abs(d1.Y_R - d2.Y_R).max(), " Y_A", np.abs(d1.Y_A - d2.Y_A).max())
