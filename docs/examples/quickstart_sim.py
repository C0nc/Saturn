"""Quickstart: simulate RNA + ATAC with a known cross-modal coupling, fit Saturn, read the result.

    python docs/examples/quickstart_sim.py            (~1-2 min on CPU)
"""
import numpy as np
from scipy.stats import spearmanr
import saturn
from saturn.sim import SimConfig, simulate

data = simulate(SimConfig(n_side=30, cross_RA=0.5, cross_AR=0.2, seed=0))   # 900 spots
xy = saturn.zscore_coords(data.coords)                                     # one common scale

res = saturn.fit([data.Y_R, data.Y_A], xy, modalities=["RNA", "ATAC"], seeds=[0, 1],
                 epochs=4000, n_restarts=2, restart_epochs=1500, refit_all_epochs=1000, threads=4)

for k, m in enumerate(res.modalities):
    print(f"{m:4s} axis vs truth |Spearman| = {abs(spearmanr(res.axes[:, k], data.d[:, k]).statistic):.3f}"
          f"   R2 full {res.r2[m]['full']:.3f}  own-only {res.r2[m]['own_axis_only']:.3f}")
print("coupling (row = modality, column = driving axis):\n", np.round(res.coupling, 3))
print("true cross share  RNA<-ATAC %.2f  ATAC<-RNA %.2f" % (data.cross_share["R_from_A"], data.cross_share["A_from_R"]))

# GASTON-style domains along the RNA axis, then a per-domain Gaussian fit of the targets
try:
    lab = saturn.domains.gaston_domains(res.axes[:, 0], data.Y_R, n_domains=4)
    print("RNA domains:", np.bincount(lab).tolist())
except ImportError:
    print("install gaston-spatial for domains: pip install 'saturn-isodepth[gaston]'")
res.save("outputs/quickstart", coords=data.coords)
