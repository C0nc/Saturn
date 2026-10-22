"""grad(rho) by autograd through the Saturn isodepth network -- GASTON's method.

GASTON computes the isodepth gradient analytically (cluster_plotting.py:102):

    x = torch.tensor(S, requires_grad=True)
    G = torch.autograd.grad(mod.spatial_embedding(x).flatten(), x,
                            grad_outputs=torch.ones_like(x[:,0]))[0]

Saturn's `iso[m]` is the direct analogue, so the same call gives grad(rho_m) exactly,
rather than the local-plane approximation. On lpc10_fov the two agree at cos 0.94 for
most spots but diverge completely for the worst 10% (cos -> 0), because the iso net is a
ReLU MLP whose autograd gradient is piecewise-constant and jumps at activation
boundaries, while a plane fit smooths those jumps away.

The model is fitted on z-scored coordinates, so the chain rule back to microns is
    d rho / d x_raw = (d rho / d x_norm) / std(x_raw)

MAGNITUDE CAVEAT (GASTON's own): |grad rho| is in arbitrary network units and is not
comparable between domains until the isodepth is rescaled per domain to physical
distance -- gaston.isodepth_scaling.adjust_isodepth. Directions are unaffected.
"""
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/scripts")
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from domain_annot import R
import saturn
from saturn.gradient import frames  # noqa: F401  (re-exported for saturn_cis_gradient_decomp)


def load_fit(ds):
    z = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    mods = [str(m) for m in z["modalities"]]
    m = saturn.load_model(f"{R}/{ds}/saturn_best.pt", [z[f"Y_{m}"].shape[1] for m in mods])
    return m, mods, z


def rho_gradient(ds, xy=None, per_axis_scaling=True):
    """returns {modality: (n_spots, 2) d rho / d(x,y) in raw coordinate units}, saturn.gradient.rho_gradient.
    per_axis_scaling=True is the manuscript run (network evaluated at per-axis z-scored coordinates);
    False is the correct chain rule through zscore_coords."""
    mod, mods, z = load_fit(ds)
    ax = pd.read_csv(f"{R}/{ds}/axes.csv", index_col=0)
    if xy is None:
        xy = np.c_[ax.x.values, ax.y.values].astype(np.float64)
    G = saturn.gradient.rho_gradient(mod, xy, per_axis_scaling=per_axis_scaling)
    return {m: G[:, i].astype(np.float32) for i, m in enumerate(mods)}, xy


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "lpc10_fov"
    g, xy = rho_gradient(ds)
    for m, G in g.items():
        u, v, n = frames(G)
        print(f"{ds} d_{m}: |grad rho| median {np.median(n):.5g}  "
              f"(1st-99th pct {np.percentile(n,1):.3g} - {np.percentile(n,99):.3g})")
