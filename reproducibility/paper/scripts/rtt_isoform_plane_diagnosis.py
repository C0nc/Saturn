"""Why does rho_Isoform come out as a plane? Three hypotheses, one test each.

(a) DATA: no curved axis explains the isoform target better than a plane.
    Fixed-plane baseline: for 36 directions theta, coordinate t = x cos(theta) + y sin(theta); fit the same
    pathway MLP (1 -> 20 -> 20 -> K) t -> Y_Isoform; best R2 over theta = what a plane can do. Compared with
    the own-axis R2 of the learned axis.
(b) COUPLING: the isoform head leans on rho_RNA and its own axis is never trained.
    Solo fit: train_saturn on the isoform target ALONE (M = 1, i.e. GASTON). Non-planar solo axis -> coupling.
(c) TRAINING: the isodepth net stays linear, or a stage/restart pushes it back to a plane.
    Joint fit (exact run_real2 settings) with snapshots after warm-up (stage 1, isoform = solo GASTON),
    after the coupled stage, and the final model (restarts + refit); plane R2 + isoform own/full R2 at each,
    restart log; ReLU activity: share of hidden units of the isoform isodepth net that switch state across
    spots (a unit active everywhere or nowhere contributes only a linear term).

Usage: python scripts/rtt_isoform_plane_diagnosis.py <target dataset>
"""
import json, os, sys, copy
import numpy as np, torch
sys.path.insert(0, "<workspace>/saturn_repo/src")
from saturn import train_saturn
from saturn.preprocess import zscore_coords
from saturn.core import _mlp
import torch.nn as nn

R = "<workspace>/Hole/gaston_mix_mm/results/real_saturn"
OUT = f"{R}/rtt_isoform_variants/plane_diagnosis"
torch.set_num_threads(int(os.environ.get("SAT_THREADS", "8")))


def r2(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def plane_r2(xy, d):
    X = np.c_[np.ones(len(xy)), xy]; b = np.linalg.lstsq(X, d, rcond=None)[0]
    return float(1 - ((d - X @ b) ** 2).sum() / ((d - d.mean()) ** 2).sum())


@torch.no_grad()
def state(model, xy_t, Y, xy, m_iso):
    cross = model.cross_on; model.cross_on = True
    d, full, own, _ = model(xy_t)
    model.cross_on = cross
    di = d[:, m_iso].numpy()
    # ReLU activity of the isoform isodepth net
    h = xy_t; active = []
    for layer in model.iso[m_iso]:
        h = layer(h)
        if isinstance(layer, nn.ReLU):
            on = (h > 0).float().mean(0).numpy(); active.append(((on > 0.02) & (on < 0.98)).mean())
    return dict(plane_R2=round(plane_r2(xy, di), 3), own_R2=round(r2(Y[m_iso], own[m_iso].numpy()), 3),
                full_R2=round(r2(Y[m_iso], full[m_iso].numpy()), 3),
                relu_switching_share=[round(float(a), 2) for a in active])


def fixed_plane_baseline(xy, Y, n_dir=36, epochs=3000):
    best = (-1, None); Yt = torch.as_tensor(Y, dtype=torch.float32)
    for th in np.linspace(0, np.pi, n_dir, endpoint=False):
        t = xy[:, 0] * np.cos(th) + xy[:, 1] * np.sin(th)
        tt = torch.as_tensor((t - t.mean()) / t.std(), dtype=torch.float32)[:, None]
        torch.manual_seed(0); net = _mlp([1, 20, 20, Y.shape[1]], act=nn.ReLU); opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(epochs):
            opt.zero_grad(); l = ((net(tt) - Yt) ** 2).mean(); l.backward(); opt.step()
        with torch.no_grad(): s = r2(Y, net(tt).numpy())
        if s > best[0]: best = (s, float(np.degrees(th)))
    return dict(best_plane_R2=round(best[0], 3), best_direction_deg=best[1])


def main(ds):
    os.makedirs(OUT, exist_ok=True)
    z = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    mods = [str(m) for m in z["modalities"]]; m_iso = mods.index("Isoform")
    xy = zscore_coords(np.asarray(z["coords"], float)); xy_t = torch.as_tensor(xy, dtype=torch.float32)
    Y = [np.asarray(z[f"Y_{m}"], np.float32) for m in mods]
    rep = dict(dataset=ds)
    # (c) joint fit with stage snapshots
    snaps = {}
    def cb(ep, model, n_warm):
        if ep == n_warm - 1: snaps["after_warmup (isoform solo GASTON stage)"] = state(model, xy_t, Y, xy, m_iso)
        if ep == 9999: snaps["after_coupled_stage"] = state(model, xy_t, Y, xy, m_iso)
    model, hist = train_saturn(Y, xy, seed=42, device="cpu", refit_all_epochs=3000, callback=cb)
    snaps["final (restarts + refit)"] = state(model, xy_t, Y, xy, m_iso)
    rep["joint"] = snaps
    rep["restarts"] = [dict(round=r, head=mods[h], before=round(b, 4), after=round(a, 4), accepted=bool(ok)) for r, h, b, a, ok in hist["restarts"]]
    rep["coupling"] = np.round(model.coupling(xy_t), 3).tolist()
    print(json.dumps(rep, indent=1), flush=True)
    # (b) isoform alone
    solo, _ = train_saturn([Y[m_iso]], xy, seed=42, device="cpu", refit_all_epochs=3000)
    rep["solo_isoform"] = state(solo, xy_t, [Y[m_iso]], xy, 0)
    print("solo:", rep["solo_isoform"], flush=True)
    # (a) fixed plane
    rep["fixed_plane"] = fixed_plane_baseline(xy, Y[m_iso])
    print("fixed plane:", rep["fixed_plane"], flush=True)
    json.dump(rep, open(f"{OUT}/{ds}.json", "w"), indent=1)
    print("->", f"{OUT}/{ds}.json")


if __name__ == "__main__":
    main(sys.argv[1])
