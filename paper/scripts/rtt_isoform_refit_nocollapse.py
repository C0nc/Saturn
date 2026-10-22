"""Joint Saturn refit of RNA + isoform with the collapse mechanisms switched off.

rtt_isoform_plane_diagnosis.py showed rho_Isoform flattens because (b) the RNA pathway absorbs the isoform
target (coupling isoform row ~0.6 from the RNA axis; own pathway R2 < 0) and (c) isoform-head restarts are
accepted on 1-2% held-out gains and leave a fresh, under-trained, near-planar head. Configs:
  norestart_lc0.3    n_restarts = 0, lam_cross = 0.3   (no restarts + cross pathways must pay 10x more)
  norestart_lc0.03   n_restarts = 0, lam_cross = 0.03  (no restarts only; default cross cost)
Everything else = run_real2 defaults (seed 42, 10,000 epochs, warm-up 0.3, residual anchor).
With n_restarts = 0 there is no held-out split, so the model is trained on every spot directly.
Writes results/real_saturn/<ds>_<config>/axes.csv + diag.json (stage snapshots as in the diagnosis).
Usage: python scripts/rtt_isoform_refit_nocollapse.py <target dataset> <config> [--weights]
  --weights  write to <ds>_<config>_weights/ instead and also save saturn_best.pt (published axes untouched)
"""
import json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from saturn import train_saturn
from saturn.preprocess import zscore_coords
from rtt_isoform_plane_diagnosis import state, r2

R = "/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn"
CONFIGS = {"norestart_lc0.3": dict(n_restarts=0, lam_cross=0.3), "norestart_lc0.03": dict(n_restarts=0, lam_cross=0.03)}
torch.set_num_threads(int(os.environ.get("SAT_THREADS", "8")))


def main(ds, cfg):
    out = f"{R}/{ds}_{cfg}" + ("_weights" if "--weights" in sys.argv else ""); os.makedirs(out, exist_ok=True)
    z = dict(np.load(f"{R}/targets_{ds}.npz", allow_pickle=True))
    mods = [str(m) for m in z["modalities"]]; m_iso = mods.index("Isoform")
    raw = np.asarray(z["coords"], float); xy = zscore_coords(raw); xy_t = torch.as_tensor(xy, dtype=torch.float32)
    Y = [np.asarray(z[f"Y_{m}"], np.float32) for m in mods]
    snaps = {}
    def cb(ep, model, n_warm):
        if ep == n_warm - 1: snaps["after_warmup"] = state(model, xy_t, Y, xy, m_iso)
    model, hist = train_saturn(Y, xy, seed=42, device="cpu", callback=cb, **CONFIGS[cfg])
    snaps["final"] = state(model, xy_t, Y, xy, m_iso)
    with torch.no_grad():
        d, full, own, cross = model(xy_t)
    D = d.numpy()
    if "--weights" in sys.argv: torch.save(model.state_dict(), f"{out}/saturn_best.pt")
    pd.DataFrame(np.c_[raw, D], columns=["x", "y"] + [f"d_{m}" for m in mods], index=z["barcodes"]).to_csv(f"{out}/axes.csv")
    R2 = {m: dict(full=r2(Y[k], full[k].numpy()), own_axis_only=r2(Y[k], (full[k] - cross[k] + cross[k].mean(0)).numpy()),
                  other_axes_only=r2(Y[k], (full[k] - own[k] + own[k].mean(0)).numpy())) for k, m in enumerate(mods)}
    rep = dict(dataset=ds, config=cfg, settings=CONFIGS[cfg], stages=snaps, R2_whole=R2,
               coupling=np.round(model.coupling(xy_t), 3).tolist(), fit_all=hist["fit_all"])
    json.dump(rep, open(f"{out}/diag.json", "w"), indent=1); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
