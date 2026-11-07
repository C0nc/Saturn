"""The whole Fig 5 table chain from the raw counts with a NEW glmpca draw (no published GLM-PCs), in its own workspace
<SATURN_ROOT>/fresh_glmpc. Same scripts and settings as the notebook. Shows how far the published numbers depend on the
unseeded GLM-PCA initialisation. Writes fresh_replicate.json next to the results.
"""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; NB = HERE.parent
main = Path(os.environ["SATURN_ROOT"]).resolve(); root = main / "fresh_glmpc"
subprocess.run([sys.executable, str(NB / "setup_workspace.py"), "--root", str(root), "--figs", "fig5", "--link-from", str(main)], check=True)
os.environ["SATURN_ROOT"] = str(root)
sys.path[:0] = [str(NB), str(HERE)]
import numpy as np, pandas as pd
import nbutil as nb

SECS, FIT, G = ["A1", "B1", "D1"], "sproxseq_B1_p46_dr", "results/real_saturn/sproxseq_gc_trajectory"
T5 = "results/real_saturn/sproxseq_B1_fig5"
for s in SECS:
    nb.step(f"prep_{s}", f"python scripts/sproxseq_prep.py {s} --probe", [f"results/real_saturn/targets_sproxseq_{s}_p46.npz"])
nb.step("depthreg_B1", "python scripts/sproxseq_prep_depthreg.py B1", [f"results/real_saturn/targets_{FIT}.npz"])
nb.step("fit_B1_dr", f"python scripts/run_real2.py --dataset {FIT} --seeds 1 --seed0 42", [f"results/real_saturn/{FIT}/axes.csv"], threads=4)
for s in SECS:
    nb.step(f"fit_{s}_p46", f"python scripts/run_real2.py --dataset sproxseq_{s}_p46 --seeds 1 --seed0 42", [f"results/real_saturn/sproxseq_{s}_p46/axes.csv"], threads=4)
nb.step("gc_trajectory", "python scripts/sproxseq_gc_trajectory.py", [f"{G}/residuals_B1.npy"])
nb.step("bt_axis", "python scripts/sproxseq_B1_bt_axis.py", ["results/real_saturn/sproxseq_B1_bt_axis/complexes.csv"], omp=16)
nb.step("rna_axis", "python scripts/sproxseq_B1_rna_axis.py", ["results/real_saturn/sproxseq_B1_rna_axis/spots.csv"], omp=16)
nb.step("fig5_tables", "python scripts/sproxseq_B1_fig5_prep.py", [f"{T5}/summary.json"])

P = nb.PUB / "Hole/gaston_mix_mm/results/real_saturn"
ax = nb.compare_axes(FIT)
S, Sp = json.load(open(nb.RES / "sproxseq_B1_fig5/summary.json")), json.load(open(P / "sproxseq_B1_fig5/summary.json"))
F, Fp = json.load(open(nb.RES / FIT / "summary.json")), json.load(open(P / FIT / "summary.json"))
out = dict(axes=ax.to_dict("records"), axis_corr=F["axis_axis_corr"][0][1], axis_corr_published=Fp["axis_axis_corr"][0][1])
for m in ("RNA", "Protein"):
    a = pd.read_csv(nb.RES / f"sproxseq_B1_fig5/complexes_{m}.csv", index_col=0); b = pd.read_csv(P / f"sproxseq_B1_fig5/complexes_{m}.csv", index_col=0).loc[a.index]
    ha, hb = a.p < 0.05, b.p < 0.05
    out[m] = dict(R2=F["R2_whole"][m]["full"], R2_published=Fp["R2_whole"][m]["full"], K=S[m]["K"], K_published=Sp[m]["K"],
                  frac_p05=S[m]["frac_p05"], frac_p05_published=Sp[m]["frac_p05"], hits_shared=int((ha & hb).sum()), hits=int(ha.sum()), hits_published=int(hb.sum()),
                  slope_r=float(np.corrcoef(a.slope, b.slope)[0, 1]))
json.dump(out, open(nb.RES / "fresh_replicate.json", "w"), indent=1)
print(json.dumps(out, indent=1))
