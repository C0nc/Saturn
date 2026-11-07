"""The Fig. 2 chain from the rebuilt AnnData with a NEW glmpca draw (no published GLM-PCs), in its own workspace
<SATURN_ROOT>/fresh_glmpc. Same scripts and settings as the notebook. Shows how far the published numbers depend on the unseeded
GLM-PCA initialisation. Writes fig2_claims.json and fresh_replicate.json next to the results.
"""
import json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent; NB = HERE.parent
main = Path(os.environ["SATURN_ROOT"]).resolve(); root = main / "fresh_glmpc"
subprocess.run([sys.executable, str(NB / "setup_workspace.py"), "--root", str(root), "--figs", "fig2", "--link-from", str(main)], check=True)
os.environ["SATURN_ROOT"] = str(root)
sys.path[:0] = [str(NB), str(HERE)]
import nbutil as nb

TP = "claude_code_tutorial_lpc/results/timepoints"; (root / TP).mkdir(parents=True, exist_ok=True)
for m in ("rna", "atac"):                       # built by section 1 of the notebook
    q = root / TP / f"arcseq_LPC10S1_{m}_adata.h5ad"
    if not q.exists(): q.symlink_to(main / TP / q.name)
for rel in ("external_data/codex_lpc/overview_LPC10_S1_lvl3.npz", "catalign_lpc10/results/LPC10S1_wm/LPC10S1_spots_in_codex.csv"):   # sections 2 and 7: independent of the fit
    q = root / rel; q.parent.mkdir(parents=True, exist_ok=True)
    if not q.exists(): q.symlink_to(main / rel)
(nb.HOLE / "results/atac_tiles").mkdir(parents=True, exist_ok=True)
F, DS, R = nb.PF / "fig2/steps", "lpc10_fov", "results/real_saturn"
ENV = {"PYTHONPATH": str(nb.HOLE / "scripts")}
nb.step("s02", f"python {F}/s02_lpc10_hotspots.py", [f"{R}/lpc10_hotspots.npz"], extra_env=ENV)
nb.step("s03", f"python {F}/s03_lpc10_fov_select.py", [f"{R}/lpc10_fov.npz"], extra_env=ENV)
nb.step("prep", "python scripts/lpc10_fov_prep.py", [f"{R}/targets_{DS}.npz"])
nb.step("fit", f"python scripts/run_real2.py --dataset {DS} --seeds 1 --seed0 42", [f"{R}/{DS}/axes.csv"], threads=8, omp=8, device="cpu")
nb.step("tf_rho", f"python scripts/saturn_tf_activity_rho.py {DS}", [f"{R}/{DS}_tf_activity/tf_fits.csv"])
nb.step("cis_pairs", f"python scripts/saturn_cis_function_pairs.py {DS} 1500", [f"{R}/{DS}_cis_pairs/cis_function_pairs.csv"])
nb.step("cis_gradient", f"python scripts/saturn_cis_gradient_decomp.py {DS} 1500", [f"{R}/{DS}_cis_pairs/cis_gradient_decomp.csv"])
nb.step("posthoc", "python scripts/lpc10_fov_posthoc.py", [f"{R}/{DS}_posthoc/infl_gene_axis_fits.csv"])
nb.step("claims", f"python {HERE}/fig2_claims.py", [f"{R}/fig2_claims.json"])
ax = nb.compare_axes(DS)
out = dict(axes=ax.to_dict("records") if ax is not None else None, fresh=json.load(open(nb.RES / "fig2_claims.json")),
           pinned=json.load(open(main / "Hole/gaston_mix_mm" / R / "fig2_claims.json")))
json.dump(out, open(nb.RES / "fresh_replicate.json", "w"), indent=1)
print(json.dumps(out, indent=1))
