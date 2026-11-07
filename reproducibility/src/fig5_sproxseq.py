# %% [markdown]
# # Figure 5: Sprox-seq human tonsil, section B1
#
# From the GEO count matrices (GSE304749) to every panel of Fig. 5: RNA and 46-probe protein targets, Saturn fit (seed 42),
# marker compartments, axis readouts, domains, and the 1,692 held-out PLA products along both isodepths.
# All panels of Fig. 5 involve Saturn, so none is dropped. The baseline benchmark (`sproxseq_baselines`, the main of
# `sproxseq_ccc`) is not part of the figure and is not run here; the figure only takes the rotation helper and the
# contact-pair list from `sproxseq_ccc`.
#
# **Conda env:** `spatialmultivelo`. **Device:** CPU.

# %%
import sys; sys.path[:0] = [".", "steps"]
from pathlib import Path
import json
import nbutil as nb
from fig5_compare import compare_npz, compare_json, compare_png, show_small
nb.init("fig5")
STEPS = Path("steps").resolve()
SECS = ["A1", "B1", "D1"]; FIT = "sproxseq_B1_p46_dr"; T5 = "results/real_saturn/sproxseq_B1_fig5"

# %% [markdown]
# ## 1. Targets
# RNA: Visium counts to 10 GLM-PCs. Protein: every PLA product containing a probe is summed into 46 probe marginals, then
# CLR and 10 PCs. The PLA products are stored next to the target and never enter the loss. All three sections are prepared
# because step 4 (`sproxseq_gc_trajectory.py`) loops over them.
#
# **The GLM-PCs are not reproducible from the raw counts.** `glmpca` starts from an unseeded `np.random` draw, so each run
# gives another rotation of the factors. The first comparison below (fresh draw vs published) shows it: everything is
# bitwise identical except `Y_RNA`. The published draw is stored in `steps/fig5_glmpc_published.npz` (published `Y_RNA` only)
# and is put back by `steps/fig5_pin_glmpc.py`, so the rest of the notebook can be checked against the published results.
# Section 8 runs the same chain on a fresh draw to show what that draw changes.

# %%
for s in SECS:
    nb.step(f"prep_{s}", f"python scripts/sproxseq_prep.py {s} --probe",
            [f"results/real_saturn/targets_sproxseq_{s}_p46.npz", f"results/real_saturn/sproxseq_{s}_p46_pla.npz"])
    compare_npz(f"sproxseq_{s}_p46_pla.npz")
PINNED = nb.RES / "fig5_glmpc_pinned.txt"
if not PINNED.exists():
    for s in SECS: compare_npz(f"targets_sproxseq_{s}_p46.npz")     # fresh draw: only Y_RNA should differ
nb.step("pin_glmpc", f"python {STEPS}/fig5_pin_glmpc.py A1 B1 D1", [PINNED])
for s in SECS: compare_npz(f"targets_sproxseq_{s}_p46.npz")         # after pinning: nothing should differ

# %% [markdown]
# Each modality's own log depth is regressed from its factors and the factors rescaled. Fig. 5 uses section B1 only.

# %%
nb.step("depthreg_B1", "python scripts/sproxseq_prep_depthreg.py B1", [f"results/real_saturn/targets_{FIT}.npz"])
compare_npz(f"targets_{FIT}.npz")

# %% [markdown]
# ## 2. Saturn fit on B1
# `run_real2.py --dataset sproxseq_B1_p46_dr --seeds 1 --seed0 42`, the command in the session of 2026-10-14. That command
# set no thread variable, so `SAT_THREADS` takes the script default of 4 and `OMP_NUM_THREADS` stays unset. Thread count
# changes the fit. About 5 min.
#
# Tested result: `axes.csv` and the weights are identical to the published ones. Two summary fields still differ: R2 by
# about 2e-8, and `axis_axis_corr`, which is +0.107 in the published summary but -0.107 here. The published `axes.csv`
# itself gives Spearman -0.107, so the published summary does not match its own axes in sign. No panel reads this field.

# %%
nb.step("fit_B1_dr", f"python scripts/run_real2.py --dataset {FIT} --seeds 1 --seed0 42", [f"results/real_saturn/{FIT}/axes.csv"], threads=4)
nb.compare_axes(FIT)
compare_json(f"{FIT}/summary.json")

# %% [markdown]
# ## 3. Fits without depth regression
# `sproxseq_gc_trajectory.py` reads the barcodes, coordinates and axes of the non-regressed p46 fits of all three sections.
# The Fig. 5 tables only take barcodes and coordinates from them. The command of 2026-09-20 is not in the transcripts, so
# the run uses the same defaults as step 2. These axes are compared but do not feed any panel.

# %%
for s in SECS:
    nb.step(f"fit_{s}_p46", f"python scripts/run_real2.py --dataset sproxseq_{s}_p46 --seeds 1 --seed0 42",
            [f"results/real_saturn/sproxseq_{s}_p46/axes.csv"], threads=4)
    nb.compare_axes(f"sproxseq_{s}_p46")

# %% [markdown]
# ## 4. Marker zones and Pearson residuals
# Follicle zones from the authors' marker lists, per-spot Pearson residuals of the PLA products. The script also runs its
# 200-rotation tests on all three sections; those are not used by the figure. OMP unset, as published.

# %%
G = "results/real_saturn/sproxseq_gc_trajectory"
nb.step("gc_trajectory", "python scripts/sproxseq_gc_trajectory.py", [f"{G}/spots_{s}.csv" for s in SECS] + [f"{G}/residuals_B1.npy"])
nb.compare_table("sproxseq_gc_trajectory/spots_B1.csv", tol=1e-6)

# %% [markdown]
# ## 5. Axis readouts on B1
# The protein-axis readout (`bt_axis`) feeds the joint table of the RNA-axis readout (`rna_axis`), which writes the
# compartments, axis orientation, mitochondrial share and depth used by the figure. 200 rotations for genes and complexes.
# `OMP_NUM_THREADS=16`, as published.

# %%
nb.step("bt_axis", "python scripts/sproxseq_B1_bt_axis.py", ["results/real_saturn/sproxseq_B1_bt_axis/complexes.csv"], omp=16)
nb.step("rna_axis", "python scripts/sproxseq_B1_rna_axis.py", ["results/real_saturn/sproxseq_B1_rna_axis/spots.csv"], omp=16)
for t in ("sproxseq_B1_bt_axis/complexes.csv", "sproxseq_B1_rna_axis/spots.csv", "sproxseq_B1_rna_axis/complexes.csv", "sproxseq_B1_rna_axis/modules.csv"):
    nb.compare_table(t, tol=1e-4)

# %% [markdown]
# ## 6. Fig. 5 tables
# GASTON DP domains on each axis with K from the elbow, proteins with A/B probes summed, isotype-corrected complex
# specificity per bin, and the slope of each complex against 200 rotations of the axis field. About 30 min, OMP unset.

# %%
nb.step("fig5_tables", "python scripts/sproxseq_B1_fig5_prep.py", [f"{T5}/{f}" for f in ("spots.csv", "summary.json", "layers.npz", "proteins.npz", "complexes_RNA.csv", "complexes_Protein.csv")])
for t in ("spots.csv", "bins_RNA.csv", "bins_Protein.csv", "complexes_RNA.csv", "complexes_Protein.csv", "profiles_RNA.csv", "profiles_Protein.csv"):
    nb.compare_table(f"sproxseq_B1_fig5/{t}", tol=1e-4)
compare_json("sproxseq_B1_fig5/summary.json"); compare_npz("sproxseq_B1_fig5/proteins.npz"); compare_npz("sproxseq_B1_fig5/layers.npz")

# %% [markdown]
# ## 7. Figure
# `fig5_B1.py` builds both pages (panels a to j) and one SVG per panel. Panel j fits the contact complexes inside the
# builder. No panel is dropped.

# %%
nb.step("figure", f"python {nb.PF}/fig5/fig5_B1.py")
compare_json("fig5/fig5_part1.json", root="pf"); compare_json("fig5/fig5_part2.json", root="pf")     # only the output paths differ
compare_png("fig5/fig5_part1.png"); compare_png("fig5/fig5_part2.png")
show_small("fig5/fig5_part1.png", "fig5/fig5_part2.png")

# %% [markdown]
# ## 8. Replicate with a fresh GLM-PCA draw
# The chain of sections 1 to 6 in its own workspace (`fresh_glmpc/`) without the published GLM-PCs. This is what a reader
# gets from the raw counts alone. About 1 h.
#
# Tested result (one draw): the published numbers depend on the draw. Axes against published: rho_RNA |r| 0.77, rho_Protein
# |r| 0.88. RNA elbow K 3 instead of 4. Complexes with rotation p < 0.05: rho_RNA 20.0% (published 22.9%), 100 of 148 hits
# shared; rho_Protein 24.0% (published 7.0%), 16 of 45 hits shared. The statement that rho_Protein is near chance does not
# hold on this draw. Other draws may differ again.

# %%
nb.step("fresh_replicate", f"python {STEPS}/fig5_fresh_replicate.py", [nb.ROOT / "fresh_glmpc/Hole/gaston_mix_mm/results/real_saturn/fresh_replicate.json"])
print(json.dumps(json.load(open(nb.ROOT / "fresh_glmpc/Hole/gaston_mix_mm/results/real_saturn/fresh_replicate.json")), indent=1))
