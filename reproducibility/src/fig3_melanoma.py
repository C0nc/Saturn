# %% [markdown]
# # Figure 3: melanoma RNA + ATAC (DBiT)
#
# From the raw count matrix, ATAC fragments and H&E to the Saturn panels of Fig. 3: objects, targets, Saturn fit (seed 42),
# post-hoc tables, chromVAR and the figure. iStar is read from its saved maps.
#
# **Envs:** `spatialmultivelo` (this kernel), `cellcharter` (tumour call).
# **Time:** about 2 h. Most of that is the fit on 8 CPU threads.
# **Hardware:** the tumour call uses a GPU for scVI when available; the rest runs on CPU.
#
# **Left out (comparison methods only):** `mel_baseline_mellon.py` and the Mellon columns of panels l-n; the DPT kernel column
# of panel k. `mel_cellrank_kernels.py` still computes DPT, because it is one kernel in an unchanged script, but it is not drawn.
# The rho and fusion CellRank kernels stay: they are Saturn results. The fit runs `run_real2.py` without `--bench`, so no GASTON.
# Supplementary figS3 is not built.
#
# **Inputs that are not raw:** `melanoma_with_cnv_in_tissue.h5ad`, `starfysh_decon.h5ad` and `targets_melanoma_tissue.npz`, see section 2.

# %%
import sys; sys.path[:0] = [".", "steps"]
import nbutil as nb, fig3_checks as chk
nb.init("fig3")
PY, STEPS = sys.executable, nb.Path(nb.__file__).parent / "steps"
OBJ = "results/melanoma"; RS = "results/real_saturn"; DS = "melanoma_tissue"
SAT = f"{nb.ROOT}/Hole/gaston_mix_mm/results/real_saturn"

# %% [markdown]
# ## 1. RNA and ATAC objects
# RNA from the raw matrix and ATAC 5 kb tiles from the fragments with SnapATAC2. Both steps are recovered from
# `upstream/melanoma_prep.py` and `upstream/melanoma_multimodal.py`, without their GASTON-Mix-MM fits. Under 5 min.

# %%
nb.step("melanoma_rna", f"python {STEPS}/fig3_upstream_rna.py", [f"{OBJ}/melanoma_rna.h5ad"])
nb.step("atac_snap", f"python {STEPS}/fig3_upstream_atac.py", [f"{OBJ}/atac_snap.h5ad"])
if nb.PUB: chk.upstream_objects()

# %% [markdown]
# ## 2. Tumour call and Saturn targets
# `upstream/melanoma_tumor_id.py` (scVI, CellCharter-style clusters, marker scores, env `cellcharter`) writes the per-spot call.
# Its spot list decides which spots are kept, and its marker columns are the programme scores of panels l-n. It reads two
# inputs that this notebook does not rebuild: `melanoma_with_cnv_in_tissue.h5ad` (needs a live BioMart query, plus an old Saturn
# and Mellon fit in the same script) and `starfysh_decon.h5ad` (a Starfysh run). Note that the first object already holds
# log-normalised values in X, which this step stores as counts and normalises again; that is how the published call was made.
#
# Targets: QC keeps spots with a tumour call, RNA depth >= 500 and ATAC depth >= 2000. RNA GLM-PCA with log depth regressed out
# and ATAC spectral embedding are recomputed on the kept spots.
#
# **The published target is an input here.** `melanoma_tissue_prep.py` runs GLM-PCA from an unseeded `np.random` start, and
# `rna_glmpc_real` retries with penalties 1, 5, 20, 100 when a start diverges, so both the values of `Y_RNA` and the penalty
# change between runs. The per-component spread of the published `Y_RNA` matches penalty 20, not the penalty 1 that new runs mostly
# reach. Fits on new draws: penalty 1, |r| with published 0.46 (rho_RNA), 0.29 (rho_ATAC), R2 RNA 0.70, ATAC 0.77; two penalty 20
# draws, rho_RNA 0.98 and 0.96, rho_ATAC 0.49 and 0.68, R2 RNA 0.80 and 0.79, ATAC 0.81 and 0.82 (published R2 0.79, 0.82).
# So the fit below reads the published `targets_melanoma_tissue.npz`, and this section only reruns the script into
# `target_recheck/` to show which arrays reproduce.

# %%
CALL = "Saturn/paper_figures/figure2/melanoma_tumor_id/per_spot_tumor_call.csv"
nb.step("tumor_call", f"python {nb.PF}/fig3/upstream/melanoma_tumor_id.py", [str(nb.ROOT / CALL)], env="cellcharter")
if nb.PUB: chk.csv_abs(CALL, "cc_K7")
nb.step("target_recheck", f"python {STEPS}/fig3_target_recheck.py", [f"{RS}/target_recheck/targets_{DS}.npz"])
if nb.PUB: chk.target_recheck(DS)

# %% [markdown]
# ## 3. Saturn fit
# Seed 42 on CPU with `SAT_THREADS=8` and `OMP_NUM_THREADS` unset, as published. Thread settings change the fit, so this is
# not the Fig. 2 setting. All post-hoc steps below run with `OMP_NUM_THREADS` unset.

# %%
nb.step("fit", f"python scripts/run_real2.py --dataset {DS} --seeds 1 --seed0 42", [f"{RS}/{DS}/axes.csv"], threads=8, device="cpu")
nb.compare_axes(DS)
if nb.PUB: chk.json_numbers(f"{DS}/summary.json", ["R2_whole", "coupling_best"])

# %% [markdown]
# ## 4. Feature tables (panels c, e, f, g)
# Gene and peak screens, own vs cross contribution per feature, function-matched peaks and DORC chromatin potential.

# %%
nb.step("screens", f"python scripts/marker_function_panels.py {DS} 8 8", [f"{RS}/{DS}_marker_functions/gene_screen.csv", f"{RS}/{DS}_marker_functions/peak_screen.csv"], extra_env=dict(RESCREEN="1"))
nb.step("own_vs_cross", f"python scripts/saturn_own_vs_cross.py {DS}", [f"{RS}/{DS}_own_vs_cross/own_vs_cross.csv"])
nb.step("function_matched", "python scripts/melanoma_chromatin_potential.py 30000", [f"{RS}/{DS}_chromatin_potential/function_matched_peaks.csv"])
nb.step("dorc", "python scripts/melanoma_chromatin_potential_dorc.py", [f"{RS}/{DS}_chromatin_potential/dorc_cp_pairs.csv", f"{RS}/{DS}_chromatin_potential/dorc_curves_RNA.npz"])
for t in ("marker_functions/gene_screen.csv", "marker_functions/peak_screen.csv", "own_vs_cross/own_vs_cross.csv",
          "chromatin_potential/function_matched_peaks.csv", "chromatin_potential/dorc_cp_pairs.csv", "chromatin_potential/dorc_cp_domain_summary.csv"):
    nb.compare_table(f"{DS}_{t}")

# %% [markdown]
# ## 5. Transition zones and CellRank 2 on the Saturn axes (panels k-n)
# The gradient step runs first because the kernel step reads its spots table. `run_fig3.sh` lists them the other way round,
# which only works when the gradient output already exists.

# %%
nb.step("gradient_zones", "python scripts/mel_gradient_transition.py", [f"{RS}/{DS}_gradient/spots.csv", f"{RS}/{DS}_gradient/zones.csv"])
nb.step("cellrank_kernels", "python scripts/mel_cellrank_kernels.py", [f"{RS}/{DS}_cellrank_kernels/kernels_summary.csv"])
nb.step("cellrank_arrows", "python scripts/mel_cellrank_arrows.py", [f"{RS}/{DS}_cellrank_kernels/arrows.npz"])
for t in ("gradient/spots.csv", "gradient/zones.csv", "cellrank_kernels/kernels_summary.csv", "cellrank_kernels/obs.csv"):
    nb.compare_table(f"{DS}_{t}")
if nb.PUB: chk.npz(f"{DS}_cellrank_kernels/arrows.npz")

# %% [markdown]
# ## 6. chromVAR TF activity per spot
# JASPAR 2022 motifs on the top 12,000 tiles, GC and accessibility matched backgrounds. The motif scan takes a few minutes.

# %%
nb.step("tf_activity_rho", f"python scripts/saturn_tf_activity_rho.py {DS}", [f"{RS}/{DS}_tf_activity/tf_fits.csv"])
nb.step("tf_activity_spot", f"python scripts/saturn_tf_activity_spatial.py {DS}", [f"{RS}/{DS}_tf_activity/tf_z_per_spot.npz"])
if nb.PUB: chk.npz(f"{DS}_tf_activity/tf_z_per_spot.npz", ["z", "tfs", "barcodes"])

# %% [markdown]
# ## 7. Saved iStar maps (panels a, h-j)
# iStar is not rerun here. It trains on the GPU without fixed seeds, so its maps and the numbers of panel j change between runs. The
# figure reads the saved maps and H&E crops listed in `inputs/fig3.json`.

# %% [markdown]
# ## 8. Figure
# Parts 1 and 2 use the figure's own builders and layouts. Part 3 keeps the Saturn columns only, drawn by
# `steps/fig3_saturn_panels.py`; panel n then draws the rho_RNA permutation p values from a different part of the random
# stream than the published version, because the Mellon permutations are gone.

# %%
nb.step("figure", f"python {STEPS}/fig3_saturn_panels.py", [str(nb.PF / "fig3/fig3_saturn_stats.json")])
if nb.PUB:
    chk.png_identical("fig3/fig3_part1.png", "fig3/fig3_part2.png")
nb.show("fig3/fig3_part1.png", "fig3/fig3_part2.png", "fig3/fig3_part3.png")
