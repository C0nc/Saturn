# %% [markdown]
# # Figure 4: RT&T-MERFISH isoform usage in the mouse brain
#
# From the published cell-by-isoform matrix to every panel of Fig. 4: isoform target, Saturn fit (seed 42), domain and
# switch tables, the Sst state transition, and the figure.
#
# **Conda envs:** `spatialmultivelo` (this kernel), `tutorial_lpc` (scVI), `stories` (CellRank 2).
# **Time:** about 1 h on 8 CPU threads for the fit, plus SPLISOSM and the 500 permutations of the Sst step.

# %%
import sys; sys.path.insert(0, ".")
import nbutil as nb
nb.init("fig4")
DS = "rtt_merfish_s5_iso100_sv_pca_shrunk"; FIT = f"results/real_saturn/{DS}_norestart_lc0.3"
WS = "results/real_saturn/rtt_isoform_variants/whole_sv"
nb.mkdirs(WS)

# %% [markdown]
# ## 1. Isoform target
# Section 5, 100 um bins. SPLISOSM selects genes with spatially variable isoform usage; the isoform target is a PCA of
# their shrunken isoform ratios. The first step builds the unfiltered binomial target, which the second step only reports
# as a comparison of the spatial ceiling.
#
# The RNA GLM-PCA starts from an unseeded random state, so the published factors cannot be regenerated. The last step
# puts them into the target (`steps/fig4_glmpc_published.npz`); everything else is rebuilt from the raw data.

# %%
nb.step("binomial_hv1000", "python scripts/rtt_isoform_binomial_glmpca.py --depthcov --hv 1000",
        ["results/real_saturn/targets_rtt_merfish_s5_iso100_binom_depthcov_hv1000.npz"])
nb.step("splisosm_multinomial", "python scripts/rtt_isoform_multinomial.py",
        [f"{WS}/sv_results.csv", "results/real_saturn/targets_rtt_merfish_s5_iso100_sv_multinom.npz"])
nb.step("isoform_ratio_pca", "python scripts/rtt_isoform_ratio_pca.py", [f"results/real_saturn/targets_{DS}.npz"])
STEPS = nb.Path(nb.__file__).resolve().parent / "steps"
nb.step("pin_glmpc", f"python {STEPS}/fig4_pin_glmpc.py {DS}", ["results/real_saturn/fig4_glmpc_pinned.txt"])

# %% [markdown]
# ## 2. Saturn fit
# Seed 42 with no restarts and a higher cross-pathway cost (`lam_cross = 0.3`), so the isoform axis does not collapse onto
# the RNA axis. Published settings: `SAT_THREADS=8`. Thread count changes the fit. The `--weights` run keeps the trained
# model, which Fig. 7 needs.

# %%
nb.step("fit", f"python scripts/rtt_isoform_refit_nocollapse.py {DS} norestart_lc0.3", [f"{FIT}/axes.csv"], threads=8)
nb.step("fit_weights", f"python scripts/rtt_isoform_refit_nocollapse.py {DS} norestart_lc0.3 --weights", [f"{FIT}_weights/saturn_best.pt"], threads=8)
nb.compare_axes(f"{DS}_norestart_lc0.3")

# %% [markdown]
# ## 3. Domains, usage along the isodepth and the boundary switch

# %%
nb.step("isodepth_tables", "python scripts/rtt_isoform_sv_isodepth.py", [f"{FIT}/isodepth/usage_screen.csv"])
nb.step("fig4_tables", "python scripts/rtt_isoform_sv_fig4_prep.py", [f"{FIT}/isodepth/domain_slopes.csv", f"{FIT}/isodepth/switch_23.csv"])
for t in ("domain_slopes.csv", "switch_23.csv"):
    nb.compare_table(f"{DS}_norestart_lc0.3/isodepth/{t}", tol=1e-4)

# %% [markdown]
# ## 4. Sst state transition (panels k-n)
# Sst interneurons on both sides of the boundary. scVI runs in `tutorial_lpc` and CellRank 2 in `stories`.

# %%
SST = "results/real_saturn/rtt_sst_transition"
nb.step("sst_prep", "python scripts/rtt_sst_isoform_transition.py prep", [f"{SST}/sst_cells.h5ad"])
nb.step("sst_scvi", "python scripts/rtt_sst_isoform_transition.py scvi", [f"{SST}/X_scvi.npy"], env="tutorial_lpc")
nb.step("sst_cellrank", "python scripts/rtt_sst_isoform_transition.py cellrank", [f"{SST}/cellrank_summary.json"], env="stories")
nb.step("sst_dynamics", "python scripts/rtt_sst_isoform_transition.py dynamics", [f"{SST}/isoform_dynamics.csv"])
nb.compare_table("rtt_sst_transition/isoform_dynamics.csv", tol=1e-4)

# %% [markdown]
# ## 5. Figure

# %%
nb.step("figure", f"python {nb.PF}/fig4/fig4_parts.py")
nb.show("fig4/fig4_part1.png", "fig4/fig4_part2.png")
