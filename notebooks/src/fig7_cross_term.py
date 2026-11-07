# %% [markdown]
# # Figure 7: what the Saturn cross pathways explain
#
# From the five Saturn fits of Figs 2-6 to the Saturn panels of Fig. 7: per-spot decomposition and local cross gain, per-gene
# cross test, cis concordance, what absorbs the ATAC cross term, the ATAC-steep zones, and the figure.
#
# **No fit is rerun here.** This notebook starts from the fits. Each input and the notebook that makes it:
#
# | Input | Made by |
# |---|---|
# | `lpc10_fov`: target, `saturn_best.pt`, `axes.csv`; chromVAR `tf_z_per_spot.npz`; RNA and ATAC AnnData (`arcseq_LPC10S1_*`) | `fig2_lpc` |
# | `melanoma_tissue`: target, `saturn_best.pt`, `axes.csv`; chromVAR `tf_z_per_spot.npz`; `melanoma_rna.h5ad`, `atac_snap.h5ad` | `fig3_melanoma` |
# | `rtt_merfish_s5_iso100_sv_pca_shrunk`: target, `_norestart_lc0.3/axes.csv`, `_norestart_lc0.3_weights/saturn_best.pt` | `fig4_merfish_isoform` |
# | `sproxseq_B1_p46_dr`: target, `saturn_best.pt`; `sproxseq_B1_fig5/spots.csv` (axis orientation) | `fig5_sproxseq` |
# | `mel_hic_abins_dr50_51_rnadr`: target, `saturn_best.pt`, `axes.csv` | `fig6_melanoma_hic` |
# | RNA matrices (MERFISH, Sprox-seq B1, SpHIC117), GTFs | raw inputs, shared with Figs 4-6 |
#
# **Lab intermediates that no notebook remakes:** `LPC10_lesion_FOV.h5mu` (cell2location proportions; `lpc10_fov_h5mu.py`, env `c2l`,
# GPU), `starfysh_composition.csv` (Starfysh run), `LPC10S1_codex_per_spot.h5ad` (Catalign script 05; `fig2_lpc` recovers only its
# coordinate part).
#
# **Env:** `spatialmultivelo`, all steps. OMP unset. **Time:** about 30 min on a shared machine (cross term and mechanism 12 min, transect 14 min, figure 2 min; 15 min in total on a quieter machine).
#
# **Panels dropped:** a and S6a, the GASTON benchmark. Their slots stay empty; every other panel comes from the builder, unmodified.
# Whole-section R2 of Saturn is still shown in b.

# %%
import os, sys; sys.path.insert(0, ".")
from pathlib import Path
import nbutil as nb
nb.init("fig7")
HERE = Path(nb.__file__).resolve().parent; STEPS = HERE / "steps"; ENV = dict(SATURN_ROOT=str(nb.ROOT))
X = "results/real_saturn/saturn_cross_term"; KEYS = ["lpc10", "melanoma", "merfish", "sprox", "hic"]; ATAC = ["lpc10", "melanoma"]


def compare(what):
    """text columns, JSON, npz and pixel checks that nb.compare_table does not do"""
    if nb.PUB is None: print("SATURN_PUBLISHED not set: no comparison"); return
    nb.step(f"compare_{what}", f"python {STEPS}/fig7_compare.py {what}", extra_env=ENV); print(open(nb.HOLE / f"logs/compare_{what}.log").read())


# %% [markdown]
# ## 1. Cross term per dataset
# From each saved model: own and cross term per spot, local cross gain (25-NN), gradients of both axes; per-gene hinge test of the
# other axis beyond rho_RNA (partial F, BH), for genes detected in at least 5% of spots; for ATAC, the gene's +-50 kb accessibility
# vs its cross profile against 200 random windows. Then the steepness test (200 rotations) and gene length.

# %%
for k in KEYS:
    nb.step(f"cross_term_{k}", f"python scripts/saturn_cross_term.py {k}", [f"{X}/{k}_spots.csv", f"{X}/{k}_genes.csv", f"{X}/{k}_profiles.npz", f"{X}/{k}_length.csv"])
for k in KEYS:
    nb.compare_table(f"saturn_cross_term/{k}_spots.csv"); nb.compare_table(f"saturn_cross_term/{k}_genes.csv")

# %% [markdown]
# ## 2. What absorbs the ATAC cross term (panel h)
# For each cross-explained gene: the share of its gain from rho_ATAC absorbed by its own cis chromatin, TF activity, cell-type
# composition, all three, or a rotated block of equal width. Also named drivers and each feature's boundary in the steep zones.

# %%
nb.step("mechanism", "python scripts/saturn_cross_mechanism.py", [f"{X}/{k}_mech_{t}" for k in ATAC for t in ("genes.csv", "drivers.csv", "boundary.csv", "summary.json")])
for k in ATAC:
    nb.compare_table(f"saturn_cross_term/{k}_mech_genes.csv")

# %% [markdown]
# ## 3. The ATAC-steep zones (panels g, i and S8)
# The builder's `steep_cases` ranks the cross-explained genes by how much of their cross component sits in the steep zone; its cache
# `fig7_steep.json` is an input of the transect. The transect runs across the largest steep zone along the gradient of rho_ATAC.
# The follow-up gives zone switches (200 rotations), edge and depth controls, model vs raw gradients and CODEX boundaries.

# %%
nb.step("steep_cases", f"python {STEPS}/fig7_saturn_panels.py steep", [f"{X}/fig7_steep.json"], extra_env=ENV)
nb.step("transition", "python scripts/saturn_steep_transition.py", [f"{X}/{k}_transition_{t}" for k in ATAC for t in ("fits.csv", "profiles.csv", "spots.csv", "summary.json")])
nb.step("steep_followup", "python scripts/saturn_steep_followup.py", [f"{X}/steep_followup/followup.json", f"{X}/steep_followup/lpc10_codex.csv"])
for k in ATAC:
    nb.compare_table(f"saturn_cross_term/{k}_transition_fits.csv", tol=1e-4)
    nb.compare_table(f"saturn_cross_term/{k}_transition_profiles.csv")
    nb.compare_table(f"saturn_cross_term/steep_followup/{k}_zone_switch.csv")
nb.compare_table("saturn_cross_term/steep_followup/lpc10_codex.csv")

# %% [markdown]
# ## 4. All tables and summaries against published
# Includes text columns (named drivers) and the JSON summaries (including `followup.json`).

# %%
for w in ("tables", "json"): compare(w)

# %% [markdown]
# ## 5. Figure
# `steps/fig7_saturn_panels.py` calls the builder's own layouts and panel functions, without a and S6a. Main panels b-i, supplements
# S6b-f, S7 and S8. The builder's example cache `fig7_examples.npz` is rebuilt first, in its own process, as in the published run.

# %%
nb.step("examples", f"python {STEPS}/fig7_saturn_panels.py examples", [f"{X}/fig7_examples.npz"], extra_env=ENV)
nb.step("figure", f"python {STEPS}/fig7_saturn_panels.py", [nb.PF / "fig7/fig7_part3.png", nb.PF / "fig7/figS8_steepness_checks.png", nb.PF / "fig7/fig7_saturn_stats.json"], extra_env=ENV)
for w in ("npz", "stats", "pixels", "svg"): compare(w)

# %% [markdown]
# One difference from the published PNGs is expected. `fig7_part1` has a smaller canvas because panel a is gone, and the tight crop
# moves by a fraction of a pixel, so its pixels are not comparable; the per-panel SVG check is the exact test of b and c.

# %%
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
for f in ("fig7_part1", "fig7_part2", "fig7_part3"):
    im = Image.open(nb.PF / f"fig7/{f}.png"); im.thumbnail((1000, 10 ** 5)); im.convert("RGB").save(nb.PF / f"fig7/{f}_preview.jpg", quality=80)
    nb.show(f"fig7/{f}_preview.jpg", width=800)
