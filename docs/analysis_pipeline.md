# The Saturn analysis pipeline

This is the full process used for the manuscript, from raw matrices to the cross-term mechanism. Each
step gives the call, the settings that were used, and why. The "Traps" boxes record failures that
have already happened once.

```
0 choose spots ─► 1 targets ─► 2 fit ─► 3 QC ─► 4 domains ─► 5 annotate ─► 6 feature functions
                                                                      │
                     7 cross term ─► 8 steepness ─► 9 mechanism ─► 10 transition
```

---

## 0. Choose the spots

**Spot selection is the strongest lever on fit quality, stronger than any model change.**

| fit | before | after |
|---|---|---|
| LPC10: cortex subset → lesion + distal white-matter FOV | R² RNA 0.13–0.30 / ATAC 0.05–0.25 | **0.643 / 0.674** |
| melanoma: tumour subset → whole slide → QC-filtered tissue | RNA 0.70 / ATAC 0.36 → 0.716 / 0.788 | **0.786 / 0.822** |

Rules:
- Make the biology of interest the dominant variance on the selected spots. For LPC10, the FOV was
  cut from DBiT microglia hotspots (Poisson z > 3 components) with the cortex excluded.
- Apply QC that removes coverage-driven spots. The melanoma tissue fit kept spots with an assigned
  tumour call, RNA depth ≥ 500 and ATAC depth ≥ 2000. The 490 unassigned spots had median depth 767
  against 8,274, and they made ρ_ATAC partly a depth axis (+0.43 with log depth).
- **Recompute every reduction on the selected spots** (step 1). A reduction computed on the whole
  slide carries the whole slide's variance.

## 1. Build one target per modality

Saturn fits low-dimensional continuous embeddings with MSE, as GASTON does, not raw counts. Every
target is **globally scaled**: one scale per modality so that the mean column variance is 1.

| modality | call | published setting |
|---|---|---|
| RNA counts | `rna_glmpc_real(X, K=10)` | Poisson GLM-PCA (Townes) over 3,000 Pearson-residual HVGs detected in ≥ 1 % of spots; falls back to residual PCA if GLM-PCA diverges at every penalty |
| ATAC fragments | `atac_spectral(tiles, n_comps=15)` | SnapATAC2 5 kb tiles → `select_features(250000)` → spectral → drop component 1 (depth) |
| ATAC peaks | `lsi_real(X, K=15)` | TF-IDF → log1p → truncated SVD, drop component 1 |
| Hi-C | `compartment_pca(AB, K=10)` | A/B compartment scores per bin → PCA (Fast-Higashi was rejected: R² 0.12–0.20 vs 0.37) |
| protein / ADT | `protein_clr_pca(X, K=10)` | CLR per spot → PCA, all proteins |
| long-read isoforms | `isoform_usage_pca(X, iso_gene, K=10, depth=d)` | shrunken within-gene usage → CLR → depth regressed → PCA |
| coordinates | `zscore_coords(xy)` | one common scale (GASTON scales x and y separately, which distorts gradient directions) |

Why global scaling and not GASTON's per-component z-score: a z-score gives a 1 %-variance component
the same weight as a 35 % one. On simulations, one isodepth fit to 15 z-scored RNA GLM-PCs found the
**wrong** axis (own 0.43, other 0.89); with global scaling it found its own axis at 0.999.

> **Traps**
> - Always check every leading component against depth, not only component 1.
> - Sparse chromatin (median < ~1,500 fragments per spot) gives held-out R² of about 0–5 % for every
>   method. Use fewer components, or gene-activity targets.
> - Save `coords`, `coords_z`, `barcodes`, `modalities`, `Y_<mod>` and `depth_<mod>`. Also save
>   annotations as `ann_*` and marker log-CPMs as `mk_*`, so every later step reads one file.

## 2. Fit

```python
res = saturn.fit(targets, coords_z, modalities, seeds=[42], threads=8, depth={...})
```

`train_saturn` defaults are the published settings. Each is a measured fix:

| stage | what | why |
|---|---|---|
| 1 (30 % of 10,000 epochs) | cross pathways OFF → M independent GASTON fits | additivity leaves a discrete label swap; with a single stage a head recovered the other axis (0.747) over its own (0.465) |
| 2 | cross ON (zero-initialised); own-axis anchor on the **residual** Y − cross | the full anchor rewards mixing the other axis into d_m: estimated coupling 0.09 against a true 0.42 |
| 3 | `n_restarts=4`: re-initialise head m and the pathways reading it; keep it only if the fit on **20 % held-out spots** improves by > 0.5 % | training-loss acceptance fails both ways: genuine rescues were chains of < 3 % steps, while spurious restarts also gained 1–2 % |
| refit | `refit_all_epochs=3000` on every spot | the reported model uses the whole section |

Other settings:
- `lam_cross=0.03`: 0.1 over-shrinks the coupling; 0 leaks.
- `standardize_d=False`: back-propagating through d's std trapped 8 of 32 initialisations.
- Network widths are (20, 20) for both the isodepth and pathway networks.
- `lam_tail=0`: a kurtosis penalty was tested and not adopted.

With several seeds, `fit` keeps the seed with the lowest whole-section loss and reports
`seed_stability` (mean pairwise Spearman after sign alignment).

> **Trap: reproducibility.** Seed and CPU thread count together define the fit (see the README).
> Record `SAT_THREADS` and `OMP_NUM_THREADS` with every published fit.

## 3. Check the fit before any biology

| check | from | pass if |
|---|---|---|
| variance explained | `res.r2[m]["full"]`, `["own_axis_only"]` | full ≥ GASTON; the gap between full and own-only is what the cross term adds |
| held-out R² | `heldout_r2(..., gaston=True)` | Saturn ≥ the GASTON baseline on 20 % unseen spots |
| depth confound | `res.axis_depth_corr` (\|Spearman\| with log depth) | well below the axis's biological correlations; Pearson on raw depth understates it (−0.41 vs −0.71) |
| seed stability | `res.seed_stability` (≥ 3 seeds) | ≳ 0.8. Below that, ensemble seeds before interpreting |
| axis collinearity | `res.axis_axis_corr` | coupling is **not identifiable** when the axes correlate > ~0.8 |
| axis kurtosis | `kurtosis(res.axes)` | < ~6. Higher means one axis spends its range on one small region (unstable: 0.36–0.53) |
| orientation | `orient(res.axes, marker_scores)` | always, before comparing fits or sections; sign is arbitrary |

**Reading the coupling matrix.** Entry [m, k] is the share of modality m's predicted variance carried
by axis k. It is a variance share, **not a causal arrow**. In melanoma, RNA→ATAC 0.26 > ATAC→RNA 0.09
probably reflects cell composition.

## 4. Domains along each axis

```python
kR, curve = saturn.domains.choose_k(res.axes[:, 0], Y_rna)                   # elbow of the DP error curve
labR = saturn.domains.gaston_domains(res.axes[:, 0], Y_rna, n_domains=kR)   # GASTON DP on the RNA targets
kA, _ = saturn.domains.choose_k(res.axes[:, 1], Y_atac)
labA = saturn.domains.gaston_domains(res.axes[:, 1], Y_atac, n_domains=kA)
bR = saturn.domains.boundaries(res.axes[:, 0], labR)                         # contour levels for maps
```

Each modality is segmented on **its own axis with its own targets**. Domains belong to the axis.
- **Choose K per axis.** The DP error always falls as K grows, so K must be chosen. `choose_k` takes
  the kneedle elbow of the unexplained-variance curve for K = 1–10. The published figures use it:

  | dataset | RNA axis | second axis |
  |---|---|---|
  | LPC10 | 3 | ATAC 2 |
  | melanoma | 4 | ATAC 3 |
  | melanoma Hi-C (A/B + insulation target) | 4 | Hi-C 2 |
  | MERFISH isoform | 4 | isoform 3 |
  | tonsil Sprox-seq B1 | 4 | protein 3 |

  `results/real_saturn/domain_k_elbow/k_elbow.json` still lists the superseded Fast-Higashi Hi-C fit (3 / 3); Fig 6
  computes its elbow on the current fit when it loads.

  Earlier diagnostic runs used a fixed K = 4.
- Use 50 buckets by default.
- Use 20 buckets for sparse ATAC panels: they need about 50 counts per domain for the LRT to have power.
- Domains under 1 % of spots are DP tail artefacts and are merged.

## 5. Annotate each domain

```python
T = saturn.annotate.annotate_counts(rna.X, rna.var_names, labR, dR, "RNA", depth=depth_rna,
                                    cats={"cluster": cc, "region": region}, conts={"microglia_z": mg})
T = saturn.annotate.annotate_counts(atac.X, atac.var_names, labA, dA, "ATAC", gtf="gencode.gtf", ...)
T = saturn.annotate.annotate_continuous(V, names, labM, dM, "HiC", ...)
```

Each row of the table is one domain. Name a domain from **all four** kinds of evidence, never from one.

| evidence | column | how |
|---|---|---|
| composition | categorical annotations as JSON fractions; continuous as means | clusters, cell-type proportions, anatomy (e.g. LPC10: lesion / distal WM / other) |
| markers | `markers` | counts: Poisson enrichment z = (obs − exp)/√exp, with exp from the domain's share of library, a count floor of 100 and log2FC ≥ 0.5. Continuous: Welch t |
| ATAC context | `markers` + `gene_level` | peaks → nearest TSS with class P / p / l (promoter ≤ 2 kb, proximal ≤ 20 kb, long-distance), and peaks summed into nearest genes |
| gradient | `gradient_feats` | of the 500 most-counted features, how many keep a slope **inside** the domain (LRT p < 0.1), i.e. vary continuously rather than only jumping at a boundary |
| coverage | `median_depth` | the cheapest check that a domain is biology rather than coverage |

Write the label back with `labels={k: "lesion core", ...}`. Use anatomical names (lesion core, lesion
rim, distal WM; melanocytic tumour, invasive front, immune niche), not "D0–D3". Domain **indices are
not stable** across fits or datasets. Map colours in paper figures by label.

> **Traps**
> - Ranking by log2FC alone puts sparse predicted genes (Gm\*, RP11-\*) on top. Rank by z instead.
> - The LPC ATAC "low-signal" domain turned out to be the **highest**-coverage one. Always report
>   median depth.
> - `np.argsort(-tot)` on a uint32 matrix wraps around and ranks empty features. Cast to float
>   first (done in the library).
> - Measure promoter distance to the peak or tile **edge** (`peak_class(dist, half_width)`). Measured
>   from the midpoint, a 5 kb tile that contains the TSS fails a 2 kb cutoff.

## 6. Which features are functions of the axis

```python
scr, slope, inter, idx = saturn.annotate.screen(X, names, lab, d, kind="poisson", lib=lib)   # or "gauss" / "binomial"
top = scr[scr.n_sloped_domains >= 2].head(8)
pos = pd.Index(idx).get_indexer(top["_i"])   # slope and inter follow idx; the table is sorted by rank
fig = saturn.plotting.function_figure(xy, d, lab, X[:, idx], slope, inter, names[idx],
                                      pos, bnds, lib=lib, link="log")
```

GASTON keeps a domain's slope only when an LRT beats the zero-slope fit at p < 0.1. The gate is right
for every modality, but the likelihood must match the data:

| data | fitter |
|---|---|
| counts with a library (genes, ATAC) | `fit_poisson` (GASTON's own) |
| signed continuous (A/B, protein CLR) | `fit_gauss` (per-domain OLS, F gate) |
| successes out of trials (isoform usage, clone composition) | `fit_binomial` (logit IRLS, LRT gate) |
| over-dispersed counts | check `dispersion()`. About 1 means Poisson is fine (ATAC tiles: 1.07–1.09) |

The wrong likelihood is not cosmetic. With Gaussian fits on isoform CLR values, every isoform of a
spatially varying gene looked sloped. The binomial with the gene total as denominator gave 117 of 800
isoforms, against 915 of 1,500 genes.

**Selection rule:** keep features sloped in ≥ 2 domains (a real multi-domain function), then rank by
R² against the binned observed values. Report `N(≥2) / N screened` with every figure: that count is
the result.

**Figure contract:** five columns per feature, `ρ | domains | RAW | FITTED | fitted function`. Clip
the function panel to the observed range, because per-domain lines extrapolate.

## 6b. Chromatin and RNA along one axis (Figs 2–3)

These analyses ask how a modality's features change along an isodepth. Inputs are k-NN smoothed
(15 spots), z-scored log1p CP10k of genes and peaks.

| question | call | settings / reading |
|---|---|---|
| which gene–peak pairs are both functions of ρ, and how alike | `cis.cis_pairs` → `cis.function_pairs(G, P, d, lab, pairs)` | ±1 Mb cis peaks (top 25 by count) and ≥ 5 Mb distal peaks of the same gene as calibration. Pairs count only where **both** slopes pass the gate in the same domain. Similarity = z-scored curve correlation on a 120-point grid. The lag is descriptive: Granger on fitted curves has a false-positive rate of 10–18 % at a nominal 5 % |
| does gene–peak agreement live along ρ or across it | `gradient.rho_gradient`, `gradient.spatial_gradients`, `cis.along_across_corr` | per domain, gene–peak correlation of gradient components along vs across ρ. LPC10: 50 % of gene-gradient energy lies along ρ |
| TF activity as a function of ρ | `tfactivity.scan_motifs` → `background_sets` → `rho_binned_activity` | JASPAR2022, top 1 % PWM scores = hits; 60 equal-count ρ bins; chromVAR z against GC- and accessibility-matched sets; per-domain Gaussian fit. Per spot: `deviations_matched` on spots, then smooth |
| is a feature explained by its own axis or the other | `crossterm.own_vs_cross` | per-feature commonality analysis (unique own, unique cross, shared) and per-domain R² |

## 7. What the cross pathways explain

```python
T, S = saturn.crossterm.spot_decomposition(res.model, coords_z, coords, targets, i_rna, i_x, signs=(+1, +1))
G, prof, genes = saturn.crossterm.gene_cross_test(rna_counts, gene_names, T.rho_RNA, T.rho_X)
sig = genes[np.isin(genes, G.gene[G.q < 0.05])]                  # FDR hits, in tested order
keep, W, _ = saturn.crossterm.cis_windows(atac_counts, peak_names, sig, gtf)
cis, cS = saturn.crossterm.cis_concordance(prof, genes, W, keep, T.rho_RNA, T.rho_X)
```

- **Gradients** come from autograd through the isodepth networks, with the chain rule through
  `zscore_coords`: one scalar scale, so directions are not bent.
  - The first Fig 7 run divided by the per-axis coordinate std; `per_axis_sd=True` reproduces it
    (the manuscript figures were rerun with the correct scaling on 2026-10-22).
  - The difference is negligible on square sections (melanoma, Hi-C).
  - On LPC10 (x/y std ratio 0.76) it moves the steepness ρ from 0.388 to 0.393 and the rotation p
    from 0.085 to 0.075.
- **Local gain**: ‖Y − P_own‖² − ‖Y − P_full‖², divided by the mean total sum of squares per spot
  and averaged over 25 neighbours. It shows where the other axis explains this modality. P_own holds
  the cross term at its mean.
- **Per-gene test**: log1p CP10k of every gene detected in ≥ 5 % of spots, ~ B(ρ_RNA) vs
  ~ B(ρ_RNA) + B(ρ_X).
  - B is a linear term plus 5 ReLU hinges at the 10–90 % quantiles (the pathway family).
  - Partial F test, BH correction.
  - The gene's **cross profile** is the B(ρ_X) part of the fit.
- **cis concordance** (ATAC). Correlate each FDR-significant gene's cross profile with the cross
  profile of its own ±50 kb accessibility. Compare with random other genes' windows (200 draws).
  - LPC10: 0.39 vs 0.12 (z ≈ 24–25).
  - Melanoma: 0.59 vs 0.33.

  This suggests that the cross term follows chromatin at the gene's own locus, not only a generic smooth
  field. It does not show that chromatin acts on the gene. The z depends on the order genes are fed to the permutation; pass them in a fixed order.
- **Gene length** (`gene_length_test`): for Hi-C, long genes carry the cross gain.
- **Example genes**: show the cross-explained genes as RNA map, ρ_ATAC map, and own vs own + cross fit.

## 8. Is the cross term used where the other axis is steeper?

```python
T["depth"] = rna_depth
st = saturn.crossterm.steepness_test(T)     # Spearman, rotation p, gain by quintile, depth-partial rho
```

Steepness is log((|∇ρ_X|/median) / (|∇ρ_RNA|/median)). The null rotates the gain field, which keeps
its spatial autocorrelation.

| case | gain by steepness quintile | ρ (depth-controlled) | rotation p |
|---|---|---|---|
| LPC10 | 0.12 → 0.62 | 0.39 (0.39) | 0.075 |
| melanoma | 0.06 → 0.19 | 0.39 (0.32) | 0.085 |

This is a **trend, not significant** under the rotation null, and it failed as a general hypothesis
across all five datasets. Present two cases in the main figure and the rest in the supplement.

## 9. Why the other axis explains a gene (mechanism)

```python
D = saturn.mechanism.absorption(Y_cross_genes, names, rho_rna, rho_x, xy,
                                blocks={"TF": chromvar_z, "comp": proportions}, cis=(keep, W))
dr = saturn.mechanism.drivers(Y_cross_genes, names, rho_rna, rho_x, xy, {"type": props_df, "tf": chromvar_df})
B = saturn.mechanism.boundary_switch({"type": props_df, "TF": chromvar_df}, xy, saturn.crossterm.steepness(T))
```

Absorbed share = 1 − ΔR²_X | K / ΔR²_X for each candidate block K. Every block is 15-NN smoothed and
6 columns wide. The null rotates each block separately, with 5 rotations each.

| block | LPC10 | melanoma |
|---|---|---|
| cis (own ±50 kb window) | 8 % | 19 % |
| TF activity (chromVAR, ATAC-only) | 43 % | 18 % |
| composition (cell2location / Starfysh) | 87 % | 59 % |
| all three | 87 % | 70 % |
| rotated null | 20 % | 15 % |

Read it as: the ATAC cross term mostly reports **which cells are there**. TF activity, which is
independent of RNA, explains a large share in LPC10 (e.g. Trem2: macrophages +0.82, SPI1 +0.74).
Interferon genes such as IFI44L keep a cis and TF component (IRF8, IRF1).

> **Circularity caveat.** Proportions from cell2location or Starfysh are estimated from the same RNA,
> so composition absorption is partly circular. Remove it with leave-gene-out or ATAC-derived
> proportions before claiming it.

## 10. Spatial transition through the steep zone

```python
tr = saturn.transition.transect(xy, T.rho_X.values, saturn.crossterm.steepness(T))
fits, prof = saturn.transition.transition_table(F_smoothed, tr)   # F: axes, gain, cell types, TFs, genes
```

- **Zone**: the largest DBSCAN component (eps 1.6 spot spacings) of the top-20 % steep spots.
- **Transect**: along the mean ∇ρ_X in the zone. The band covers the zone's 10–90 % perpendicular
  offset, extended upstream.
- **Per feature**:
  - sigmoid switch point s0 and width w;
  - bump peak, marked transient when R²_bump − R²_sig > 0.15;
  - the **steepest change** of the binned profile with a 200-resample bootstrap CI. This is the
    switch point used in the figure.

Result: each axis puts its steepness on a different tissue transition, as summarised in the table
below. Where ρ_RNA is flat but cells still switch, only the ATAC pathway can fit the RNA genes there.

| case | ρ_RNA steep at | then | ρ_ATAC steep at |
|---|---|---|---|
| LPC10 | lesion edge (macrophage, microglia, Trem2, SPI1 fall, s ≈ −17) | transient interface: leptomeninges, Ttr/Enpp2 and RNA gain all peak at s ≈ −10; cortical neurons and Pcsk2/Ntrk3 rise at −7 | inside the cortex (+4) |
| melanoma | invasive / neural-crest / myCAF stretch | PMEL, MITF, BCAN, melanocytic tumour rise and IRF1 falls (−7 to −5) | −3; then FOSB, tolerant CD8 and ANXA1 rise (+2 to +4) |

Caveats:
- This is one transect per dataset: a description, not a test.
- The melanoma zone sits at the section edge with about 2.5× the depth.
- The bootstrap random stream differs from the paper run, so the CIs differ slightly; point
  estimates are identical.

---

## Figures

Diagnostic figures (`saturn.plotting`) are for review. Paper figures follow these house rules:
- Arial, minimum 5 pt.
- PNG at 600 dpi plus SVG, with one SVG per sub-panel.
- Dense points rasterised inside the SVG.
- Sticky colours: a term keeps its colour in every figure.
- Colour maps from `saturn.theme`: `saturn_div` for ρ and scores (centred), `saturn_seq_r` for expression and distance,
  `saturn_seq` for pseudotime, `saturn_mono` for 0–1 quality and gains, `saturn_cat` for domains.

`saturn.theme.set_theme()` applies the fonts and colours, and `saturn.plotting` uses the same colours. See also
`reproducibility/paper/style/paper_style.py` (`save_dual` refuses fonts below 5 pt).

## Where the manuscript numbers come from

See [`reproducibility/paper/README.md`](https://github.com/C0nc/Saturn/blob/main/reproducibility/paper/README.md) for the script behind each figure.
