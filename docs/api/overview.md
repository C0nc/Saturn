# Function overview

Arrays are spots × features unless stated otherwise. `d` is one isodepth (N,), `lab` the integer domain
labels (N,), and `xy` the raw coordinates (N, 2) unless a name says `coords_z`.

## saturn.preprocess: targets

| function | input | returns |
|---|---|---|
| `rna_glmpc_real(counts, K=10, n_hvg=3000, min_frac=0.01, method="glmpca")` | RNA counts, sparse OK | (Y, info{genes, depth, method}) |
| `atac_spectral(tiles, n_comps=15, n_features=250000, drop_first=1)` | SnapATAC2 tile AnnData | (Y, info{depth, depth_corr_dropped}) |
| `lsi_real(counts, K=15, drop_first=True)` | peak counts | (Y, info{peaks, depth, depth_corr, dropped}) |
| `compartment_pca(AB, K=10)` | A/B scores per bin | (Y, info) |
| `protein_clr_pca(counts, K=10)` | ADT counts | (Y, info) |
| `isoform_usage_pca(iso_counts, iso_gene, K=10, alpha=5, depth=None)` | isoform counts | (Y, info) |
| `rna_glmpc`, `atac_lsi` | small dense matrices (simulation) | Y |
| `check_target(Y, depth, coords, labels)` | candidate embedding | depth_leak, label_knn, moran, var_share, elbow |
| `scale_target(P, "global")` | any embedding | centred, one scale per modality |
| `zscore_coords(xy)` | coordinates | centred, one common scale |

## saturn.core: model

- `Saturn(K, iso_hidden=(20,20), path_hidden=(20,20))`
  - `.isodepth(xy)` → (N, M) isodepths.
  - `forward(xy)` → `d, full, own, cross`.
  - `.coupling(xy)` → (M, M) variance shares.
- `train_saturn(targets, coords_z, ...)` → `(model, hist)`. Key arguments:
  - training: `epochs=10000`, `warmup_frac=0.3`, `lam_self=1`, `lam_cross=0.03`;
  - restarts: `n_restarts=4`, `restart_epochs=4000`, `val_frac=0.2`, `restart_tol=0.005`;
  - other: `refit_all_epochs`, `masks`, `device`, `seed`.

## saturn.fit: fitting and read-outs

| function | purpose |
|---|---|
| `fit(targets, coords_z, modalities, seeds, masks, depth, refit_all_epochs=3000, device, threads, **train_kw)` | best-of-seeds fit → `SaturnFit` |
| `SaturnFit` | `.axes .coupling .r2 .seed_stability .axis_axis_corr .axis_depth_corr .summary() .save(dir, coords, barcodes)` |
| `decompose(model, coords_z)` | `d, full, own, cross` as numpy |
| `r2_split(model, coords_z, targets, masks, modalities)` | full / own_axis_only / other_axes_only R² |
| `orient(axes, reference)` | flip axes to correlate positively with a reference |
| `heldout_r2(targets, coords_z, modalities, test_frac=0.2, gaston=False)` | held-out R² table |
| `load_model(path, K)` | rebuild from `saturn_best.pt` |

## saturn.domains

- `gaston_domains(d, Y, n_domains=4, n_buckets=50, min_frac=0.01)`: GASTON DP, tail domains merged.
- `choose_k(d, Y, k_max=10)` → (K, curve): K at the kneedle elbow of `dp_error_curve`; `kneedle(u)`.
- `boundaries(d, lab)`: isodepth values where the label changes.
- `isodepth_bins(d, n_bins)`: equal-width bins, as in GASTON.

## saturn.piecewise

- Fitters. Each returns `(slope, inter)` shaped (features, domains), with slope = 0 wherever the gate
  (p < 0.1) rejects it:
  - `fit_poisson(counts, lib, lab, d, t=0.1)`
  - `fit_gauss(Y, lab, d, t=0.1)`
  - `fit_binomial(S, N, lab, d, t=0.1)`
- `n_sloped(slope)`: the number of domains in which each feature keeps a slope.
- `predict(slope, inter, lab, d, link)`: the fitted per-domain function at every spot.
- `binned_r2(y, d, lab, slope_j, inter_j, n_bins, lib, link)`: R² against the binned observed values.
- `dispersion(counts, lib)`: a quick over-dispersion check.

## saturn.annotate

| function | purpose |
|---|---|
| `annotate_counts(Mat, feats, lab, d, modality, depth, cats, conts, gtf=None, labels=None)` | domain table for RNA / ATAC |
| `annotate_continuous(V, names, lab, d, modality, ...)` | domain table for continuous modalities |
| `domain_table(...)` | the generic table builder |
| `poisson_markers`, `continuous_markers` | one-vs-rest markers |
| `screen(values, names, lab, d, kind, lib, trials)` | multi-domain function screen → (table, slope, inter, candidate idx) |
| `gradient_count(Mat, lib, lab, d, n_top=500)` | sloped features per domain |
| `nearest_gene(regions, gtf)`, `peak_class(dist, half_width)`, `peaks_to_genes(A, near_g)`, `all_tss(gtf)`, `gene_lengths(gtf)` | genomic context |

## saturn.crossterm

| function | purpose |
|---|---|
| `spot_decomposition(model, coords_z, coords, targets, i_rna, i_x, signs, per_axis_sd=False)` | per-spot ρ, gradients, misalignment, rate difference, local gain, cross share, and a summary |
| `gene_cross_test(counts, genes, rho_rna, rho_x, min_det=0.05)` | per-gene own vs own + cross test → (table, cross profiles, genes) |
| `cis_windows(atac, peak_names, genes, gtf, window=50000)` | ±window accessibility per gene |
| `cis_concordance(prof, prof_genes, W, W_genes, rho_rna, rho_x)` | cis concordance vs random-window control |
| `steepness(T)` | per-spot steepness of the X axis over the RNA axis |
| `steepness_test(T, nrot=200)` | gain vs steepness, with a rotation null |
| `gene_length_test(G, lengths)` | gene length vs cross gain |
| `own_vs_cross(Y, d_own, lab_own, d_cross, lab_cross)` | per-feature commonality: R² own / cross / joint, unique and shared parts, per-domain R² |
| `rank_resid(a, b)` | ranks of a with the ranks of b regressed out (depth-controlled Spearman) |
| `hinge`, `fit_split`, `bh`, `logcpm`, `rotate`, `corr_cols` | building blocks |

## saturn.mechanism

| function | purpose |
|---|---|
| `absorption(Y, genes, rho_rna, rho_x, xy, blocks, cis=None, n_rot=5)` | absorbed share per block per gene |
| `drivers(Y, genes, rho_rna, rho_x, xy, tables, n=3)` | the top correlates of each gene's cross profile |
| `boundary_switch(tables, xy, steep)` | the features that change most where X is steep |
| `smooth`, `pcs`, `local_grad`, `r2`, `dR2` | building blocks |

## saturn.transition

| function | purpose |
|---|---|
| `transect(xy, rho_x, steep, top_q=0.8, eps=1.6)` | the steep zone, the band and the coordinate s |
| `transition_table(F, tr, n_boot=200, n_bin=18)` | per-feature s0, w, R²_sig, R²_bump, peak, steepest point and CI, transient flag; binned profiles |
| `fit_sigmoid_bump`, `steepest`, `grad_vec` | building blocks |

## saturn.gradient

| function | purpose |
|---|---|
| `rho_gradient(model, coords, knn=0, per_axis_scaling=False)` | (N, M, 2) d ρ / d(x, y) by autograd. The chain rule goes through `zscore_coords` (one scalar scale); `per_axis_scaling=True` reproduces the earlier per-axis-scaled Fig 2i run |
| `frames(G)` | unit vectors along / across ρ, and \|∇ρ\| |
| `spatial_gradients(V, xy, k=12)` | (N, F, 2) local-plane gradients of every feature |
| `along_across(Gf, G_rho)` | feature gradient components along and across ρ |

## saturn.cis

| function | purpose |
|---|---|
| `tss_table(gtf)`, `peak_positions(names)` | TSS per gene; chromosome and midpoint per peak |
| `cis_pairs(gene_idx, genes, peaks, peak_counts, tss, window=1e6, max_peaks=25, distal_min=None)` | cis peaks per gene (and distal background peaks) |
| `smoothed_logcpm(X, cols, xy, k=15)` | k-NN smoothed, z-scored log1p CP10k |
| `function_pairs(G, P, d, lab, pairs)` | co-fitted gene–peak pairs per domain: slopes, speed ratio, concordance, curve similarity, lag |
| `gradient_cosine(gG, gP, ga, pb, lab)` | magnitude-weighted cos between gene and peak spatial gradients, per domain |
| `along_across_corr(gG, gP, G_rho, ga, pb, lab)` | gene–peak correlation of gradient components along vs across ρ |
| `curves`, `best_lag_many`, `zz` | building blocks |

## saturn.tfactivity

| function | purpose |
|---|---|
| `jaspar_pwms(tfs)`, `peak_sequences(fasta, peaks, idx)`, `scan_motifs(seqs, pwm, q=0.99)` | motif hits per peak (both strands) |
| `peak_gc(fasta, peaks, valid)`, `background_sets(gc, acc, hits)` | chromVAR's GC- and accessibility-matched backgrounds |
| `deviations_matched(M, hits, BG)`, `deviations_fast(M, hits)` | chromVAR-style z per row (bins or spots) |
| `rho_bins(d, lab, n_bins=60)`, `log_accessibility(A)` | equal-count isodepth bins; depth-normalised log accessibility |
| `rho_binned_activity(A, d, lab, hits, tfs, BG)` | TF activity as a per-domain function of ρ → (fits, bins, groups) |

## saturn.plotting

- `spatial(ax, xy, val, ...)`: one spot map. Markers are sized to tile the spot grid; continuous values use
  `saturn_div` by default, and `discrete=True` uses the theme's categorical colours.
- `spot_size(ax, xy)`: the marker area and shape that `spatial` uses.
- `function_row(axes5, ...)`: one feature drawn into the five columns.
- `function_figure(xy, d, lab, Ymat, slope, inter, names, idx, ...)`: the full 5-column figure.

## saturn.theme

- `set_theme(base_pt=7)`: the matplotlib settings of the paper figures.
- `CMAP`: colormap name per data type: `score` and isodepth `saturn_div`, `expression` `saturn_seq_r`, `pseudotime`
  `saturn_seq`, `recovery` and gains `saturn_mono`, `cluster` `saturn_cat`.
- `MODALITY`: one colour per modality (RNA navy, ATAC ochre, Protein plum, Isoform blue-grey, Hi-C rust).
- `ANCHOR`, `CATEGORICAL`, `CMAPS`: the theme colours and the registered colormaps.

## saturn.sim

`simulate(SimConfig(...))` returns `SimData` with these fields:
- `coords`;
- `d`: the true axes;
- `domains`;
- `counts_R`, `counts_A`;
- `Y_R`, `Y_A`: the targets;
- `parts`;
- `cross_share`.

Knobs: `axis_angle_deg`, `cross_RA`, `cross_AR`, `coupled_frac`, `jump_frac`, depths.
