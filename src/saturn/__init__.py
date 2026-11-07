"""Saturn: interpretable per-modality isodepths for spatial multi-omics.

One isodepth per modality, M x M additive piecewise-linear pathways, fitted to per-modality
low-dimensional targets. See docs/analysis_pipeline.md.

  saturn.preprocess   per-modality targets (RNA GLM-PCA, ATAC LSI, Hi-C, protein, isoform usage)
                      and coordinate scaling
  saturn.core         the model (Saturn) and its 3-stage trainer (train_saturn)
  saturn.fit          best-of-seeds fitting, R2 split, coupling, held-out R2, save / load
  saturn.domains      GASTON DP domains along each axis, K chosen at the DP-error elbow
  saturn.piecewise    per-domain fits behind GASTON's LRT gate (Poisson / Gaussian / binomial)
  saturn.annotate     domain annotation tables, markers, nearest genes, multi-domain feature screens
  saturn.crossterm    what the cross pathways explain: local gain, per-gene test, cis windows, steepness
  saturn.mechanism    absorption of the cross signal by cis / TF / composition blocks; boundary switches
  saturn.transition   transect through the steep zone: switch points and transient states
  saturn.gradient     isodepth gradients (autograd), feature gradients, along / across decomposition
  saturn.cis          gene vs cis-peak function pairs, gradient agreement along / across rho
  saturn.tfactivity   chromVAR-style TF activity on isodepth bins and per spot
  saturn.plotting     domain maps and the 5-column function figure
  saturn.theme        the paper's colours and colormaps (saturn_div, saturn_seq, saturn_mono, saturn_cat)
  saturn.sim          two-modality simulator of Saturn's generative model
"""
__version__ = "1.0.0"

from .core import Saturn, train_saturn
from .preprocess import (rna_glmpc, rna_glmpc_real, atac_lsi, lsi_real, compartment_pca,
                         protein_clr_pca, isoform_usage_pca, atac_spectral, check_target,
                         scale_target, zscore_coords)
from .fit import fit, SaturnFit, decompose, r2_split, orient, load_model, heldout_r2
from . import domains, piecewise, annotate, crossterm, mechanism, transition, plotting, theme, sim, gradient, cis, tfactivity

__all__ = ["Saturn", "train_saturn", "fit", "SaturnFit", "decompose", "r2_split", "orient", "load_model",
           "heldout_r2", "rna_glmpc", "rna_glmpc_real", "atac_lsi", "lsi_real", "compartment_pca",
           "protein_clr_pca", "isoform_usage_pca", "atac_spectral", "check_target", "scale_target", "zscore_coords",
           "domains", "piecewise", "annotate", "crossterm", "mechanism", "transition", "plotting", "theme", "sim",
           "gradient", "cis", "tfactivity"]
