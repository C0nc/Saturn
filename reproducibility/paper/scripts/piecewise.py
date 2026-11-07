"""Per-domain piecewise fits behind GASTON's LRT gate -- now provided by the Saturn package.

Kept as a module so existing `from piecewise import fit_gauss` imports keep working; the code lives in
saturn.piecewise (saturn_repo/src). See that module for the fitters and their gate.
"""
import sys
sys.path.insert(0, "/mnt/HDD2B/Users/yancui/saturn_repo/src")
from saturn.piecewise import (_gate, fit_gauss, fit_binomial, fit_poisson, dispersion, n_sloped,  # noqa: F401
                              predict, binned_r2, T_LRT)
