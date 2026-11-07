"""Run melanoma_tissue_prep.py, unchanged, but write its targets to results/real_saturn/target_recheck/.

The notebook fits on the published targets_melanoma_tissue.npz, a declared input that is a symlink to the lab file, so the script
must not write to its normal path. Everything in the target reproduces except Y_RNA: GLM-PCA starts from np.random without a seed.
"""
import os, runpy, sys

S = f"{os.environ['SATURN_ROOT']}/Hole/gaston_mix_mm/scripts"
sys.path.insert(0, S)
import prep_real2

prep_real2.OUT = f"{os.path.dirname(prep_real2.OUT)}/real_saturn/target_recheck"
os.makedirs(prep_real2.OUT, exist_ok=True)
runpy.run_path(f"{S}/melanoma_tissue_prep.py", run_name="__main__")
