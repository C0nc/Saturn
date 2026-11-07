# API

```python
import saturn
```

Start with the [function overview](overview.md). Each module has its own page.

## Package layout

```
src/saturn/
  core.py         Saturn model + train_saturn (3 stages: warm-up, cross on, held-out restarts)
  preprocess.py   per-modality targets, global scaling, coordinate scaling
  fit.py          fit (best-of-seeds), SaturnFit, decompose, r2_split, orient, heldout_r2, load_model
  domains.py      GASTON DP domains per axis, K at the DP-error elbow, boundaries, isodepth bins
  piecewise.py    per-domain Poisson / Gaussian / binomial fits behind GASTON's LRT gate
  annotate.py     domain tables, Poisson-z and Welch-t markers, nearest genes, feature screens
  crossterm.py    local gain, gradients, per-gene cross test, cis windows, steepness, own vs cross
  mechanism.py    absorption by cis / TF / composition blocks, drivers, boundary switches
  transition.py   steep-zone transect, sigmoid / bump switch points, bootstrap CIs
  gradient.py     isodepth gradients (autograd), feature gradients, along / across ρ
  cis.py          gene vs cis-peak function pairs, gradient agreement along / across ρ
  tfactivity.py   chromVAR-style TF activity on isodepth bins and per spot
  plotting.py     domain maps and the 5-column function figure
  sim.py          two-modality simulator of the generative model
```


```{toctree}
:hidden:

overview
preprocess
core
fit
domains
piecewise
annotate
crossterm
mechanism
transition
gradient
cis
tfactivity
plotting
sim
repro
```
