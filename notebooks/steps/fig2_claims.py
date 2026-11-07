"""The numbers the Fig. 2 panels print, from the figure's own panel functions, for the workspace in SATURN_ROOT. Writes <results>/fig2_claims.json."""
import json, os, sys
ROOT = os.environ["SATURN_ROOT"]
sys.path.insert(0, f"{ROOT}/Saturn/published_figures/fig2")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
import fig2_lpc as F2
from paper_style import apply
apply(base_pt=6.5)
D = F2.load()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)): return round(float(x), 4)
    if isinstance(x, np.integer): return int(x)
    return x if isinstance(x, (int, str, bool, type(None))) else str(x)


ax = plt.figure().add_subplot()
auc, med = F2.panel_layers(ax, D)
out = dict(R2_whole=D["summary"]["R2_whole"], coupling=D["summary"]["coupling_best"], axis_corr=D["summary"]["axis_axis_corr"][0][1],
           domain_sizes={m: np.bincount(D["lab"][m]).tolist() for m in D["lab"]}, c_auc=auc, c_median_percentile=med)
for k, fn in (("h", F2.panel_cosign), ("i", F2.panel_gradient), ("j", F2.panel_infl)):
    out[k] = fn(plt.figure().add_subplot())
for m, x, g in [("RNA", "RNA", "Trem2"), ("RNA", "RNA", "Mbp")]:
    out[f"e_{g}_R2"] = F2.feature_function(D, m, g)["r2"]
out = clean(out)
json.dump(out, open(f"{ROOT}/Hole/gaston_mix_mm/results/real_saturn/fig2_claims.json", "w"), indent=1)
print(json.dumps(out, indent=1))
