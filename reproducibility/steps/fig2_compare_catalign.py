"""DBiT spots on CODEX: this workspace vs the published csv, in CODEX level-0 px (0.507 um/px; one DBiT spot is about 62 px)."""
import os
import numpy as np, pandas as pd
ROOT, PUB = os.environ["SATURN_ROOT"], os.environ.get("SATURN_PUBLISHED")
if not PUB:
    print("SATURN_PUBLISHED not set: no comparison"); raise SystemExit
rel = "catalign_lpc10/results/LPC10S1_wm/LPC10S1_spots_in_codex.csv"
a, b = pd.read_csv(f"{ROOT}/{rel}", index_col=0), pd.read_csv(f"{PUB}/{rel}", index_col=0).loc[lambda d: d.index]
b = b.loc[a.index]; c = ["codex_lvl0_x", "codex_lvl0_y"]
d = np.hypot(*(a[c].values - b[c].values).T)
print(f"spots {len(a)}; displacement vs published px: median {np.median(d):.3g}, max {d.max():.3g}")
