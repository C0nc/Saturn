"""Step 01 — CODEX CD11c+ foci inside white matter and the FOV box that covers the two largest.

Recovered verbatim (logic unchanged) from the interactive session of 2026-09-20 that produced
external_data/codex_lpc/cd11c_hotspots_lvl3.json and fov_lesion_plus_SM.json. Only the paths
were made absolute and the diagnostic PNG dropped.

NOTE: this CODEX box is drawn in Fig 2a as CONTEXT. The spots Saturn was fitted on were
selected on the DBiT data itself (steps s02-s03), not from this box.
"""
import json, re
import numpy as np, tifffile
from scipy import ndimage as ndi

D = "<workspace>/external_data/codex_lpc"
SC = 8                                    # level 3 -> level 0


def main():
    z = np.load(f"{D}/overview_LPC10_S1_lvl3.npz")
    cd11c, mbp = z["CD11c"], z["MBP"]
    desc = tifffile.TiffFile(f"{D}/Codex_LPC10_S1.qptiff").series[0].pages[0].description
    m = re.search(r"<ScaleFactor>([\d.]+)</ScaleFactor>", desc)
    um_per_px0 = float(m.group(1)) / 2 if m else 0.5
    wm = mbp > np.percentile(mbp[mbp > 0.02], 60)                     # white-matter mask
    hot = (cd11c > np.percentile(cd11c, 99.3)) & ndi.binary_dilation(wm, np.ones((9, 9)))
    hot = ndi.binary_closing(hot, np.ones((7, 7)))
    lab, n = ndi.label(hot)
    sizes = ndi.sum(hot, lab, range(1, n + 1))
    cands = []
    for k in np.argsort(-sizes)[:6]:
        i = k + 1; msk = lab == i; ys, xs = np.where(msk)
        cands.append(dict(id=int(i), size=int(sizes[k]), cy=float(ys.mean()), cx=float(xs.mean()),
                          y0=int(ys.min()), y1=int(ys.max()), x0=int(xs.min()), x1=int(xs.max()),
                          cd11c=float(cd11c[msk].mean())))
    json.dump(cands, open(f"{D}/cd11c_hotspots_lvl3.json", "w"), indent=1)
    top2 = cands[:2]; pad = 40
    y0 = max(min(c["y0"] for c in top2) - pad, 0); y1 = min(max(c["y1"] for c in top2) + pad, cd11c.shape[0] - 1)
    x0 = max(min(c["x0"] for c in top2) - pad, 0); x1 = min(max(c["x1"] for c in top2) + pad, cd11c.shape[1] - 1)
    json.dump(dict(level3=[y0, y1, x0, x1], level0=[y0 * SC, y1 * SC, x0 * SC, x1 * SC], um_per_px0=um_per_px0),
              open(f"{D}/fov_lesion_plus_SM.json", "w"), indent=1)
    print(f"CODEX FOV (level 3) y {y0}-{y1}, x {x0}-{x1}; {len(cands)} CD11c+ foci")


if __name__ == "__main__":
    main()
