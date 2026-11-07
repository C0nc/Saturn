"""Step 00 — CODEX overview: pull pyramid level 3 (~8x downsample) of the LPC10 S1 qptiff and
save percentile-normalised channels used by Fig 2a.

Recovered verbatim (logic unchanged) from the interactive session of 2026-09-20 that produced
external_data/codex_lpc/overview_LPC10_S1_lvl3.npz. Only the paths were made absolute.
"""
import json, os
import numpy as np, tifffile

D = "<workspace>/external_data/codex_lpc"


def main():
    bm = json.load(open(f"{D}/channels_LPC10_S1.json"))
    A = tifffile.TiffFile(f"{D}/Codex_LPC10_S1.qptiff").series[0].levels[3].asarray()   # 1890 x 2520
    print("overview array", A.shape, A.dtype)
    ch = {b: i for i, b in enumerate(bm)}

    def n(c, lo=1, hi=99.7):
        x = A[ch[c]].astype(float)
        a, b = np.percentile(x, [lo, hi])
        return np.clip((x - a) / max(b - a, 1e-9), 0, 1)

    dapi, mbp, cd11c, iba1, cd11b = [n(k) for k in ["DAPI", "MBP", "CD11c", "IBA1", "CD11b"]]
    np.savez_compressed(f"{D}/overview_LPC10_S1_lvl3.npz", DAPI=dapi, MBP=mbp, CD11c=cd11c, IBA1=iba1, CD11b=cd11b)
    print(f"-> {D}/overview_LPC10_S1_lvl3.npz")


if __name__ == "__main__":
    main()
