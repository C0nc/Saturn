"""DBiT spots on CODEX: the coordinate part of catalign_lpc10/scripts/05_apply_codex_per_spot.py.

Fig. 2a needs only LPC10S1_spots_in_codex.csv (codex_lvl0_x, codex_lvl0_y, in D order). The rest of 05 (per-spot CODEX protein,
join to the Saturn FOV h5mu) feeds other analyses and is left out. The transform is the published one.

    fig2_catalign_spots.py config   write the Catalign config for this workspace
    fig2_catalign_spots.py spots    apply the registration to every spot
"""
import os, sys
from pathlib import Path
import numpy as np, pandas as pd

H = os.environ["SATURN_ROOT"]
C = f"{H}/catalign_lpc10"; RES = f"{C}/results/LPC10S1_wm"
POS = f"{H}/external_data/dbit_lpc10_spatial/GSM9247593_07_LPC10S1_tissue_positions_list.csv"

if sys.argv[1] == "config":
    t = Path(__file__).with_name("fig2_catalign_config.yaml").read_text().replace("POSITIONS", POS)
    Path(f"{C}/config_wm.yaml").write_text(t); print(t)
else:
    from Catalign import Registration
    reg = Registration.load(f"{RES}/registration.json")
    p = pd.read_csv(POS, header=None, names=["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]).set_index("barcode")
    chip_xy = p[["pxl_col", "pxl_row"]].to_numpy(float)
    l1 = reg.chip_to_histology(chip_xy)
    pd.DataFrame(np.c_[chip_xy, l1 * 2.0], index=p.index, columns=["chip_x", "chip_y", "codex_lvl0_x", "codex_lvl0_y"]).join(
        p[["in_tissue"]]).to_csv(f"{RES}/LPC10S1_spots_in_codex.csv")
    print("spots", len(p))
