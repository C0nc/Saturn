"""Number of GASTON domains per (dataset, axis): the elbow K from domain_k_elbow.py.

Every figure analysis that segments a rho axis asks kdom(ds, axis) instead of a fixed K.
Datasets not in the table fall back to 4.
"""
import json
from pathlib import Path

_R = Path("<workspace>/Hole/gaston_mix_mm/results/real_saturn")
_ELBOW = json.load(open(_R / "domain_k_elbow" / "k_elbow.json"))          # "fig|axis" -> K
FIG_OF = {"lpc10_fov": "fig2", "melanoma_tissue": "fig3", "mel_hic_fh_dr50_51": "fig6",
          "rtt_merfish_s5_iso100_sv_pca_shrunk": "fig4", "rtt_merfish_s5_iso100_sv_pca_shrunk_norestart_lc0.3": "fig4"}


def kdom(ds, axis, default=4):
    fig = FIG_OF.get(str(ds))
    return int(_ELBOW.get(f"{fig}|{axis}", default)) if fig else default
