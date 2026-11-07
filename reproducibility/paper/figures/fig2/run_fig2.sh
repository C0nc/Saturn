#!/usr/bin/env bash
# Reproduce Figure 2 (LPC10 FOV) end to end: CODEX context -> FOV selection -> Saturn targets ->
# Saturn fit (seed 42) -> post-hoc analyses -> figure + per-panel SVG/PNG.
#
#   bash run_fig2.sh            # everything
#   bash run_fig2.sh post       # skip CODEX/FOV/prep/fit, rerun post-hoc + figure on the existing fit
#   bash run_fig2.sh figure     # figure only
#
# WARNING: steps write to the canonical results folder (see README "Where outputs go"),
# overwriting the files the published figure was made from. Back them up first if you need to diff.
set -euo pipefail
cd "$(dirname "$0")"
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate spatialmultivelo
# Saturn package (github.com/C0nc/Saturn) at the commit in SATURN_VERSION; scripts find it through $SATURN_SRC
export SATURN_SRC="${SATURN_SRC:-<workspace>/saturn_repo/src}"
want=$(awk '{print $3}' SATURN_VERSION); have=$(git -C "$SATURN_SRC/.." rev-parse HEAD 2>/dev/null || echo unknown)
[[ "$want" == "$have" ]] || echo "WARNING: Saturn at $have, this repo was checked with $want"
export SAT_DEVICE=cpu CUDA_VISIBLE_DEVICES=""
MODE="${1:-all}"

if [[ "$MODE" == "all" ]]; then
  python steps/s00_codex_overview.py         # CODEX qptiff level 3 -> normalised channels   (Fig 2a)
  python steps/s01_codex_hotspots_fov.py     # CODEX CD11c+ foci + context FOV box          (Fig 2a)
  python steps/s02_lpc10_hotspots.py         # DBiT microglia-score hotspots (lesion, distal WM focus)
  python steps/s03_lpc10_fov_select.py       # FOV mask Saturn is fitted on (cortex excluded)
  python scripts/lpc10_fov_prep.py           # targets: RNA GLM-PCA, ATAC 5 kb tiles -> spectral
  # EXACT published fit env: CPU, torch 8 threads, OMP 8 (other thread counts land in a different optimum)
  SAT_THREADS=8 OMP_NUM_THREADS=8 python scripts/run_real2.py --dataset lpc10_fov --seeds 1 --seed0 42
fi
if [[ "$MODE" == "all" || "$MODE" == "post" ]]; then
  # post-hoc steps run with OMP_NUM_THREADS unset, as published: BLAS thread count flips exact ties
  # (e.g. 342 of 175k curve lags in cis_function_pairs at 8 threads)
  unset OMP_NUM_THREADS
  # domain K per axis = results/real_saturn/domain_k_elbow/k_elbow.json (scripts/domain_k_elbow.py in the main
  # code; it needs the Fig 3/4/6 fits too, so it is an input here)
  python scripts/lpc10_fov_domain_annot.py                       # domain composition + markers (Supp table)
  RESCREEN=1 python scripts/marker_function_panels.py lpc10_fov 8 8   # gene / peak function screens (Fig 2e)
  python scripts/saturn_tf_activity_rho.py lpc10_fov             # rho-binned chromVAR, GC-matched   (Fig 2f)
  python scripts/saturn_tf_activity_spatial.py lpc10_fov         # per-spot TF activity               (Fig 2g)
  python scripts/saturn_cis_function_pairs.py lpc10_fov 1500     # cis slope-sign concordance         (Fig 2h)
  python scripts/saturn_cis_gradient_decomp.py lpc10_fov 1500    # gradient along vs across rho      (Fig 2i)
  python scripts/lpc10_fov_posthoc.py                            # inflammatory genes on both axes    (Fig 2j)
  python scripts/lpc10_seacells_cis_benchmark.py                 # SEACells metacells vs isodepth     (Fig 2k-l)
fi
python figure/fig2_lpc.py                                        # Fig 2 (a-l, via fig2_lpc_v2) + panels
