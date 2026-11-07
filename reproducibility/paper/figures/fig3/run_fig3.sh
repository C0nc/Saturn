#!/usr/bin/env bash
# Reproduce Figure 3 (melanoma RNA + ATAC):
#   targets -> Saturn fit (seed 42) -> post-hoc -> H&E registration -> iStar -> figure + panels.
#
#   bash run_fig3.sh            # everything
#   bash run_fig3.sh post       # reuse the fit; rerun post-hoc, registration, iStar, figure
#   bash run_fig3.sh istar      # registration + iStar + figure only
#   bash run_fig3.sh figure     # figure only
#
# Starts from three upstream objects built by other pipelines (see upstream/ and README):
#   results/melanoma/melanoma_rna.h5ad, results/melanoma/atac_snap.h5ad,
#   Saturn/paper_figures/figure2/melanoma_tumor_id/per_spot_tumor_call.csv
#
# WARNING: steps write to the canonical results folders the published figure was made from.
# The Saturn fit reproduces bit-for-bit; iStar (GPU) does not -- see REPRO_CHECK.md.
set -eo pipefail
cd "$(dirname "$0")"; REPO=$(pwd)
source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"; conda activate spatialmultivelo
# Saturn package (github.com/C0nc/Saturn) at the commit in SATURN_VERSION; scripts find it through $SATURN_SRC
export SATURN_SRC="${SATURN_SRC:-/mnt/HDD2B/Users/yancui/saturn_repo/src}"
want=$(awk '{print $3}' SATURN_VERSION); have=$(git -C "$SATURN_SRC/.." rev-parse HEAD 2>/dev/null || echo unknown)
[[ "$want" == "$have" ]] || echo "WARNING: Saturn at $have, this repo was checked with $want"
RS=/mnt/HDD2B/Users/yancui/Hole/gaston_mix_mm/results/real_saturn
MODE="${1:-all}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1        # HIPT checkpoints (official weights) hold numpy scalars

if [[ "$MODE" == "all" ]]; then
  python scripts/melanoma_tissue_prep.py         # in-tissue QC; RNA GLM-PCA (log depth regressed), ATAC spectral
  # EXACT published environment: CPU, torch threads 8, OMP_NUM_THREADS unset (other settings give a different fit)
  env -u OMP_NUM_THREADS SAT_THREADS=8 SAT_DEVICE=cpu python scripts/run_real2.py --dataset melanoma_tissue --seeds 1 --seed0 42
fi
if [[ "$MODE" == "all" || "$MODE" == "post" ]]; then
  RESCREEN=1 python scripts/marker_function_panels.py melanoma_tissue 8 8   # gene / peak screens  (Fig 3e R2)
  python scripts/saturn_own_vs_cross.py melanoma_tissue                     # own vs cross per feature (Fig 3c)
  python scripts/melanoma_chromatin_potential.py 30000                     # function-matched peaks (Fig 3f)
  python scripts/melanoma_chromatin_potential_dorc.py                      # DORC chromatin potential (Fig 3g)
  python scripts/mel_cellrank_kernels.py                                   # CellRank 2 kernels / fusion (Fig 3k-l)
  python scripts/mel_cellrank_arrows.py                                    # fate arrows on the tissue    (Fig 3k)
  python scripts/mel_gradient_transition.py                                # gradient transition zones   (Fig 3m-o)
  python scripts/mel_baseline_mellon.py                                    # Mellon low-density baseline (Fig 3n)
fi
if [[ "$MODE" == "all" || "$MODE" == "post" || "$MODE" == "istar" ]]; then
  REG=$RS/melanoma_tissue_istar/registration
  python scripts/melanoma_he_register.py $REG                             # brightfield -> adjacent H&E  (Fig 3a)
  python scripts/melanoma_istar_prep.py $REG/registration.json similarity             # iStar inputs, all spots
  python scripts/melanoma_istar_prep.py $REG/registration.json similarity --holdout 0.1  # 10% held out (seed 0)
  P=$RS/melanoma_tissue_istar/
  ( cd istar && python rescale.py $P --image --radius && python preprocess.py $P --image \
             && python extract_features.py $P --device=cuda )            # HIPT features on the 8 um superpixel grid
  for f in he.jpg he-scaled.jpg embeddings-hist.pickle; do                # held-out run reuses the same H&E + features
    ln -sf ${P}$f $RS/melanoma_tissue_istar_holdout10/$f; done; cp ${P}radius.txt $RS/melanoma_tissue_istar_holdout10/radius.txt
  python scripts/melanoma_istar_features.py                               # measured features -> feat / feat_holdout10
  for D in melanoma_tissue_istar_feat melanoma_tissue_istar_feat_holdout10; do
    ( cd istar && python impute.py $RS/$D/ --epochs=400 --device=cuda --shape=square ); done
  python scripts/melanoma_istar_boundary.py                               # tumour - stroma maps + sharpness (Fig 3h-j)
  python scripts/melanoma_istar_eval_features.py                          # per-feature held-out test (text / Supp)
fi
python figure/fig3_parts.py                                               # Fig 3 (a-o) parts 1-3 + one SVG per panel
