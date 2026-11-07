#!/usr/bin/env bash
# Execute one notebook against a workspace, compare with published results, keep outputs in the .ipynb.
#   SATURN_ROOT=... SATURN_PUBLISHED=... bash reproducibility/test_nb.sh fig4_merfish_isoform
set -eo pipefail
cd "$(dirname "$0")"
${PYTHON:-python} build.py "src/$1.py"
exec ${PYTHON:-python} -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 "$1.ipynb"
