#!/usr/bin/env bash
# Build a tutorial from docs/tutorials/_src/<name>.py, execute it in a working directory, drop progress-bar output, copy it back.
#   PYTHON=<python> bash docs/tutorials/run_tutorial.sh <name> <working directory>
# The tutorial reads its inputs from <working directory>/saturn_workspace (make_workspace.py links them there).
set -eo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; PY=${PYTHON:-python}; WD=$2; mkdir -p "$WD"
"$PY" "$HERE/../../reproducibility/build.py" "$HERE/_src/$1.py"
cp "$HERE/$1.ipynb" "$WD/"
cd "$WD"
PYTHONPATH="$HERE/../../src:$PYTHONPATH" "$PY" -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 "$1.ipynb"
"$PY" - "$1.ipynb" <<'E'
import json, sys
p = sys.argv[1]; nb = json.load(open(p))
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = [o for o in c["outputs"] if not (o["output_type"] == "stream" and o.get("name") == "stderr")]
json.dump(nb, open(p, "w"), indent=1)
E
cp "$1.ipynb" "$HERE/$1.ipynb"
