"""Fill a workspace folder with links to the files a figure notebook reads.

    python reproducibility/make_workspace.py <figure> <data root> <workspace>

<figure> names a file in reproducibility/workspace_files/ (e.g. fig4). <data root> is the folder that holds the files at
the 'from' paths. The notebook then reads its inputs as <workspace>/<name>; results made by tools outside Saturn
(other conda environments, GPU runs, live queries) sit under <workspace>/published/.
"""
import json, os, sys
from pathlib import Path

name, root, work = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
missing = []
for f in json.load(open(Path(__file__).parent / "workspace_files" / f"{name}.json")):
    src, dst = root / f["from"], work / f["name"]
    if not src.exists():
        missing.append(f["from"]); continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)
print(f"{name}: linked into {work}" + (f"; missing {missing}" if missing else ""))
sys.exit(1 if missing else 0)
