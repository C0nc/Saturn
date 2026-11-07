"""Fill a tutorial workspace folder with links to the files listed in workspace_files.json.

    python docs/tutorials/make_workspace.py <tutorial> <data root> <workspace>

<data root> is the folder that holds the files at the 'from' paths. Every tutorial then reads its inputs as
<workspace>/<name>, with the published results under <workspace>/published/.
"""
import json, os, sys
from pathlib import Path

name, root, work = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
missing = []
for f in json.load(open(Path(__file__).with_name("workspace_files.json")))[name]:
    src, dst = root / f["from"], work / f["name"]
    if not src.exists():
        missing.append(f["from"]); continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src.resolve(), dst)
print(f"{name}: linked into {work}" + (f"; missing {missing}" if missing else ""))
sys.exit(1 if missing else 0)
