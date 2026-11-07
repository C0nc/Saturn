#!/usr/bin/env python
"""Build a Saturn manuscript workspace: the folder every notebook reads from and writes to.

    python reproducibility/setup_workspace.py --root ~/saturn_workspace [--figs fig4 fig5] [--link-from /path/to/data_root]

The manuscript code in reproducibility/paper/ keeps the lab's absolute paths (it documents how every number was made). This script
copies it into <root>, with the lab prefix replaced by <root>, in the same folder layout:
    <root>/Hole/gaston_mix_mm/{scripts,sim,results}    analysis code and all intermediate results
    <root>/Saturn/published_figures/{figN,_style,...}  figure builders and panels
    <root>/saturn_repo/src -> this repo's package
Nothing outside <root> is written. Input data must sit in <root> at the relative paths listed in inputs/figN.json;
--link-from symlinks them from a folder that has the same layout. Re-running refreshes code, and keeps results.
"""
import argparse, json, os, shutil
from pathlib import Path

LAB = "/mnt/HDD2B/Users/yancui"
REPO = Path(__file__).resolve().parent.parent
TEXT = {".py", ".sh", ".json", ".txt"}


def copy_tree(src, dst, root):
    for p in Path(src).rglob("*"):
        if p.is_dir() or "__pycache__" in p.parts: continue
        q = Path(dst) / p.relative_to(src); q.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix in TEXT:
            q.write_text(p.read_text(errors="surrogateescape").replace(LAB, str(root)), errors="surrogateescape")
        else:
            shutil.copy(p, q)


def inputs_of(figs):
    M = {p.stem: json.load(open(p)) for p in (REPO / "reproducibility/inputs").glob("*.json")}
    return {f: M[f] for f in (figs or M)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--figs", nargs="*")
    ap.add_argument("--link-from"); a = ap.parse_args()
    root = Path(a.root).expanduser().resolve(); root.mkdir(parents=True, exist_ok=True)
    if str(root).startswith(LAB + "/") and not (root / ".saturn_workspace").exists() and any(root.iterdir()):
        raise SystemExit(f"{root} is not empty and is not a workspace")
    (root / ".saturn_workspace").touch()
    H = root / "Hole/gaston_mix_mm"; PF = root / "Saturn/published_figures"
    copy_tree(REPO / "reproducibility/paper/scripts", H / "scripts", root); copy_tree(REPO / "reproducibility/paper/sim", H / "sim", root)
    copy_tree(REPO / "reproducibility/paper/figures", PF, root); copy_tree(REPO / "reproducibility/paper/style", PF / "_style", root)
    copy_tree(REPO / "reproducibility/paper/palettes", PF / "_palettes", root); copy_tree(REPO / "reproducibility/paper/config", H / "results/real_saturn", root)
    (H / "logs").mkdir(exist_ok=True)
    for link in (root / "saturn_repo/src", root / "Saturn/src"):       # both names the scripts use for the package
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists(): link.unlink()
        link.symlink_to(REPO / "src")
    miss = []
    for fig, items in inputs_of(a.figs).items():
        for it in items:
            q = root / it["path"]
            if a.link_from and not (q.exists() or q.is_symlink()):
                s = Path(a.link_from) / it["path"]
                if s.exists():
                    q.parent.mkdir(parents=True, exist_ok=True); q.symlink_to(s)
            if not q.exists(): miss.append((fig, it["path"], it.get("source", "")))
    print(f"workspace: {root}")
    for m in miss: print("MISSING INPUT", *m, sep="  ")
    print("all inputs present" if not miss else f"{len(miss)} inputs missing: see reproducibility/inputs/")


if __name__ == "__main__":
    main()
