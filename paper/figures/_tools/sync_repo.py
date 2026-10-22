"""Refresh a figure reproduction repo (figN/repo) from the live code, pointed at the Saturn package.

    python published_figures/_tools/sync_repo.py fig2 [--dry]

What it does
  scripts/   every pipeline module the figure needs (run_figN.sh steps + everything the figure builders
             import, followed recursively), copied from Hole/gaston_mix_mm/scripts
  figure/    the current figure builders of this figure (+ fig2_lpc.py, whose shared panel builders every
             figure imports) and the style module (_style/paper_style.py, saturn_theme.py)
  sim/       metrics_v2.py (imported by the fitter)
  saturn_src/  REMOVED: the repo now runs on the Saturn package (github.com/C0nc/Saturn), located through
             $SATURN_SRC (default: the local clone's src/), and records its git commit in SATURN_VERSION

Paths are rewritten so nothing falls through to the live folders:
  Hole/gaston_mix_mm/scripts -> repo/scripts      Saturn/published_figures/figN -> repo/figure
  Hole/gaston_mix_mm/sim     -> repo/sim          Saturn/published_figures/_style -> repo/figure/_style
  saturn_repo/src            -> $SATURN_SRC
"""
import glob, os, re, shutil, subprocess, sys

H = "/mnt/HDD2B/Users/yancui"
SCRIPTS = f"{H}/Hole/gaston_mix_mm/scripts"
PF = f"{H}/Saturn/published_figures"
PKG = f"{H}/saturn_repo"
HERE = '__import__("os").path.dirname(__import__("os").path.abspath(__file__))'
UP = lambda sub: f'__import__("os").path.join({HERE}, "..", "{sub}")'
SATURN = f'__import__("os").environ.get("SATURN_SRC", "{PKG}/src")'
FIGURES = {"fig2": ["fig2_lpc.py", "fig2_lpc_v2.py"],
           "fig3": ["fig3_melanoma.py", "fig3_melanoma_v2.py", "fig3_parts.py"]}


def crawl(seeds):
    local = {os.path.basename(p)[:-3] for p in glob.glob(f"{SCRIPTS}/*.py")}
    seen, st = set(), [s for s in seeds if s in local]
    while st:
        m = st.pop()
        if m in seen:
            continue
        seen.add(m)
        t = open(f"{SCRIPTS}/{m}.py").read()
        st += [x for x in re.findall(r"(?:^|\n)\s*(?:from|import)\s+([A-Za-z0-9_]+)", t) if x in local and x not in seen]
    return seen


def rewrite(t, where):
    """where = 'scripts' or 'figure'. Only sys.path entries are rewritten; output paths stay canonical."""
    def target(path):
        if path == f"{PKG}/src":
            return SATURN
        if path == SCRIPTS:
            return HERE if where == "scripts" else UP("scripts")
        if path == f"{H}/Hole/gaston_mix_mm/sim":
            return UP("sim")
        if path == f"{PF}/_style":
            return f'__import__("os").path.join({HERE}, "_style")' if where == "figure" else UP("figure/_style")
        if re.fullmatch(rf"{PF}/fig\d", path):
            return HERE if where == "figure" else UP("figure")
        return None

    def sub(m):
        new = target(m.group(2))
        return m.group(0) if new is None else f"{m.group(1)}{new})"
    t = re.sub(r'(sys\.path\.insert\(0, )"([^"]+)"\)', sub, t)
    return t.replace('os.path.join(ROOT, "sim")', UP("sim"))


def main(fig, dry=False):
    repo = f"{PF}/{fig}/repo"
    run = open(f"{repo}/run_{fig}.sh").read()
    seeds = set(re.findall(r"scripts/([A-Za-z0-9_]+)\.py", run))
    figs = ["fig2_lpc.py"] + FIGURES[fig]
    for f in figs:
        t = open(f"{PF}/{f[:4]}/{f}").read()
        seeds |= set(re.findall(r"([A-Za-z0-9_]+)\.py", t)) | set(re.findall(r"(?:from|import)\s+([A-Za-z0-9_]+)", t))
    need = sorted(crawl(seeds))
    print(f"{fig}: {len(need)} scripts, {len(set(figs))} figure modules")
    if dry:
        print(need); return
    shutil.rmtree(f"{repo}/scripts"); os.makedirs(f"{repo}/scripts")
    for m in need:
        open(f"{repo}/scripts/{m}.py", "w").write(rewrite(open(f"{SCRIPTS}/{m}.py").read(), "scripts"))
    shutil.rmtree(f"{repo}/figure", ignore_errors=True); os.makedirs(f"{repo}/figure/_style")
    for f in dict.fromkeys(figs):
        open(f"{repo}/figure/{f}", "w").write(rewrite(open(f"{PF}/{f[:4]}/{f}").read(), "figure"))
    for f in ("__init__.py", "paper_style.py", "saturn_theme.py"):
        shutil.copy(f"{PF}/_style/{f}", f"{repo}/figure/_style/{f}")
    if os.path.isdir(f"{PF}/_palettes"):
        shutil.copytree(f"{PF}/_palettes", f"{repo}/figure/_palettes", dirs_exist_ok=True)
    os.makedirs(f"{repo}/sim", exist_ok=True)
    shutil.copy(f"{H}/Hole/gaston_mix_mm/sim/metrics_v2.py", f"{repo}/sim/metrics_v2.py")
    shutil.rmtree(f"{repo}/saturn_src", ignore_errors=True)
    commit = subprocess.run(["git", "-C", PKG, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    open(f"{repo}/SATURN_VERSION", "w").write(f"github.com/C0nc/Saturn @ {commit}\n")
    left = subprocess.run(["grep", "-rlE", 'sys.path.insert\\(0, "[^"]*(Hole/gaston_mix_mm|published_figures/fig|Saturn/src)',
                           f"{repo}/scripts", f"{repo}/figure"], capture_output=True, text=True).stdout
    print("modules importing from a live code folder:", left.split() or "none")


if __name__ == "__main__":
    main(sys.argv[1], dry="--dry" in sys.argv)
