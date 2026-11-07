"""Helpers shared by the manuscript notebooks.

Set SATURN_ROOT to the workspace built by setup_workspace.py. Every step runs in <root>/Hole/gaston_mix_mm with a clean
thread environment and is skipped when its outputs already exist, so a notebook can start from saved results or from
nothing. FORCE=1 reruns everything. Set SATURN_PUBLISHED to a folder with the published results to compare against them.
"""
import json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ.get("SATURN_ROOT", "~/saturn_workspace")).expanduser().resolve()
HOLE = ROOT / "Hole/gaston_mix_mm"; RES = HOLE / "results/real_saturn"; PF = ROOT / "Saturn/published_figures"
PUB = Path(os.environ["SATURN_PUBLISHED"]) if os.environ.get("SATURN_PUBLISHED") else None
FORCE = os.environ.get("FORCE") == "1"
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "SAT_THREADS", "SAT_DEVICE")


def init(fig):
    """Check the workspace and the figure's input files."""
    if not (ROOT / ".saturn_workspace").exists():
        raise SystemExit(f"{ROOT} is not a workspace: run notebooks/setup_workspace.py --root {ROOT}")
    M = json.load(open(Path(__file__).resolve().parent / "inputs" / f"{fig}.json"))
    miss = [i for i in M if not (ROOT / i["path"]).exists()]
    for i in M: print(("ok      " if (ROOT / i["path"]).exists() else "MISSING ") + i["path"] + "  [" + i.get("source", "") + "]")
    if miss: raise SystemExit(f"{len(miss)} input files are missing")


def mkdirs(*rel):
    """Create output folders the scripts expect to exist (paths relative to Hole/gaston_mix_mm)."""
    for r in rel: (HOLE / r).mkdir(parents=True, exist_ok=True)


def step(name, cmd, outputs=(), threads=None, omp=None, device=None, env=None, cwd=None, extra_env=None):
    """Run one shell command unless all `outputs` exist. threads = SAT_THREADS, omp = OMP_NUM_THREADS (None = unset),
    env = conda env for this step (default: the notebook's own). Fits depend on these settings: see each notebook."""
    outs = [HOLE / o if not str(o).startswith("/") else Path(o) for o in outputs]
    if outs and not FORCE and all(o.exists() for o in outs):
        print(f"[skip] {name}: outputs exist"); return
    e = {k: v for k, v in os.environ.items() if k not in THREAD_VARS}
    e.update(MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    if threads: e["SAT_THREADS"] = str(threads)
    if omp: e["OMP_NUM_THREADS"] = str(omp)
    if device: e["SAT_DEVICE"] = device
    e.update(extra_env or {})
    py = sys.executable if env is None else env_python(env)
    full = cmd.replace("python ", f"{py} ", 1) if cmd.startswith("python ") else cmd
    log = HOLE / "logs" / (name.replace(" ", "_").replace("/", "_") + ".log"); log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); print(f"[run ] {name}: {cmd}", flush=True)
    with open(log, "w") as f:
        r = subprocess.run(full, shell=True, cwd=cwd or HOLE, env=e, stdout=f, stderr=subprocess.STDOUT, executable="/bin/bash")
    tail = "".join(open(log, errors="replace").readlines()[-6:])
    if r.returncode: raise RuntimeError(f"step '{name}' failed ({r.returncode}); log {log}\n{tail}")
    bad = [str(o) for o in outs if not o.exists()]
    if bad: raise RuntimeError(f"step '{name}' finished without writing {bad}; log {log}")
    print(f"[done] {name}: {(time.time() - t0) / 60:.1f} min", flush=True)


def env_python(env):
    """Python of another conda env (SATURN_ENV_<NAME> overrides)."""
    p = os.environ.get(f"SATURN_ENV_{env.upper()}")
    if p: return p
    base = Path(sys.executable).resolve().parents[2]
    return str(base / env / "bin/python")


def compare_axes(ds, cols=None):
    """Published vs this workspace's isodepths: max absolute difference and |Pearson r| per axis."""
    if PUB is None: print("SATURN_PUBLISHED not set: no comparison"); return None
    pub = PUB / "Hole/gaston_mix_mm/results/real_saturn" / ds / "axes.csv"
    a = pd.read_csv(RES / ds / "axes.csv", index_col=0); b = pd.read_csv(pub, index_col=0).loc[a.index]
    rows = [dict(dataset=ds, axis=c, max_abs_diff=float(np.abs(a[c] - b[c]).max()), abs_r=float(abs(np.corrcoef(a[c], b[c])[0, 1])))
            for c in (cols or [c for c in a if c.startswith("d_")])]
    T = pd.DataFrame(rows); print(T.to_string(index=False)); return T


def compare_table(rel, key=None, tol=1e-6):
    """Published vs this workspace for one CSV under results/real_saturn: same shape, largest numeric difference."""
    if PUB is None: print("SATURN_PUBLISHED not set: no comparison"); return None
    a = pd.read_csv(RES / rel); b = pd.read_csv(PUB / "Hole/gaston_mix_mm/results/real_saturn" / rel)
    if key: a, b = a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    same = a.shape == b.shape; d = np.nan
    if same:
        n = [c for c in a.select_dtypes("number") if c in b]
        d = float(np.nanmax(np.abs(a[n].values - b[n].values))) if n else 0.0
    print(f"{rel}: shape {a.shape} vs {b.shape}; max numeric diff {d:.3g}; {'MATCH' if same and d <= tol else 'DIFFERS'}")
    return dict(file=rel, same_shape=same, max_diff=d)


def show(*paths, width=900):
    """Display panels from the workspace (paths relative to published_figures)."""
    from IPython.display import Image, display
    for p in paths: display(Image(filename=str(PF / p), width=width))
