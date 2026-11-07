"""Fig 7: workspace vs published, for what nb.compare_table does not check: text columns, JSON summaries, npz arrays and figure pixels.

    fig7_compare.py tables | json | npz | pixels | stats | svg
"""
import glob, json, os, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(os.environ["SATURN_ROOT"]); PUB = Path(os.environ["SATURN_PUBLISHED"])
X = "Hole/gaston_mix_mm/results/real_saturn/saturn_cross_term"; PF = "Saturn/published_figures/fig7"
KEYS = ["lpc10", "melanoma", "merfish", "sprox", "hic"]; ATAC = ["lpc10", "melanoma"]


def flat(o, p=""):
    if isinstance(o, dict):
        for k, v in o.items(): yield from flat(v, f"{p}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o): yield from flat(v, f"{p}/{i}")
    else:
        yield p, o


def cmp_json(a, b, drop=()):
    A, B = dict(flat(a)), dict(flat(b))
    A = {k: v for k, v in A.items() if not any(k.startswith(d) for d in drop)}; B = {k: v for k, v in B.items() if not any(k.startswith(d) for d in drop)}
    miss = sorted(set(A) ^ set(B)); d, txt = 0.0, []
    for k in set(A) & set(B):
        x, y = A[k], B[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if x == x or y == y: e = abs(x - y); d = max(d, e if e == e else np.inf)
        elif x != y: txt.append(k)
    return len(A), d, txt, miss


def tables():
    rel = [f"{k}_{t}.csv" for k in KEYS for t in ("spots", "genes", "length")]
    rel += [f"{k}_{t}.csv" for k in ATAC for t in ("cis", "mech_genes", "mech_drivers", "mech_boundary", "transition_fits", "transition_profiles", "transition_spots")]
    rel += [f"steep_followup/{k}_zone_switch.csv" for k in ATAC] + ["steep_followup/lpc10_codex.csv"]
    for r in rel:
        a, b = pd.read_csv(ROOT / X / r), pd.read_csv(PUB / X / r)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            print(f"{r}: DIFFERS shape {a.shape} vs {b.shape}"); continue
        n = [c for c in a if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])]; t = [c for c in a if c not in n]
        v = a[n].values.astype(float) - b[n].values.astype(float)
        nan = int((np.isnan(a[n].values.astype(float)) != np.isnan(b[n].values.astype(float))).sum())
        wc = n[int(np.nanargmax(np.nanmax(np.abs(v), axis=0)))] if n and np.isfinite(v).any() else ""
        txt = sum(int((a[c].astype(str) != b[c].astype(str)).sum()) for c in t)
        print(f"{r}: {a.shape}; max|diff| {np.nanmax(np.abs(v)) if n else 0:.3g} ({wc}); NaN pattern diffs {nan}; text cells differing {txt}")


def jsons():
    rel = [f"{k}_summary.json" for k in KEYS] + [f"{k}_{t}.json" for k in ATAC for t in ("mech_summary", "transition_summary")] + ["fig7_steep.json", "steep_followup/followup.json"]
    for r in rel:
        n, d, t, m = cmp_json(json.load(open(ROOT / X / r)), json.load(open(PUB / X / r)))
        print(f"{r}: {n} values; max|diff| {d:.3g}; text differing {t}; keys in one only {m}")


def npzs():
    for r in [f"{k}_profiles.npz" for k in KEYS] + ["fig7_examples.npz"]:
        a, b = np.load(ROOT / X / r, allow_pickle=True), np.load(PUB / X / r, allow_pickle=True)
        out = []
        for k in sorted(set(a.files) | set(b.files)):
            if k not in a.files or k not in b.files: out.append(f"{k}: in one only"); continue
            x, y = a[k], b[k]
            if x.shape != y.shape: out.append(f"{k}: shape {x.shape} vs {y.shape}")
            elif x.dtype.kind in "USO": out.append(f"{k}: {'identical' if (x == y).all() else 'DIFFERS'}")
            else: out.append(f"{k}: {np.nanmax(np.abs(x.astype(float) - y.astype(float))):.3g}")
        print(r, "; ".join(out))


def pixels():
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    # S6a's slot is masked. Part 1 has a different canvas: without panel a the tight crop moves by a fraction of a pixel, so it is
    # compared right-aligned, and the svg check below is the exact test of panels b and c.
    masks = {"figS6_all_datasets": (0.0, 0.0, 0.52, 0.26)}
    names = ["fig7_part1", "fig7_part2", "fig7_part3", "figS6_all_datasets", "figS7_gradient", "figS8_steepness_checks",
             "panels/fig7g_what_atac_steep_zone_marks", "panels/fig7h_what_atac_cross_term_reflects", "panels/fig7i_steep_zone_transition", "panels/fig7gi_steep_zone_block"]
    for n in names:
        a = np.asarray(Image.open(ROOT / PF / f"{n}.png").convert("RGB")).astype(np.int16); b = np.asarray(Image.open(PUB / PF / f"{n}.png").convert("RGB")).astype(np.int16)
        if n == "fig7_part1":
            print(f"{n}: size {a.shape} vs {b.shape}, tight crop without panel a"); b = b[:, b.shape[1] - a.shape[1]:]
        if a.shape != b.shape: print(f"{n}: size {a.shape} vs {b.shape}"); continue
        d = np.abs(a - b).max(2); msg = ""
        if n in masks:
            x0, y0, x1, y1 = masks[n]; H, W = d.shape; msg = f" (panel a slot masked: {x1:.2f} W x {y1:.2f} H)"
            d[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 0
        print(f"{n}: {a.shape[1]}x{a.shape[0]}; differing pixels {(d > 0).mean():.3g}; over 8/255 {(d > 8).mean():.3g}; max {d.max()}{msg}")


def stats():
    for n in ("fig7_part1", "fig7_part2", "fig7_part3", "figS6_all_datasets", "figS7_gradient", "figS8_steepness_checks"):
        a, b = json.load(open(ROOT / PF / f"{n}.json")), json.load(open(PUB / PF / f"{n}.json"))
        sa, sb = a.get("stats", a), b.get("stats", b)
        sb = {k: v for k, v in sb.items() if k in sa}
        c, d, t, m = cmp_json(sa, sb)
        print(f"{n}: panels {sorted(sa)}; {c} values; max|diff| {d:.3g}; text differing {t[:10]}; keys in one only {m[:10]}")


def svg():
    """per-panel SVGs (the only output of panels b-f and S6-S8) as text, without the date and the random ids"""
    def norm(p):
        t = open(p, errors="replace").read()
        t = re.sub(r"<dc:date>.*?</dc:date>", "", t); ids = {}
        return re.sub(r"\b(m|p|c|i|image|clip)?[0-9a-f]{10}[0-9a-f]*\b", lambda m: ids.setdefault(m.group(0), f"id{len(ids)}"), t)
    for f in sorted(glob.glob(str(ROOT / PF / "panels/*.svg"))):
        n = os.path.basename(f); q = PUB / PF / "panels" / n
        if not q.exists(): print(f"{n}: not in published"); continue
        a, b = norm(f), norm(q); same = a == b
        d = [(x, y) for x, y in zip(a.splitlines(), b.splitlines()) if x.strip() != y.strip()]
        same = same or (not d and len(a.splitlines()) == len(b.splitlines()))
        print(f"{n}: " + ("identical" if same else f"DIFFERS, {len(d)} of {len(a.splitlines())} lines ({len(b.splitlines())} published)"))
        for x, y in d[:3]: print("   here:", re.sub(r"<[^>]+>", " ", x).strip()[:100], "| published:", re.sub(r"<[^>]+>", " ", y).strip()[:100])


dict(tables=tables, json=jsons, npz=npzs, pixels=pixels, stats=stats, svg=svg)[sys.argv[1]]()
