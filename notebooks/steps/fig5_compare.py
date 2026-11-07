"""Comparisons Fig 5 needs beyond nbutil.compare_axes / compare_table: npz arrays and JSON stats, workspace vs published.
Read-only on both sides."""
import json
import numpy as np
import nbutil as nb

PUB_R = (nb.PUB / "Hole/gaston_mix_mm/results/real_saturn") if nb.PUB else None


def compare_npz(rel, keys=None):
    """Largest absolute difference per numeric array of one npz under results/real_saturn; string arrays must be equal."""
    if PUB_R is None: print("SATURN_PUBLISHED not set"); return
    a, b = np.load(nb.RES / rel, allow_pickle=True), np.load(PUB_R / rel, allow_pickle=True)
    print(rel, "keys only here", sorted(set(a.files) - set(b.files)), "only published", sorted(set(b.files) - set(a.files)))
    out = {}
    for k in keys or sorted(set(a.files) & set(b.files)):
        x, y = a[k], b[k]
        if x.shape != y.shape: out[k] = f"shape {x.shape} vs {y.shape}"
        elif x.dtype.kind in "fiu" and y.dtype.kind in "fiu": out[k] = float(np.max(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0
        else: out[k] = "equal" if (x.astype(str) == y.astype(str)).all() else "DIFFERS"
    bad = {k: v for k, v in out.items() if v not in ("equal", 0.0)}
    print(f"  {len(out)} arrays compared, {len(out) - len(bad)} identical; others: {bad if len(bad) < 15 else dict(list(bad.items())[:15])}")
    return out


def _walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b: yield path + "/" + str(k), a.get(k, "<absent>"), b.get(k, "<absent>")
            else: yield from _walk(a[k], b[k], path + "/" + str(k))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (p, q) in enumerate(zip(a, b)): yield from _walk(p, q, f"{path}[{i}]")
    elif a != b and not (isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b)):
        yield path, a, b


def compare_json(rel, root="res"):
    """Every leaf of a JSON, workspace vs published; prints the leaves that differ. root = res (results) or pf (figures)."""
    if nb.PUB is None: print("SATURN_PUBLISHED not set"); return
    x, y = (nb.RES / rel, PUB_R / rel) if root == "res" else (nb.PF / rel, nb.PUB / "Saturn/published_figures" / rel)
    d = list(_walk(json.load(open(x)), json.load(open(y))))
    print(f"{rel}: {len(d)} leaves differ")
    for p, a, b in d[:25]: print(f"  {p}: here {a} | published {b}")
    return d


def compare_png(rel):
    """Pixel comparison of one figure PNG under published_figures, workspace vs published."""
    if nb.PUB is None: print("SATURN_PUBLISHED not set"); return
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    a = np.asarray(Image.open(nb.PF / rel).convert("RGB")).astype(int)
    b = np.asarray(Image.open(nb.PUB / "Saturn/published_figures" / rel).convert("RGB")).astype(int)
    if a.shape != b.shape: print(f"{rel}: shape {a.shape} vs {b.shape}"); return
    d = np.abs(a - b).max(2)
    print(f"{rel}: {a.shape[1]}x{a.shape[0]} px; largest channel difference {d.max()}; {(d > 0).mean():.2%} of pixels differ")


def show_small(*rels, width=1100):
    """Display figure pages at reduced size, so the saved notebook stays small (nbutil.show embeds the 600 dpi PNG)."""
    import io
    from IPython.display import Image as IImage, display
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    for rel in rels:
        im = Image.open(nb.PF / rel).convert("RGB"); im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
        b = io.BytesIO(); im.save(b, "PNG", optimize=True); display(IImage(data=b.getvalue(), width=width))
