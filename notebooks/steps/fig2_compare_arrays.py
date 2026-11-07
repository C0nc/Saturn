"""Compare npz arrays and JSON files of this workspace with the published ones. Usage: fig2_compare_arrays.py <rel path> ..."""
import json, os, sys
import numpy as np

ROOT, PUB = os.environ["SATURN_ROOT"], os.environ.get("SATURN_PUBLISHED")
if not PUB:
    print("SATURN_PUBLISHED not set: no comparison"); raise SystemExit
for rel in sys.argv[1:]:
    a, b = f"{ROOT}/{rel}", f"{PUB}/{rel}"
    if rel.endswith(".json"):
        A, B = json.load(open(a)), json.load(open(b))
        same = lambda u, v: json.dumps(u, sort_keys=True) == json.dumps(v, sort_keys=True)   # dumps: NaN equals NaN
        bad = [k for k in sorted(set(A) | set(B)) if not same(A.get(k), B.get(k))] if isinstance(A, dict) and isinstance(B, dict) else ([] if same(A, B) else ["whole file"])
        print(rel, "identical" if not bad else f"DIFFERS in {bad}"); continue
    x, y = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    print(rel, "keys same" if set(x.files) == set(y.files) else f"KEYS DIFFER {set(x.files) ^ set(y.files)}")
    for k in sorted(set(x.files) & set(y.files)):
        p, q = x[k], y[k]
        if p.shape != q.shape: print(f"  {k}: SHAPE {p.shape} vs {q.shape}")
        elif p.dtype.kind in "fiub":
            m = np.nanmax(np.abs(p.astype(float) - q.astype(float))) if p.size else 0.0
            print(f"  {k}: max diff {m:.3g}") if m > 0 else None
        elif not np.array_equal(p.astype(str), q.astype(str)): print(f"  {k}: TEXT DIFFERS")
    print("  (arrays not listed are exactly equal)")
