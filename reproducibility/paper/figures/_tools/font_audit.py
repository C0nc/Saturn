"""Smallest font and any non-Arial math text actually written to each SVG (includes mathtext sub/superscripts, which check_min_font cannot see).
Usage: python _tools/font_audit.py <svg> [<svg> ...]   -> one line per file, exit 1 if any file has text below 5 pt"""
import re, sys, collections
bad = 0
for f in sys.argv[1:]:
    c = collections.Counter(round(float(x), 2) for x in re.findall(r"font(?:-size)?:\s*(?:italic |oblique |bold |normal |\d00 )*([\d.]+)px", open(f, errors="ignore").read()))
    txt = open(f, errors="ignore").read(); dj = txt.count("font-family: 'DejaVu Sans'"); bad += dj > 0
    low = {k: v for k, v in sorted(c.items()) if k < 4.995}; bad += bool(low)
    print(f"{f}: min {min(c) if c else 'no text'} pt, n={sum(c.values())}" + (f", BELOW 5: {low}" if low else "") + (f", NON-ARIAL: {dj}" if dj else ""))
sys.exit(1 if bad else 0)
