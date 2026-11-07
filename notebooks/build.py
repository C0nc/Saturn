#!/usr/bin/env python
"""src/figN_*.py (cells separated by '# %%', markdown cells '# %% [markdown]') -> figN_*.ipynb, without outputs."""
import re, sys
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent


def build(src):
    nb = nbf.v4.new_notebook(metadata=dict(kernelspec=dict(name="python3", display_name="Python 3", language="python")))
    for part in re.split(r"(?m)^# %%", src.read_text())[1:]:
        head, _, body = part.partition("\n"); body = body.strip("\n")
        if "[markdown]" in head:
            nb.cells.append(nbf.v4.new_markdown_cell(re.sub(r"(?m)^# ?", "", body)))
        else:
            nb.cells.append(nbf.v4.new_code_cell(body))
    out = src.resolve().parent.parent / (src.stem + ".ipynb"); nbf.write(nb, out); print(out)


for s in ([Path(a) for a in sys.argv[1:]] or sorted((HERE / "src").glob("*.py"))): build(s)
