"""Concatenate the main and supplementary figures into two PDFs, with a label in the upper-left corner of each page.

  main  the user's latest Illustrator assemblies (args: fig1 ... fig7 PDFs); kept at 100%; a 24-pt band is added at the top for the label
  supp  fig1/supp/S1-S3 and fig_supp/S4-S15, exact A4 pages; the label goes in the existing top margin
Figure content is not rescaled or re-rendered: fonts and images pass through. Needs pypdf and reportlab.
Usage: python concat_figures.py <out_dir> <fig1.pdf> ... <fig7.pdf>
"""
import io, sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter, PageObject, Transformation
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

PF = Path("/mnt/HDD2B/Users/yancui/Saturn/published_figures")
pdfmetrics.registerFont(TTFont("Arial", str(Path.home() / ".local/share/fonts/Arial.ttf")))
BAND, FS = 24, 9


def stamp(w, h, text, x, y):
    b = io.BytesIO(); c = canvas.Canvas(b, pagesize=(w, h), initialFontName="Arial"); c.setFont("Arial", FS); c.drawString(x, y, text); c.save(); b.seek(0)
    return PdfReader(b).pages[0]


def build(items, out, band, x, y_from_top):
    W = PdfWriter()
    for label, path in items:
        src = PdfReader(str(path)).pages[0]
        x0, y0, w, h = (float(v) for v in (src.mediabox.left, src.mediabox.bottom, src.mediabox.width, src.mediabox.height))
        H = h + band; p = PageObject.create_blank_page(width=w, height=H)
        p.merge_transformed_page(src, Transformation().translate(-x0, -y0))
        p.merge_page(stamp(w, H, label, x, H - y_from_top))
        W.add_page(p); W.add_outline_item(label, len(W.pages) - 1)
    W.add_metadata({"/Title": out.stem.replace("_", " ")})
    W.write(str(out)); print("->", out, len(W.pages), "pages", round(out.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    build([(f"Fig. {i + 1}", p) for i, p in enumerate(sys.argv[2:9])], out / "Saturn_Figures.pdf", BAND, 18, 17)
    S = [PF / f"fig1/supp/S{i}/figS{i}.pdf" for i in (1, 2, 3)] + [PF / f"fig_supp/S{i}/figS{i}.pdf" for i in range(4, 16)]
    build([(f"Supplementary Fig. {i + 1}", p) for i, p in enumerate(S)], out / "Saturn_Supplementary_Figures.pdf", 0, 40, 26)
