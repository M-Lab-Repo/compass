"""Figure 2 composite sheet from the four panels: a, c and d stacked on the left, b on the right.

    python figure2/figure2_compose.py fig2a.png fig2b.png fig2c.png fig2d.png --out figure2.png
    python figure2/figure2_compose.py fig2a.pdf fig2b.pdf fig2c.pdf fig2d.pdf --out figure2.pdf

Options
  FILE FILE FILE FILE  the four panels, in the order a b c d (positional, required): PNGs for a
                       raster sheet, or the PDFs the create scripts write with --out X.pdf for a
                       vector sheet whose text stays editable (Illustrator, Inkscape)
  --out FILE           the sheet (required); .png with PNG inputs, .pdf with PDF inputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

GUTTER = 40  # pixels between panels
LETTER = 56  # pixel size of the panel letter


def _font(size: int):
    try:
        import matplotlib
        ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        return ImageFont.truetype(str(ttf), size)
    except Exception:
        return ImageFont.load_default()


def _fit_width(img: Image.Image, width: int) -> Image.Image:
    return img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)


def compose(a, b, c, d, out_png) -> dict:
    """Left column a, c, d at a common width; b scaled to the column's height on the right."""
    pa, pb, pc, pd = (Image.open(p).convert("RGB") for p in (a, b, c, d))
    width = max(pa.width, pc.width, pd.width)
    left = [_fit_width(p, width) for p in (pa, pc, pd)]
    left_h = sum(p.height for p in left) + GUTTER * 2
    pb = pb.resize((round(pb.width * left_h / pb.height), left_h), Image.LANCZOS)
    sheet = Image.new("RGB", (width + GUTTER + pb.width, left_h), "white")
    y = 0
    font = _font(LETTER)
    for letter, p in zip("acd", left, strict=True):
        sheet.paste(p, (0, y))
        ImageDraw.Draw(sheet).text((8, y + 4), letter, fill="black", font=font)
        y += p.height + GUTTER
    sheet.paste(pb, (width + GUTTER, 0))
    ImageDraw.Draw(sheet).text((width + GUTTER + 8, 4), "b", fill="black", font=font)
    sheet.save(out_png)
    return {"out": str(out_png), "size": sheet.size}


def compose_pdf(a, b, c, d, out_pdf) -> dict:
    """Same layout as compose(), on one PDF page, keeping the panels as vector graphics."""
    from pypdf import PdfReader, PdfWriter, Transformation

    pages = {k: PdfReader(path).pages[0] for k, path in zip("abcd", (a, b, c, d), strict=True)}
    size = {k: (float(pg.mediabox.width), float(pg.mediabox.height)) for k, pg in pages.items()}
    gutter, lane = 18.0, 16.0                                 # points; the letters get a lane of their own
    width = max(size[k][0] for k in "acd")                    # the left column's common width
    scale = {k: width / size[k][0] for k in "acd"}
    left_h = sum(size[k][1] * scale[k] for k in "acd") + 2 * gutter
    scale["b"] = left_h / size["b"][1]
    x_b = lane + width + gutter + lane
    page_w, page_h = x_b + size["b"][0] * scale["b"], left_h
    writer = PdfWriter()
    sheet = writer.add_blank_page(width=page_w, height=page_h)
    y = page_h                                                # PDF origin is bottom-left; fill top-down
    letters = []
    for k in "acd":
        h = size[k][1] * scale[k]
        y -= h
        sheet.merge_transformed_page(pages[k], Transformation().scale(scale[k]).translate(lane, y))
        letters.append((k, 2, y + h - 16))
        y -= gutter
    sheet.merge_transformed_page(pages["b"], Transformation().scale(scale["b"]).translate(x_b, 0))
    letters.append(("b", x_b - lane + 2, page_h - 16))
    # panel letters as real text on an overlay page (editable like the rest)
    import matplotlib
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig = Figure(figsize=(page_w / 72, page_h / 72))
    FigureCanvasPdf(fig)
    for k, x, yy in letters:
        fig.text(x / page_w, yy / page_h, k, fontsize=14, fontweight="bold", va="baseline", ha="left")
    tmp = Path(out_pdf).with_suffix(".letters.pdf")
    fig.savefig(tmp, format="pdf", transparent=True)   # merged on top: an opaque page would hide the panels
    sheet.merge_page(PdfReader(tmp).pages[0])
    tmp.unlink()
    with open(out_pdf, "wb") as fh:
        writer.write(fh)
    return {"out": str(out_pdf), "size": (round(page_w), round(page_h))}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panels", nargs=4, metavar="PNG")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if str(args.out).lower().endswith(".pdf"):
        info = compose_pdf(*args.panels, out_pdf=args.out)
        print(f"wrote {info['out']} ({info['size'][0]}x{info['size'][1]} pt, vector, text editable)")
    else:
        info = compose(*args.panels, out_png=args.out)
        print(f"wrote {info['out']} ({info['size'][0]}x{info['size'][1]})")
    return info


if __name__ == "__main__":
    main()
