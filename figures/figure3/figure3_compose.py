"""Figure 3 composite from the panels: a and b on the first row, c and d on the second, and an
optional e (the SI real-data drift) alone on a third row.

    python figure3/figure3_compose.py fig3a.png fig3b.png fig3c.png fig3d.png --out figure3.png [--extra fig3e.png]
    python figure3/figure3_compose.py fig3a.pdf fig3b.pdf fig3c.pdf fig3d.pdf --out figure3.pdf

Options
  FILE FILE FILE FILE  the four panels in the order a b c d (positional, required): PNGs for a raster sheet,
                       or the PDFs the create scripts write with --out X.pdf for a vector sheet whose text
                       stays editable (Illustrator, Inkscape)
  --extra FILE         a fifth panel, lettered e, on its own row below (same file type as the others)
  --row-titles TEXT... one title per row, drawn in a band above the row (e.g. "When nothing is wrong: ..."
                       "When a fix exists: ..."); rows without a title get none
  --out FILE           the sheet (required); .png with PNG inputs, .pdf with PDF inputs

Every panel is scaled, aspect kept, to a common column width (the widest panel's); a row is as
tall as its taller panel, cells are aligned at their top edges, and each panel carries its
letter in a short band above its top-left corner.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

try:
    import figure3_common  # noqa: F401  puts the repo root on sys.path when run as a file
except ImportError:
    pass
from figure2.figure2_compose import _fit_width, _font  # noqa: E402

GUTTER, LETTER = 40, 56  # pixels between panels, pixel size of the panel letter


def _rows(panels: dict) -> list[tuple[str, ...]]:
    return [("a", "b"), ("c", "d")] + ([("e",)] if "e" in panels else [])


def compose(panels: dict, out_png, row_titles=()) -> dict:
    """``panels`` maps letters to PNG files; the grid is written to ``out_png``. A row title, when
    given, sits in its own band above the row, spanning the sheet."""
    imgs = {k: Image.open(p).convert("RGB") for k, p in panels.items()}
    width = max(im.width for im in imgs.values())
    imgs = {k: _fit_width(im, width) for k, im in imgs.items()}
    rows = _rows(panels)
    band, title_band = round(1.15 * LETTER), (round(1.3 * LETTER) if row_titles else 0)
    heights = [max(imgs[k].height for k in row) for row in rows]
    sheet = Image.new("RGB", (2 * width + 3 * GUTTER,
                              sum(heights) + len(rows) * (band + title_band + GUTTER) + GUTTER), "white")
    draw, font, title_font, y = ImageDraw.Draw(sheet), _font(LETTER), _font(round(0.75 * LETTER)), GUTTER
    for i, (row, h) in enumerate(zip(rows, heights, strict=True)):
        if i < len(row_titles):
            draw.text((GUTTER, y), row_titles[i], fill="#333333", font=title_font)
        y += title_band
        for j, k in enumerate(row):
            x = GUTTER + j * (width + GUTTER)
            draw.text((x, y), k, fill="black", font=font)
            sheet.paste(imgs[k], (x, y + band))
        y += band + h + GUTTER
    sheet.save(out_png)
    return {"out": str(out_png), "size": sheet.size}


def compose_pdf(panels: dict, out_pdf, row_titles=()) -> dict:
    """Same grid on one PDF page, keeping the panels as vector graphics."""
    import matplotlib
    from matplotlib.backends.backend_pdf import FigureCanvasPdf
    from matplotlib.figure import Figure
    from pypdf import PdfReader, PdfWriter, Transformation

    pages = {k: PdfReader(p).pages[0] for k, p in panels.items()}
    size = {k: (float(pg.mediabox.width), float(pg.mediabox.height)) for k, pg in pages.items()}
    gutter, band, title_band = 18.0, 20.0, (22.0 if row_titles else 0.0)  # points
    width = max(w for w, _ in size.values())
    scale = {k: width / w for k, (w, _) in size.items()}
    rows = _rows(panels)
    heights = [max(size[k][1] * scale[k] for k in row) for row in rows]
    page_w = 2 * width + 3 * gutter
    page_h = sum(heights) + len(rows) * (band + title_band + gutter) + gutter
    writer = PdfWriter()
    sheet = writer.add_blank_page(width=page_w, height=page_h)
    texts, y_top = [], page_h - gutter  # PDF origin is bottom-left; fill top-down
    for i, (row, h) in enumerate(zip(rows, heights, strict=True)):
        if i < len(row_titles):
            texts.append((row_titles[i], gutter, y_top - title_band + 6, 11))
        y_top -= title_band
        for j, k in enumerate(row):
            x = gutter + j * (width + gutter)
            y = y_top - band - size[k][1] * scale[k]
            sheet.merge_transformed_page(pages[k], Transformation().scale(scale[k]).translate(x, y))
            texts.append((k, x, y_top - band + 4, 14))
        y_top -= band + h + gutter
    matplotlib.rcParams["pdf.fonttype"] = 42
    fig = Figure(figsize=(page_w / 72, page_h / 72))
    FigureCanvasPdf(fig)
    for text, x, y, size_pt in texts:  # real text, editable like the rest
        fig.text(x / page_w, y / page_h, text, fontsize=size_pt, fontweight="bold", va="baseline", ha="left",
                 color="black" if size_pt == 14 else "#333333")
    tmp = Path(out_pdf).with_suffix(".letters.pdf")
    fig.savefig(tmp, format="pdf", transparent=True)
    sheet.merge_page(PdfReader(tmp).pages[0])
    tmp.unlink()
    with open(out_pdf, "wb") as fh:
        writer.write(fh)
    return {"out": str(out_pdf), "size": (round(page_w), round(page_h))}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("panels", nargs=4, metavar="FILE")
    ap.add_argument("--extra", metavar="FILE")
    ap.add_argument("--row-titles", nargs="+", metavar="TEXT", default=())
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    panels = dict(zip("abcd", args.panels, strict=True))
    if args.extra:
        panels["e"] = args.extra
    if str(args.out).lower().endswith(".pdf"):
        info = compose_pdf(panels, args.out, row_titles=args.row_titles)
        print(f"wrote {info['out']} ({info['size'][0]}x{info['size'][1]} pt, vector, text editable)")
    else:
        info = compose(panels, args.out, row_titles=args.row_titles)
        print(f"wrote {info['out']} ({info['size'][0]}x{info['size'][1]})")
    return info


if __name__ == "__main__":
    main()
