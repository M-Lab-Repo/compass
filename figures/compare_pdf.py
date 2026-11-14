#!/usr/bin/env python3
"""Compare rendered manuscript PDFs with the reference PDFs, dates and other metadata aside.

    python figures/compare_pdf.py --rendered DIR --reference DIR_OR_MANIFEST [--names a.pdf,b.pdf]
    python figures/compare_pdf.py --write-manifest OUT.json --reference DIR --source-label TEXT

Method. The scripts under ``figures/`` write deterministic PDFs: matplotlib emits the
objects, the embedded TrueType font subsets and the content streams in a fixed order, and
pypdf lays the composite pages out without any date. The one thing that changes between two
renders of the same records is the date stamp of a matplotlib page, the value of
``/CreationDate`` in the Info dictionary (a ``/ModDate`` would be another such value). Its form
depends on where the render ran: ``D:YYYYMMDDhhmmssZ`` under UTC, ``D:YYYYMMDDhhmmss-05'00'``
(six bytes longer) in any other timezone, and ``D:19700101000000Z`` when ``SOURCE_DATE_EPOCH=0``
is set, which ``render.py`` does unless the caller chose an epoch, so that its renders are
byte-identical on every machine. Two hashes cover the three cases:

``sha256_dateless``
    SHA-256 of the PDF bytes after the value of every ``/CreationDate`` and ``/ModDate``, in
    either form and of any length, is replaced by zeros of the same length (the same length
    keeps the byte offsets of the cross-reference table). Equal hashes mean that every other
    byte agrees, fonts, coordinates and text included. A date stamp of another length (a UTC
    render against a non-UTC one without ``SOURCE_DATE_EPOCH``) moves the cross-reference
    offsets after the Info dictionary and the ``startxref`` pointer, so this hash differs
    although nothing drawn does.
``sha256_drawing``
    SHA-256 of a canonical serialisation of every page as pypdf parses it: the page dictionary
    without ``/Parent``, indirect references resolved, dictionary keys sorted, streams decoded
    (content streams, font programs, images, shadings). It covers what is drawn, with the
    fonts, and nothing about how the file is stored: not the Info dictionary, the object
    numbers, the compression or the byte offsets.

The report against a reference directory says ``BYTE-IDENTICAL`` when even the dates agree,
``IDENTICAL (dates aside)`` when only date values of the same length differ, ``IDENTICAL
(drawing); metadata differs`` when the pages agree in ``sha256_drawing`` but the Info dictionary
or the file layout does not (which Info keys differ is listed), and otherwise ``DIFFERENT`` with
the sizes, the page sizes and whether the text pypdf extracts differs. Against a manifest the
same three verdicts are given from its hashes. The exit status is 0 only when every expected
file is present on both sides and identical in drawing. A manifest written with
``--write-manifest`` stores, per file, both hashes, the byte size, the page count, the page size
in points and the number of date values neutralised, with a symbolic ``source`` label (never a
path), so that a verifier without the research checkout can still check a render.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DictionaryObject,
    FloatObject,
    IndirectObject,
    NameObject,
    NullObject,
    NumberObject,
    StreamObject,
)

# the eleven files of figures/render.py, in its order
EXPECTED = (
    "fig2a_editable.pdf", "fig2b_editable.pdf", "fig2c_editable.pdf", "fig2d_editable.pdf",
    "fig2_composite_editable.pdf",
    "fig3a_editable.pdf", "fig3b_editable.pdf", "fig3c_editable.pdf", "fig3d_editable.pdf",
    "fig3_composite_editable.pdf",
    "fig4_cost_editable.pdf",
)
METHOD = ("sha256_dateless: SHA-256 of the PDF bytes after the value of every /CreationDate and "
          "/ModDate (D:YYYYMMDDhhmmss followed by Z or by a timezone offset such as -05'00', any "
          "length) is replaced by zeros of the same length, so every other byte, fonts, "
          "coordinates and text included, must agree; sha256_drawing: SHA-256 of a canonical "
          "serialisation of every page as pypdf parses it (the page dictionary without /Parent, "
          "indirect references resolved, dictionary keys sorted, streams decoded), which depends "
          "on what is drawn and not on the Info dictionary, object numbers, compression or byte "
          "offsets. A render in a non-UTC timezone without SOURCE_DATE_EPOCH carries a date stamp "
          "six bytes longer than the UTC form and so agrees only in sha256_drawing "
          "(figures/compare_pdf.py)")
_DATE = re.compile(rb"(/(?:CreationDate|ModDate)\s*\((?:D:)?)([^)]*)(\))")
# how a stream is stored, not what it holds: left out of the drawing serialisation
_STREAM_STORAGE = {"/Length", "/Filter", "/DecodeParms", "/DL"}


def neutralise(data: bytes) -> tuple[bytes, int]:
    """``data`` with every date value replaced by zeros of the same length, and the number of
    values replaced. Both the UTC form (``D:20261030211147Z``) and the offset form
    (``D:20261111224440-04'00'``) are matched, whatever their length; keeping the length keeps
    the byte offsets of the cross-reference table, so the result is still a well-formed PDF."""
    n = 0

    def repl(m: re.Match) -> bytes:
        nonlocal n
        n += 1
        return m.group(1) + b"0" * len(m.group(2)) + m.group(3)

    return _DATE.sub(repl, data), n


def date_values(data: bytes) -> list[str]:
    """The date stamps found in ``data``, as written (``D:20261030211147Z``, ...)."""
    return ["D:" + m.group(2).decode("latin-1") for m in _DATE.finditer(data)]


def dateless_sha256(data: bytes) -> str:
    return hashlib.sha256(neutralise(data)[0]).hexdigest()


def canonical(obj, path: tuple = ()) -> bytes:
    """A canonical serialisation of a pypdf object: indirect references resolved (a reference
    back into the chain being serialised is written as ``<cycle>``), dictionary keys sorted,
    streams decoded and written with their dictionary minus the storage keys (``/Length``,
    ``/Filter``, ``/DecodeParms``). It depends on what the object holds and on nothing about how
    the file stores it."""
    if isinstance(obj, IndirectObject):
        key = (obj.idnum, obj.generation)
        if key in path:
            return b"<cycle>"
        return canonical(obj.get_object(), (*path, key))
    if isinstance(obj, DictionaryObject):
        skip = _STREAM_STORAGE if isinstance(obj, StreamObject) else ()
        body = b"".join(str(k).encode("latin-1", "replace") + b" " + canonical(v, path)
                        for k, v in sorted(obj.items()) if k not in skip)
        if isinstance(obj, StreamObject):
            try:
                data = obj.get_data()
            except Exception:  # a filter pypdf cannot decode: hash the stored bytes
                data = obj._data
            return b"<<" + body + b">>stream[" + data + b"]"
        return b"<<" + body + b">>"
    if isinstance(obj, ArrayObject):
        return b"[" + b" ".join(canonical(v, path) for v in obj) + b"]"
    if isinstance(obj, NameObject):
        return str(obj).encode("latin-1", "replace")
    if isinstance(obj, BooleanObject):
        return b"true" if obj.value else b"false"
    if obj is None or isinstance(obj, NullObject):
        return b"null"
    if isinstance(obj, FloatObject):
        return repr(float(obj)).encode()
    if isinstance(obj, NumberObject):
        return str(int(obj)).encode()
    if isinstance(obj, bytes):
        return b"(" + obj + b")"
    if isinstance(obj, str):
        return b"(" + obj.encode("utf-8", "replace") + b")"
    return repr(obj).encode("utf-8", "replace")


def drawing_sha256(reader: PdfReader) -> str:
    """SHA-256 of the canonical serialisation of every page (``/Parent`` left out: it points
    back into the page tree), separated by page."""
    h = hashlib.sha256()
    for page in reader.pages:
        page_dict = DictionaryObject({k: v for k, v in page.items() if k != "/Parent"})
        h.update(b"\npage\n" + canonical(page_dict))
    return h.hexdigest()


def describe(path: Path) -> dict:
    data = path.read_bytes()
    neutral, n_dates = neutralise(data)
    reader = PdfReader(io.BytesIO(data))
    box = reader.pages[0].mediabox
    return {"sha256_dateless": hashlib.sha256(neutral).hexdigest(),
            "sha256_drawing": drawing_sha256(reader), "bytes": len(data),
            "pages": len(reader.pages), "width_pt": round(float(box.width), 2),
            "height_pt": round(float(box.height), 2), "dates_neutralised": n_dates}


def text_of(path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def info_of(path: Path) -> dict[str, str]:
    """The Info dictionary as strings (``/Producer``, ``/Creator``, ``/CreationDate``, ...)."""
    meta = PdfReader(str(path)).metadata or {}
    return {str(k): str(v) for k, v in meta.items()}


def _shape(d: dict) -> str:
    return f"{d['bytes']} bytes, {d['pages']} page(s), {d['width_pt']}x{d['height_pt']} pt"


def _metadata_difference(rendered: Path, reference: Path) -> str:
    """What differs outside the pages: the Info keys whose values differ (dates shown as
    written) and the byte sizes."""
    a, b = info_of(rendered), info_of(reference)
    keys = sorted(set(a) | set(b))
    parts = [f"{k} {a.get(k, '(absent)')} vs {b.get(k, '(absent)')}" for k in keys
             if a.get(k) != b.get(k)]
    sizes = f"{rendered.stat().st_size} vs {reference.stat().st_size} bytes"
    return "; ".join([*parts, sizes]) if parts else f"{sizes}, Info dictionary equal"


def write_manifest(reference: Path, out: Path, names: tuple[str, ...], source_label: str) -> dict:
    entries = {}
    for name in names:
        p = reference / name
        if not p.is_file():
            raise SystemExit(f"reference file missing: {p}")
        entries[name] = describe(p)
    manifest = {"source": source_label, "method": METHOD, "files": entries}
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def compare(rendered: Path, reference: Path, names: tuple[str, ...]) -> tuple[list[str], int]:
    """Lines to print and the number of files that are not identical in drawing."""
    manifest = None
    if reference.is_file():
        manifest = json.loads(reference.read_text())["files"]
    lines, bad = [], 0
    for name in names:
        r = rendered / name
        if not r.is_file():
            lines.append(f"{name:36s} MISSING in rendered")
            bad += 1
            continue
        got = describe(r)
        if manifest is not None:
            ref = manifest.get(name)
            if ref is None:
                lines.append(f"{name:36s} MISSING in manifest")
                bad += 1
            elif got["sha256_dateless"] == ref["sha256_dateless"]:
                lines.append(f"{name:36s} IDENTICAL (dates aside, manifest)")
            elif got["sha256_drawing"] == ref.get("sha256_drawing"):
                dates = ", ".join(date_values(r.read_bytes())) or "none"
                lines.append(f"{name:36s} IDENTICAL (drawing); metadata differs from manifest: "
                             f"{got['bytes']} vs {ref['bytes']} bytes, date stamp {dates}")
            else:
                lines.append(f"{name:36s} DIFFERENT from manifest: {_shape(got)} vs {_shape(ref)}")
                bad += 1
            continue
        f = reference / name
        if not f.is_file():
            lines.append(f"{name:36s} MISSING in reference")
            bad += 1
            continue
        data_r, data_f = r.read_bytes(), f.read_bytes()
        if data_r == data_f:
            lines.append(f"{name:36s} BYTE-IDENTICAL")
            continue
        if got["sha256_dateless"] == dateless_sha256(data_f):
            lines.append(f"{name:36s} IDENTICAL (dates aside; {got['dates_neutralised']} date "
                         "value(s) differ)")
            continue
        ref = describe(f)
        if got["sha256_drawing"] == ref["sha256_drawing"]:
            lines.append(f"{name:36s} IDENTICAL (drawing); metadata differs: "
                         f"{_metadata_difference(r, f)}")
            continue
        text = "text identical" if text_of(r) == text_of(f) else "text differs"
        lines.append(f"{name:36s} DIFFERENT: {_shape(got)} vs {_shape(ref)}; {text}")
        bad += 1
    return lines, bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rendered", type=Path, help="directory with the freshly rendered PDFs")
    ap.add_argument("--reference", type=Path, required=True,
                    help="directory of reference PDFs, or a manifest JSON")
    ap.add_argument("--names", help="comma-separated file names (default: the eleven files)")
    ap.add_argument("--write-manifest", type=Path,
                    help="write a manifest of --reference to this path and exit")
    ap.add_argument("--source-label",
                    help="with --write-manifest: what the reference files are, symbolically "
                         "(for example 'renders of the research figure scripts at commit "
                         "835b750 from the records at 2c66219'); never a path")
    args = ap.parse_args(argv)
    names = tuple(s.strip() for s in args.names.split(",")) if args.names else EXPECTED
    if args.write_manifest:
        if not args.source_label:
            ap.error("--write-manifest needs --source-label (a symbolic description of the "
                     "reference files, never a path)")
        if not args.reference.is_dir():
            ap.error("--write-manifest needs --reference to be a directory")
        m = write_manifest(args.reference, args.write_manifest, names, args.source_label)
        print(f"wrote {args.write_manifest} ({len(m['files'])} files; source {m['source']!r})")
        return 0
    if args.rendered is None:
        ap.error("--rendered is required unless --write-manifest is given")
    lines, bad = compare(args.rendered, args.reference, names)
    print("\n".join(lines))
    print(f"{len(names) - bad}/{len(names)} identical in drawing (dates and other metadata aside)",
          file=sys.stderr if bad else sys.stdout)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
