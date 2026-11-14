"""``figures/compare_pdf.py``: the date-neutral hash, the drawing hash, the manifest round trip
and the report on tiny PDFs written by matplotlib under ``SOURCE_DATE_EPOCH`` (which fixes the
``/CreationDate`` matplotlib embeds) and with a non-UTC date stamp (``time.timezone``
monkeypatched, the variable unset), plus the shape of the committed reference manifest
(``figures/reference_manifest.json``: the eleven files of ``figures/render.py``). No test reads
the benchmark records.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from pypdf import PdfReader
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    ByteStringObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NullObject,
    NumberObject,
    TextStringObject,
)

import compare_pdf
import render

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
MANIFEST = FIGURES_DIR / "reference_manifest.json"


def _pdf(path: Path, text: str, epoch: int, monkeypatch) -> Path:
    """A one-page PDF (144 x 72 pt) with one line of text, dated ``epoch``."""
    monkeypatch.setenv("SOURCE_DATE_EPOCH", str(epoch))
    fig = Figure(figsize=(2, 1))
    FigureCanvasPdf(fig)
    fig.text(0.1, 0.5, text)
    fig.savefig(path, format="pdf")
    return path


def _offset_pdf(path: Path, text: str, monkeypatch) -> Path:
    """The same one-page PDF written without ``SOURCE_DATE_EPOCH`` in a non-UTC timezone:
    matplotlib stamps the offset of ``time.timezone`` when ``time.daylight`` is 0, here EST, so
    the date is ``D:YYYYMMDDhhmmss-05'00'``, six bytes longer than the UTC form."""
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    monkeypatch.setattr(time, "daylight", 0)
    monkeypatch.setattr(time, "timezone", 5 * 3600)
    fig = Figure(figsize=(2, 1))
    FigureCanvasPdf(fig)
    fig.text(0.1, 0.5, text)
    fig.savefig(path, format="pdf")
    return path


def test_neutralise_blanks_every_date_value_and_keeps_the_length():
    data = (b"%PDF-1.4\n<< /CreationDate (D:20261030211147Z) /ModDate (D:20261030211147+00'00') "
            b"/Producer (x) >>\n")
    out, n = compare_pdf.neutralise(data)
    assert n == 2 and len(out) == len(data)
    assert b"/CreationDate (D:" + b"0" * len(b"20261030211147Z") + b")" in out
    assert b"/ModDate (D:" + b"0" * len(b"20261030211147+00'00'") + b")" in out
    assert b"/Producer (x)" in out and b"2026" not in out
    assert compare_pdf.neutralise(b"no dates here")[1] == 0
    assert compare_pdf.date_values(data) == ["D:20261030211147Z", "D:20261030211147+00'00'"]
    # the offset form of any length, and a value without the D: prefix, are neutralised too
    out, n = compare_pdf.neutralise(b"/CreationDate (D:20261111224440-04'00') /ModDate (20261111224440)")
    assert n == 2 and out == b"/CreationDate (D:" + b"0" * 21 + b") /ModDate (" + b"0" * 14 + b")"


def test_same_drawing_at_different_dates_is_identical_dates_aside(tmp_path, monkeypatch):
    a = _pdf(tmp_path / "a.pdf", "hello", 1_000_000, monkeypatch)
    b = _pdf(tmp_path / "b.pdf", "hello", 2_000_000, monkeypatch)
    assert a.read_bytes() != b.read_bytes()
    da, db = compare_pdf.describe(a), compare_pdf.describe(b)
    assert da["sha256_dateless"] == db["sha256_dateless"]
    assert da["dates_neutralised"] == 1 and da["pages"] == 1
    assert (da["width_pt"], da["height_pt"]) == (144.0, 72.0)
    assert da["sha256_dateless"] != hashlib.sha256(a.read_bytes()).hexdigest()
    c = _pdf(tmp_path / "c.pdf", "other", 1_000_000, monkeypatch)
    assert compare_pdf.describe(c)["sha256_dateless"] != da["sha256_dateless"]


def test_drawing_hash_ignores_the_info_dictionary_and_sees_the_pages(tmp_path, monkeypatch):
    a = _pdf(tmp_path / "a.pdf", "hello", 1, monkeypatch)
    fig = Figure(figsize=(2, 1))
    FigureCanvasPdf(fig)
    fig.text(0.1, 0.5, "hello")
    b = tmp_path / "b.pdf"
    fig.savefig(b, format="pdf", metadata={"Creator": "someone else", "Producer": "another producer"})
    da, db = compare_pdf.describe(a), compare_pdf.describe(b)
    assert len(da["sha256_drawing"]) == 64 and int(da["sha256_drawing"], 16) >= 0
    assert da["sha256_dateless"] != db["sha256_dateless"]  # the Producer string is in the bytes
    assert da["sha256_drawing"] == db["sha256_drawing"]  # but not on any page
    assert compare_pdf.describe(_pdf(tmp_path / "c.pdf", "hellp", 1, monkeypatch))["sha256_drawing"] != da["sha256_drawing"]
    assert compare_pdf.info_of(b) == {"/Creator": "someone else", "/Producer": "another producer",
                                      "/CreationDate": "D:19700101000001Z"}
    # the page dictionary minus /Parent serialises without a cycle; with /Parent the page tree
    # leads back to the page and the chain is cut at the marker
    page = PdfReader(str(a)).pages[0]
    without_parent = DictionaryObject({k: v for k, v in page.items() if k != "/Parent"})
    assert b"<cycle>" not in compare_pdf.canonical(without_parent)
    assert b"<cycle>" in compare_pdf.canonical(page)


def test_canonical_serialisation_of_the_object_kinds():
    c = compare_pdf.canonical
    assert c(FloatObject(1.5)) == b"1.5" and c(NumberObject(3)) == b"3"
    assert c(BooleanObject(True)) == b"true" and c(BooleanObject(False)) == b"false"
    assert c(NullObject()) == b"null" and c(None) == b"null"
    assert c(NameObject("/Type")) == b"/Type"
    assert c(TextStringObject("abc")) == b"(abc)" and c(ByteStringObject(b"ab")) == b"(ab)"
    assert c(ArrayObject([NumberObject(1), NameObject("/X")])) == b"[1 /X]"
    d = DictionaryObject({NameObject("/B"): NumberObject(2), NameObject("/A"): FloatObject(0.5)})
    assert c(d) == b"<</A 0.5/B 2>>"  # keys sorted, whatever the insertion order


def test_a_date_stamp_of_another_length_is_identical_in_drawing(tmp_path, monkeypatch):
    utc = _pdf(tmp_path / "utc.pdf", "same", 0, monkeypatch)
    est = _offset_pdf(tmp_path / "est.pdf", "same", monkeypatch)
    assert compare_pdf.date_values(utc.read_bytes()) == ["D:19700101000000Z"]
    (stamp,) = compare_pdf.date_values(est.read_bytes())
    assert stamp.endswith("-05'00'") and len(est.read_bytes()) == len(utc.read_bytes()) + 6
    du, de = compare_pdf.describe(utc), compare_pdf.describe(est)
    assert du["dates_neutralised"] == de["dates_neutralised"] == 1
    assert du["sha256_dateless"] != de["sha256_dateless"]  # the longer stamp moves startxref
    assert du["sha256_drawing"] == de["sha256_drawing"]
    ref, got = tmp_path / "ref", tmp_path / "got"
    ref.mkdir()
    got.mkdir()
    shutil.copy(utc, ref / "p.pdf")
    shutil.copy(est, got / "p.pdf")
    lines, bad = compare_pdf.compare(got, ref, ("p.pdf",))
    assert bad == 0
    verdict = lines[0].split(maxsplit=1)[1]
    assert verdict.startswith("IDENTICAL (drawing); metadata differs: /CreationDate D:")
    assert f"{stamp} vs D:19700101000000Z" in verdict and f"{de['bytes']} vs {du['bytes']} bytes" in verdict
    manifest = tmp_path / "m.json"
    compare_pdf.write_manifest(ref, manifest, ("p.pdf",), "a label")
    lines, bad = compare_pdf.compare(got, manifest, ("p.pdf",))
    assert bad == 0 and "IDENTICAL (drawing); metadata differs from manifest:" in lines[0]
    assert f"date stamp {stamp}" in lines[0]
    # a real difference carrying an offset stamp is still DIFFERENT, in both forms
    _offset_pdf(got / "p.pdf", "other", monkeypatch)
    lines, bad = compare_pdf.compare(got, ref, ("p.pdf",))
    assert bad == 1 and "DIFFERENT:" in lines[0] and "text differs" in lines[0]
    lines, bad = compare_pdf.compare(got, manifest, ("p.pdf",))
    assert bad == 1 and "DIFFERENT from manifest" in lines[0]
    # a manifest without the drawing hash (an older one) cannot vouch for the drawing
    m = json.loads(manifest.read_text())
    del m["files"]["p.pdf"]["sha256_drawing"]
    manifest.write_text(json.dumps(m))
    shutil.copy(est, got / "p.pdf")
    lines, bad = compare_pdf.compare(got, manifest, ("p.pdf",))
    assert bad == 1 and "DIFFERENT from manifest" in lines[0]


def test_directory_comparison_reports_every_outcome(tmp_path, monkeypatch):
    ref, got = tmp_path / "ref", tmp_path / "got"
    ref.mkdir()
    got.mkdir()
    _pdf(ref / "same.pdf", "same", 1, monkeypatch)
    shutil.copy(ref / "same.pdf", got / "same.pdf")
    _pdf(ref / "dated.pdf", "dated", 1, monkeypatch)
    _pdf(got / "dated.pdf", "dated", 2, monkeypatch)
    _pdf(ref / "other.pdf", "one text", 1, monkeypatch)
    _pdf(got / "other.pdf", "another text", 1, monkeypatch)
    _pdf(ref / "only_ref.pdf", "x", 1, monkeypatch)
    _pdf(got / "only_got.pdf", "x", 1, monkeypatch)
    names = ("same.pdf", "dated.pdf", "other.pdf", "only_ref.pdf", "only_got.pdf")
    lines, bad = compare_pdf.compare(got, ref, names)
    report = dict(line.split(maxsplit=1) for line in lines)
    assert report["same.pdf"] == "BYTE-IDENTICAL"
    assert report["dated.pdf"].startswith("IDENTICAL (dates aside; 1 date value(s) differ)")
    assert report["other.pdf"].startswith("DIFFERENT:") and "text differs" in report["other.pdf"]
    assert report["only_ref.pdf"] == "MISSING in rendered"
    assert report["only_got.pdf"] == "MISSING in reference"
    assert bad == 3


def test_manifest_round_trip_and_command_line(tmp_path, monkeypatch, capsys):
    ref, got = tmp_path / "ref", tmp_path / "got"
    ref.mkdir()
    got.mkdir()
    for name, text in (("a.pdf", "alpha"), ("b.pdf", "beta")):
        _pdf(ref / name, text, 1, monkeypatch)
        _pdf(got / name, text, 2, monkeypatch)
    manifest = tmp_path / "m.json"
    # --write-manifest refuses to run without a symbolic label (no absolute path in the manifest)
    with pytest.raises(SystemExit) as exc:
        compare_pdf.main(["--write-manifest", str(manifest), "--reference", str(ref), "--names", "a.pdf,b.pdf"])
    assert exc.value.code == 2 and "--source-label" in capsys.readouterr().err
    rc = compare_pdf.main(["--write-manifest", str(manifest), "--reference", str(ref),
                           "--names", "a.pdf,b.pdf", "--source-label", "a label, not a path"])
    assert rc == 0
    m = json.loads(manifest.read_text())
    assert m["source"] == "a label, not a path" and str(ref) not in manifest.read_text()
    assert set(m["files"]) == {"a.pdf", "b.pdf"} and "sha256_dateless" in m["method"]
    assert m["files"]["a.pdf"] == compare_pdf.describe(ref / "a.pdf")
    rc = compare_pdf.main(["--rendered", str(got), "--reference", str(manifest), "--names", "a.pdf,b.pdf"])
    out = capsys.readouterr().out
    assert rc == 0 and out.count("IDENTICAL (dates aside, manifest)") == 2
    assert "2/2 identical in drawing (dates and other metadata aside)" in out
    _pdf(got / "b.pdf", "changed", 2, monkeypatch)
    rc = compare_pdf.main(["--rendered", str(got), "--reference", str(manifest), "--names", "a.pdf,b.pdf"])
    cap = capsys.readouterr()
    assert rc == 1 and "b.pdf" in cap.out and "DIFFERENT from manifest" in cap.out
    assert "1/2 identical" in cap.err
    (got / "a.pdf").unlink()
    rc = compare_pdf.main(["--rendered", str(got), "--reference", str(ref), "--names", "a.pdf"])
    assert rc == 1 and "MISSING in rendered" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        compare_pdf.main(["--reference", str(ref)])  # --rendered or --write-manifest is needed


def test_expected_names_are_the_files_render_writes():
    assert [render.output(Path("."), n, "pdf").name for n in render.NAMES] == list(compare_pdf.EXPECTED)
    assert len(compare_pdf.EXPECTED) == 11 and all(n.endswith("_editable.pdf") for n in compare_pdf.EXPECTED)


def test_committed_reference_manifest_covers_the_eleven_files_with_a_symbolic_source():
    m = json.loads(MANIFEST.read_text())
    assert list(m["files"]) == list(compare_pdf.EXPECTED) and len(m["files"]) == 11
    assert not Path(m["source"]).is_absolute() and "/" not in m["source"].split()[0]
    for word in ("835b750", "2c66219", "SOURCE_DATE_EPOCH=0", "not committed"):
        assert word in m["source"], word
    assert m["method"] == compare_pdf.METHOD and "sha256_drawing" in m["method"]
    for name, entry in m["files"].items():
        assert set(entry) == {"sha256_dateless", "sha256_drawing", "bytes", "pages", "width_pt",
                              "height_pt", "dates_neutralised"}, name
        for key in ("sha256_dateless", "sha256_drawing"):
            assert len(entry[key]) == 64 and int(entry[key], 16) >= 0, (name, key)
        assert entry["pages"] == 1 and entry["bytes"] > 10_000, name
        # matplotlib pages carry one /CreationDate; the two composites pypdf writes carry none
        assert entry["dates_neutralised"] == (0 if "composite" in name else 1), name
