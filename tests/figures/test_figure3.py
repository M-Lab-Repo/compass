"""The figure3/*_create.py scripts (``figures/figure3/``): the drift medians, the mechanism
counts and the real-data matrix on the committed arms, the round rule and the latent-site count
by hand, unscored rounds and side files on a fixture, every panel rendered as PNG and as PDF, the
composed sheet with its two row titles, and the scripts' independence from the pipeline (a
script imports nothing from ``compass`` or ``experiments``, and runs as a file from any
directory or as a module from ``figures/``).

The tests that read the committed arms take them from ``render.arms_dir()`` (``COMPASS_ARMS_DIR``
when it is set, else ``<repo>/../bench/arms``, the research layout) and skip when the arms are
absent: the synthetic drift arm ``e2_anthropic/drift``, the real drift arm
``e2_anthropic_real/drift`` and the real-data census arm ``e1_anthropic_realdata_x2``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

import render
from figure2 import figure2_common as f2
from figure3 import (
    figure3_common as f3,
)
from figure3 import (
    figure3_compose,
    figure3a_create,
    figure3b_create,
    figure3c_create,
)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
ARMS = render.arms_dir()
DRIFT = ARMS / "e2_anthropic" / "drift"
REAL_DRIFT = ARMS / "e2_anthropic_real" / "drift"
MATRIX = ARMS / "e1_anthropic_realdata_x2"
RULES = ["execution", "self", "control", "referee"]  # the four rules of the drift arms, in order
ROUNDS = [0, 1, 2, 3, 4, 5, 6]                       # round 0 is the starting model


def _need(*paths_):
    for p in paths_:
        if not p.is_dir() or not any(p.glob("*.json")):
            pytest.skip(f"{p} not committed")


# ---------------------------------------------------------------------------
# The counts on the committed arms
# ---------------------------------------------------------------------------

def test_3a_medians_on_the_committed_drift_arm():
    _need(DRIFT)
    numbers, series = figure3a_create.count(DRIFT)
    assert list(numbers["rules"]) == list(series) == RULES
    assert (numbers["n_dropped"], numbers["n_dropped_rounds"]) == (0, 8)
    for rule, d in numbers["rules"].items():
        assert d["n"] == len(series[rule]) == 8, rule          # the eight synthetic problems
        assert d["rounds"] == ROUNDS and d["median"][0] == d["q1"][0] == d["q3"][0] == 0.0
        assert len(d["median"]) == len(d["q1"]) == len(d["q3"]) == len(ROUNDS)
        assert all(q1 <= m <= q3 for q1, m, q3 in zip(d["q1"], d["median"], d["q3"], strict=True))
        for pts in series[rule]:
            assert pts[0] == 0.0 and set(pts) <= set(ROUNDS)
    assert numbers["rules"]["referee"]["median"] == [0.0] * len(ROUNDS)  # the referee stayed put


def test_3c_real_drift_on_the_committed_real_arm():
    _need(REAL_DRIFT)
    numbers, series = figure3a_create.count(REAL_DRIFT)
    assert list(numbers["rules"]) == RULES
    assert (numbers["n_dropped"], numbers["n_dropped_rounds"]) == (0, 1)
    for rule, d in numbers["rules"].items():
        assert d["n"] == len(series[rule]) == 7, rule          # the seven textbook datasets
        assert d["rounds"] == ROUNDS and d["median"][0] == 0.0
    assert numbers["rules"]["referee"]["median"] == [0.0] * len(ROUNDS)
    assert max(numbers["rules"]["execution"]["median"]) > 5.0  # why the panel is drawn --symlog


def test_3b_mechanism_on_the_committed_drift_arm():
    _need(DRIFT)
    numbers = figure3b_create.count(DRIFT)
    assert list(numbers) == RULES
    for rule, d in numbers.items():
        assert d["rounds"] == ROUNDS and d["frac_changed"][0] == 0.0, rule
        assert len(d["frac_changed"]) == len(d["sites"]) == len(ROUNDS)
        assert all(0.0 <= f <= 1.0 for f in d["frac_changed"])
        assert (d["n"], d["n_rounds"]) == (8, 48) and 0 <= d["changed"] <= d["n_rounds"]
        assert d["sites"][0] == numbers["execution"]["sites"][0]  # the same eight starting models
    assert numbers["execution"]["changed"] == 48                   # changed in every round
    assert numbers["referee"]["changed"] == 1
    assert numbers["referee"]["sites"] == [numbers["referee"]["sites"][0]] * len(ROUNDS)


def test_3d_matrix_on_the_committed_real_arm():
    _need(MATRIX)
    numbers = figure3c_create.count([MATRIX])
    assert numbers["datasets"] == list(f3.REAL_ORDER)
    assert numbers["rules"][-1] == "referee" and numbers["rules"][:3] == ["execution", "self", "control"]
    assert (numbers["n_records"], numbers["n_crashed"]) == (21, 0)  # three runs per dataset
    for ds in numbers["datasets"]:
        assert list(numbers["rows"][ds]) == numbers["rules"]
        for cell in numbers["rows"][ds].values():
            assert 0 <= cell["k"] <= cell["n"] <= 3 and (cell["label"] is None) == (cell["n"] == 0)
        assert numbers["deployed"][ds]["n"] == 3 >= numbers["deployed"][ds]["deployed"]
    small = {ds for ds, n in numbers["n_hidden"].items() if n < f2.SMALL_FOLD}
    assert small == {"ucbadmit", "baseball", "seeds", "golf"}
    for ds in small:
        for cell in numbers["rows"][ds].values():
            assert cell["label"] in ("underpowered", figure3c_create.NOTHING, figure3c_create.UNSCORED, None)
    assert numbers["rows"]["roaches"]["referee"] == {"label": "genuine", "k": 3, "n": 3}
    assert numbers["deployed"]["roaches"]["deployed"] == 3


# ---------------------------------------------------------------------------
# The rules by hand and on fixtures
# ---------------------------------------------------------------------------

def test_round_delta_and_site_count_by_hand():
    rec = {"type": "overdispersion"}
    fresh, hidden = {"available": True, "delta": 2.0, "label": "genuine"}, {"available": True, "delta": -1.0, "label": "harmful"}
    assert f3.round_delta({"fresh": fresh, "hidden": hidden}, rec) == 2.0
    assert f3.round_delta({"fresh": {"available": False, "label": "unavailable"}, "hidden": hidden}, rec) == -1.0
    assert f3.round_delta({"fresh": fresh, "hidden": hidden}, {"type": "roaches"}) == -1.0  # real data: no fresh draw
    assert f3.round_delta({"hidden": {"available": False}}, rec) is None
    assert f3.series({"type": "roaches", "rounds": [{"round": 1, "hidden": hidden}, {"round": 2, "hidden": {"available": False}}]}) == {0: 0.0, 1: -1.0}
    code = ('def model(x=None, y=None):\n    a = numpyro.sample("a", dist.Normal(0, 1))\n'
            '    with numpyro.plate("obs", 3):\n        numpyro.sample("y", dist.Normal(a, 1), obs=y)\n'
            '    numpyro.sample("b", dist.Normal(0, (1)))\n')
    assert f3.count_latent_sites(code) == 2                       # "a" and "b"; "y" carries obs=
    assert f3.count_latent_sites("") == f3.count_latent_sites(None) == 0
    assert f3.count_latent_sites('numpyro.sample("c", dist.Normal(0, "(")\n') == 1  # unbalanced: the line
    assert f3.REAL_ORDER == ("roaches", "ucbadmit", "baseball", "seeds", "radon", "kidiq", "golf")
    assert f3.dataset_order(["zzz", "golf", "roaches"]) == ["roaches", "golf", "zzz"]
    assert f3.dataset_order(["b", "a", "kidiq", "radon", "a"]) == ["radon", "kidiq", "a", "b"]
    assert (f3.dataset_name("ucbadmit"), f3.dataset_name("kidiq"), f3.dataset_name("golf")) == ("UCBadmit", "KidIQ", "golf")
    assert f3.legend_name("control") == "Control judge" and f3.legend_name("mcmc") == "Referee"


def _trajectory(rule, deltas):
    return {"type": "overdispersion", "seed": 0, "rule": rule, "baseline0_code": "", "rounds": [
        {"round": k, "changed": True, "n_sample_sites": 3,
         "fresh": ({"available": True, "delta": x, "label": "genuine"} if x is not None
                   else {"available": False, "label": "unavailable"})} for k, x in enumerate(deltas, 1)]}


def test_trajectory_keys_come_from_the_file_name_when_the_record_lacks_them(tmp_path):
    d = tmp_path / "drift"
    d.mkdir()
    (d / "heavy_tails_seed4_control.json").write_text(json.dumps({"rounds": []}))
    (d / "overdispersion_seed0_self.json").write_text(json.dumps(_trajectory("self", [1.0])))
    (d / "no_rounds.json").write_text(json.dumps({"type": "heavy_tails", "seed": 1, "rule": "self"}))
    recs = f3.load_trajectories(d)
    assert [(r["type"], r["seed"], r["rule"]) for r in recs] == [("heavy_tails", 4, "control"),
                                                                 ("overdispersion", 0, "self")]
    assert recs[1]["seed"] == 0 and recs[1]["rounds"][0]["round"] == 1  # the record's own keys stay


def test_unscored_rounds_are_left_out_and_side_files_win(tmp_path):
    d = tmp_path / "drift"
    d.mkdir()
    (d / "overdispersion_seed0_self.json").write_text(json.dumps(_trajectory("self", [1.0, None, 3.0])))
    (d / "overdispersion_seed1_self.json").write_text(json.dumps(_trajectory("self", [None, None, None])))
    (d / "notes.json").write_text("{}")
    (d / "broken.json").write_text("{not json")
    assert len(f3.load_trajectories(d)) == 2  # the notes and the broken file are skipped
    numbers, series = figure3a_create.count(d)
    assert numbers["rules"]["self"]["rounds"] == [0, 1, 3]
    assert (numbers["n_dropped"], numbers["n_dropped_rounds"]) == (1, 4)
    assert series["self"] == [{0: 0.0, 1: 1.0, 3: 3.0}]
    (d / "scores").mkdir()
    (d / "scores" / "overdispersion_seed0_self.json").write_text(json.dumps(
        {"rounds": {"2": {"fresh": {"available": True, "delta": 2.0, "label": "genuine"}}}}))
    numbers, _ = figure3a_create.count(d)
    assert numbers["rules"]["self"]["rounds"] == [0, 1, 2, 3] and numbers["rules"]["self"]["median"] == [0.0, 1.0, 2.0, 3.0]
    mech = figure3b_create.count(d)
    assert mech["self"] == {"rounds": [0, 1, 2, 3], "frac_changed": [0.0, 1.0, 1.0, 1.0], "sites": [0.0, 3.0, 3.0, 3.0],
                            "n": 2, "changed": 6, "n_rounds": 6}
    with pytest.raises(SystemExit, match="no scored trajectories"):
        figure3a_create.main([str(tmp_path), "--out", str(tmp_path / "none.png")])
    with pytest.raises(SystemExit, match="no trajectories"):
        figure3b_create.main([str(tmp_path), "--out", str(tmp_path / "none.png")])


# ---------------------------------------------------------------------------
# Rendering and independence from the pipeline
# ---------------------------------------------------------------------------

def test_every_script_renders_png_and_pdf_and_composes(tmp_path, capsys):
    """Panels a (drift), b (mechanism), c (real drift, --symlog) and d (matrix), as PNG and as
    PDF, and the sheet of figure3_compose with the two row titles of render.ROW_TITLES."""
    _need(DRIFT, REAL_DRIFT, MATRIX)
    a = figure3a_create.main([str(DRIFT), "--out", str(tmp_path / "a.png"), "--json", str(tmp_path / "a.json")])
    figure3b_create.main([str(DRIFT), "--out", str(tmp_path / "b.png")])
    figure3a_create.main([str(REAL_DRIFT), "--out", str(tmp_path / "c.png"), "--symlog"])
    figure3c_create.main([str(MATRIX), "--out", str(tmp_path / "d.png")])
    for p in "abcd":
        assert (tmp_path / f"{p}.png").stat().st_size > 10_000, p
    out = capsys.readouterr().out
    assert all(s in out for s in ("figure3a", "figure3b", "figure3c", "left out:", "deployed 3/3"))
    assert json.loads((tmp_path / "a.json").read_text()) == a and a["rules"]["referee"]["n"] == 8
    panels = [str(tmp_path / f"{p}.png") for p in "abcd"]
    info = figure3_compose.main([*panels, "--out", str(tmp_path / "sheet.png"), "--row-titles", *render.ROW_TITLES])
    assert (tmp_path / "sheet.png").stat().st_size > 20_000
    # two columns at the widest panel's width, with a gutter on either side and between them
    widest = max(Image.open(p).width for p in panels)
    assert info["size"][0] == 2 * widest + 3 * figure3_compose.GUTTER
    figure3a_create.main([str(DRIFT), "--out", str(tmp_path / "a.pdf")])
    figure3b_create.main([str(DRIFT), "--out", str(tmp_path / "b.pdf")])
    figure3a_create.main([str(REAL_DRIFT), "--out", str(tmp_path / "c.pdf"), "--symlog"])
    figure3c_create.main([str(MATRIX), "--out", str(tmp_path / "d.pdf")])
    figure3_compose.main([str(tmp_path / f"{p}.pdf") for p in "abcd"]
                         + ["--out", str(tmp_path / "sheet.pdf"), "--row-titles", *render.ROW_TITLES])
    from pypdf import PdfReader
    sheet = PdfReader(str(tmp_path / "sheet.pdf"))
    text = sheet.pages[0].extract_text()
    assert len(sheet.pages) == 1 and not (tmp_path / "sheet.letters.pdf").exists()
    for title in (*render.ROW_TITLES, "Deployed model over successive rounds", "Why the deployed model drifts",
                  "Outcome of the accepted revision"):
        assert title in text, title
    fonts = {str(f.get_object().get("/Subtype")) for f in sheet.pages[0]["/Resources"]["/Font"].values()}
    assert fonts <= {"/TrueType", "/Type0"}


def test_drawing_scripts_do_not_import_the_pipeline_and_run_both_ways(tmp_path):
    drawing = sorted(p for p in (FIGURES_DIR / "figure3").glob("*.py")
                     if p.name.endswith(("_create.py", "_common.py", "_compose.py")))
    assert [p.stem for p in drawing] == ["figure3_common", "figure3_compose", "figure3a_create",
                                         "figure3b_create", "figure3c_create"]
    for path in drawing:
        text = path.read_text()
        assert "import bench" not in text and "from bench" not in text, path.name
        for root in ("compass",):
            assert f"import {root}" not in text and f"from {root}" not in text, (path.name, root)
    _need(DRIFT)
    # as a file from any directory, and as a module from the figures/ directory
    for argv, cwd in (([str(FIGURES_DIR / "figure3" / "figure3b_create.py")], tmp_path),
                      (["-m", "figure3.figure3b_create"], FIGURES_DIR)):
        out = tmp_path / f"{Path(argv[-1]).name}.png"
        proc = subprocess.run([sys.executable, *argv, str(DRIFT), "--out", str(out)], cwd=cwd,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-800:]
        assert out.exists() and "figure3b" in proc.stdout
