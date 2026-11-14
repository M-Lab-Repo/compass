"""The figure2/*_create.py scripts (``figures/figure2/``): the counts behind each panel on the
committed benchmark arms and on an edge-case arm built from the record fixtures, the label rule
by hand, every panel rendered as PNG and as PDF with editable text, the composed sheet, and the
scripts' independence from the pipeline (a script imports nothing from ``compass`` or
``experiments``, and runs as a file from any directory or as a module from ``figures/``).

The tests that read the committed arms take them from ``render.arms_dir()`` (``COMPASS_ARMS_DIR``
when it is set, else ``<repo>/../bench/arms``, the research layout) and skip when the arms are
absent; everything else builds its records under ``tmp_path`` from ``figure_records.py``.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import figure_records
import numpy as np
import pytest

import render
from figure2 import (
    figure2_common as f2,
)
from figure2 import (
    figure2_compose,
    figure2a_create,
    figure2b_create,
    figure2c_create,
    figure2d_create,
)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
ARMS = render.arms_dir()
E5_ARMS = tuple(a for a in render.CENSUS_ARMS if a.startswith("e5_"))  # the correct-baseline arms
RULES_IN_FIXTURE = ("execution", "self", "control", "mcmc")


def _census_dirs() -> list[Path]:
    dirs = [ARMS / a for a in render.CENSUS_ARMS if (ARMS / a).is_dir()]
    if not dirs:
        pytest.skip(f"no census arm under {ARMS}")
    return dirs


def _candidates(d: Path) -> int:
    return sum(len((rec.get("pipeline") or {}).get("candidates") or []) for rec in f2.load_records(d))


# ---------------------------------------------------------------------------
# The counts on the committed arms
# ---------------------------------------------------------------------------

def test_2a_counts_every_candidate_of_the_committed_arms_once():
    dirs = _census_dirs()
    numbers = figure2a_create.count(dirs)
    assert list(numbers) == [d.name for d in dirs]
    for d in dirs:
        assert tuple(numbers[d.name]) == f2.LABELS
        assert sum(numbers[d.name].values()) == _candidates(d) > 0, d.name
    # a real-data arm has no fresh draw, so its labels come from the hidden fold alone
    for d in dirs:
        if "realdata" in d.name:
            assert f2.condition(d.name, f2.load_records(d)) == "real data"


def test_2b_cells_are_accepted_over_judged_per_rule_and_source():
    dirs = _census_dirs()
    numbers = figure2b_create.count(dirs)
    assert numbers["rules"] == f2.rule_order(numbers["rules"]) and numbers["rules"][-1] == "mcmc"
    assert {"execution", "self", "control"} <= set(numbers["rules"])
    for d in dirs:
        cells = numbers["arms"][d.name]
        assert list(cells) == numbers["rules"]
        for rule in numbers["rules"]:
            assert set(cells[rule]) == set(figure2b_create.SOURCES)
            for n, k in cells[rule].values():
                assert 0 <= k <= n
        # every pipeline row the execution check judged is one of the arm's candidates
        assert cells["execution"]["pipeline"][0] <= _candidates(d)


def test_2d_counts_runs_not_candidates_on_the_correct_baseline_arms():
    e5 = [d for d in _census_dirs() if d.name in E5_ARMS]
    if not e5:
        pytest.skip("no correct-baseline arm committed")
    numbers = figure2d_create.count(e5)
    assert numbers["rules"][-1] == "referee" and numbers["k_alpha"] == 4.6
    for d in e5:
        arm = numbers["arms"][d.name]
        assert arm["n_records"] + arm["n_crashed"] == len(f2.load_records(d))
        assert arm["replaced"]["referee"][0] == arm["n_records"]
        for runs, replaced in arm["replaced"].values():
            assert 0 <= replaced <= runs <= arm["n_records"]
    # one arm alone gives panel d itself, the same numbers for that arm
    assert figure2d_create.count([e5[0]])["arms"][e5[0].name] == numbers["arms"][e5[0].name]
    e1 = [d for d in _census_dirs() if d.name.startswith("e1_")]  # misspecified baselines: refused
    if e1:
        with pytest.raises(ValueError, match="null_baseline"):
            figure2d_create.count(e1[:1])


def test_2c_curves_rise_with_n_under_p_plus_and_stay_under_the_bound_under_p_minus():
    numbers = figure2c_create.simulate(reps=400, seed=0)
    assert numbers["n"] == list(figure2c_create.NS) and numbers["bound"] == pytest.approx(0.023)
    assert numbers["accept_plus"][-1] > 0.99 > 0.5 > numbers["accept_plus"][0]
    # a weaker help is accepted later: well under the strong curve at n = 500, past one half at n = 1000
    assert numbers["accept_plus_weak"][-2] < 0.3 < 0.5 < numbers["accept_plus_weak"][-1] <= numbers["accept_plus"][-1]
    assert max(numbers["accept_minus"]) <= 0.05  # the bound is Phi(-2) = 0.023
    # the same seed gives the same curves; the closed-form S gate accepts the obvious case
    assert figure2c_create.simulate(reps=400, seed=0) == numbers
    y = np.full((1, 50), 3.0)
    y[0, ::2] = 2.0  # a nonzero spread keeps SE > 0
    assert figure2c_create.s_gate_accepts(y).tolist() == [True]
    assert figure2c_create.s_gate_accepts(np.zeros((1, 50))).tolist() == [False]
    arm = ARMS / "e1_anthropic"
    if not arm.is_dir():
        pytest.skip("e1_anthropic not committed")
    flat = figure2c_create.flat_rates(f2.load_records(arm))
    assert tuple(flat) == figure2c_create.FLAT_RULES
    for rule in flat.values():
        assert 0 < rule["accepted"] <= rule["n"] <= _candidates(arm)
        assert rule["rate"] == pytest.approx(rule["accepted"] / rule["n"])


# ---------------------------------------------------------------------------
# Edge cases the committed arms do not exercise
# ---------------------------------------------------------------------------

# Counts of the edge-case arms below, pinned on 2026-11-14 against the research tree's older
# figure code (its census, acceptance and counterfactual tallies agreed on these fixtures).
PINNED_2A = {
    "synth_edge": {"genuine": 1, "confabulated": 4, "harmful": 1, "underpowered": 0, "unavailable": 1},
    "real_edge": {"genuine": 0, "confabulated": 0, "harmful": 0, "underpowered": 2, "unavailable": 0},
}
PINNED_2B = {
    "synth_edge": {
        "execution": {"pipeline": [5, 5], "control": [6, 6], "arbiter": [2, 2]},
        "self": {"pipeline": [4, 4], "control": [6, 2], "arbiter": [2, 2]},
        "control": {"pipeline": [5, 4], "control": [6, 2], "arbiter": [0, 0]},
        "mcmc": {"pipeline": [5, 1], "control": [6, 0], "arbiter": [2, 0]},
    },
    # every row of the real arm is "underpowered", a label panel b does not count
    "real_edge": {rule: {"pipeline": [0, 0], "control": [0, 0], "arbiter": [0, 0]}
                  for rule in RULES_IN_FIXTURE},
}
PINNED_2D_REPLACED = {"execution": [4, 3], "self": [4, 2], "control": [4, 2], "referee": [4, 2]}


@pytest.fixture
def edge_arms(tmp_path):
    """A synthetic arm with None verdicts, flat-key arbiter rows, a crashed run and a run with
    no candidate; a real-data arm with a 5-point fold; plus a summary and a broken file."""
    a, b, c = figure_records.records()
    crashed = copy.deepcopy(a)
    crashed["seed"] = 7
    crashed["pipeline"]["ok"] = False
    empty = copy.deepcopy(c)
    empty["seed"] = 8
    empty["pipeline"]["candidates"] = []
    empty["pipeline"]["validation_passed"] = False
    synth = figure_records.write_arm(tmp_path / "arms" / "synth_edge", [a, b, c, crashed, empty])
    real = copy.deepcopy(a)
    real["type"], real["seed"] = "roaches", 0
    for row in real["pipeline"]["candidates"] + real["control"]:
        row["fresh"] = None                       # real data has no fresh draw
        row["hidden"]["n_hidden"] = 5             # a 5-point fold: confabulated -> underpowered
    real_dir = figure_records.write_arm(tmp_path / "arms" / "real_edge", [real])
    return synth, real_dir


def test_edge_fixture_counts(edge_arms):
    synth, real = edge_arms
    assert figure2a_create.count([synth, real]) == PINNED_2A
    assert figure2a_create.count([real])["real_edge"]["underpowered"] == 2
    numbers = figure2b_create.count([synth, real])
    assert numbers["rules"] == list(RULES_IN_FIXTURE)
    assert numbers["arms"] == PINNED_2B
    d2 = figure2d_create.count([synth])
    assert d2["rules"] == ["execution", "self", "control", "referee"] and d2["k_alpha"] == 4.6
    assert d2["arms"]["synth_edge"]["replaced"] == PINNED_2D_REPLACED
    assert d2["arms"]["synth_edge"]["n_crashed"] == 1 and d2["arms"]["synth_edge"]["n_records"] == 4
    with pytest.raises(ValueError, match="null_baseline"):
        bad = json.loads((synth / "overdispersion_seed0_r1.json").read_text())
        bad["flags"]["null_baseline"] = False
        (synth / "overdispersion_seed0_r1.json").write_text(json.dumps(bad))
        figure2d_create.count([synth])


def test_label_rule_by_hand():
    rec = {"type": "overdispersion"}
    assert f2.label_of({"fresh": {"label": "harmful"}, "hidden": {"label": "genuine"}}, rec) == "harmful"
    assert f2.label_of({"fresh": {"label": "unavailable"}, "hidden": {"label": "genuine"}}, rec) == "genuine"
    assert f2.label_of({"hidden": {"label": "confabulated", "n_hidden": 5}}, {"type": "roaches"}) == "underpowered"
    assert f2.label_of({"hidden": {"label": "confabulated", "n_hidden": 38}}, {"type": "roaches"}) == "confabulated"
    assert f2.label_of({"hidden_label": "genuine", "hidden_delta": 1.0}, rec) == "genuine"
    assert f2.label_of({}, rec) == "unavailable"
    assert f2.verdict({"mcmc_passed": True}, "mcmc") is True
    assert f2.verdict({"judges": {"self": None}}, "self") is None
    assert f2.rule_order({"mcmc", "independent:b", "independent:a", "control", "self", "execution"}) == [
        "execution", "self", "control", "independent:a", "independent:b", "mcmc"]
    assert f2.SMALL_FOLD == 10 and len(f2.SYNTHETIC_TYPES) == 8
    assert f2.condition("e5_anthropic", [{"type": "overdispersion"}]) == "E5"
    assert f2.condition("e1_anthropic_realdata_x2", [{"type": "roaches"}]) == "real data"
    assert f2.arm_label("e1_haiku", [{"type": "heavy_tails", "model": "claude-haiku-4-5-20251001"}]) == "Haiku 4.5, E1"


# ---------------------------------------------------------------------------
# Rendering and independence from the pipeline
# ---------------------------------------------------------------------------

def test_every_script_renders(tmp_path, capsys):
    dirs = _census_dirs()
    two = [str(d) for d in dirs[:2]]
    e5 = [str(d) for d in dirs if d.name in E5_ARMS] or [str(dirs[0])]
    outs = {
        "a": figure2a_create.main([*two, "--out", str(tmp_path / "a.png"), "--json", str(tmp_path / "a.json")]),
        "b": figure2b_create.main([*two, "--out", str(tmp_path / "b.png")]),
        "c": figure2c_create.main([str(dirs[-1]), "--out", str(tmp_path / "c.png"), "--reps", "50"]),
        "d": figure2d_create.main([e5[0], "--out", str(tmp_path / "d.png")]),
        "d_fam": figure2d_create.main([*e5, "--out", str(tmp_path / "d_fam.png")]),
    }
    for name in ("a", "b", "c", "d", "d_fam"):
        assert (tmp_path / f"{name}.png").stat().st_size > 10_000, name
    assert json.loads((tmp_path / "a.json").read_text()) == outs["a"]
    out = capsys.readouterr().out
    total = sum(outs["a"][dirs[0].name].values())
    assert f"{total:5d}" in out and "figure2b" in out and "flat: execution" in out
    info = figure2_compose.main([str(tmp_path / f"{p}.png") for p in "abcd"] + ["--out", str(tmp_path / "sheet.png")])
    assert (tmp_path / "sheet.png").stat().st_size > 20_000 and info["size"][0] > info["size"][1] / 2
    # vector output: every panel as PDF with TrueType text, composed on one editable page
    from pypdf import PdfReader
    figure2a_create.main([*two, "--out", str(tmp_path / "a.pdf")])
    figure2b_create.main([*two, "--out", str(tmp_path / "b.pdf")])
    figure2c_create.main([str(dirs[-1]), "--out", str(tmp_path / "c.pdf"), "--reps", "50"])
    figure2d_create.main([e5[0], "--out", str(tmp_path / "d.pdf")])
    figure2_compose.main([str(tmp_path / f"{p}.pdf") for p in "abcd"] + ["--out", str(tmp_path / "sheet.pdf")])
    sheet = PdfReader(str(tmp_path / "sheet.pdf"))
    assert len(sheet.pages) == 1 and not (tmp_path / "sheet.letters.pdf").exists()
    text = sheet.pages[0].extract_text()
    assert "When nothing is wrong" in text and "Who accepts them" in text        # text, not outlines
    fonts = {str(f.get_object().get("/Subtype")) for f in sheet.pages[0]["/Resources"]["/Font"].values()}
    assert fonts <= {"/TrueType", "/Type0"} and "/Type3" not in fonts
    pymupdf = pytest.importorskip("pymupdf")  # rasteriser for the overlay check; not a requirement
    pix = pymupdf.open(str(tmp_path / "sheet.pdf"))[0].get_pixmap(dpi=36)
    dark = sum(1 for i in range(0, len(pix.samples), pix.n) if pix.samples[i] < 200)
    assert dark > 0.02 * pix.width * pix.height          # the panels are drawn, not hidden by the overlay


def test_scripts_do_not_import_the_pipeline_and_run_both_ways(tmp_path):
    # the drawing scripts read records with the standard library and draw with matplotlib;
    # neither the pipeline nor the benchmark modules that the run and score scripts use is
    # imported by them
    drawing = sorted(p for p in (FIGURES_DIR / "figure2").glob("*.py")
                     if p.name.endswith(("_create.py", "_common.py", "_compose.py")))
    assert [p.stem for p in drawing] == ["figure2_common", "figure2_compose", "figure2a_create",
                                         "figure2b_create", "figure2c_create", "figure2d_create"]
    for path in drawing:
        text = path.read_text()
        assert "import bench" not in text and "from bench" not in text, path.name
        for root in ("compass",):
            assert f"import {root}" not in text and f"from {root}" not in text, (path.name, root)
    dirs = [str(d) for d in _census_dirs()[:1]]
    # the two ways a person runs a script: as a file from any directory (its own directory is
    # then on the path) and as a module from the figures/ directory
    for argv, cwd in (([str(FIGURES_DIR / "figure2" / "figure2a_create.py")], tmp_path),
                      (["-m", "figure2.figure2a_create"], FIGURES_DIR)):
        out = tmp_path / f"{Path(argv[-1]).name}.png"
        proc = subprocess.run([sys.executable, *argv, *dirs, "--out", str(out)], cwd=cwd,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-800:]
        assert out.exists() and "figure2a" in proc.stdout
