"""``figures/figure4/figure4_create.py``: the cost per verdict on the committed cost arms and on
an arm built from the record fixtures with known cost blocks, one and several arms rendered (PNG
and PDF, the grouped sheet of render.py), the refusal of a directory without a cost block, and
the script's independence from the pipeline (it imports nothing from ``compass`` or
``experiments``, and runs as a file from any directory or as a module from ``figures/``).

The tests that read the committed arms take them from ``render.arms_dir()`` (``COMPASS_ARMS_DIR``
when it is set, else ``<repo>/../bench/arms``, the research layout) and skip when the arms are
absent.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import figure_records
import pytest

import render
from figure2 import figure2_common as f2
from figure4 import figure4_create

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
ARMS = render.arms_dir()


def _dirs() -> list[Path]:
    dirs = [ARMS / a for a in render.COST_ARMS if (ARMS / a).is_dir()]
    if not dirs:
        pytest.skip(f"no cost arm under {ARMS}")
    return dirs


# ---------------------------------------------------------------------------
# The counts
# ---------------------------------------------------------------------------

def test_cost_per_verdict_on_the_committed_arms():
    dirs = _dirs()
    numbers = figure4_create.count(dirs)
    assert list(numbers) == [d.name for d in dirs]
    for d in dirs:
        n, records = numbers[d.name], f2.load_records(d)
        with_cost = [r for r in records if isinstance(r.get("cost"), dict)]
        assert (n["n_with_cost"], n["n_without_cost"]) == (len(with_cost), len(records) - len(with_cost))
        assert n["n_with_cost"] > 0 and list(n["rules"]) == f2.rule_order(n["rules"])
        assert set(n["mcmc_seconds"]) == {"pipeline_validation", *figure4_create.EXCLUDED}
        # the referee: the validator's MCMC seconds over the candidates it judged, no tokens
        referee = n["rules"]["referee"]
        assert referee["verdicts"] == n["n_candidates"] == sum(
            len((r.get("pipeline") or {}).get("candidates") or []) for r in with_cost) > 0
        assert referee["tokens"] is None
        assert referee["seconds"] == pytest.approx(n["mcmc_seconds"]["pipeline_validation"] / referee["verdicts"])
        # a reading rule: the sums of its cost blocks over the records, per verdict
        for rule, v in n["rules"].items():
            if rule == "referee":
                continue
            blocks = [(r["cost"].get("by_rule") or {}).get(rule) or {} for r in with_cost]
            verdicts = sum(int(b.get("verdicts") or 0) for b in blocks)
            assert v["verdicts"] == verdicts > 0, (d.name, rule)
            assert v["seconds"] == pytest.approx(sum(float(b.get("wall_seconds") or 0.0) for b in blocks) / verdicts)
            assert v["tokens"] == pytest.approx(sum(int(b.get("total_tokens") or 0) for b in blocks) / verdicts)


def _with_cost(records: list[dict]) -> list[dict]:
    """A cost block per record: one verdict per row (candidates, controls, arbiter cases) for
    three reading rules at 3 s and 900 tokens each; 120 s of validator MCMC."""
    for rec in records:
        rows = (len(rec["pipeline"]["candidates"]) + len(rec.get("control") or [])
                + len((((rec.get("arbiter") or {}).get("rules") or {}).get("rows")) or []))
        rec["cost"] = {
            "by_rule": {rule: {"verdicts": rows, "wall_seconds": 3.0 * rows, "total_tokens": 900 * rows}
                        for rule in ("execution", "self", "control")},
            "mcmc_seconds": {"pipeline_validation": 120.0, "hidden_scoring": 60.0, "arbiter": 30.0,
                             "control": 30.0}}
    return records


def test_cost_per_verdict_on_a_fixture_arm(tmp_path):
    a, b, c = _with_cost([copy.deepcopy(r) for r in figure_records.records()])  # 5, 3 and 4 rows
    no_cost = copy.deepcopy(a)
    no_cost["seed"] = 9
    del no_cost["cost"]
    c["cost"]["by_rule"]["unknown_rule"] = {"verdicts": 4, "wall_seconds": 1.0, "total_tokens": 1}  # no family: ignored
    arm = figure_records.write_arm(tmp_path / "arms" / "cost", [a, b, c, no_cost])
    (arm / "scores").mkdir()
    (arm / "scores" / "zero_inflation_seed2_r1.json").write_text(json.dumps(
        {"pipeline": {"0": {"hidden": {"seconds": 7.5}, "fresh": {"seconds": 2.5}}}}))
    numbers = figure4_create.count([arm])["cost"]
    assert (numbers["n_with_cost"], numbers["n_without_cost"], numbers["n_candidates"]) == (3, 1, 5)
    assert list(numbers["rules"]) == ["execution", "self", "control", "referee"]
    for rule in ("execution", "self", "control"):
        assert numbers["rules"][rule] == {"verdicts": 12, "seconds": 3.0, "tokens": 900.0}
    assert numbers["rules"]["referee"] == {"verdicts": 5, "seconds": 72.0, "tokens": None}
    assert numbers["mcmc_seconds"] == {"pipeline_validation": 360.0, "hidden_scoring": 190.0,
                                       "arbiter": 90.0, "control": 90.0}
    assert figure4_create.side_file_seconds(arm) == 10.0
    assert figure4_create.row_label("cost", f2.load_records(arm)) == "cost"  # no model recorded
    assert figure4_create.row_label("e1_haiku", [{"type": "heavy_tails", "model": "claude-haiku-4-5-20251001"}]) == "Haiku 4.5\n(synthetic problems)"
    figure4_create.main([str(arm), "--out", str(tmp_path / "one.pdf")])
    figure4_create.main([str(arm), str(arm), "--out", str(tmp_path / "grouped.pdf"), "--grouped"])
    assert (tmp_path / "one.pdf").stat().st_size > 5_000 and (tmp_path / "grouped.pdf").stat().st_size > 5_000


# ---------------------------------------------------------------------------
# Rendering and independence from the pipeline
# ---------------------------------------------------------------------------

def test_renders_one_and_several_arms(tmp_path, capsys):
    dirs = _dirs()
    one = figure4_create.main([str(dirs[0]), "--out", str(tmp_path / "one.png"), "--json", str(tmp_path / "one.json")])
    figure4_create.main([str(d) for d in dirs[:2]] + ["--out", str(tmp_path / "two.png")])
    figure4_create.main([str(dirs[0]), "--out", str(tmp_path / "one.pdf")])
    figure4_create.main([str(d) for d in dirs] + ["--out", str(tmp_path / "grouped.pdf"), "--grouped"])
    assert (tmp_path / "one.png").stat().st_size > 10_000 and (tmp_path / "two.png").stat().st_size > 10_000
    assert (tmp_path / "one.pdf").stat().st_size > 5_000 and (tmp_path / "grouped.pdf").stat().st_size > 5_000
    assert json.loads((tmp_path / "one.json").read_text()) == one
    out = capsys.readouterr().out
    assert "figure4" in out and "verdicts" in out and "hidden-fold scoring" in out
    with pytest.raises(SystemExit, match="cost block"):
        figure4_create.main([str(tmp_path), "--out", str(tmp_path / "none.png")])


def test_script_does_not_import_the_pipeline_and_runs_both_ways(tmp_path):
    text = (FIGURES_DIR / "figure4" / "figure4_create.py").read_text()
    assert "import bench" not in text and "from bench" not in text
    for root in ("compass", "experiments"):
        assert f"import {root}" not in text and f"from {root}" not in text, root
    arm = _dirs()[-1]
    # as a file from any directory, and as a module from the figures/ directory
    for argv, cwd in (([str(FIGURES_DIR / "figure4" / "figure4_create.py")], tmp_path),
                      (["-m", "figure4.figure4_create"], FIGURES_DIR)):
        out = tmp_path / f"{Path(argv[-1]).name}.png"
        proc = subprocess.run([sys.executable, *argv, str(arm), "--out", str(out)], cwd=cwd,
                              capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, proc.stderr[-800:]
        assert out.is_file() and "figure4" in proc.stdout
