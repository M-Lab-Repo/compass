"""``figures/render.py``: the eleven names and their files, ``--list``, the data-root precedence
and the missing-arms stop, the composite-input check, failure isolation, the byte-identical
render across timezones (``SOURCE_DATE_EPOCH``, in subprocesses under ``TZ=UTC`` and
``TZ=EST5EDT``), and a full render of all eleven PDFs from a synthetic data root built under
``tmp_path`` with the record fixtures of ``figure_records.py`` (no test reads the committed
benchmark records).
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import figure_records
import numpy as np
import pytest
from pypdf import PdfReader

import compare_pdf
import render

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"
NAMES = ["fig2a", "fig2b", "fig2c", "fig2d", "fig2_composite", "fig3a", "fig3b", "fig3c", "fig3d",
         "fig3_composite", "fig4_cost"]
REQUIRED = [*render.CENSUS_ARMS, "e2_anthropic/drift", "e2_anthropic_real/drift", "e1_anthropic_realdata"]
MODEL = '''def model(y=None):
    a = numpyro.sample("a", dist.Normal(0, 1))
    numpyro.sample("y", dist.Normal(a, 1), obs=y)
'''


@pytest.fixture(autouse=True)
def _no_env_root(monkeypatch):
    """Nothing in this file may pick up a data root from the caller's environment."""
    monkeypatch.delenv(render.ENV_VAR, raising=False)


def _main(argv: list[str]) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return render.main(argv)


# ---- the table ---------------------------------------------------------------------------------

def test_names_files_scripts_and_inputs():
    assert list(render.NAMES) == NAMES
    for entry in render.ENTRIES:
        script = FIGURES_DIR / entry.script.split()[0]
        assert script.is_file(), entry.script
        assert render.output(Path("o"), entry.name, "pdf") == Path("o") / f"{entry.name}_editable.pdf"
        assert render.output(Path("o"), entry.name, "png") == Path("o") / f"{entry.name}.png"
        assert all(i in render.BY_NAME for i in entry.inputs)
        assert all(isinstance(a, str) for a in entry.arms)
        assert bool(entry.arms) != bool(entry.inputs)  # a panel reads arms, a composite reads panels
        argv = entry.argv(Path("/root"), Path("/out"), "pdf")
        assert "--out" in argv and argv[argv.index("--out") + 1] == f"/out/{entry.name}_editable.pdf"
    assert render.BY_NAME["fig2_composite"].inputs == ("fig2a", "fig2b", "fig2c", "fig2d")
    assert render.BY_NAME["fig3_composite"].inputs == ("fig3a", "fig3b", "fig3c", "fig3d")
    assert render.BY_NAME["fig3d"].arms == ("e1_anthropic_realdata_x2",)
    assert render.BY_NAME["fig4_cost"].arms == render.COST_ARMS and len(render.COST_ARMS) == 5
    assert "--symlog" in render.BY_NAME["fig3c"].argv(Path("r"), Path("o"), "pdf")
    assert "--grouped" in render.BY_NAME["fig4_cost"].argv(Path("r"), Path("o"), "pdf")
    argv = render.BY_NAME["fig3_composite"].argv(Path("r"), Path("o"), "pdf")
    assert argv[argv.index("--row-titles") + 1:] == list(render.ROW_TITLES) and len(render.ROW_TITLES) == 2
    assert render.required_arms(NAMES) == REQUIRED
    assert render.required_arms(["fig2_composite"]) == []
    assert render.required_arms(["fig3c", "fig3a"]) == ["e2_anthropic_real/drift", "e2_anthropic/drift"]


def test_list_prints_the_names_and_exits_zero(capsys):
    assert _main(["--list"]) == 0
    assert capsys.readouterr().out.split() == NAMES
    proc = subprocess.run([sys.executable, "-I", str(FIGURES_DIR / "render.py"), "--list"],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and proc.stdout.split() == NAMES


def test_unknown_only_name_returns_two(tmp_path, capsys):
    out = tmp_path / "out"
    assert _main(["--only", "fig2a,nope", "--arms", str(tmp_path), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "unknown name(s) ['nope']" in err and "fig2a" in err
    assert not out.exists()


# ---- the data root -------------------------------------------------------------------------------

def test_data_root_precedence(tmp_path, monkeypatch):
    assert render.arms_dir() == render.DEFAULT_ARMS_DIR and render.arms_source() == "default"
    assert render.DEFAULT_ARMS_DIR == FIGURES_DIR.parents[1] / "bench" / "arms"
    monkeypatch.setenv(render.ENV_VAR, str(tmp_path / "env"))
    assert render.arms_dir() == (tmp_path / "env").resolve() and render.arms_source() == "env"
    assert render.arms_dir(tmp_path / "flag") == (tmp_path / "flag").resolve()
    assert render.arms_source(tmp_path / "flag") == "flag"


def test_missing_data_root_stops_before_drawing(tmp_path):
    root = (tmp_path / "arms").resolve()
    root.mkdir()
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exc:
        _main(["--arms", str(root), "--out", str(out)])
    msg = str(exc.value)
    assert f"data root {root} (chosen by flag) is missing {len(REQUIRED)} item(s)" in msg
    for arm in REQUIRED:
        assert f"{root / arm}  (directory missing)" in msg
    assert "COMPASS_ARMS_DIR" in msg and "Nothing was rendered." in msg
    assert not out.exists()
    # a subset needs only its own arms; an empty directory is named as such
    (root / "e2_anthropic" / "drift").mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        _main(["--arms", str(root), "--out", str(out), "--only", "fig3a,fig3b"])
    msg = str(exc.value)
    assert "is missing 1 item(s)" in msg and f"{root / 'e2_anthropic' / 'drift'}  (no *.json records)" in msg


def test_missing_root_from_the_environment_is_named_as_such(tmp_path, monkeypatch):
    env_root = (tmp_path / "env_arms").resolve()
    monkeypatch.setenv(render.ENV_VAR, str(env_root))
    with pytest.raises(SystemExit) as exc:
        _main(["--out", str(tmp_path / "out"), "--only", "fig2d"])
    assert f"data root {env_root} (chosen by env)" in str(exc.value)
    assert not (tmp_path / "out").exists()


# ---- the date stamp: byte-identical across timezones ----------------------------------------------

def _subprocess_env(**over: str) -> dict[str, str]:
    """The test's environment without SOURCE_DATE_EPOCH (render.py must set it itself)."""
    env = {k: v for k, v in os.environ.items() if k not in (render.ENV_VAR, "SOURCE_DATE_EPOCH")}
    env.update(MPLBACKEND="Agg", **over)
    return env


def _run(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    proc = subprocess.run([sys.executable, *argv], capture_output=True, text=True, timeout=300,
                          env=env, check=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc


def test_render_is_byte_identical_across_timezones_and_a_bare_script_differs_only_in_metadata(tmp_path):
    """render.py sets SOURCE_DATE_EPOCH=0 before matplotlib is imported: the mechanism panel of
    a synthetic drift root rendered under TZ=UTC and under TZ=EST5EDT is the same file, stamped
    D:19700101000000Z; a caller's epoch is kept; the panel script run on its own in EST5EDT
    stamps a timezone offset (six bytes longer, the cross-reference pointer moves) and
    compare_pdf.py then reports the same drawing with differing metadata instead of DIFFERENT."""
    name = "fig3b_editable.pdf"
    root = tmp_path / "arms"
    drift = root / "e2_anthropic" / "drift"
    _drift(drift, synthetic=True)
    script = str(FIGURES_DIR / "render.py")
    rendered: dict[str, bytes] = {}
    for tz in ("UTC", "EST5EDT"):
        out = tmp_path / tz
        _run([script, "--arms", str(root), "--out", str(out), "--only", "fig3b"], _subprocess_env(TZ=tz))
        rendered[tz] = (out / name).read_bytes()
    assert rendered["UTC"] == rendered["EST5EDT"]
    assert compare_pdf.date_values(rendered["UTC"]) == ["D:19700101000000Z"]
    assert compare_pdf.describe(tmp_path / "UTC" / name)["dates_neutralised"] == 1
    # the caller's epoch wins over the default: another stamp of the same length, nothing else
    out = tmp_path / "epoch"
    _run([script, "--arms", str(root), "--out", str(out), "--only", "fig3b"],
         _subprocess_env(TZ="EST5EDT", SOURCE_DATE_EPOCH="1234567890"))
    data = (out / name).read_bytes()
    assert compare_pdf.date_values(data) == ["D:20090213233130Z"] and len(data) == len(rendered["UTC"])
    lines, bad = compare_pdf.compare(out, tmp_path / "UTC", (name,))
    assert bad == 0 and "IDENTICAL (dates aside; 1 date value(s) differ)" in lines[0]
    # the panel script on its own sets no epoch: in EST5EDT the stamp carries an offset
    bare = tmp_path / "bare"
    bare.mkdir()
    _run([str(FIGURES_DIR / "figure3" / "figure3b_create.py"), str(drift), "--out", str(bare / name)],
         _subprocess_env(TZ="EST5EDT"))
    data = (bare / name).read_bytes()
    (stamp,) = compare_pdf.date_values(data)
    assert stamp[-7:] in ("-05'00'", "-04'00'") and len(data) == len(rendered["UTC"]) + 6
    got, ref = compare_pdf.describe(bare / name), compare_pdf.describe(tmp_path / "UTC" / name)
    assert got["sha256_dateless"] != ref["sha256_dateless"] and got["sha256_drawing"] == ref["sha256_drawing"]
    lines, bad = compare_pdf.compare(bare, tmp_path / "UTC", (name,))
    assert bad == 0 and "IDENTICAL (drawing); metadata differs: /CreationDate " in lines[0]
    assert f"{stamp} vs D:19700101000000Z" in lines[0]
    manifest = tmp_path / "m.json"
    compare_pdf.write_manifest(tmp_path / "UTC", manifest, (name,), "the UTC render")
    lines, bad = compare_pdf.compare(bare, manifest, (name,))
    assert bad == 0 and "IDENTICAL (drawing); metadata differs from manifest" in lines[0]


def test_composite_without_its_panels_fails_clearly(tmp_path, capsys):
    root = tmp_path / "arms"
    root.mkdir()
    out = tmp_path / "out"
    rc = _main(["--arms", str(root), "--out", str(out), "--only", "fig2_composite"])
    cap = capsys.readouterr()
    assert rc == 1
    assert "FAILED fig2_composite (pdf): input(s) not in" in cap.out
    assert "fig2a_editable.pdf" in cap.out and "render fig2a, fig2b, fig2c, fig2d first" in cap.out
    assert "1 of 1 file(s) failed" in cap.err
    assert not list(out.glob("*.pdf"))


def test_render_isolates_failures_and_checks_outputs(tmp_path, monkeypatch, capsys):
    def entry(name, main, inputs=()):
        return render.Entry(name, "x", "x.py", (), inputs,
                            lambda root, out, ext: [str(render.output(out, name, ext))], main)

    def good(argv):
        Path(argv[0]).write_bytes(b"%PDF")

    fakes = {
        "good": entry("good", good),
        "silent": entry("silent", lambda argv: None),
        "refuses": entry("refuses", lambda argv: (_ for _ in ()).throw(SystemExit("no scored trajectories"))),
        "boom": entry("boom", lambda argv: 1 / 0),
        "sheet": entry("sheet", good, inputs=("good", "silent")),
    }
    monkeypatch.setattr(render, "BY_NAME", fakes)
    out = tmp_path / "out"
    out.mkdir()
    failures = render.render(list(fakes), tmp_path, out, ("pdf",))
    cap = capsys.readouterr()
    assert failures == ["silent (pdf)", "refuses (pdf)", "boom (pdf)", "sheet (pdf)"]
    assert "rendered good -> " in cap.out
    assert "FAILED silent (pdf): expected output not written" in cap.out
    assert "FAILED refuses (pdf): no scored trajectories" in cap.out
    assert "FAILED boom (pdf): ZeroDivisionError" in cap.out and "ZeroDivisionError" in cap.err
    assert "FAILED sheet (pdf): input(s) not in" in cap.out and "['silent_editable.pdf']" in cap.out
    assert render.render(["good"], tmp_path, out, ("pdf", "png")) == []
    assert (out / "good.png").is_file()


# ---- a full render from synthetic records --------------------------------------------------------

def _with_cost(records: list[dict]) -> list[dict]:
    for rec in records:
        rows = (len(rec["pipeline"]["candidates"]) + len(rec.get("control") or [])
                + len((((rec.get("arbiter") or {}).get("rules") or {}).get("rows")) or []))
        rec["cost"] = {
            "by_rule": {rule: {"verdicts": rows, "wall_seconds": 3.0 * rows, "total_tokens": 900 * rows}
                        for rule in ("execution", "self", "control")},
            "mcmc_seconds": {"pipeline_validation": 120.0, "hidden_scoring": 60.0, "arbiter": 30.0,
                             "control": 30.0}}
    return records


def _census() -> list[dict]:
    return _with_cost([copy.deepcopy(r) for r in figure_records.records()])


def _real() -> list[dict]:
    out = []
    for rec, ds, n_hidden in zip(_census(), ("roaches", "radon", "golf"), (38, 38, 5), strict=True):
        rec["type"] = ds
        rec["pipeline"]["split_info"]["n_hidden"] = n_hidden
        for row in rec["pipeline"]["candidates"] + (rec.get("control") or []):
            row["fresh"] = None  # real data has no fresh draw
        out.append(rec)
    return out


def _drift(d: Path, *, synthetic: bool) -> None:
    """A drift directory of eight trajectories (four rules, two starts), their rounds scored
    on the hidden fold; the referee's deployed model never changes."""
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for rule, slope in (("execution", -0.4), ("self", -0.2), ("control", -0.1), ("referee", 0.3)):
        for seed in range(2):
            rec = figure_records.trajectory(rule, slope, rng)
            rec["baseline0_code"] = MODEL
            if not synthetic:
                rec["type"] = "roaches"
            for r in rec["rounds"]:
                r["changed"] = rule != "referee"
                r["n_sample_sites"] = 1 if rule == "referee" else 1 + r["round"]
            (d / f"{rec['type']}_seed{seed}_{rule}.json").write_text(json.dumps(rec))


@pytest.fixture
def synthetic_root(tmp_path) -> Path:
    root = tmp_path / "arms"
    for arm in (*render.CENSUS_ARMS, "e1_anthropic_realdata"):
        figure_records.write_arm(root / arm, _real() if "realdata" in arm else _census())
    _drift(root / "e2_anthropic" / "drift", synthetic=True)
    _drift(root / "e2_anthropic_real" / "drift", synthetic=False)
    return root


def test_all_eleven_files_render_from_a_synthetic_root(synthetic_root, tmp_path, capsys):
    out = tmp_path / "out"
    rc = _main(["--arms", str(synthetic_root), "--out", str(out)])
    text = capsys.readouterr().out
    assert rc == 0, text[-3000:]
    assert f"data root {synthetic_root.resolve()} (chosen by flag)" in text
    for name in NAMES:
        assert f"rendered {name} -> {out / (name + '_editable.pdf')}" in text, name
        assert (out / f"{name}_editable.pdf").stat().st_size > 5_000, name
    assert "11 file(s) rendered" in text
    assert not list(out.glob("*.letters.pdf"))  # the composites' overlay is a temporary file
    assert sorted(p.name for p in out.iterdir()) == sorted(f"{n}_editable.pdf" for n in NAMES)
    assert sorted(p.name for p in out.iterdir()) == sorted(compare_pdf.EXPECTED)
    fig2 = PdfReader(str(out / "fig2_composite_editable.pdf")).pages[0].extract_text()
    for title in ("What the agent proposed", "Who accepts them", "The theorem in one picture",
                  "When nothing is wrong"):
        assert title in fig2, title
    fig3 = PdfReader(str(out / "fig3_composite_editable.pdf")).pages[0].extract_text()
    for title in (*render.ROW_TITLES, "Deployed model over successive rounds",
                  "Why the deployed model drifts", "Outcome of the accepted revision"):
        assert title in fig3, title
    assert "What a verdict costs" in PdfReader(str(out / "fig4_cost_editable.pdf")).pages[0].extract_text()
    # the scripts printed the numbers they drew
    assert "figure2a  the agent's revisions" in text and "figure4  what a verdict costs" in text
