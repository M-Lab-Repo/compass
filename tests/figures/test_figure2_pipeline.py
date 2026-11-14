"""figure2_run / figure2_score, every step with stubbed models and no MCMC: a raw record per
run, resume by skipping, nothing written on a pipeline error; labels and readers in side files
that figure2_common joins; the protocol table against the committed records; the per-panel
thin files and their dry runs. One slow test fits for real and reproduces the labels stored in
the smoke record under fixtures/.

The tests that read the committed arms take them from ``render.arms_dir()`` (``COMPASS_ARMS_DIR``
when it is set) and skip when the arms are absent."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

import render
from bench import dgp
from compass.llm import FixtureLLM
from compass.schemas import JudgeVerdict
from compass.state import ValidationResult
from figure2 import (
    figure2_common as common,
)
from figure2 import (
    figure2_protocols as protocols,
)
from figure2 import (
    figure2_run,
    figure2_score,
    figure2a_create,
    figure2a_run,
    figure2a_score,
    figure2b_run,
    figure2c_run,
    figure2d_run,
)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"   # the scripts run from here
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ARMS = render.arms_dir()                                          # the committed records
PASS = ValidationResult(converged=True, rhat_max=1.0, ess_min=900.0, loo_delta=-40.0,
                        failure_reasons=[], confidence="high")
FAIL = ValidationResult(converged=True, rhat_max=1.0, ess_min=800.0, loo_delta=0.5,
                        failure_reasons=["no significant improvement"], confidence="low")

# pinned on 2026-11-14 against the research benchmark engines
HIER_BUDGET = {"num_warmup": 1000, "num_samples": 2000, "num_chains": 4, "target_accept": 0.95}
HEAVY_TAILS_BUDGET_X2 = {"num_warmup": 1000, "num_samples": 2000, "num_chains": 2, "target_accept": 0.9}
DATASET_SUMMARY_5_P = "N=5, p"


# ---------------------------------------------------------------------------
# Stubs: the pipeline, the referee and the execution check never run for real here
# ---------------------------------------------------------------------------

def _final(inst):
    """A pipeline result with two candidates: the decoy (refused) then the gold fix (deployed)."""
    hist = [{"iteration": 0, "refined_model_code": inst["decoy_code"], "proposal_names": ["pin"],
             "proposals": [{"name": "pin"}], "rationale": "pin it", "passed": False, "validation": FAIL},
            {"iteration": 1, "refined_model_code": inst["gold_code"], "proposal_names": ["negbin"],
             "proposals": [{"name": "negbin"}], "rationale": "negbin it", "passed": True, "validation": PASS}]
    return {"status": "complete", "validation": PASS, "iteration": 1, "refined_model_code": inst["gold_code"],
            "proposals": [{"name": "negbin"}], "candidate_history": hist, "literature_findings": []}


GOLD = {dgp.make_instance(t, s)["gold_code"] for t in dgp.TYPES for s in (0, 1, 2)}


def _seams(calls: dict):
    def run_refinement(**kw):
        calls.setdefault("run", []).append(kw)
        # thread_id is "<arm>-<type>-<seed>-s<split>-r<k>"; the arm name itself contains hyphens
        _, type_name, seed, _, _ = kw["thread_id"].rsplit("-", 4)
        inst = dgp.make_instance(type_name, int(seed))
        return _final(inst)

    def validate(refined, original, data, budget, proposals=None):
        calls.setdefault("validate", []).append((len(data["y"]), budget["num_samples"]))
        return PASS if refined in GOLD and refined != original else FAIL

    def execution_check(shapes, timeout=120, *, data=None):
        def check(refined_code, original_code, **kw):
            return JudgeVerdict(passed=True, confidence="high", reasoning="compiled")
        check.role = "execution"
        return check

    return {"run_refinement": run_refinement, "validate": validate, "execution_check": execution_check}


def ctx_visible(ran):
    rec = json.loads((ran["dir"] / "heavy_tails_seed0_r1.json").read_text())
    return figure2_score.instance_for(rec)["visible"]["y"]


def _judge_llm():
    return FixtureLLM(outputs=[JudgeVerdict(passed=True, reasoning="looks fine")])


def _scorer(calls: dict):
    def score(code, baseline, visible, fold, *, budget, seed, cache_dir):
        calls.setdefault("score", []).append((len(fold["y"]), seed, cache_dir))
        good = code in GOLD and code != baseline
        delta = 12.0 if good else 0.1
        return {"available": True, "delta": delta, "se": 2.0, "n_hidden": len(fold["y"]),
                "label": "genuine" if good else "confabulated", "per_obs_delta": [0.0]}
    return score


@pytest.fixture
def ran(tmp_path):
    """One misspecified-baseline arm, one type, stubbed end to end: 9 raw records."""
    calls: dict = {}
    out_root = tmp_path / "runs"
    statuses = figure2_run.run_arm("misspecified-baseline_haiku-4.5", out_root, llm=_judge_llm(), only="heavy_tails", **_seams(calls))
    return {"root": out_root, "dir": out_root / "misspecified-baseline_haiku-4.5", "statuses": statuses, "calls": calls}


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def test_run_arm_writes_one_raw_record_per_run_and_resumes(ran):
    assert [s for _, s in ran["statuses"]] == ["done"] * 9
    files = sorted(p.name for p in ran["dir"].glob("*.json"))
    assert files == sorted(f"heavy_tails_seed{s}_r{r}.json" for s in (0, 1, 2) for r in (1, 2, 3))
    rec = json.loads((ran["dir"] / "heavy_tails_seed0_r1.json").read_text())
    assert {"type", "seed", "arm", "repeat", "provider", "model", "run_date", "flags", "pipeline",
            "control", "arbiter", "budget", "usage"} <= set(rec)
    assert rec["flags"]["discovery"] and not rec["flags"]["null_baseline"] and rec["model"] == "claude-haiku-4-5-20251001"
    cands = rec["pipeline"]["candidates"]
    assert [c["iteration"] for c in cands] == [0, 1] and [c["deployed"] for c in cands] == [False, True]
    assert set(cands[0]) == {"iteration", "code", "proposal_names", "rationale", "mcmc", "judges", "deployed"}
    assert set(cands[1]["judges"]) == {"execution", "self", "control"}           # live rules only
    assert "hidden" not in cands[0] and "fresh" not in cands[0]                    # nothing derived
    assert len(rec["control"]) == 3 and rec["control"][0]["rationale"] == ""
    assert [r["name"].split(" ")[0] for r in rec["arbiter"]["rules"]["rows"]] == ["gold", "decoy", "no_change"]
    assert [r["mcmc_passed"] for r in rec["arbiter"]["rules"]["rows"]] == [True, False, False]
    assert rec["pipeline"]["split_info"]["n_hidden"] > 0 and rec["pipeline"]["validation_passed"] is True
    # the pipeline was called on the visible fold with the symptom-free statement and this arm's budget
    kw = ran["calls"]["run"][0]
    assert len(kw["data"]["y"]) == rec["pipeline"]["split_info"]["n_visible"]
    assert kw["problem_description"] == "Regression of a continuous response on a continuous covariate."  # symptom-free
    assert kw["verdict_only"] is True and kw["scout_kwargs"] == {"live": False}
    assert all(budget == 1000 for _, budget in ran["calls"]["validate"])
    # a second pass skips everything
    again = figure2_run.run_arm("misspecified-baseline_haiku-4.5", ran["root"], llm=_judge_llm(), only="heavy_tails", **_seams({}))
    assert [s for _, s in again] == ["skip"] * 9


def test_e5_arm_has_no_arbiter_and_starts_from_the_gold_model(tmp_path):
    calls: dict = {}
    figure2_run.run_arm("correct-baseline_haiku-4.5", tmp_path, llm=_judge_llm(), only="overdispersion_seed0_r1", **_seams(calls))
    rec = json.loads((tmp_path / "correct-baseline_haiku-4.5" / "overdispersion_seed0_r1.json").read_text())
    assert rec["arbiter"] is None and rec["flags"]["null_baseline"] is True
    assert calls["run"][0]["model_code"] == dgp.make_instance("overdispersion", 0)["gold_code"]


def test_pipeline_error_writes_no_record_and_the_committed_arms_root_is_refused(tmp_path):
    seams = _seams({})

    def boom(**kw):
        raise RuntimeError("provider down")
    seams["run_refinement"] = boom
    statuses = figure2_run.run_arm("misspecified-baseline_haiku-4.5", tmp_path, llm=_judge_llm(), only="heavy_tails_seed0_r1", **seams)
    assert statuses[0][1].startswith("ERROR RuntimeError") and not list((tmp_path / "misspecified-baseline_haiku-4.5").glob("*.json"))
    with pytest.raises(ValueError, match="committed bench/arms"):
        figure2_run.run_arm("misspecified-baseline_haiku-4.5", figure2_run.COMMITTED_ARMS / "x", dry_run=True)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def test_score_arm_writes_labels_in_side_files_and_load_records_joins_them(ran):
    calls: dict = {}
    report = figure2_score.score_arm(ran["dir"], scorer=_scorer(calls))
    assert len(report) == 9 and all(r["fits"] == (2 + 3 + 3) * 2 and r["kept"] == 0 for r in report)
    scores = json.loads((ran["dir"] / "scores" / "heavy_tails_seed0_r1.json").read_text())
    assert set(scores) == {"budget", "split_check", "pipeline", "control", "arbiter", "date"}
    assert set(scores["pipeline"]) == {"0", "1"} and set(scores["pipeline"]["1"]) == {"hidden", "fresh"}
    assert scores["pipeline"]["1"]["fresh"]["label"] == "genuine" and scores["pipeline"]["0"]["hidden"]["label"] == "confabulated"
    assert len(scores["arbiter"]) == 3 and "gold (genuine)" in scores["arbiter"] and "no_change" in scores["arbiter"]
    assert "per_obs_delta" not in scores["pipeline"]["1"]["hidden"]
    ctx = figure2_score.instance_for(json.loads((ran["dir"] / "heavy_tails_seed0_r1.json").read_text()))
    assert {n for n, _, _ in calls["score"]} == {len(ctx["hidden"]["y"]), len(ctx["fresh"]["y"])}  # hidden fold, fresh draw
    # the raw record was not touched; the loader joins the labels onto the rows
    raw = json.loads((ran["dir"] / "heavy_tails_seed0_r1.json").read_text())
    assert "hidden" not in raw["pipeline"]["candidates"][0]
    rec = common.load_records(ran["dir"])[0]
    assert rec["pipeline"]["candidates"][1]["fresh"]["label"] == "genuine"
    assert common.label_of(rec["pipeline"]["candidates"][1], rec) == "genuine"
    assert common.label_of(rec["control"][0], rec) == "confabulated"
    assert rec["arbiter"]["rules"]["rows"][0]["hidden"]["label"] == "genuine"
    counts = figure2a_create.count([ran["dir"]])["misspecified-baseline_haiku-4.5"]
    assert counts["genuine"] == 9 and counts["confabulated"] == 9 and counts["unavailable"] == 0
    # resume: a second pass fits nothing; --force fits everything again
    again = figure2_score.score_arm(ran["dir"], scorer=_scorer(calls2 := {}))
    assert "score" not in calls2 and all(r["kept"] == 16 for r in again)
    figure2_score.score_arm(ran["dir"], scorer=_scorer(calls3 := {}), force=True)
    assert len(calls3["score"]) == 9 * 16


def test_readers_pass_writes_verdicts_with_the_live_prompt_inputs(ran):
    seen: list = []

    def factory(provider, model):
        def judge(code, baseline, *, proposals=None, dataset_summary="", problem="", rationale=""):
            seen.append((model, dataset_summary, problem, rationale, code[:12]))
            return JudgeVerdict(passed=True, reasoning=f"{model} ok")
        judge.role = "independent"
        return judge

    report = figure2_score.score_arm(ran["dir"], scorer=_scorer({}), readers=(("anthropic", "m1"),),
                                     judge_factory=factory)
    assert all(r["verdicts"] == 8 and r["errors"] == 0 for r in report)
    readers = json.loads((ran["dir"] / "readers" / "heavy_tails_seed0_r1.json").read_text())
    assert readers["models"] == ["m1"] and "overdispersed" not in readers["problem_shown"]
    v = readers["pipeline"]["1"]["m1"]
    assert v["passed"] is True and v["role"] == "independent" and v["discovery_neutralized"] is True
    # what the reader saw: the neutral statement in both fields, the stored rationale, the program
    model, summary, problem, rationale, code = seen[0]
    assert summary == f"N={len(ctx_visible(ran))}, {problem}" and rationale == "pin it" and problem == readers["problem_shown"]
    assert any(r[3] == "" for r in seen)                                        # controls: no rationale
    rec = common.load_records(ran["dir"])[0]
    assert common.verdict(rec["pipeline"]["candidates"][1], "independent:m1") is True
    assert common.verdict(rec["arbiter"]["rules"]["rows"][1], "independent:m1") is True
    # resume: nothing new; a second reader is added without redoing the first
    n_before = len(seen)
    figure2_score.score_arm(ran["dir"], scorer=_scorer({}), readers=(("anthropic", "m1"),), judge_factory=factory)
    assert len(seen) == n_before
    figure2_score.score_arm(ran["dir"], scorer=_scorer({}), readers=(("anthropic", "m1"), ("openai", "m2")),
                            judge_factory=factory)
    assert len(seen) == n_before + 9 * 8
    readers = json.loads((ran["dir"] / "readers" / "heavy_tails_seed0_r1.json").read_text())
    assert readers["models"] == ["m1", "m2"] and set(next(iter(readers["control"].values()))) == {"m1", "m2"}


def test_instance_for_refuses_a_record_that_does_not_match(ran):
    rec = json.loads((ran["dir"] / "heavy_tails_seed0_r1.json").read_text())
    assert figure2_score.instance_for(rec)["fresh"]["y"].shape == (150,)
    bad = copy.deepcopy(rec)
    bad["pipeline"]["split_info"]["n_hidden"] = 5
    with pytest.raises(ValueError, match="split mismatch"):
        figure2_score.instance_for(bad)
    bad = copy.deepcopy(rec)
    bad["budget"]["num_samples"] = 7
    with pytest.raises(ValueError, match="budget"):
        figure2_score.instance_for(bad)


# ---------------------------------------------------------------------------
# The protocol table against the committed arms
# ---------------------------------------------------------------------------

def test_protocol_flags_match_the_committed_records():
    seen = 0
    for arm in protocols.ARM_PROTOCOLS:  # Figure 2's ten and the one Figure 4 reads
        d = ARMS / protocols.LEGACY_DIR[arm]  # the committed directory of the arm
        for path in sorted(d.glob("*.json"))[:3] if d.is_dir() else []:
            rec = json.loads(path.read_text())
            if "type" not in rec:
                continue
            ours = protocols.record_flags(arm, rec["flags"].get("split_seed", 0))
            for key, value in rec["flags"].items():
                if key in ("arm", "provider", "model", "judge_providers", "repeat"):
                    continue  # provenance and pass bookkeeping, not protocol
                assert ours[key] == value, (arm, path.name, key)
            seen += 1
    if seen == 0:
        pytest.skip(f"no committed census arms under {ARMS}")
    assert sum(len(protocols.instances_of(a)) for a in protocols.CENSUS_ARMS) == 8 * 72 + 2 * 21


def test_stems_budgets_and_dataset_summary():
    inst = dgp.make_instance("missing_hierarchy", 1)
    assert protocols.budget_for(inst) == HIER_BUDGET == protocols.HIER_BUDGET
    assert protocols.budget_for(dgp.make_instance("heavy_tails", 0), 2.0) == HEAVY_TAILS_BUDGET_X2
    assert protocols.record_stem("roaches", 0, 2, 1) == "roaches_seed0_s2_r1"
    assert protocols.record_stem("heavy_tails", 1, 0, 3) == "heavy_tails_seed1_r3"
    assert common.dataset_summary(5, "p") == DATASET_SUMMARY_5_P
    assert protocols.readers_of("correct-baseline_haiku-4.5")[0] == ("anthropic", "claude-sonnet-4-6") and protocols.readers_of("correct-baseline_gpt-5") == ()


def test_thin_files_list_the_protocol_in_a_dry_run(tmp_path, capsys):
    out = figure2a_run.main(["--dry-run", "--out-root", str(tmp_path)])
    assert set(out) == set(protocols.CENSUS_ARMS)
    assert sum(len(v) for v in out.values()) == 8 * 72 + 2 * 21
    text = capsys.readouterr().out
    assert text.count("would run") == 618 and "(dry run)" in text and not any(tmp_path.iterdir())
    assert set(figure2c_run.main(["--dry-run", "--out-root", str(tmp_path)])) == {"misspecified-baseline_sonnet-4.6"}
    assert set(figure2d_run.main(["--dry-run", "--out-root", str(tmp_path)])) == set(protocols.E5_ARMS)
    assert figure2b_run.ARMS == figure2a_run.ARMS
    only = figure2a_run.main(["--dry-run", "--out-root", str(tmp_path), "--arm", "correct-baseline_gpt-5", "--only", "heavy_tails"])
    assert list(only) == ["correct-baseline_gpt-5"] and len(only["correct-baseline_gpt-5"]) == 9
    with pytest.raises(SystemExit):
        figure2a_run.main(["--help"])
    text = capsys.readouterr().out
    assert "Options" in text and "--only PREFIX" in text and "arms of this panel" in text
    assert all(arm in text for arm in protocols.CENSUS_ARMS) and "starts from the correct model" in text
    assert protocols.describe("correct-baseline_haiku-4.5").startswith("correct-baseline_haiku-4.5") and "72 runs" in protocols.describe("correct-baseline_haiku-4.5")
    # the values spelled out in each thin file are the protocol table's, and its ARMS are literal
    for mod in (figure2a_run, figure2b_run, figure2c_run, figure2d_run, figure2a_score):
        src = Path(mod.__file__).read_text()
        assert all(f'"{arm}"' in src for arm in mod.ARMS), mod.__name__
        assert set(mod.ARMS) <= set(protocols.ARM_PROTOCOLS)
        if mod.__name__.endswith("_run"):
            assert all(t in mod.__doc__ for t in protocols.SYNTHETIC_TYPES), mod.__name__
    assert all(d in figure2a_run.__doc__ for d in protocols.REAL_SEVEN)
    assert set(figure2a_run.ARMS) == set(protocols.CENSUS_ARMS) and set(figure2d_run.ARMS) == set(protocols.E5_ARMS)
    assert all(m in figure2a_score.__doc__ for _, m in protocols.SONNET_READERS + protocols.HAIKU_READERS)
    # the script runs as a file from the figures/ directory
    proc = subprocess.run([sys.executable, "figure2/figure2a_run.py", "--dry-run", "--out-root", str(tmp_path),
                           "--arm", "misspecified-baseline_haiku-4.5"], cwd=FIGURES_DIR, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and proc.stdout.count("would run") == 72, proc.stderr[-500:]


def test_thin_score_file_passes_protocol_readers(ran, monkeypatch):
    got = {}

    def fake_score_arm(arm_dir, *, readers=(), force=False, **kw):
        got[Path(arm_dir).name] = (readers, force)
        return []
    monkeypatch.setattr(figure2_score, "score_arm", fake_score_arm)
    figure2a_score.main(["--out-root", str(ran["root"]), "--arm", "misspecified-baseline_haiku-4.5"])
    assert got["misspecified-baseline_haiku-4.5"] == (protocols.HAIKU_READERS, False)
    figure2a_score.main(["--out-root", str(ran["root"]), "--arm", "misspecified-baseline_haiku-4.5", "--no-readers", "--force"])
    assert got["misspecified-baseline_haiku-4.5"] == ((), True)
    figure2a_score.main(["--out-root", str(ran["root"]), "--arm", "correct-baseline_gpt-5", "--readers", "openai:gpt-5"])
    assert got["correct-baseline_gpt-5"] == ((("openai", "gpt-5"),), False)


# ---------------------------------------------------------------------------
# Real MCMC: the score step reproduces the labels stored in the smoke record
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_score_arm_reproduces_the_committed_smoke_labels(tmp_path, monkeypatch, trace_cache_dir):
    monkeypatch.setattr(figure2_run, "CACHE", trace_cache_dir)   # the session's trace cache, not the checkout's
    src = FIXTURES / "smoke_overdispersion_seed0_r1.json"
    rec = json.loads(src.read_text())
    if any("trimmed" in row["code"] for row in rec["pipeline"]["candidates"] + rec["control"]):
        pytest.skip(f"{src.name} carries stub programs; the labels are reproduced only from the record with its programs")
    expected = {("pipeline", str(c["iteration"])): (c["hidden"]["label"], c["fresh"]["label"])
                for c in rec["pipeline"]["candidates"]}
    expected.update({("control", c["name"]): (c["hidden"]["label"], c["fresh"]["label"]) for c in rec["control"]})
    for row in rec["pipeline"]["candidates"] + rec["control"]:
        row.pop("hidden"), row.pop("fresh")
    (tmp_path / "smoke").mkdir()
    (tmp_path / "smoke" / "overdispersion_seed0_r1.json").write_text(json.dumps(rec))  # the name the runner gave it
    figure2_score.score_arm(tmp_path / "smoke")
    scores = json.loads((tmp_path / "smoke" / "scores" / "overdispersion_seed0_r1.json").read_text())
    got = {(w, k): (v["hidden"]["label"], v["fresh"]["label"]) for w in ("pipeline", "control")
           for k, v in scores[w].items()}
    assert got == expected
