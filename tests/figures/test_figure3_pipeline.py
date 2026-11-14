"""figure3_run / figure3_score, every step with stubbed models and no MCMC: one raw trajectory
per (starting model, rule), rewritten after every round and resumed from its last deployed
model; scores in side files that figure3_common joins; the protocol table against the committed
trajectories; the per-panel thin files and their dry runs, panel c's going through figure2's
census engine.

The tests that read the committed arms take them from ``render.arms_dir()`` (``COMPASS_ARMS_DIR``
when it is set) and skip when the arms are absent."""

from __future__ import annotations

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
from figure2 import figure2_run as census_run
from figure2 import figure2_score as census_score
from figure3 import (
    figure3_common as f3,
)
from figure3 import (
    figure3_protocols as protocols,
)
from figure3 import (
    figure3_run,
    figure3_score,
    figure3a_create,
    figure3a_run,
    figure3a_score,
    figure3b_create,
    figure3b_run,
    figure3c_run,
    figure3c_score,
    figure3e_run,
    figure3e_score,
)

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures"   # the scripts run from here
ARMS = render.arms_dir()                                          # the committed records
PASS = ValidationResult(converged=True, failure_reasons=[])
FAIL = ValidationResult(converged=True, failure_reasons=["no significant improvement"])
NEW_SITE = '    numpyro.sample("extra{g}", dist.Normal(0, 1))\n'
ARM = "drift_correct-baseline_sonnet-4.6"

# pinned on 2026-11-14 against the research benchmark engines: the acceptance rules of a drift
# trajectory and the defaults of the drift configuration whose flags the trajectories carry
DRIFT_RULES = ("execution", "self", "control", "referee")
DRIFT_DEFAULTS = dict(arm="drift", rounds=8, hidden_frac=0.25, split_seed=0, discovery=False,
                      null_baseline=False, live_scout=False, provider=None, model=None, smoke_on_data=False,
                      guard_replicate_sites=False, budget_scale=1.0, lit_threshold=6, force=False)


# ---------------------------------------------------------------------------
# Stubs: the pipeline appends one latent site per call; the referee is a switch; no MCMC
# ---------------------------------------------------------------------------

def _seams(calls: dict, state: dict) -> dict:
    def run_refinement(**kw):
        calls.setdefault("run", []).append(kw)
        cand = kw["model_code"] + NEW_SITE.format(g=len(calls["run"]))
        v = state["validation"]
        return {"status": "complete", "iteration": 1, "refined_model_code": cand, "proposals": [{"name": "tweak"}],
                "validation": v, "error_log": [],
                "candidate_history": [{"iteration": 0, "refined_model_code": cand, "proposal_names": ["tweak"],
                                       "proposals": [{"name": "tweak"}], "rationale": "tweak: adds a site",
                                       "passed": True, "validation": v}]}

    def validate(refined, original, data, budget, proposals=None):
        calls.setdefault("validate", []).append(budget)
        return state["validation"]

    def execution_check(shapes, timeout=120, *, data=None):
        def check(code, baseline, **kw):
            return JudgeVerdict(passed=True, confidence="high", reasoning="compiled")
        check.role = "execution"
        return check

    return dict(run_refinement=run_refinement, validate=validate, execution_check=execution_check)


def _scorer(calls: dict):
    def score(code, baseline, visible, fold, *, budget, seed, cache_dir):
        calls.setdefault("score", []).append((len(fold["y"]), seed))
        delta = float(code.count('"extra') - baseline.count('"extra'))
        return {"available": True, "delta": delta, "se": 0.1, "n_hidden": len(fold["y"]),
                "label": "genuine" if delta > 0 else "confabulated", "per_obs_delta": [delta]}
    return score


def _judge_llm(passed=True):
    return FixtureLLM(outputs=[JudgeVerdict(passed=passed, reasoning="fixture")])


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def test_execution_rule_deploys_every_round_and_the_model_grows(tmp_path):
    calls, state = {}, {"validation": PASS}
    statuses = figure3_run.run_arm(ARM, tmp_path / "runs", llm=_judge_llm(), only="overdispersion_seed0_execution",
                                   **_seams(calls, state))
    assert statuses == [("overdispersion_seed0_execution", "done")]
    rec = json.loads((tmp_path / "runs" / ARM / "drift" / "overdispersion_seed0_execution.json").read_text())
    assert rec["rule"] == "execution" and len(rec["rounds"]) == 6
    assert all(r["accepted"] and r["changed"] for r in rec["rounds"])
    assert [r["n_sample_sites"] for r in rec["rounds"]] == list(range(4, 10))  # the gold NegBin has 3 latent sites
    assert rec["baseline0_code"] == dgp.make_instance("overdispersion", 0)["gold_code"]
    assert rec["flags"]["null_baseline"] and rec["flags"]["rounds"] == 6 and rec["split_info"]["n_hidden"] == 38
    assert all(kw["max_iterations"] == 1 and kw["verdict_only"] for kw in calls["run"])
    assert calls["run"][1]["model_code"] == rec["rounds"][0]["baseline_code_after"]  # the next round starts there
    assert "hidden" not in rec["rounds"][0] and "validate" not in calls  # scores come later; the referee never ran
    stub = calls["run"][0]["validator_kwargs"]["validation_fn"]  # a reading rule gets a stub that always passes
    assert stub("cand", "base", {"y": [0.0]}).converged and "validate" not in calls
    assert rec["rounds"][0]["cost"]["mcmc_seconds"] == {"referee": 0.0}


def test_referee_rule_keeps_the_model_when_the_verifier_refuses(tmp_path):
    calls, state = {}, {"validation": FAIL}
    figure3_run.run_arm(ARM, tmp_path / "runs", llm=_judge_llm(), only="overdispersion_seed0_referee", **_seams(calls, state))
    rec = json.loads((tmp_path / "runs" / ARM / "drift" / "overdispersion_seed0_referee.json").read_text())
    assert not any(r["accepted"] for r in rec["rounds"])
    assert all(r["baseline_code_after"] == rec["baseline0_code"] for r in rec["rounds"])
    assert rec["rounds"][0]["rule_verdict"] == {"passed": False, "error": None, "reasoning": "no significant improvement"}
    # the pipeline was handed the real referee at the arm's budget (the stub pipeline does not call it itself)
    referee = calls["run"][0]["validator_kwargs"]["validation_fn"]
    assert referee("cand", "base", {"y": [0.0]}) is FAIL and calls["validate"] == [{**calls["validate"][0]}]
    assert calls["validate"][0]["num_samples"] == 1000 and rec["rounds"][0]["cost"]["mcmc_seconds"]["referee"] == 0.0


def test_self_rule_follows_the_judge_and_a_partial_trajectory_resumes(tmp_path):
    calls, state = {}, {"validation": PASS}
    seams, root = _seams(calls, state), tmp_path / "runs"
    figure3_run.run_arm(ARM, root, llm=_judge_llm(passed=False), only="overdispersion_seed0_self", **seams)
    path = root / ARM / "drift" / "overdispersion_seed0_self.json"
    rec = json.loads(path.read_text())
    assert len(rec["rounds"]) == 6 and not any(r["accepted"] for r in rec["rounds"])
    # cut to two rounds: the next pass resumes from the stored model and redoes nothing
    rec["rounds"] = rec["rounds"][:2]
    path.write_text(json.dumps(rec))
    before = len(calls["run"])
    statuses = figure3_run.run_arm(ARM, root, llm=_judge_llm(), only="overdispersion_seed0_self", **seams)
    assert statuses == [("overdispersion_seed0_self", "resumed")] and len(calls["run"]) == before + 4
    rec = json.loads(path.read_text())
    assert len(rec["rounds"]) == 6 and all(r["accepted"] for r in rec["rounds"][2:])
    assert figure3_run.run_arm(ARM, root, llm=_judge_llm(), only="overdispersion_seed0_self", **seams) == [
        ("overdispersion_seed0_self", "skip")]
    # a stored trajectory from another starting model is refused, not silently continued
    rec["baseline0_code"], rec["rounds"] = "def model(): pass", rec["rounds"][:1]
    path.write_text(json.dumps(rec))
    (stem, status), = figure3_run.run_arm(ARM, root, llm=_judge_llm(), only="overdispersion_seed0_self", **seams)
    assert status.startswith("ERROR ValueError") and "another model" in status


def test_the_committed_arms_root_is_refused_and_no_candidate_keeps_the_model(tmp_path):
    with pytest.raises(ValueError, match="committed bench/arms"):
        figure3_run.run_arm(ARM, census_run.COMMITTED_ARMS / "x", llm=_judge_llm())
    calls, state = {}, {"validation": PASS}
    seams = _seams(calls, state)

    def nothing(**kw):
        calls.setdefault("run", []).append(kw)
        return {"status": "failed", "refined_model_code": "", "candidate_history": [], "error_log": ["boom"]}

    seams["run_refinement"] = nothing
    figure3_run.run_arm(ARM, tmp_path / "runs", llm=_judge_llm(), only="heavy_tails_seed0_execution", **seams)
    rec = json.loads((tmp_path / "runs" / ARM / "drift" / "heavy_tails_seed0_execution.json").read_text())
    assert all(r["candidate"] is None and not r["accepted"] and r["attempts"] == 2 for r in rec["rounds"])
    assert len(calls["run"]) == 12 and rec["rounds"][0]["pipeline_errors"] == ["boom"]
    assert rec["rounds"][0]["rule_verdict"]["reasoning"] == "no candidate produced"


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def test_score_arm_writes_side_files_that_the_panels_join(tmp_path):
    calls, state = {}, {"validation": PASS}
    root = tmp_path / "runs"
    figure3_run.run_arm(ARM, root, llm=_judge_llm(), only="overdispersion_seed0", **_seams(calls, state))  # 4 rules
    drift = root / ARM / "drift"
    report = figure3_score.score_arm(drift, scorer=_scorer(calls))
    assert len(report) == 4 and all(r["fits"] == 18 and r["kept"] == 0 for r in report)  # 6 rounds x (hidden, fresh, step)
    side = json.loads((drift / "scores" / "overdispersion_seed0_execution.json").read_text())
    assert set(side["rounds"]) == {str(g) for g in range(1, 7)}
    assert side["rounds"]["6"]["hidden"]["delta"] == 6.0 and side["rounds"]["6"]["step"]["delta"] == 1.0
    assert "seconds" in side["rounds"]["6"]["fresh"] and "per_obs_delta" not in side["rounds"]["6"]["fresh"]
    assert {n for n, _ in calls["score"]} == {38, 150}  # the hidden fold and the fresh draw
    assert all(r["kept"] == 18 and r["fits"] == 0 for r in figure3_score.score_arm(drift, scorer=_scorer(calls)))
    by_rule = {t["rule"]: t for t in f3.load_trajectories(drift)}
    assert f3.series(by_rule["execution"]) == {k: float(k) for k in range(7)}
    numbers, _ = figure3a_create.count(drift)
    assert numbers["rules"]["execution"]["median"] == [float(k) for k in range(7)] and numbers["n_dropped_rounds"] == 0
    mech = figure3b_create.count(drift)
    assert mech["execution"]["sites"] == [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0] and mech["referee"]["changed"] == 6


def test_instance_for_refuses_a_mismatched_trajectory():
    rec = {"type": "overdispersion", "seed": 0, "rule": "self", "flags": protocols.record_flags(ARM),
           "baseline0_code": "nope", "split_info": {"n_hidden": 38}, "rounds": []}
    with pytest.raises(ValueError, match="starting model"):
        figure3_score.instance_for(rec)
    rec["baseline0_code"], rec["split_info"] = dgp.make_instance("overdispersion", 0)["gold_code"], {"n_hidden": 5}
    with pytest.raises(ValueError, match="split"):
        figure3_score.instance_for(rec)
    rec["split_info"] = {"n_hidden": 38}
    ctx = figure3_score.instance_for(rec)
    assert len(ctx["fresh"]["y"]) == 150 and ctx["budget"]["num_samples"] == 1000


# ---------------------------------------------------------------------------
# The protocol table against the committed trajectories
# ---------------------------------------------------------------------------

def test_protocol_flags_match_the_committed_trajectories():
    seen = 0
    for arm, committed in protocols.LEGACY_DIR.items():  # the committed directory of each arm
        d = ARMS / committed / "drift"
        for path in sorted(d.glob("*.json"))[:4] if d.is_dir() else []:
            rec = json.loads(path.read_text())
            ours = protocols.record_flags(arm)
            for key, value in rec["flags"].items():
                if key not in ("arm", "provider", "model"):
                    assert ours[key] == value, (arm, path.name, key)
            assert rec["rule"] in protocols.RULES and len(rec["rounds"]) == protocols.ROUNDS
            seen += 1
    if not seen:
        pytest.skip(f"no committed drift arms under {ARMS}")
    assert len(protocols.trajectories_of("drift_correct-baseline_sonnet-4.6")) == 32
    assert len(protocols.trajectories_of("drift_real-data_sonnet-4.6")) == 28
    assert protocols.trajectory_stem("roaches", 0, "referee") == "roaches_seed0_referee"
    assert set(protocols.RULES) == set(DRIFT_RULES)
    for key, value in protocols.FLAG_DEFAULTS.items():
        if key not in ("arm", "rounds"):  # the campaign ran 6 rounds, not the CLI default
            assert DRIFT_DEFAULTS[key] == value, key


def test_thin_files_list_the_protocol_in_a_dry_run(tmp_path, capsys):
    out = figure3a_run.main(["--dry-run", "--out-root", str(tmp_path)])
    assert list(out) == [ARM] and len(out[ARM]) == 32 and not any(tmp_path.iterdir())
    assert figure3b_run.ARMS == figure3a_run.ARMS
    assert set(figure3e_run.main(["--dry-run", "--out-root", str(tmp_path)])) == {"drift_real-data_sonnet-4.6"}
    c = figure3c_run.main(["--dry-run", "--out-root", str(tmp_path)])
    assert set(c) == set(protocols.MATRIX_ARMS) == {"real-data_sonnet-4.6"} and sum(len(v) for v in c.values()) == 21
    only = figure3a_run.main(["--dry-run", "--out-root", str(tmp_path), "--only", "heavy_tails"])
    assert len(only[ARM]) == 4
    assert capsys.readouterr().out.count("would run") == 32 + 28 + 21 + 4
    with pytest.raises(SystemExit):
        figure3a_run.main(["--help"])
    text = capsys.readouterr().out
    assert all(s in text for s in ("Options", "--only PREFIX", "arms of this panel", "starting from the correct model"))
    for mod in (figure3a_run, figure3b_run, figure3e_run, figure3a_score, figure3e_score):
        src = Path(mod.__file__).read_text()
        assert all(f'"{arm}"' in src for arm in mod.ARMS) and set(mod.ARMS) <= set(protocols.ARM_PROTOCOLS), mod.__name__
    assert set(figure3c_run.ARMS) == set(figure3c_score.ARMS) == set(protocols.MATRIX_ARMS)
    assert all(t in figure3a_run.__doc__ for t in f3.common.SYNTHETIC_TYPES)
    assert all(r in figure3a_run.__doc__ for r in protocols.RULES)
    # the script runs as a file from the figures/ directory
    proc = subprocess.run([sys.executable, "figure3/figure3e_run.py", "--dry-run", "--out-root", str(tmp_path)],
                          cwd=FIGURES_DIR, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0 and proc.stdout.count("would run") == 28, proc.stderr[-500:]


def test_thin_score_files_pass_the_right_directories(tmp_path, monkeypatch):
    got = {}

    def fake_drift_score(d, *, force=False, **kw):
        got[str(d)] = force
        return []

    monkeypatch.setattr(figure3_score, "score_arm", fake_drift_score)
    figure3a_score.main(["--out-root", str(tmp_path), "--force"])
    figure3e_score.main(["--out-root", str(tmp_path)])
    assert got == {str(tmp_path / ARM / "drift"): True, str(tmp_path / "drift_real-data_sonnet-4.6" / "drift"): False}
    got2 = {}

    def fake_census_score(d, *, readers=(), force=False, **kw):
        got2[Path(d).name] = readers
        return []

    monkeypatch.setattr(census_score, "score_arm", fake_census_score)
    figure3c_score.main(["--out-root", str(tmp_path)])
    assert got2 == {"real-data_sonnet-4.6": (("openai", "gpt-5"),)}   # the matrix's one arm, its GPT-5 reader
