"""LLM-judge baseline and the alternative acceptance rules (``compass.judge``).

The judge is an EVALUATION baseline, never an arbiter: these tests check the verdict
plumbing, the control arm that withholds the proposer's rationale, the execution
check, the single-judge baseline tally (false acceptance against the MCMC arbiter,
judge errors kept as rows) and the multi-rule tally. FixtureLLM only; no API key, no MCMC, no subprocess.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import compass.agents.synthesizer as synthesizer
import compass.judge as judge_mod
from compass.judge import (
    JUDGE_PROMPT,
    _call_judge,
    arbiter_agreement,
    create_control_judge,
    create_execution_check,
    create_independent_judge,
    create_llm_judge,
    create_self_judge,
    format_arbiter_comparison,
    format_judge_baseline,
    run_arbiter_comparison,
    run_judge_baseline,
)
from compass.llm import FixtureLLM
from compass.schemas import JudgeVerdict
from compass.state import ValidationResult


def _verdict(passed: bool, reasoning: str = "x", confidence: str = "medium") -> JudgeVerdict:
    return JudgeVerdict(passed=passed, confidence=confidence, reasoning=reasoning)


def _vr(passed: bool) -> ValidationResult:
    return ValidationResult(converged=passed,
                            failure_reasons=[] if passed else ["no significant improvement"])


# --- prompt and the baseline judge -----------------------------------------------


def test_judge_prompt_takes_the_rationale():
    assert "rationale" in JUDGE_PROMPT.input_variables
    assert {"problem", "dataset_summary", "proposals", "original_code",
            "refined_code"} <= set(JUDGE_PROMPT.input_variables)


def test_create_llm_judge_round_trip_carries_code_and_context():
    fake = FixtureLLM(outputs=[_verdict(True, "sound reparameterization", "high")])
    judge = create_llm_judge(fake)
    v = judge("def model(): pass  # REFINED-MARK", "def model(): pass  # ORIGINAL-MARK",
              proposals=[{"name": "non_centered"}], dataset_summary="N=100",
              problem="eight schools", rationale="RATIONALE-MARK the funnel geometry")
    assert v.passed is True and v.confidence == "high"
    prompt = fake.calls[0]
    assert "REFINED-MARK" in prompt and "ORIGINAL-MARK" in prompt
    assert "non_centered" in prompt and "eight schools" in prompt and "N=100" in prompt
    assert "## Proposer's justification" in prompt
    assert "RATIONALE-MARK the funnel geometry" in prompt
    assert "(no rationale provided)" not in prompt
    assert judge.role == "llm" and judge.model_id is None  # FixtureLLM names no model


def test_self_judge_sees_the_rationale_and_control_judge_does_not():
    self_llm, control_llm = FixtureLLM(outputs=[_verdict(True)]), FixtureLLM(outputs=[_verdict(True)])
    self_judge, control = create_self_judge(self_llm), create_control_judge(control_llm)
    common = dict(proposals=[{"name": "non_centered"}], dataset_summary="N=8",
                  problem="eight schools", rationale="RATIONALE-MARK must not reach the control")
    assert self_judge("REFINED-MARK", "ORIGINAL-MARK", **common).passed is True
    assert control("REFINED-MARK", "ORIGINAL-MARK", **common).passed is True

    assert "RATIONALE-MARK" in self_llm.calls[0]
    control_prompt = control_llm.calls[0]
    assert "RATIONALE-MARK" not in control_prompt
    assert "## Proposer's justification\n(no rationale provided)" in control_prompt
    # everything else still reaches the control arm
    assert "REFINED-MARK" in control_prompt and "ORIGINAL-MARK" in control_prompt
    assert "non_centered" in control_prompt and "N=8" in control_prompt
    assert "eight schools" in control_prompt
    assert self_judge.role == "self" and control.role == "control"
    assert create_independent_judge(self_llm).role == "independent"


def test_model_id_is_read_from_the_llm():
    fake = FixtureLLM(outputs=[_verdict(True)])
    anthropic_like = SimpleNamespace(model="claude-test-1",
                                     with_structured_output=fake.with_structured_output)
    openai_like = SimpleNamespace(model_name="gpt-test-1",
                                  with_structured_output=fake.with_structured_output)
    assert create_self_judge(anthropic_like).model_id == "claude-test-1"
    assert create_independent_judge(openai_like).model_id == "gpt-test-1"
    assert create_control_judge(anthropic_like).model_id == "claude-test-1"


# --- execution check ---------------------------------------------------------------


def test_execution_check_passes_data_through_to_the_smoke_validator(monkeypatch):
    seen: dict = {}

    def fake_validate(code, data_shapes, timeout=120, *, data=None):
        seen.update(code=code, data_shapes=data_shapes, timeout=timeout, data=data)
        return {"compiled": True, "stage": "smoke", "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(synthesizer, "validate_numpyro_code", fake_validate)
    check = create_execution_check({"y": (10,)}, timeout=7)
    v = check("REFINED", "ORIGINAL", proposals=[{"name": "p"}], dataset_summary="d",
              problem="p", rationale="ignored")
    assert v.passed is True and v.confidence == "high"
    assert v.reasoning == "execution check: stage=smoke"
    assert seen == {"code": "REFINED", "data_shapes": {"y": (10,)}, "timeout": 7, "data": None}
    assert check.role == "execution" and check.model_id is None

    visible = {"tries": [5, 7], "y": [1, 2]}
    v = create_execution_check({"tries": (2,), "y": (2,)}, data=visible)("R", "O")
    assert v.passed is True
    assert seen["data"] is visible and seen["data_shapes"] == {"tries": (2,), "y": (2,)}


def test_execution_check_fails_and_truncates_stderr(monkeypatch):
    long_err = "SyntaxError: " + "x" * 1000
    monkeypatch.setattr(
        synthesizer, "validate_numpyro_code",
        lambda code, shapes, timeout=120, data=None: {"compiled": False, "stage": "ast",
                                                      "stdout": "", "stderr": long_err},
    )
    v = create_execution_check({})("def model(:", "o")
    assert v.passed is False and v.confidence == "high"
    assert v.reasoning == "execution check: stage=ast; stderr: " + long_err[:300]


# --- agreement and the multi-rule comparison --------------------------------------


def test_arbiter_agreement_flags_the_two_error_types():
    fa = arbiter_agreement(_verdict(True), _vr(False))
    assert fa["false_accept"] is True and fa["agree"] is False and fa["false_reject"] is False
    fr = arbiter_agreement(_verdict(False), _vr(True))
    assert fr["false_reject"] is True and fr["false_accept"] is False
    ag = arbiter_agreement(_verdict(True), _vr(True))
    assert ag["agree"] is True and not ag["false_accept"] and not ag["false_reject"]


def _cases() -> list[dict]:
    return [
        {"name": "c1", "refined_code": "r1", "original_code": "o", "rationale": "because-c1"},
        {"name": "c2", "refined_code": "r2", "original_code": "o"},
        {"name": "c3", "refined_code": "r3", "original_code": "o"},
    ]


# c1 converged but failed the strict criterion -> MCMC rejects (exercises the
# `converged and not failure_reasons` predicate, not just `converged`).
_MCMC = {
    "c1": SimpleNamespace(converged=True, failure_reasons=["no significant improvement"]),
    "c2": SimpleNamespace(converged=True, failure_reasons=[]),
    "c3": SimpleNamespace(converged=False, failure_reasons=["divergences"]),
}
_HIDDEN = {
    "c1": {"label": "confabulated", "delta": -1.0, "se": 0.5},
    "c2": {"label": "genuine", "delta": 5.0, "se": 1.0},
    "c3": {"label": "harmful", "delta": -8.0},
}


def _comparison():
    self_llm = FixtureLLM(outputs=[_verdict(True), _verdict(True), _verdict(False)])
    indep_llm = FixtureLLM(outputs=[_verdict(False), _verdict(False), _verdict(True)])
    judges = {"self": create_self_judge(self_llm), "indep": create_independent_judge(indep_llm)}
    res = run_arbiter_comparison(_cases(), judges, validate=lambda c: _MCMC[c["name"]],
                                 hidden_scorer=lambda c: _HIDDEN[c["name"]])
    return res, self_llm


def test_comparison_rows_carry_ground_truth_and_the_rationale_reaches_the_judge():
    res, self_llm = _comparison()
    assert res["n"] == 3 and res["judges"] == ["self", "indep"]
    assert set(res["summary"]) == {"self", "indep", "mcmc"}
    rows = res["rows"]
    assert [r["name"] for r in rows] == ["c1", "c2", "c3"]
    assert [r["mcmc_passed"] for r in rows] == [False, True, False]
    assert [r["hidden_label"] for r in rows] == ["confabulated", "genuine", "harmful"]
    assert [r["hidden_se"] for r in rows] == [0.5, 1.0, None]
    assert rows[0]["judges"]["self"] == {"passed": True, "confidence": "medium",
                                         "reasoning": "x", "error": None}
    assert "because-c1" in self_llm.calls[0]
    assert "(no rationale provided)" in self_llm.calls[1]


def test_comparison_summary_numbers_by_hand():
    res, _ = _comparison()
    s, i, m = res["summary"]["self"], res["summary"]["indep"], res["summary"]["mcmc"]

    # self: accepts c1, c2; rejects c3. MCMC accepts only c2.
    assert (s["n_judged"], s["n_errors"], s["n_accepted"]) == (3, 0, 2)
    assert s["false_accept_n"] == 1 and abs(s["false_accept_rate"] - 1 / 3) < 1e-9  # c1
    assert s["false_reject_n"] == 0 and s["false_reject_rate"] == 0.0
    assert (s["n_mcmc_failed"], s["n_mcmc_passed"]) == (2, 1)
    assert s["false_accept_cond_rate"] == 0.5 and s["false_reject_cond_rate"] == 0.0
    assert s["by_hidden_label"]["confabulated"] == {"n": 1, "n_accepted": 1, "rate": 1.0}
    assert s["false_accept_hidden"] == {"n": 2, "n_accepted": 1, "rate": 0.5}

    # indep: accepts only c3 (harmful, MCMC-rejected); rejects c2 (genuine, MCMC-accepted).
    assert (i["n_judged"], i["n_errors"], i["n_accepted"]) == (3, 0, 1)
    assert i["false_accept_n"] == 1 and i["false_reject_n"] == 1
    assert i["false_accept_cond_rate"] == 0.5 and i["false_reject_cond_rate"] == 1.0
    assert i["by_hidden_label"]["harmful"] == {"n": 1, "n_accepted": 1, "rate": 1.0}

    # mcmc pseudo-rule: accepts c2 only; no FA-vs-itself columns.
    assert (m["n_judged"], m["n_errors"], m["n_accepted"]) == (3, 0, 1)
    assert "false_accept_rate" not in m and "n_mcmc_failed" not in m
    assert m["by_hidden_label"]["genuine"] == {"n": 1, "n_accepted": 1, "rate": 1.0}

    md = format_arbiter_comparison(res)
    assert "| self | 67% (2/3) | 50% (1/2) | 100% (1/1) | 0% (0/1) | 100% (1/1) |" in md
    assert "| indep | 33% (1/3) | 50% (1/2) | 0% (0/1) | 100% (1/1) | 0% (0/1) |" in md
    assert "| mcmc | 33% (1/3) | — | 0% (0/1) | 0% (0/1) | 100% (1/1) |" in md


def test_raising_or_non_verdict_rules_are_errors_on_that_case_only():
    def flaky(refined_code, original_code, **_):
        if refined_code == "r2":
            raise RuntimeError("boom")
        return _verdict(refined_code == "r1")

    def refuses(*_, **__):
        return None  # with_structured_output yields None when there is no tool call

    res = run_arbiter_comparison(_cases(), {"flaky": flaky, "refuses": refuses},
                                 validate=lambda c: _MCMC[c["name"]])
    assert res["rows"][1]["judges"]["flaky"] == {
        "passed": None, "confidence": None, "reasoning": None, "error": "RuntimeError: boom"}
    assert res["rows"][0]["judges"]["refuses"]["error"] == (
        "TypeError: judge returned NoneType, not JudgeVerdict")
    s = res["summary"]["flaky"]
    assert (s["n_judged"], s["n_errors"], s["n_accepted"]) == (2, 1, 1)
    assert s["false_accept_cond_rate"] == 0.5 and s["false_reject_cond_rate"] is None
    assert res["summary"]["refuses"]["acceptance_rate"] is None
    assert res["summary"]["mcmc"]["n_judged"] == 3  # the pseudo-rule is unaffected
    assert "flaky (1), refuses (3)" in format_arbiter_comparison(res)


# --- the single-judge baseline tally (run_judge_baseline / format_judge_baseline) --------
# Ported from the research suite's tests/test_judge.py. Its three plumbing tests (the
# create_llm_judge round trip and the two arbiter_agreement cases) are covered above.


def _baseline_result():
    verdicts = [
        JudgeVerdict(passed=True, reasoning="a"),    # case1: judge PASS
        JudgeVerdict(passed=True, reasoning="b"),    # case2: judge PASS
        JudgeVerdict(passed=False, reasoning="c"),   # case3: judge FAIL
    ]
    judge = create_llm_judge(FixtureLLM(outputs=verdicts))
    mcmc = {"case1": _vr(False), "case2": _vr(True), "case3": _vr(False)}
    cases = [
        {"name": "case1", "refined_code": "r1", "original_code": "o"},
        {"name": "case2", "refined_code": "r2", "original_code": "o"},
        {"name": "case3", "refined_code": "r3", "original_code": "o"},
    ]
    return run_judge_baseline(cases, judge, validate=lambda c: mcmc[c["name"]])


def test_run_judge_baseline_tallies_false_acceptance():
    res = _baseline_result()
    assert res["n"] == 3
    # case1: judge passed a model MCMC rejects -> false accept
    assert abs(res["false_accept_rate"] - 1 / 3) < 1e-9
    # case2 (both pass) + case3 (both fail) agree
    assert abs(res["agreement_rate"] - 2 / 3) < 1e-9
    assert res["false_reject_rate"] == 0.0
    fa = [r for r in res["rows"] if r["false_accept"]]
    assert len(fa) == 1 and fa[0]["name"] == "case1"


def test_format_judge_baseline_markdown():
    md = format_judge_baseline(_baseline_result())
    assert "LLM-judge" in md and "false-acceptance" in md
    assert "FALSE ACCEPT" in md  # case1's outcome


# ---------------------------------------------------------------------------
# A judge that yields no verdict must not lose the rows judged so far
# ---------------------------------------------------------------------------

_BASELINE_CASES = [
    {"name": "case1", "refined_code": "r1", "original_code": "o"},
    {"name": "case2", "refined_code": "r2", "original_code": "o"},
    {"name": "case3", "refined_code": "r3", "original_code": "o"},
]
_BASELINE_MCMC = {"case1": _vr(False), "case2": _vr(True), "case3": _vr(False)}


def _scripted_judge(*verdicts):
    """A judge callable that returns (or raises) the scripted outcomes in order."""
    queue = list(verdicts)

    def judge(refined_code, original_code, **kwargs):
        out = queue.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    return judge


def test_run_judge_baseline_records_a_none_verdict_as_an_error_row():
    """``with_structured_output`` yields None when the model answers without a tool call.
    That used to raise AttributeError on ``verdict.confidence`` and lose every row judged so
    far; it is now an error row (judge_passed None) excluded from the rates."""
    judge = _scripted_judge(JudgeVerdict(passed=True, reasoning="a"), None,
                            JudgeVerdict(passed=False, reasoning="c"))
    res = run_judge_baseline(_BASELINE_CASES, judge, validate=lambda c: _BASELINE_MCMC[c["name"]])
    assert res["n"] == 3 and res["n_errors"] == 1
    err = res["rows"][1]
    assert err["name"] == "case2" and err["judge_passed"] is None and err["agree"] is None
    assert err["false_accept"] is False and err["false_reject"] is False
    assert err["judge_confidence"] is None and err["judge_reasoning"] is None
    assert err["judge_error"].startswith("TypeError") and "NoneType" in err["judge_error"]
    assert err["mcmc_passed"] is True  # the MCMC verdict is still recorded
    # rates over the two judged cases: case1 false accept, case3 agree
    assert abs(res["false_accept_rate"] - 0.5) < 1e-9
    assert abs(res["agreement_rate"] - 0.5) < 1e-9
    assert res["false_reject_rate"] == 0.0
    ok = res["rows"][0]
    assert ok["judge_passed"] is True and ok["judge_error"] is None
    assert ok["judge_confidence"] == "medium" and ok["judge_reasoning"] == "a"
    md = format_judge_baseline(res)
    assert "| case2 | pass | error | judge error |" in md
    assert "1 judge error(s) excluded from the rates" in md


def test_run_judge_baseline_records_a_raising_judge_as_an_error_row():
    """Same contract as ``run_arbiter_comparison``: a rule that raises on one case is an
    error on that case, not an abort of the whole tally."""
    judge = _scripted_judge(RuntimeError("overloaded"), JudgeVerdict(passed=True, reasoning="b"),
                            JudgeVerdict(passed=False, reasoning="c"))
    res = run_judge_baseline(_BASELINE_CASES, judge, validate=lambda c: _BASELINE_MCMC[c["name"]])
    assert res["n_errors"] == 1 and res["rows"][0]["judge_error"] == "RuntimeError: overloaded"
    assert [r["judge_passed"] for r in res["rows"]] == [None, True, False]
    assert res["agreement_rate"] == 1.0 and res["false_accept_rate"] == 0.0


def test_run_judge_baseline_all_errors_reports_no_rates():
    """P5: a judge that errored on every case has no agreement / false-acceptance rate.
    The three rates are None (as ``run_arbiter_comparison`` reports them), never a 0.0
    that reads as a perfect score, and the markdown renders them as a dash."""
    judge = _scripted_judge(None, None, None)
    res = run_judge_baseline(_BASELINE_CASES, judge, validate=lambda c: _BASELINE_MCMC[c["name"]])
    assert res["n"] == 3 and res["n_errors"] == 3
    assert res["agreement_rate"] is None
    assert res["false_accept_rate"] is None and res["false_reject_rate"] is None
    assert all(r["judge_passed"] is None and r["mcmc_passed"] is not None for r in res["rows"])
    md = format_judge_baseline(res)
    assert "agreement —" in md and "false-acceptance —" in md and "false-rejection —" in md
    assert "3 judge error(s) excluded from the rates" in md
    assert md.count("| judge error |") == 3


def test_rates_are_numbers_as_soon_as_one_row_is_judged():
    judge = _scripted_judge(None, JudgeVerdict(passed=True, reasoning="b"), None)
    res = run_judge_baseline(_BASELINE_CASES, judge, validate=lambda c: _BASELINE_MCMC[c["name"]])
    assert res["n_errors"] == 2
    assert res["agreement_rate"] == 1.0 and res["false_accept_rate"] == 0.0
    assert res["false_reject_rate"] == 0.0
    assert "agreement 100%" in format_judge_baseline(res)


def test_a_malformed_case_is_the_callers_bug_not_a_judge_error_row():
    """P5: ``_call_judge`` reads refined_code / original_code before it guards the judge
    call, so a case without them raises KeyError instead of being booked as a judge
    error — which would silently drop the case from every rate."""
    judge = _scripted_judge(JudgeVerdict(passed=True, reasoning="a"))
    with pytest.raises(KeyError, match="original_code"):
        _call_judge(judge, {"name": "broken", "refined_code": "r"})
    with pytest.raises(KeyError, match="refined_code"):
        run_judge_baseline([{"name": "broken", "original_code": "o"}], judge,
                           validate=lambda c: _vr(True))
    # a well-formed case with a raising judge is still an error row, not a raise
    boom = _scripted_judge(RuntimeError("overloaded"))
    row = _call_judge(boom, {"name": "ok", "refined_code": "r", "original_code": "o"})
    assert row["passed"] is None and row["error"] == "RuntimeError: overloaded"


def test_run_judge_baseline_keeps_full_reasoning_and_forwards_rationale():
    """The baseline stores the judge's whole reasoning (no 500-char cap, as before) and,
    like ``run_arbiter_comparison``, forwards a case's ``rationale`` when it has one."""
    long = "r" * 900
    fake = FixtureLLM(outputs=[JudgeVerdict(passed=True, reasoning=long)])
    case = {"name": "c", "refined_code": "r", "original_code": "o", "rationale": "RATIONALE-MARK"}
    res = run_judge_baseline([case], create_llm_judge(fake), validate=lambda c: _vr(True))
    assert res["rows"][0]["judge_reasoning"] == long and res["n_errors"] == 0
    assert "RATIONALE-MARK" in fake.calls[0]


def test_all_exports_match_the_research_module():
    assert set(judge_mod.__all__) == {
        "create_llm_judge", "create_self_judge", "create_independent_judge",
        "create_control_judge", "create_execution_check", "arbiter_agreement",
        "run_judge_baseline", "format_judge_baseline", "run_arbiter_comparison",
        "format_arbiter_comparison", "JUDGE_PROMPT",
    }
    for name in judge_mod.__all__:
        assert hasattr(judge_mod, name)
