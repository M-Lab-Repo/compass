"""LLM-judge baseline (v3) and the alternative acceptance rules it is compared to.

Contemporary agentic-science systems adjudicate their own outputs with LLM or
human review. To show empirically that our MCMC-as-sole-arbiter design avoids
the resulting false acceptances, we add an LLM JUDGE that decides — from the
model code alone, WITHOUT running MCMC — whether a refined model improves on the
original, and compare its verdict to the validator's ground truth.

`run_arbiter_comparison` runs several acceptance rules side by side — the LLM
judge reading code + the proposer's rationale (the proposer's own model, or an
independent one), the same judge with no rationale (control), and a
compile/smoke execution check — against the MCMC verdict and, when available,
a hidden held-out label (genuine / confabulated / harmful).

This module is EVALUATION-ONLY. No judge here is wired into the refinement
pipeline as an arbiter; `graph.validation_passed` (the MCMC validator) remains
the sole arbiter. The headline metric is a rule's FALSE-ACCEPTANCE rate: how
often it passes a model that real MCMC (or the hidden label) rejects.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from langchain_core.prompts import ChatPromptTemplate

from compass.agents import synthesizer
from compass.schemas import JudgeVerdict

_JUDGE_SYSTEM = """You are a senior Bayesian statistician reviewing a proposed \
refinement to a probabilistic model. You are given the ORIGINAL model and a \
REFINED model. Decide whether the refined model is a GENUINE improvement — \
better-specified priors, sounder geometry, better-calibrated likelihood — that \
would converge cleanly and predict at least as well out of sample.

You CANNOT run the model. Judge from the code, the stated refinements, and your \
expertise alone. Be discerning: a plausible-looking change is not automatically \
an improvement, and a change can diverge, over-fit, or fail a posterior \
predictive check. Return `passed` (true only if you believe it is a real, \
safe improvement), a `confidence`, and a one-paragraph `reasoning`."""

_JUDGE_HUMAN = """## Problem
{problem}

## Dataset
{dataset_summary}

## Refinements proposed
{proposals}

## Proposer's justification
{rationale}

## Original model
```
{original_code}
```

## Refined model
```
{refined_code}
```

Is the refined model a genuine improvement over the original? Decide WITHOUT \
running it."""

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _JUDGE_SYSTEM), ("human", _JUDGE_HUMAN)]
)

_HIDDEN_LABELS = ("genuine", "confabulated", "harmful")
_HIDDEN_BAD = ("confabulated", "harmful")


def _model_id(llm) -> str | None:
    for attr in ("model", "model_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def create_llm_judge(llm) -> Callable:
    """Build the baseline judge: ``judge(refined, original, ...) -> JudgeVerdict``."""
    chain = JUDGE_PROMPT | llm.with_structured_output(JudgeVerdict)

    def judge(refined_code: str, original_code: str, *, proposals=None,
              dataset_summary: str = "", problem: str = "",
              rationale: str = "") -> JudgeVerdict:
        names = []
        for p in proposals or []:
            p = p.model_dump() if hasattr(p, "model_dump") else p
            names.append(p.get("name", "?") if isinstance(p, dict) else str(p))
        return chain.invoke({
            "problem": problem or "(unspecified)",
            "dataset_summary": dataset_summary or "(unspecified)",
            "proposals": ", ".join(names) or "(not provided)",
            "rationale": rationale or "(no rationale provided)",
            "original_code": original_code,
            "refined_code": refined_code,
        })

    judge.role = "llm"
    judge.model_id = _model_id(llm)
    return judge


def create_self_judge(llm) -> Callable:
    """The proposer's own model judging its proposal from code + rationale."""
    judge = create_llm_judge(llm)
    judge.role = "self"
    return judge


def create_independent_judge(llm) -> Callable:
    """A different model judging the proposal from code + rationale."""
    judge = create_llm_judge(llm)
    judge.role = "independent"
    return judge


def create_control_judge(llm) -> Callable:
    """The same LLM judge with the rationale withheld (control arm): code only."""
    base = create_llm_judge(llm)

    def judge(refined_code: str, original_code: str, *, proposals=None,
              dataset_summary: str = "", problem: str = "",
              rationale: str = "") -> JudgeVerdict:
        return base(refined_code, original_code, proposals=proposals,
                    dataset_summary=dataset_summary, problem=problem)

    judge.role = "control"
    judge.model_id = base.model_id
    return judge


def create_execution_check(data_shapes: dict, *, timeout: int = 120,
                           data: dict | None = None) -> Callable:
    """Acceptance rule: the refined model parses and survives a 10-draw NUTS smoke test
    (on ``data`` when given, else on dummy data shaped like ``data_shapes``)."""

    def check(refined_code: str, original_code: str, *, proposals=None,
              dataset_summary: str = "", problem: str = "",
              rationale: str = "") -> JudgeVerdict:
        res = synthesizer.validate_numpyro_code(refined_code, data_shapes, timeout, data=data)
        reasoning = f"execution check: stage={res.get('stage')}"
        # the smoke harness reports candidate exceptions on stdout as FAILURE lines
        failures = [ln for ln in (res.get("stdout") or "").splitlines()
                    if ln.startswith("FAILURE:")]
        if failures:
            reasoning += f"; {failures[-1][:300]}"
        stderr = (res.get("stderr") or "").strip()
        if stderr:
            reasoning += f"; stderr: {stderr[:300]}"
        return JudgeVerdict(passed=bool(res["compiled"]), confidence="high",
                            reasoning=reasoning)

    check.role = "execution"
    check.model_id = None
    return check


def _mcmc_passed(validation) -> bool:
    return bool(getattr(validation, "converged", False)
                and not getattr(validation, "failure_reasons", []))


def _agreement(judge_passed: bool, mcmc_passed: bool) -> dict:
    return {
        "judge_passed": judge_passed,
        "mcmc_passed": mcmc_passed,
        "agree": judge_passed == mcmc_passed,
        "false_accept": judge_passed and not mcmc_passed,
        "false_reject": (not judge_passed) and mcmc_passed,
    }


def arbiter_agreement(verdict: JudgeVerdict, validation) -> dict:
    """Compare the LLM judge's pass/fail to the MCMC validator's ground truth.

    ``validation`` is a ``state.ValidationResult`` (the sole arbiter). Flags the
    two error types — ``false_accept`` (judge passes what MCMC rejects, the
    dangerous one) and ``false_reject`` (judge rejects what MCMC accepts).
    """
    return _agreement(bool(getattr(verdict, "passed", False)), _mcmc_passed(validation))


def run_judge_baseline(cases, judge: Callable, validate: Callable) -> dict:
    """Run the judge vs the MCMC validator over ``cases`` and tally agreement.

    Each case is a dict with ``refined_code``, ``original_code`` and optionally
    ``name`` / ``proposals`` / ``dataset_summary`` / ``problem`` / ``rationale``.
    ``validate(case) -> ValidationResult`` is the ground-truth MCMC verdict (run
    it, or pass a precomputed result through a closure). Returns per-case rows
    plus the headline rates — chiefly ``false_accept_rate``.

    The judge call goes through ``_call_judge``, as in ``run_arbiter_comparison``:
    a judge that raises, or returns anything but a ``JudgeVerdict`` (a
    ``with_structured_output`` chain yields ``None`` when the model answers
    without a tool call), is recorded as an error row — ``judge_passed`` /
    ``agree`` / ``judge_confidence`` / ``judge_reasoning`` ``None``,
    ``judge_error`` set — and excluded from the rates (``n_errors`` counts them).
    Fix: previously such a verdict raised ``AttributeError`` on
    ``verdict.confidence`` and lost every row judged so far.
    """
    rows = []
    for c in cases:
        res = _call_judge(judge, c, max_reasoning=None)
        mcmc_passed = _mcmc_passed(validate(c))
        if res["error"] is None:
            cmp = _agreement(res["passed"], mcmc_passed)
        else:
            cmp = {"judge_passed": None, "mcmc_passed": mcmc_passed, "agree": None,
                   "false_accept": False, "false_reject": False}
        rows.append({**cmp, "name": c.get("name", "?"),
                     "judge_confidence": res["confidence"],
                     "judge_reasoning": res["reasoning"],
                     "judge_error": res["error"]})
    judged = [r for r in rows if r["judge_passed"] is not None]
    n = len(judged)
    return {
        "n": len(rows),
        "n_errors": len(rows) - n,
        # None, not 0.0, when no row was judged: a judge that errored on every case
        # has no agreement or false-acceptance rate (``run_arbiter_comparison``
        # reports None the same way, through ``_rate``); a 0.0 read as a perfect
        # score. ``format_judge_baseline`` renders None as a dash.
        "agreement_rate": _rate(sum(r["agree"] for r in judged), n),
        "false_accept_rate": _rate(sum(r["false_accept"] for r in judged), n),
        "false_reject_rate": _rate(sum(r["false_reject"] for r in judged), n),
        "rows": rows,
    }


def _pct(rate: float | None) -> str:
    """A rate for the markdown summary; ``None`` (nothing judged) renders as a dash."""
    return "—" if rate is None else f"{rate:.0%}"


def format_judge_baseline(result: dict) -> str:
    """Deterministic markdown summary of the MCMC-arbiter vs LLM-judge comparison."""
    n_errors = result.get("n_errors", 0)
    lines = [
        "## MCMC arbiter vs. LLM-judge baseline",
        f"\n{result['n']} cases · agreement {_pct(result['agreement_rate'])} · "
        f"**LLM-judge false-acceptance {_pct(result['false_accept_rate'])}** "
        f"(passed a model real MCMC rejected) · "
        f"false-rejection {_pct(result['false_reject_rate'])}"
        + (f" · {n_errors} judge error(s) excluded from the rates" if n_errors else "") + "\n",
        "| case | MCMC | LLM-judge | outcome |",
        "|---|---|---|---|",
    ]
    for r in result["rows"]:
        if r["judge_passed"] is None:  # the judge raised or returned no verdict
            judge_cell, outcome = "error", "judge error"
        else:
            judge_cell = "pass" if r["judge_passed"] else "fail"
            outcome = ("FALSE ACCEPT" if r["false_accept"] else
                       "false reject" if r["false_reject"] else "agree")
        lines.append(
            f"| {r['name']} | {'pass' if r['mcmc_passed'] else 'fail'} | "
            f"{judge_cell} | {outcome} |"
        )
    return "\n".join(lines) + "\n"


def _call_judge(judge: Callable, case: dict, *, max_reasoning: int | None = 500) -> dict:
    """Call ``judge`` on ``case`` and reduce the outcome to a row fragment.

    Never raises FOR THE JUDGE: a judge that raises, or returns anything but a
    ``JudgeVerdict``, yields ``passed=None`` with ``error`` set, so one bad verdict
    cannot lose the rows already collected. A malformed case (no ``refined_code`` /
    ``original_code``) is the caller's bug and raises ``KeyError`` — it is read
    before the guarded call, so it can never be booked as a judge error and vanish
    from the rates. ``max_reasoning`` caps the stored reasoning text (``None`` keeps
    it whole, as ``run_judge_baseline`` always did).
    """
    refined_code, original_code = case["refined_code"], case["original_code"]
    try:
        v = judge(
            refined_code, original_code,
            proposals=case.get("proposals"), dataset_summary=case.get("dataset_summary", ""),
            problem=case.get("problem", ""), rationale=case.get("rationale", ""),
        )
        # with_structured_output(first_tool_only) yields None when the model
        # answers without a tool call; that is a data point, not a crash.
        if not isinstance(v, JudgeVerdict):
            raise TypeError(f"judge returned {type(v).__name__}, not JudgeVerdict")
        reasoning = v.reasoning or ""
        return {"passed": bool(v.passed), "confidence": v.confidence,
                "reasoning": reasoning if max_reasoning is None else reasoning[:max_reasoning],
                "error": None}
    except Exception as e:
        return {"passed": None, "confidence": None, "reasoning": None,
                "error": f"{type(e).__name__}: {e}"[:500]}


def _rate(hits: int, n: int) -> float | None:
    return hits / n if n else None


def _acceptance(rows: list[dict], passed: list[bool | None], *,
                with_mcmc: bool, with_hidden: bool, labels: tuple[str, ...]) -> dict:
    judged = [(r, p) for r, p in zip(rows, passed, strict=True) if p is not None]
    n = len(judged)
    accepted = sum(p for _, p in judged)
    out = {"n_judged": n, "n_errors": len(rows) - n, "n_accepted": accepted,
           "acceptance_rate": _rate(accepted, n)}
    if with_mcmc:
        n_failed = sum(1 for r, _ in judged if not r["mcmc_passed"])
        n_passed = n - n_failed
        fa = sum(1 for r, p in judged if p and not r["mcmc_passed"])
        fr = sum(1 for r, p in judged if not p and r["mcmc_passed"])
        out.update(n_mcmc_failed=n_failed, n_mcmc_passed=n_passed,
                   false_accept_n=fa, false_accept_rate=_rate(fa, n),
                   false_accept_cond_rate=_rate(fa, n_failed),
                   false_reject_n=fr, false_reject_rate=_rate(fr, n),
                   false_reject_cond_rate=_rate(fr, n_passed))
    if with_hidden:
        def tally(sub: list[bool]) -> dict:
            return {"n": len(sub), "n_accepted": sum(sub), "rate": _rate(sum(sub), len(sub))}

        out["by_hidden_label"] = {
            lab: tally([p for r, p in judged if r["hidden_label"] == lab]) for lab in labels
        }
        out["false_accept_hidden"] = tally(
            [p for r, p in judged if r["hidden_label"] in _HIDDEN_BAD]
        )
    return out


def run_arbiter_comparison(cases: list[dict], judges: dict[str, Callable], *,
                           validate: Callable | None = None,
                           hidden_scorer: Callable | None = None) -> dict:
    """Run every acceptance rule in ``judges`` over ``cases`` and tally each one.

    Cases are dicts as in ``run_judge_baseline`` plus an optional ``rationale``.
    ``validate(case) -> ValidationResult`` gives the MCMC verdict (same pass
    predicate as ``arbiter_agreement``); ``hidden_scorer(case) -> {"label",
    "delta"?, "se"?}`` gives the held-out label. A rule that raises, or returns
    anything but a ``JudgeVerdict``, is recorded as an error on that case and
    drops out of its own denominators. When ``validate`` is given, the MCMC
    verdict is tallied as the pseudo-rule ``"mcmc"`` so it can sit in the same
    per-label table.

    Two false-acceptance conventions are reported per rule: ``false_accept_rate``
    is the joint P(accept and MCMC rejected) over the rule's judged cases, as in
    ``run_judge_baseline``; ``false_accept_cond_rate`` is the conditional
    P(accept | MCMC rejected) with denominator ``n_mcmc_failed`` — the paper's
    convention (``bench.aggregate.arbiter_stats``) and what the table shows.
    Likewise ``false_reject_rate`` / ``false_reject_cond_rate`` (``n_mcmc_passed``).
    """
    rows = []
    for c in cases:
        row = {"name": c.get("name", "?"),
               "judges": {name: _call_judge(j, c) for name, j in judges.items()},
               "mcmc_passed": None, "hidden_label": None,
               "hidden_delta": None, "hidden_se": None}
        if validate is not None:
            row["mcmc_passed"] = _mcmc_passed(validate(c))
        if hidden_scorer is not None:
            h = hidden_scorer(c)
            row["hidden_label"] = h["label"]
            row["hidden_delta"] = h.get("delta")
            row["hidden_se"] = h.get("se")
        rows.append(row)

    with_mcmc, with_hidden = validate is not None, hidden_scorer is not None
    extra = {r["hidden_label"] for r in rows} - set(_HIDDEN_LABELS) - {None}
    labels = (*_HIDDEN_LABELS, *sorted(extra))
    summary = {}
    for name in judges:
        summary[name] = _acceptance(rows, [r["judges"][name]["passed"] for r in rows],
                                    with_mcmc=with_mcmc, with_hidden=with_hidden, labels=labels)
    if with_mcmc:
        summary["mcmc"] = _acceptance(rows, [r["mcmc_passed"] for r in rows],
                                      with_mcmc=False, with_hidden=with_hidden, labels=labels)
    return {"n": len(rows), "judges": list(judges), "summary": summary, "rows": rows}


def _cell(rate: float | None, hits: int | None, n: int | None) -> str:
    return "—" if rate is None else f"{rate:.0%} ({hits}/{n})"


def format_arbiter_comparison(result: dict) -> str:
    """Deterministic markdown table: acceptance rules × (accept, FA vs MCMC, accept by label).

    The FA column is the conditional rate — acceptance among the rule's
    MCMC-rejected cases (``false_accept_cond_rate``), the paper's convention.
    """
    summary = result["summary"]
    names = list(result["judges"]) + (["mcmc"] if "mcmc" in summary else [])
    lines = [
        "## Acceptance rules vs. ground truth",
        f"\n{result['n']} cases · rules: {', '.join(result['judges'])}"
        + (" · MCMC validator as arbiter · FA vs MCMC = acceptance among MCMC-rejected cases"
           if "mcmc" in summary else "") + "\n",
        "| rule | accept | FA vs MCMC | accept on confabulated | accept on harmful "
        "| accept on genuine |",
        "|---|---|---|---|---|---|",
    ]
    for name in names:
        s = summary[name]
        by = s.get("by_hidden_label", {})
        cells = [
            name,
            _cell(s["acceptance_rate"], s["n_accepted"], s["n_judged"]),
            _cell(s.get("false_accept_cond_rate"), s.get("false_accept_n"),
                  s.get("n_mcmc_failed")),
        ]
        for lab in ("confabulated", "harmful", "genuine"):
            cells.append(_cell(by[lab]["rate"], by[lab]["n_accepted"], by[lab]["n"])
                         if lab in by else "—")
        lines.append("| " + " | ".join(cells) + " |")
    errors = [f"{name} ({summary[name]['n_errors']})" for name in names
              if summary[name]["n_errors"]]
    if errors:
        lines.append(f"\nrules that raised on some cases (excluded from their rates): "
                     f"{', '.join(errors)}")
    extra = Counter(r["hidden_label"] for r in result["rows"]
                    if r["hidden_label"] not in (*_HIDDEN_LABELS, None))
    if extra:
        seen = ", ".join(f"{lab} ({n})" for lab, n in sorted(extra.items()))
        lines.append(f"\nhidden labels outside {{{', '.join(_HIDDEN_LABELS)}}} — not shown "
                     f"in the table, tallied in summary.by_hidden_label: {seen}")
    return "\n".join(lines) + "\n"


__all__ = ["create_llm_judge", "create_self_judge", "create_independent_judge",
           "create_control_judge", "create_execution_check", "arbiter_agreement",
           "run_judge_baseline", "format_judge_baseline", "run_arbiter_comparison",
           "format_arbiter_comparison", "JUDGE_PROMPT"]
