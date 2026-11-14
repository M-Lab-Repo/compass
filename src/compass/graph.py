"""LangGraph orchestrator — wires the six agents into the refinement state machine.

Spec reference: §5.6. Nodes are the agent ``run(state) -> update-dict``
callables built by each ``create_*_agent`` factory; edges are conditional on
state. The feedback loop is the architectural core: the validator emits
structured failure reasons, ``validate`` records them into
``feedback_history``, and ``route_after_validation`` decides which agent to
re-invoke.

Two deliberate deviations from the spec's illustrative snippet, both documented
where they occur:

* Routing functions are PURE (LangGraph conditional-edge callables do not
  persist state mutations) — iteration/feedback bookkeeping happens inside the
  ``validate`` node instead.
* The success criterion is the spec §13 *tightened* one: converged AND zero
  failure reasons. The validator records "no significant improvement" /
  PPC failure as (non-convergence) failure reasons, so an empty list already
  means "converged, significantly better (|ΔeLPD| ≥ 2·SE), PPC passed".
* Synthesis exhaustion routes to ``prepare_report`` (not bare END) so every
  terminal path lands on status "complete" or "failed" with a report.
"""

from __future__ import annotations

import html as _html
import re
import uuid
from typing import Any

import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from compass.agents.analyst import create_analyst_agent
from compass.agents.scout import create_scout_agent
from compass.agents.strategist import (
    EMPTY_BUNDLE_LOG_PREFIX,
    create_strategist_agent,
    proposals_rationale,
)
from compass.agents.synthesizer import create_synthesizer_agent
from compass.agents.validator import create_validator_agent
from compass.state import ModelSpec, WorkflowState

# ---------------------------------------------------------------------------
# Routing (pure functions — unit-testable, no side effects)
# ---------------------------------------------------------------------------


def validation_passed(state: WorkflowState) -> bool:
    """Spec §13 tightened criterion: converged and nothing left to complain about."""
    v = state.validation
    return v is not None and v.converged and not v.failure_reasons


def route_after_synthesis(state: WorkflowState) -> str:
    """The synthesizer already runs its own generate->validate->fix loop
    (max_attempts), so an empty ``refined_model_code`` here means the proposal
    is likely fundamentally incompatible — report failure rather than retry."""
    if state.refined_model_code.strip():
        return "validate"
    return "fail"


def route_after_validation(state: WorkflowState) -> str:
    """success | retry_with_feedback (computational issue) | search_more
    (converged but no gain) | fail (iteration budget exhausted)."""
    v = state.validation
    if v is None:
        return "fail"
    if validation_passed(state):
        return "success"
    if state.iteration >= state.max_iterations:
        return "fail"
    if not v.converged:
        return "retry_with_feedback"
    return "search_more"


def route_after_feasibility(state: WorkflowState) -> str:
    """synthesize (survivors) | retry_with_feedback (all dropped, budget left) |
    fail (all dropped, budget exhausted). Pure: the gate node does the
    iteration/feedback bookkeeping (LangGraph routers cannot persist state)."""
    if state.proposals:
        return "synthesize"
    if state.iteration >= state.max_iterations:
        return "fail"
    return "retry_with_feedback"


# ---------------------------------------------------------------------------
# Report agent (spec §5.6 ``create_report_agent``)
# ---------------------------------------------------------------------------

REPORT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a Bayesian statistician writing the executive summary "
               "of a model-refinement run for the model's owner. Be concrete: "
               "name the changes, the evidence, and the validation numbers. "
               "3-6 sentences, no headings."),
    ("human", "## Outcome\n{outcome}\n\n## Proposals applied\n{proposals}\n\n"
              "## Validation\n{validation}\n\n## Iterations\n{history}"),
])


def _render_validation(state: WorkflowState) -> str:
    v = state.validation
    if v is None:
        return "(validation never ran)"
    lines = [
        f"- converged: {v.converged}",
        f"- R-hat max: {v.rhat_max:.4f}   ESS min: {v.ess_min:.0f}   "
        f"divergences: {v.divergences}",
        f"- WAIC: {v.waic:.1f} (Δ vs original: {v.waic_delta})   "
        f"LOO: {v.loo:.1f} (Δ: {v.loo_delta})",
        f"- posterior predictive check: {'pass' if v.posterior_predictive_check else 'FAIL'}",
    ]
    if getattr(v, "s_gate", None):
        lines.append(f"- significance gate: {v.s_gate}")
    if getattr(v, "repair", None):
        lines.append(f"- REPAIR CLAUSE: {v.repair}")
    pm = getattr(v, "ppc_mixed", None)
    if pm:
        if pm.get("error"):
            lines.append(f"- mixed-replication PPC (descriptive): unavailable ({pm['error']})")
        else:
            failed = [d.get("stat") for d in pm.get("details", [])
                      if not d.get("passed")]
            lines.append("- mixed-replication PPC (descriptive): "
                         + ("pass" if pm.get("passed")
                            else "FAIL — " + ", ".join(str(s) for s in failed)))
    if getattr(v, "confidence", ""):
        lines.append(f"- confidence (margin-calibrated): {v.confidence}")
    _marks = {True: "PASS", False: "FAIL", None: "n/a"}
    if getattr(v, "proposal_scores", None):
        lines.append("- per-proposal predictions (scored vs MCMC):")
        for s in v.proposal_scores:
            lines.append(
                f"  - {s.get('name')}: predicted {s.get('predicted')} -> "
                f"observed {s.get('observed')} [{_marks.get(s.get('passed'), 'n/a')}]"
            )
    if v.failure_reasons:
        lines.append("- failure reasons: " + "; ".join(str(r) for r in v.failure_reasons))
    return "\n".join(lines)


def build_grounding_block(state: WorkflowState) -> str:
    """Deterministic, code-generated evidence-grounding block (MEDEA borrow).

    For each proposal: its provenance (literature | model_structure | llm_prior),
    its cited evidence, and — when the validator scored it — the verdict of its
    own falsifiable prediction against the real MCMC result. A summary line
    classifies the whole refinement as full | partial | llm_only grounded. This
    is NOT an LLM judgment: provenance comes from the strategist's declared (and
    deterministically cross-checked) evidence_basis, the verdicts come from the
    MCMC-driven validator.
    """
    proposals = state.proposals or []
    if not proposals:
        return ""
    bases = [p.get("evidence_basis", "llm_prior") for p in proposals]
    n_lit = sum(b == "literature" for b in bases)
    grounding = "full" if n_lit == len(proposals) else \
        "llm_only" if n_lit == 0 else "partial"

    v = state.validation
    scores = {s.get("name"): s for s in (getattr(v, "proposal_scores", None) or [])}
    marks = {True: "✓", False: "✗", None: "·"}

    lines = [
        "## Evidence grounding",
        f"\n**Grounding:** {grounding} "
        f"({n_lit}/{len(proposals)} proposals literature-backed)\n",
    ]
    if v is not None and getattr(v, "confidence", ""):
        lines.append(f"**Validator confidence:** {v.confidence} (ΔLOO {v.loo_delta})\n")
    for p in proposals:
        name = p.get("name", "?")
        basis = p.get("evidence_basis", "llm_prior")
        papers = ", ".join(p.get("supporting_papers", [])) or "—"
        s = scores.get(name)
        verdict = ""
        if s is not None:
            verdict = (f" — prediction *{s.get('predicted')}* "
                       f"{marks.get(s.get('passed'), '·')} (observed {s.get('observed')})")
        lines.append(f"- **{name}** [{basis}] evidence: {papers}{verdict}")
    return "\n".join(lines) + "\n"


def _failure_diagnosis(state: WorkflowState) -> str:
    """On a failed run, say WHY in the arbiter's terms: was a fix proposed but its
    code never ran (completeness failure), or did the code run and the verifier
    reject it (a certified 'no')? These are very different messages to a user and
    otherwise only recoverable by digging through error_log / feedback state."""
    if validation_passed(state):
        return ""
    v = state.validation
    lines: list[str] = []
    # Verified-and-refused rounds on record (candidate_history keeps every
    # candidate the verifier judged; refined_model_code / validation are
    # overwritten each iteration). On a failed run every entry is a refusal.
    refused = [c for c in (state.candidate_history or []) if not c.get("passed")]
    if not state.proposals and refused:
        # The P1 terminal path: the final strategist round delivered nothing
        # that survived post-validation / the feasibility gate (details in the
        # Errors section) after earlier candidates were refused. Checked first
        # because refined_model_code / validation still hold the LAST refused
        # candidate's code and verdict (the synthesizer never ran again), which
        # would otherwise make this read as a plain refusal.
        lines.append(
            "The final round **proposed nothing that survived post-validation "
            "or the feasibility gate**, so no new candidate was synthesized; "
            f"before it, {_n_candidates(refused)} verified and refused by the "
            "arbiter (see Errors for what was dropped and why).")
        lines.extend(_last_refusal_lines(refused))
    elif state.proposals and not state.refined_model_code:
        names = "; ".join(p.get("name", "?") for p in state.proposals)
        smoke = next((e for e in (state.error_log or []) if "synthes" in e.lower()), "")
        reason = ""
        for tok in ("FAILURE:", "last error:"):
            if tok in smoke:
                reason = smoke.split(tok, 1)[1].strip().splitlines()[0][:200]
                break
        if refused:
            # Defect fixed here (P6): when an earlier round WAS verified and
            # refused and a later round's synthesis failed, the report claimed
            # "nothing reached the verifier" — false for the run as a whole.
            lines.append(
                "The **last** proposed refinement's code never passed the "
                "smoke/convergence stage (a completeness failure of code synthesis, "
                f"not a rejection by the arbiter); before it, {_n_candidates(refused)} "
                "verified and refused by the arbiter.")
            lines.append(f"- Proposed last (code never ran): {names}")
            if reason:
                lines.append(f"- Synthesis blocker: `{reason}`")
            lines.extend(_last_refusal_lines(refused))
        else:
            lines.append(
                "A refinement **was proposed, but its code never passed the "
                "smoke/convergence stage**, so nothing reached the verifier. This is a "
                "completeness failure (code synthesis), not a rejection by the arbiter — "
                "the proposed change may still be sound.")
            lines.append(f"- Proposed: {names}")
            if reason:
                lines.append(f"- Synthesis blocker: `{reason}`")
    elif state.refined_model_code and v is not None and v.failure_reasons:
        lines.append(
            "A refined model **was synthesized and fit, but the verifier did not "
            "certify it** — the arbiter's reasons:")
        lines.extend(f"- {r}" for r in v.failure_reasons)
    elif not state.proposals and (summary := _empty_round_summary(state)) is not None:
        # Zero verified candidates AND an empty final bundle (P3): every strategist
        # round was rejected before synthesis — by the strategist's own
        # post-validation or by the pre-MCMC feasibility gate — so nothing was ever
        # synthesized or verified. This used to read "ended before any refinement
        # was proposed", which is false: proposals were made and dropped, and the
        # Errors section says which and why.
        lines.append(
            "Every round's proposals were rejected before synthesis (post-validation "
            "or the feasibility gate); see Errors for what was dropped and why.")
        lines.append(f"- Last round: {summary}")
    elif not state.proposals:
        lines.append("The run ended before any refinement was proposed.")
    if not lines:
        return ""
    return "## Why no model was accepted\n" + "\n".join(lines) + "\n"


def _n_candidates(refused: list) -> str:
    n = len(refused)
    return f"{n} earlier candidate{'s were' if n != 1 else ' was'}"


def _last_refusal_lines(refused: list) -> list[str]:
    """``- Last refused candidate: <names> — arbiter's reasons: ...`` for the most
    recent refused candidate_history entry (its ``validation`` is a
    ValidationResult, or a dict after a to_dict round-trip)."""
    last = refused[-1]
    names = ", ".join(str(n) for n in (last.get("proposal_names") or [])) or "?"
    val = last.get("validation")
    if isinstance(val, dict):
        reasons = val.get("failure_reasons") or []
    else:
        reasons = getattr(val, "failure_reasons", None) or []
    reason_text = "; ".join(str(r) for r in reasons) or "(no reasons recorded)"
    return [f"- Last refused candidate: {names} — arbiter's reasons: {reason_text}"]


# First words of the result the feasibility gate books for a round that reached it
# with nothing to synthesize: the strategist's own summary (its EMPTY_BUNDLE_LOG_PREFIX
# line minus the agent prefix) or the gate's own line when it dropped everything.
_POST_VALIDATION_REJECTED = "post-validation rejected"
_GATE_DROPPED_ALL = "pre-MCMC feasibility gate dropped every proposal"


def _empty_round_summary(state: WorkflowState) -> str | None:
    """The summary line of the LAST round that ended with an empty bundle, or None.

    Read from error_log first (the strategist's ``EMPTY_BUNDLE_LOG_PREFIX`` line
    without its ``strategist: `` prefix — exactly what the feasibility gate copies
    into its feedback entry), else from feedback_history: an entry with ``tried``
    None whose result starts "post-validation rejected", or one whose result starts
    with the gate's own "dropped every proposal" line (``tried`` names the first
    dropped proposal there). None when no round ended that way."""
    for e in reversed(state.error_log or []):
        if str(e).startswith(EMPTY_BUNDLE_LOG_PREFIX):
            return str(e).split("strategist: ", 1)[1]
    for h in reversed(state.feedback_history or []):
        if not isinstance(h, dict):
            continue
        result = h.get("result") or []
        first = (str(result[0]) if isinstance(result, (list, tuple)) and result
                 else str(result or ""))
        if first.startswith(_GATE_DROPPED_ALL) or (
                h.get("tried") is None and first.startswith(_POST_VALIDATION_REJECTED)):
            return first
    return None


def _tried_label(entry: dict) -> str:
    """``tried`` for an iteration-history line: the name in backticks, or
    ``(no technique)`` when the entry names none (the feasibility gate books an
    empty bundle with ``tried`` None; the verdict-only arm redacts it) — never a
    literal `None`."""
    tried = entry.get("tried")
    return "(no technique)" if tried is None else f"`{tried}`"


def _tried_html(entry: dict) -> str:
    """HTML twin of ``_tried_label``."""
    tried = entry.get("tried")
    return "(no technique)" if tried is None else f"<code>{_html.escape(str(tried))}</code>"


def build_report(state: WorkflowState, narrative: str = "", extra_block: str = "") -> str:
    """Deterministic markdown report; ``narrative`` is the optional LLM summary.

    ``extra_block`` (optional) is the DESIGN RESOLUTION text (resolution.py),
    rendered under its own heading so the report discloses the structural
    facts + hint_level the strategist was shown.
    """
    passed = validation_passed(state)
    outcome = "SUCCESS — refined model validated" if passed else \
        "FAILED — no validated improvement within the iteration budget"
    parts = [
        "# Bayesian Model Refinement Report",
        f"\n**Outcome:** {outcome}\n",
    ]
    diag = _failure_diagnosis(state)
    if diag:
        parts.append(diag)
    if narrative:
        parts.append(f"{narrative}\n")
    parts.append(f"## Problem\n{state.problem_description}\n")
    if state.analysis:
        issues = state.analysis.get("potential_issues", [])
        parts.append("## Identified weaknesses\n" + "\n".join(
            f"- **{i.get('severity', '?')}**: {i.get('issue', '')}" for i in issues) + "\n")
    if state.literature_findings:
        parts.append("## Literature\n" + "\n".join(
            f"- {f.get('paper_title', '?')} ({f.get('year', '?')}) — "
            f"{f.get('technique', '')} [relevance {f.get('relevance_score', 0):.2f}]"
            for f in state.literature_findings) + "\n")
    if state.proposals:
        parts.append("## Proposals (ranked)\n" + "\n".join(
            f"{p.get('priority', '?')}. **{p.get('name', '?')}** ({p.get('change_type', '')}) — "
            f"{p.get('description', '')} [evidence: {', '.join(p.get('supporting_papers', []))}]"
            for p in state.proposals) + "\n")
    if state.refined_model_code:
        parts.append("## Refined model\n```python\n" + state.refined_model_code.strip()
                     + "\n```\n")
    parts.append("## Validation\n" + _render_validation(state) + "\n")
    grounding = build_grounding_block(state)
    if grounding:
        parts.append(grounding)
    if extra_block:
        parts.append("## Design resolution\n" + extra_block + "\n")
    if state.feedback_history:
        parts.append("## Iteration history\n" + "\n".join(
            f"- iteration {h.get('iteration')}: tried {_tried_label(h)} -> "
            f"{'; '.join(str(r) for r in h.get('result', []))}"
            for h in state.feedback_history) + "\n")
    if state.error_log:
        parts.append("## Errors\n" + "\n".join(f"- {e}" for e in state.error_log) + "\n")
    return "\n".join(parts)


def _validation_html(state: WorkflowState) -> str:
    v = state.validation
    if v is None:
        return "<p>(validation never ran)</p>"
    esc = _html.escape
    rows = [
        ("converged", v.converged), ("R-hat max", f"{v.rhat_max:.4f}"),
        ("ESS min", f"{v.ess_min:.0f}"), ("divergences", v.divergences),
        ("WAIC (Δ vs original)", f"{v.waic:.1f} ({v.waic_delta})"),
        ("LOO (Δ)", f"{v.loo:.1f} ({v.loo_delta})"),
        ("posterior predictive check", "pass" if v.posterior_predictive_check else "FAIL"),
    ]
    if getattr(v, "confidence", ""):
        rows.append(("confidence", v.confidence))
    body = "".join(f"<tr><th>{esc(str(k))}</th><td>{esc(str(val))}</td></tr>"
                   for k, val in rows)
    out = f"<table>{body}</table>"
    for s in (getattr(v, "proposal_scores", None) or []):
        mark = {True: "PASS", False: "FAIL", None: "n/a"}.get(s.get("passed"), "n/a")
        out += (f"<p class='small'>{esc(str(s.get('name')))}: predicted "
                f"{esc(str(s.get('predicted')))} → observed {esc(str(s.get('observed')))} "
                f"[{mark}]</p>")
    if v.failure_reasons:
        out += ("<p><strong>failure reasons:</strong> "
                + esc("; ".join(str(r) for r in v.failure_reasons)) + "</p>")
    return out


def build_html_report(state: WorkflowState, narrative: str = "") -> str:
    """Deterministic standalone HTML export of the refinement report (v3).

    Mirrors ``build_report``'s core sections as a self-contained HTML document
    (no external CSS/JS, no LLM); the Errors section and the evidence-grounding
    block are intentionally omitted from the HTML export — consult the markdown
    report for those. ``narrative`` is the optional executive summary.
    """
    esc = _html.escape
    passed = validation_passed(state)
    outcome = ("SUCCESS — refined model validated" if passed else
               "FAILED — no validated improvement within the iteration budget")
    p = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<title>Bayesian Model Refinement Report</title><style>",
        "body{font-family:system-ui,-apple-system,sans-serif;max-width:820px;",
        "margin:2rem auto;padding:0 1rem;line-height:1.5;color:#222}",
        "h1,h2{color:#1f3550} h2{border-bottom:1px solid #c9d4e2;padding-bottom:2px;margin-top:1.4rem}",
        "pre{background:#f6f4ef;border:1px solid #d8d2c4;padding:8px;overflow:auto;font-size:.85em}",
        "table{border-collapse:collapse;margin:.4rem 0} td,th{border:1px solid #c9d4e2;padding:3px 8px;text-align:left}",
        ".pass{color:#2e7d32;font-weight:bold}.fail{color:#b03a2e;font-weight:bold}.small{color:#5a6b80;font-size:.9em;margin:.1rem 0}",
        "</style></head><body>",
        "<h1>Bayesian Model Refinement Report</h1>",
        f"<p><strong>Outcome:</strong> "
        f"<span class='{'pass' if passed else 'fail'}'>{esc(outcome)}</span></p>",
    ]
    diag = _failure_diagnosis(state)
    if diag:
        # reuse the deterministic markdown diagnosis: strip the heading, escape,
        # keep line breaks. Bold markers (**) are flattened — content, not styling.
        body = esc(diag.split("\n", 1)[1].replace("**", "")).replace("\n", "<br>")
        p.append(f"<h2>Why no model was accepted</h2><p class='fail'>{body}</p>")
    if narrative:
        p.append(f"<p>{esc(narrative)}</p>")
    p.append(f"<h2>Problem</h2><p>{esc(state.problem_description)}</p>")
    if state.analysis:
        issues = state.analysis.get("potential_issues", [])
        p.append("<h2>Identified weaknesses</h2><ul>" + "".join(
            f"<li><strong>{esc(str(i.get('severity', '?')))}</strong>: "
            f"{esc(str(i.get('issue', '')))}</li>" for i in issues) + "</ul>")
    if state.literature_findings:
        p.append("<h2>Literature</h2><ul>" + "".join(
            f"<li>{esc(str(f.get('paper_title', '?')))} "
            f"({esc(str(f.get('year', '?')))}) — {esc(str(f.get('technique', '')))}</li>"
            for f in state.literature_findings) + "</ul>")
    if state.proposals:
        p.append("<h2>Proposals (ranked)</h2><ol>" + "".join(
            f"<li><strong>{esc(str(pr.get('name', '?')))}</strong> "
            f"[{esc(str(pr.get('evidence_basis', 'llm_prior')))}] "
            f"({esc(str(pr.get('change_type', '')))}) — "
            f"{esc(str(pr.get('description', '')))}</li>" for pr in state.proposals) + "</ol>")
    if state.refined_model_code:
        p.append("<h2>Refined model</h2><pre><code>"
                 + esc(state.refined_model_code.strip()) + "</code></pre>")
    p.append("<h2>Validation</h2>" + _validation_html(state))
    if state.feedback_history:
        p.append("<h2>Iteration history</h2><ul>" + "".join(
            f"<li>iteration {esc(str(h.get('iteration')))}: tried "
            f"{_tried_html(h)} → "
            f"{esc('; '.join(str(r) for r in h.get('result', [])))}</li>"
            for h in state.feedback_history) + "</ul>")
    p.append("</body></html>")
    return "".join(p)


def create_report_agent(llm=None, extra_block_fn=None):
    """Final node: render the report, set terminal status complete|failed.

    ``llm=None`` produces the deterministic report only (no API needed);
    with an llm, an executive-summary narrative is prepended.
    ``extra_block_fn(state) -> str`` (optional) supplies the DESIGN RESOLUTION
    block so the report discloses the same structural facts + hint_level the
    strategist saw.
    """

    def run(state: WorkflowState) -> dict:
        extra_block = ""
        if extra_block_fn is not None:
            try:
                extra_block = extra_block_fn(state) or ""
            except Exception:  # advisory only — never break a finished run
                extra_block = ""
        narrative = ""
        if llm is not None:
            try:
                msg = (REPORT_PROMPT | llm).invoke({
                    "outcome": "success" if validation_passed(state) else "failure",
                    "proposals": ", ".join(p.get("name", "?") for p in state.proposals) or "none",
                    "validation": _render_validation(state),
                    "history": str(state.feedback_history or "first iteration"),
                })
                narrative = getattr(msg, "content", str(msg))
                if not isinstance(narrative, str):  # anthropic content blocks
                    narrative = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b) for b in narrative)
            except Exception as exc:  # report must never kill a finished run
                # copy-on-write (matches the feasibility gate): never mutate
                # the LangGraph channel list in place.
                state.error_log = state.error_log + [f"report narrative failed: {exc}"]
        report = build_report(state, narrative, extra_block=extra_block)
        status = "complete" if validation_passed(state) else "failed"
        state.report = report
        state.status = status
        return {"report": report, "status": status, "error_log": state.error_log}

    return run


# ---------------------------------------------------------------------------
# Pre-MCMC feasibility gate (MEDEA borrow) — advisory, between strategist and
# synthesizer; never approves, only prevents wasted synthesis + MCMC.
# ---------------------------------------------------------------------------

# The soft-flag matcher over a likelihood change row's lowercased `proposed`
# text: a Student-t family spelling (StudentT / student_t / student-t /
# Student's t), or "cauchy" NOT preceded by "half" / "half_" / "half-" /
# "half " — HalfCauchy is a scale PRIOR, never a likelihood (P8). The trailing
# `t` must be followed by a non-letter or the end (P4): a word boundary `\b`
# fails before "_", so "StudentT_obs" / "student_t_likelihood" went unflagged;
# "students" / "studentized" still do not match.
_HEAVY_TAIL_RE = re.compile(r"student(?:'s)?[\s_-]?t(?![a-z])|(?<!half)(?<!half[\s_-])cauchy")
_HEAVY_TAIL_DATA_CUES = ("outlier", "heavy tail", "heavy-tail", "fat tail",
                         "fat-tail", "kurtos")
# A change whose `current` reads like "no existing prior" is ADDING a parameter
# (e.g. Student-t's nu, a non-centered z) rather than modifying a missing one.
_NEW_PARAM_CUES = ("", "none", "n/a", "na", "-", "new", "(none)", "added", "add")


def _is_phantom_modification(change: dict, params: set) -> bool:
    """True iff a change claims to MODIFY a parameter the analyst never found.

    A genuine new-parameter addition (``current`` empty / "none" / "new") is NOT
    phantom — proposals legitimately introduce parameters (Student-t's nu, a
    non-centered z). Only an unknown parameter paired with a concrete existing
    prior to "replace" is a hallucinated modification worth dropping pre-MCMC.
    """
    param = change.get("parameter")
    if not params or param in params:
        return False
    current = str(change.get("current", "")).strip().lower()
    introduces_new = (current in _NEW_PARAM_CUES
                      or "new" in current or "add" in current)
    return not introduces_new


def create_feasibility_gate():
    """Build the feasibility-gate node: run(state) -> dict of changed fields.

    Cheap, deterministic screen run BEFORE synthesis/MCMC:
    * HARD fail (drop): a proposal whose ``changes`` reference a parameter the
      analyst never identified — synthesizing it would waste an MCMC budget on
      code that cannot match the model.
    * SOFT fail (flag, keep): a data-conditioned mismatch, e.g. a heavy-tailed
      likelihood (Student-t / Cauchy, read off likelihood change rows only — a
      HalfCauchy scale prior is not one) when the dataset summary shows no sign
      of outliers. Advisory only — recorded to ``error_log``, the proposal stays.

    If EVERY proposal is dropped, the gate records feedback and increments the
    iteration counter (mirroring the validate node) so ``route_after_feasibility``
    can bounce back to the strategist within the iteration budget — or fail.
    """

    def run(state: WorkflowState) -> dict:
        original = list(state.proposals or [])
        analysis = state.analysis or {}
        params = {str(p.get("name")) for p in analysis.get("parameters", [])}
        # Known referenceable names also include the observed/response variable —
        # a likelihood change legitimately targets it (y ~ Normal -> y ~ StudentT).
        lik = analysis.get("likelihood", {}) or {}
        observed = str(lik.get("observed_variable", "")).strip()
        known = (params | {observed}) - {""}
        text = f"{state.dataset_summary} {state.problem_description}".lower()
        heavy_tailed = any(cue in text for cue in _HEAVY_TAIL_DATA_CUES)

        kept: list = []
        logged: list[str] = []
        for p in original:
            name = p.get("name", "?")
            changes = p.get("changes") or []
            # Hard-drop ONLY a `prior` proposal that retunes a non-existent
            # parameter — the unambiguous hallucination. likelihood / structure /
            # hierarchy changes legitimately reference the response variable or
            # introduce new parameters (Student-t's nu, a non-centered z), so they
            # pass through to the synthesizer's own smoke test rather than risk a
            # false drop here.
            phantom = (sorted({c.get("parameter") for c in changes
                               if _is_phantom_modification(c, known)})
                       if p.get("change_type") == "prior" else [])
            if phantom:
                logged.append(
                    f"feasibility: dropped {name!r} — `prior` proposal retunes "
                    f"parameter(s) {phantom} not present in the analysis "
                    "(and not introduced as new)"
                )
                continue
            # Defect fixed here (P8): the heavy-tail check was a bare substring
            # match over EVERY change row, so a HalfCauchy(2.5) scale PRIOR was
            # flagged as a heavy-tailed LIKELIHOOD. Only likelihood rows count —
            # the proposal's change_type is "likelihood", or the row rewrites
            # the observed variable itself — and "half...cauchy" never matches.
            lik_rows = [c for c in changes
                        if p.get("change_type") == "likelihood"
                        or (observed and str(c.get("parameter", "")).strip() == observed)]
            proposed = " ".join(str(c.get("proposed", "")) for c in lik_rows).lower()
            if _HEAVY_TAIL_RE.search(proposed) and not heavy_tailed:
                logged.append(
                    f"feasibility: SOFT-FLAG {name!r} — proposes a heavy-tailed "
                    "likelihood but the data shows no sign of outliers/heavy tails"
                )
            kept.append(p)

        update: dict = {}
        if logged:
            state.error_log = state.error_log + logged
            update["error_log"] = state.error_log

        if kept:
            if len(kept) != len(original):  # some dropped; status untouched
                state.proposals = kept
                update["proposals"] = kept
            return update

        # Every proposal infeasible — or (P1) the strategist delivered an EMPTY
        # bundle because its post-validation rejected every proposal: book a
        # failed iteration so the budget bounds the loop back to the strategist
        # (route_after_feasibility). The empty-bundle entry names no technique
        # (nothing new is banned: the dropped names were banned or invalid
        # already) and carries the strategist's own summary line from error_log
        # so the retry prompt says why the last round produced nothing.
        iteration = state.iteration + 1
        if original:
            result = ["pre-MCMC feasibility gate dropped every proposal"] + logged
        else:
            summary = next((str(e) for e in reversed(state.error_log or [])
                            if str(e).startswith(EMPTY_BUNDLE_LOG_PREFIX)), None)
            result = [summary.split("strategist: ", 1)[1] if summary else
                      "the strategist delivered no proposals; nothing reached synthesis"]
        entry = {
            "iteration": iteration,
            "tried": original[0].get("name") if original else None,
            "tried_all": [p.get("name", "?") for p in original],
            "result": result,
            "waic_delta": None,
        }
        state.iteration = iteration
        state.feedback_history = state.feedback_history + [entry]
        state.proposals = []
        update.update({
            "iteration": iteration,
            "feedback_history": state.feedback_history,
            "proposals": [],
        })
        return update

    return run


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_refinement_graph(
    llm=None,
    *,
    data: dict | None = None,
    analyst_llm=None,
    scout_llm=None,
    strategist_llm=None,
    synthesizer_llm=None,
    report_llm=None,
    scout_kwargs: dict | None = None,
    synthesizer_kwargs: dict | None = None,
    validator_kwargs: dict | None = None,
    strategist_directive: str = "",
    checkpointer: Any = None,
    interrupt_before: list | None = None,
    hint_level: str = "structure+ladder",
    verdict_only: bool = False,
    smoke_on_data: bool = False,
    guard_replicate_sites: bool = False,
):
    """Compile the refinement state machine (spec §5.6).

    ``llm`` is the shared default model; per-agent overrides let tests inject
    independent FixtureLLM queues (and let production mix model tiers).
    ``data`` is the actual dataset dict for the validator — WorkflowState
    deliberately carries only a textual summary.
    ``interrupt_before=["synthesize_code"]`` enables the human-in-the-loop
    approval pattern from §5.6.

    ``hint_level`` governs the DESIGN RESOLUTION block (resolution.py) shown
    to the strategist and rendered in the report: "none" | "structure" |
    "structure+ladder" (default). Advisory only — it steers proposals, never
    the verdict — and the active level is recorded inside the block itself so
    every report discloses how much steering the strategist received.

    ``verdict_only`` (benchmark arm) REDACTS the verifier's feedback to the
    proposer: after a refused candidate the ``feedback_history`` entry carries
    only the iteration number and ``verdict_only: True`` — no failure reasons,
    no PPC / LOO / ΔeLPD numbers or confidence, no failed falsifiable
    predictions, no misspecification (diagnostic) hints, and no proposal names
    (``tried`` is None and ``tried_all`` empty, so the strategist's
    deterministic DO-NOT-PROPOSE ban bans nothing). The strategist then sees a
    single line stating that a previous candidate was refused. The pre-MCMC
    feasibility gate's own entry is proposer-side and is not redacted; the
    validator still runs in full and ``candidate_history`` keeps every verdict.

    ``smoke_on_data`` (default False): when True and ``data`` is given, the
    synthesizer's NUTS smoke test runs candidates on the real ``data`` arrays
    instead of dummy data derived from the recorded shapes (dummies turn
    integer count/index columns into negative floats, which kills every
    candidate that uses them semantically). Forwarded as ``data=`` to
    ``create_synthesizer_agent``; explicit ``synthesizer_kwargs`` win.

    ``guard_replicate_sites`` (default False): when True the synthesizer's
    prompts carry the NumPyro site guard (no unobserved replicate or discrete
    sample sites, which NUTS cannot sample) and its fix prompt gets a targeted
    hint on such errors. Forwarded as ``guard_replicate_sites=True`` to
    ``create_synthesizer_agent``; explicit ``synthesizer_kwargs`` win.
    """
    resolved: dict[str, Any] = {}

    def _llm(specific):
        if specific is not None:
            return specific
        if "shared" not in resolved:
            if llm is not None:
                resolved["shared"] = llm
            else:
                from compass.llm import get_llm  # late import: only hit when truly needed

                resolved["shared"] = get_llm()
        return resolved["shared"]

    # DESIGN RESOLUTION block: computed from the original model + data (both
    # available only at run time via state / closure), memoized PER MODEL, and
    # shared verbatim by the strategist prompt and the report so both always
    # tell the same story. Failure -> "" (advisory).
    # Defect fixed here (P3): the memo lived under one constant key per
    # compiled graph, so an app compiled once and invoked for a second model
    # (a benchmark driver, a tournament bracket) carried the FIRST model's
    # structural audit into the second model's prompts and report. The key is
    # now the model source plus the identity and shapes of the data object.
    _resolution_cache: dict[tuple, str] = {}

    def _resolution_key(state: WorkflowState) -> tuple:
        code = state.original_model.source_code if state.original_model is not None else ""
        shapes = repr(sorted(infer_data_shapes(data or {}).items(), key=lambda kv: kv[0]))
        return (code, id(data), shapes)

    def _resolution_block(state: WorkflowState) -> str:
        key = _resolution_key(state)
        if key not in _resolution_cache:
            block = ""
            try:
                code = key[0]
                if code and (data or {}) and hint_level != "none":
                    from compass.resolution import audit_resolution, format_resolution_block

                    block = format_resolution_block(
                        audit_resolution(code, data or {}), hint_level)
            except Exception:
                block = ""
            _resolution_cache[key] = block
        return _resolution_cache[key]

    graph = StateGraph(WorkflowState)
    graph.add_node("analyze", create_analyst_agent(_llm(analyst_llm)))
    graph.add_node("search_literature",
                   create_scout_agent(_llm(scout_llm), **(scout_kwargs or {})))
    graph.add_node("propose_refinements",
                   create_strategist_agent(_llm(strategist_llm),
                                           directive=strategist_directive,
                                           context_fn=_resolution_block))
    synth_kwargs = dict(synthesizer_kwargs or {})
    if smoke_on_data and data is not None:
        synth_kwargs = {"data": data, **synth_kwargs}
    if guard_replicate_sites:
        synth_kwargs = {"guard_replicate_sites": True, **synth_kwargs}
    graph.add_node("synthesize_code",
                   create_synthesizer_agent(_llm(synthesizer_llm), **synth_kwargs))

    validator_run = create_validator_agent(data=data, **(validator_kwargs or {}))

    def validate(state: WorkflowState) -> dict:
        """Validator + feedback bookkeeping (kept here because LangGraph
        conditional-edge callables cannot persist mutations — spec §5.6's
        route_after_validation mutated state inside the router instead)."""
        update = validator_run(state)
        passed = validation_passed(state)
        proposals = [p.model_dump() if hasattr(p, "model_dump") else dict(p)
                     for p in state.proposals]
        # NOTE (known, deliberately kept): candidate_history.iteration is the
        # 0-based counter at validation time, while the feedback_history entry
        # booked below for the same refusal carries the 1-based value
        # (state.iteration + 1). The experiment layer (bench/, figure2/) keys
        # its records on the 0-based value, so renumbering would change record
        # semantics; readers correlating the two channels must add 1.
        candidate = {
            "iteration": state.iteration,
            "refined_model_code": state.refined_model_code,
            "proposal_names": [p.get("name", "?") for p in proposals],
            "proposals": proposals,
            "rationale": proposals_rationale(proposals),
            "passed": passed,
            "validation": state.validation,
        }
        state.candidate_history = state.candidate_history + [candidate]
        update["candidate_history"] = state.candidate_history
        if not passed and verdict_only:
            iteration = state.iteration + 1
            entry = {
                "iteration": iteration, "tried": None, "tried_all": [], "result": [],
                "waic_delta": None, "confidence": None, "failed_predictions": [],
                "hints": [], "verdict_only": True,
            }
            state.iteration = iteration
            state.feedback_history = state.feedback_history + [entry]
            update.update({"iteration": iteration,
                           "feedback_history": state.feedback_history})
        elif not passed:
            iteration = state.iteration + 1
            v = state.validation
            # Name which proposals failed their own falsifiable prediction so the
            # next strategist round's DO-NOT-PROPOSE list is proposal-specific
            # (asta-theorizer), not just a bundle of convergence complaints.
            failed_preds = [
                f"{s.get('name')} failed its prediction "
                f"({s.get('predicted')}; observed {s.get('observed')})"
                for s in (getattr(v, "proposal_scores", None) or [])
                if s.get("passed") is False
            ]
            entry = {
                "iteration": iteration,
                # The synthesizer applies the WHOLE ranked bundle, so the ban
                # list must name every applied proposal, not just the first:
                # 'tried' stays a string (backward-compatible), 'tried_all'
                # carries the full bundle for render_feedback.
                "tried": state.proposals[0].get("name") if state.proposals else None,
                "tried_all": [p.get("name", "?") for p in state.proposals],
                "result": (list(v.failure_reasons) if v else
                           ["validator produced no result"]) + failed_preds,
                "waic_delta": v.waic_delta if v else None,
                "confidence": getattr(v, "confidence", "") if v else "",
                "failed_predictions": failed_preds,
                # localization borrow: what the diagnostics suggest trying next
                # (guidance, kept separate from the DO-NOT-PROPOSE ban list).
                "hints": list(getattr(v, "misspecification_hints", None) or []),
            }
            state.iteration = iteration
            state.feedback_history = state.feedback_history + [entry]
            update.update({
                "iteration": iteration,
                "feedback_history": state.feedback_history,
            })
        update["status"] = "validating"
        state.status = "validating"
        return update

    graph.add_node("validate", validate)
    graph.add_node("feasibility_gate", create_feasibility_gate())
    graph.add_node("prepare_report",
                   create_report_agent(report_llm, extra_block_fn=_resolution_block))

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "search_literature")
    graph.add_edge("search_literature", "propose_refinements")
    # Pre-MCMC feasibility gate sits between the strategist and the synthesizer.
    graph.add_edge("propose_refinements", "feasibility_gate")
    graph.add_conditional_edges("feasibility_gate", route_after_feasibility, {
        "synthesize": "synthesize_code",
        "retry_with_feedback": "propose_refinements",  # all dropped, budget left
        "fail": "prepare_report",                      # all dropped, budget gone
    })
    graph.add_conditional_edges("synthesize_code", route_after_synthesis, {
        "validate": "validate",
        "retry_synthesis": "synthesize_code",  # kept for spec parity; the
        "fail": "prepare_report",              # internal fix loop handles retries
    })
    graph.add_conditional_edges("validate", route_after_validation, {
        "success": "prepare_report",
        "retry_with_feedback": "propose_refinements",
        "search_more": "search_literature",
        "fail": "prepare_report",
    })
    graph.add_edge("prepare_report", END)

    return graph.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver(),
        interrupt_before=interrupt_before,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def infer_data_shapes(data: dict) -> dict:
    """ModelSpec.data_shapes from a dataset dict: arrays AND plain lists/tuples
    -> shape list (generate_dummy_data interprets lists as dimension specs, so
    passing raw list values through would turn data VALUES into shapes);
    true scalars pass through."""
    shapes: dict[str, Any] = {}
    for key, value in (data or {}).items():
        if isinstance(value, (np.ndarray, list, tuple)):
            shapes[key] = list(np.asarray(value).shape)
        else:
            shapes[key] = value
    return shapes


def summarize_dataset(data: dict) -> str:
    """Spec §5.1: pre-compute the dataset summary (N, per-variable stats,
    missingness, correlations) with pandas, so callers may pass raw data only.

    The ``N=<count>`` prefix is load-bearing: the scout's applicability filter
    regex-parses it for the data-size question. N is the largest LEADING
    dimension across the array-likes — an (N, P) design matrix counts as N
    observations, not N*P — and variables whose shape does not line up with N
    (or that are multi-dimensional) get their own shape-summary line rather
    than being dropped.
    """
    import pandas as pd

    arrays = {k: np.asarray(v) for k, v in (data or {}).items()
              if isinstance(v, (np.ndarray, list, tuple))}
    arrays = {k: v for k, v in arrays.items() if v.ndim >= 1}
    if not arrays:
        return ""
    n = max(v.shape[0] for v in arrays.values())
    df = pd.DataFrame({k: v for k, v in arrays.items()
                       if v.ndim == 1 and len(v) == n})
    lines = [f"N={n} observations."]
    numeric = df.select_dtypes("number")           # describe() drops object cols
    desc = numeric.describe() if numeric.shape[1] else None  # and bools -> categorical
    for col in df.columns:
        if col in numeric.columns:
            lines.append(
                f"{col}: mean={desc[col]['mean']:.3g}, sd={desc[col]['std']:.3g}, "
                f"missing={int(df[col].isna().sum())}"
            )
        else:
            lines.append(f"{col}: non-numeric, {df[col].nunique()} unique values")
    # Multi-dimensional / off-length variables: summarize by shape + flat stats.
    for key, v in arrays.items():
        if key in df.columns:
            continue
        if np.issubdtype(v.dtype, np.number):
            lines.append(f"{key}: shape {tuple(v.shape)}, "
                         f"mean={np.nanmean(v):.3g}, sd={np.nanstd(v):.3g}")
        else:
            lines.append(f"{key}: shape {tuple(v.shape)}, non-numeric")
    num = numeric
    if num.shape[1] >= 2:
        corr = num.corr().round(2)
        pairs = [f"corr({a},{b})={corr.loc[a, b]}"
                 for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]]
        lines.append("; ".join(pairs))
    # Anything not array-like above (incl. 0-d arrays) reports as a constant —
    # keeps the tuple/list/ndarray split consistent with infer_data_shapes.
    scalars = {k: v for k, v in (data or {}).items() if k not in arrays}
    if scalars:
        lines.append("Constants: " + ", ".join(f"{k}={v}" for k, v in scalars.items()))
    return "\n".join(lines)


def run_refinement(
    problem_description: str,
    dataset_summary: str = "",
    model_code: str = "",
    framework: str = "numpyro",
    data: dict | None = None,
    max_iterations: int = 3,
    thread_id: str | None = None,
    recursion_limit: int = 50,
    **graph_kwargs,
) -> dict:
    """One-call refinement: build state + graph, run to completion, return the
    final state dict (LangGraph returns channel values, not the dataclass).

    If ``dataset_summary`` is empty it is computed from ``data`` with pandas
    (``summarize_dataset``, spec §5.1).

    NOTE on retrieval defaults: the scout searches the bundled 15-paper test
    corpus with live channels OFF. Production callers should pass
    ``scout_kwargs={"corpus": <curated papers>, "live": True}``.
    """
    app = build_refinement_graph(data=data, **graph_kwargs)
    initial = WorkflowState(
        problem_description=problem_description,
        dataset_summary=dataset_summary or summarize_dataset(data or {}),
        original_model=ModelSpec(
            framework=framework,
            source_code=model_code,
            data_shapes=infer_data_shapes(data or {}),
        ),
        max_iterations=max_iterations,
        status="initialized",
    )
    config = {
        "configurable": {"thread_id": thread_id or uuid.uuid4().hex},
        # Worst-case loop is search_more: 5 nodes/iteration with the feasibility
        # gate (search -> propose -> gate -> synthesize -> validate), plus
        # analyze and prepare_report. Guarantee every run can reach
        # prepare_report instead of dying in GraphRecursionError.
        "recursion_limit": max(recursion_limit, 5 * max_iterations + 10),
    }
    return app.invoke(initial, config)


__all__ = [
    "build_refinement_graph",
    "run_refinement",
    "route_after_synthesis",
    "route_after_validation",
    "route_after_feasibility",
    "validation_passed",
    "create_feasibility_gate",
    "create_report_agent",
    "build_report",
    "build_html_report",
    "build_grounding_block",
    "infer_data_shapes",
    "summarize_dataset",
]
