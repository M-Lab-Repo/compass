"""Refinement Strategist agent — the judgment core (spec §5.3).

Merges the analyst's structural read of the model with the scout's literature
findings (plus any validator failure feedback) into 3-5 concrete, cited,
ranked `Proposal`s. Not every technique applies: the strategist must reject
topically-related-but-inapplicable findings (the horseshoe trap) and must
never re-propose a technique that already failed in a previous iteration.

Contract (system-wide conventions):
- ``create_strategist_agent(llm) -> run``; ``run(state) -> dict`` of changed
  fields, while also setting them on the passed ``WorkflowState``.
- ``state.proposals`` = list of ``Proposal.model_dump()`` dicts sorted by
  priority ascending; ``state.status = "proposed"`` on success.
- Post-validation: trim to 5 proposals max, drop uncited / invalid-change-type
  proposals and any proposal whose normalized name exactly matches a failed
  technique from ``feedback_history`` (the DO-NOT-PROPOSE ban, enforced
  deterministically, not just via the prompt; logging every trim/drop to
  ``state.error_log``). If nothing survives the node does NOT raise: it logs
  one summary line (``EMPTY_BUNDLE_LOG_PREFIX``) and returns an EMPTY bundle
  (``proposals=[]``, status untouched); the graph's feasibility gate then
  books the failed iteration and ``route_after_feasibility`` retries the
  strategist within ``max_iterations`` or routes to ``prepare_report``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from compass.llm import invoke_structured
from compass.schemas import (
    CHANGE_TYPES,
    EVIDENCE_BASES,
    FalsifiablePrediction,
    StrategyOutput,
)
from compass.state import WorkflowState

MAX_PROPOSALS = 5  # spec §5.3: limit to 3-5 proposals per iteration

# Deterministic fallback prediction metric per change_type, used only when the
# LLM omits a proposal's `prediction` — so every surviving proposal is
# validator-scorable even on a thin fixture.
_DEFAULT_PREDICTION_METRIC = {
    "hierarchy": "divergences",   # reparameterizations target funnel divergences
    "prior": "loo_delta",         # regularizing priors target predictive accuracy
    "likelihood": "loo_delta",
    "structure": "loo_delta",
}

_SYSTEM = """You are the Refinement Strategist in a multi-agent Bayesian model \
refinement system. Upstream agents give you (1) a structural analysis of a \
probabilistic model and (2) vetted literature findings. Your job is judgment: \
merge them into concrete, ranked, implementable refinement proposals.

Hard requirements — every one of them is enforced downstream:
1. CONCRETE. Every proposal must name exact distributions WITH hyperparameters \
and exact structure changes. Write `beta ~ Normal(0, 10)`, never "use a tighter \
prior". Every entry in `changes` must be an object with keys "parameter", \
"current", "proposed", "reason", where "parameter" is a real parameter name \
from the analysis.
2. CITED. `supporting_papers` must contain at least one paper title copied \
verbatim from the literature findings below. Never cite a paper that is not in \
the findings. Proposals without a citation are discarded.
3. RANKED. Set `priority` by expected impact divided by implementation risk: \
priority 1 is the single highest-leverage, lowest-risk change to do first. \
List `implementation_order` to match.
4. INTERACTIONS. Consider how proposals interact: changing a prior AND the \
hierarchy at the same time can create identifiability problems. Use \
`combine_with` to name proposals that should be applied together (or that this \
one depends on), and flag conflicts in `risk_assessment`.
5. JUDGMENT. Not every finding applies. Reject techniques that do not fit this \
specific model and data (for example, a sparsity-inducing prior for a \
single-predictor regression), and explain each rejection in `reasoning`.
6. LIMIT. Produce at least 3 and at most 5 proposals.
7. `change_type` must be exactly one of: prior, hierarchy, likelihood, \
structure.
8. FAILED-TECHNIQUE BAN. If the human message contains a "DO NOT PROPOSE" \
list, those techniques already failed validation in earlier iterations. Do not \
re-propose any of them in their plain form. Your `reasoning` MUST explicitly \
explain how each new proposal avoids the listed failure modes. A materially \
modified variant (e.g. a regularized version of a plainly-failed prior) is \
acceptable only if you justify why it escapes the recorded failure.
9. PREDICT (falsifiable). Every proposal MUST carry a `prediction`: the single \
validator-measured diagnostic it most directly targets (`metric` is one of \
divergences, rhat_max, ess_min, ebfmi_min, loo_delta, waic_delta, ppc), the \
`direction` of the expected effect, a one-line `rationale` (what SUPPORT looks \
like) and a `failure_mode` (the observation that would FALSIFY it). The \
validator scores each proposal against its own prediction using real MCMC, so \
make the prediction concrete and checkable — not a restatement of the change.
10. PROVENANCE. Set `evidence_basis` for each proposal: "literature" if a real \
finding above drives it, "model_structure" if it is entailed by the model's \
geometry or identifiability (the citation is incidental — the reason is the \
structure), or "llm_prior" if it is your own judgment without direct support."""

_HUMAN = """Devise refinement proposals for this Bayesian model.

## Problem
{problem_description}

## Dataset
{dataset_summary}

## Model analysis (from the Model Analyst)
{analysis_block}

## Literature findings (from the Literature Scout)
{findings_block}

## Design resolution (structural facts about this model + data; may be empty)
{design_resolution_block}

## Diversification directive (optional)
{diversification_directive}

## Previously failed techniques
{do_not_propose_block}

Return a StrategyOutput: `reasoning` (including rejected techniques and, if a
DO NOT PROPOSE list exists above, how the new proposals avoid those failure
modes), 3-5 concrete cited `proposals` ranked by priority (1 = do first) — each
carrying a falsifiable `prediction` and an `evidence_basis` — and
`implementation_order`."""

STRATEGIST_PROMPT = ChatPromptTemplate.from_messages(
    [("system", _SYSTEM), ("human", _HUMAN)]
)


# ---------------------------------------------------------------------------
# Rendering helpers (unit-testable, no LLM involved)
# ---------------------------------------------------------------------------
VERDICT_ONLY_NOTICE = ("- A previous candidate was refused by the verifier; "
                       "no diagnostics are disclosed.")

# First words of the error_log line the strategist writes when post-validation
# rejects its whole bundle. graph.create_feasibility_gate looks for it to carry
# the diagnosis into the failed-iteration feedback entry (the retry prompt).
EMPTY_BUNDLE_LOG_PREFIX = "strategist: post-validation rejected every proposal"


def _entry_names(entry: dict) -> list[str]:
    """Every technique name a feedback entry records, as stripped strings,
    deduplicated (case-insensitively, first spelling wins) in order: ``tried``
    (the single legacy name) first, then ``tried_all`` (the whole applied
    bundle). Both keys may be absent or None — graph.py writes ``tried``
    explicitly as None when no proposal name is available — and ``tried_all``
    may be a single string. This is the ONE place that decides which names a
    feedback entry names, so the rendered DO-NOT-PROPOSE bullet and the
    deterministic ban (``_banned_technique_names``) can never disagree."""
    raw: list = [entry.get("tried")]
    tried_all = entry.get("tried_all")
    if isinstance(tried_all, str):
        raw.append(tried_all)
    elif tried_all:
        raw.extend(tried_all)
    names: list[str] = []
    seen: set[str] = set()
    for n in raw:
        s = str(n).strip() if n is not None else ""
        if s and s.lower() not in seen:
            seen.add(s.lower())
            names.append(s)
    return names


def render_feedback(feedback_history: list) -> str:
    """Render feedback_history as a DO-NOT-PROPOSE bullet list.

    One bullet per failed iteration:
    ``- DO NOT PROPOSE: <name>, <name>, ... — failed because <result; ...>``
    naming EVERY technique the entry bans (``tried`` first, then the rest of
    ``tried_all``), followed, when the validator localized the
    misspecification, by a positive
    ``- DIAGNOSTIC HINT: <what the data suggest trying>`` line (kept separate
    from the ban so the strategist is steered toward, not away from, the fix).
    An entry that names NO technique (the feasibility gate books one, with
    ``tried`` None, for a round whose bundle was emptied before synthesis)
    bans nothing and renders as ``- Previous round: <result; ...>`` — what
    happened, not a ban on a placeholder. A redacted entry (``verdict_only``
    True, graph.build_refinement_graph's verdict-only arm) renders as the
    single ``VERDICT_ONLY_NOTICE`` line: no names, no numbers, no hints.
    Returns "" for an empty history.
    """
    lines = []
    for entry in feedback_history or []:
        entry = _as_dict(entry)
        if entry.get("verdict_only"):
            lines.append(VERDICT_ONLY_NOTICE)
            continue
        # Defect fixed here (P2): the bullet used to show entry["tried"] only,
        # while _banned_technique_names bans every name in entry["tried_all"]
        # — the proposer was told about ONE banned name and then had its
        # re-served bundle members dropped for names it was never shown. Both
        # now read the same list (_entry_names).
        names = _entry_names(entry)
        results = entry.get("result") or []
        if isinstance(results, str):
            results = [results]
        reasons = "; ".join(str(r) for r in results) or "unspecified failure"
        if names:
            lines.append(f"- DO NOT PROPOSE: {', '.join(names)} — failed because {reasons}")
        else:
            # Defect fixed here (P3): an entry without any name — the feasibility
            # gate's record of a round whose bundle was emptied before synthesis
            # — bans nothing, yet it rendered as "DO NOT PROPOSE: unknown
            # technique". It now reports what happened last round.
            lines.append(f"- Previous round: {reasons}")
        hints = entry.get("hints") or []
        if isinstance(hints, str):
            hints = [hints]
        for hint in hints:
            lines.append(f"- DIAGNOSTIC HINT (what the data suggest): {hint}")
    return "\n".join(lines)


def proposals_rationale(proposals: list) -> str:
    """The proposer's justification as one string: per proposal, the
    descriptive fields of ``schemas.Proposal`` that carry argument or evidence
    (description, expected_improvement, evidence_basis, supporting_papers,
    prediction.rationale), joined as ``<name>: <text>`` lines. This is what the
    LLM judges read as the "proposer's justification"."""
    lines = []
    for p in proposals or []:
        p = p.model_dump() if hasattr(p, "model_dump") else dict(p)
        parts = [str(p.get("description") or "").strip()]
        if p.get("expected_improvement"):
            parts.append(f"Expected improvement: {p['expected_improvement']}")
        basis = p.get("evidence_basis")
        papers = [str(x) for x in (p.get("supporting_papers") or [])]
        if basis or papers:
            parts.append("Evidence basis: " + (str(basis) if basis else "unspecified")
                         + (f" (cites: {'; '.join(papers)})" if papers else ""))
        pred = p.get("prediction") or {}
        if isinstance(pred, dict) and pred.get("rationale"):
            parts.append(f"Prediction: {pred['rationale']}")
        lines.append(f"{p.get('name', '?')}: " + " ".join(x for x in parts if x))
    return "\n".join(lines)


def _banned_technique_names(feedback_history: list) -> set[str]:
    """Normalized (stripped, lowercased) names of techniques already recorded
    as failed — the deterministic side of the DO-NOT-PROPOSE ban. Reads each
    entry's ``tried`` (single name) and, when present, ``tried_all`` (every
    name in the failed bundle); both may be absent or None."""
    banned: set[str] = set()
    for entry in feedback_history or []:
        banned.update(n.lower() for n in _entry_names(_as_dict(entry)))
    return banned


def _as_dict(obj: Any) -> dict:
    """Tolerate dataclass / pydantic inputs on the direct-call path."""
    if isinstance(obj, dict):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    return dict(obj)


def _render_analysis(analysis: Any) -> str:
    """Human-readable block: parameters / likelihood / hierarchy / issues."""
    a = _as_dict(analysis) if analysis else {}
    lines = [f"Framework: {a.get('framework', 'unknown')}", "", "Parameters:"]
    for p in a.get("parameters", []):
        lines.append(
            f"- {p.get('name')}: prior {p.get('prior_dist')} "
            f"(hyperparams: {p.get('hyperparams')}; role: {p.get('role')})"
        )
    lik = a.get("likelihood", {}) or {}
    lines += [
        "",
        f"Likelihood: family={lik.get('family')}, link={lik.get('link_function')}, "
        f"observed variable={lik.get('observed_variable')}",
        "",
        "Hierarchical structure:",
    ]
    levels = a.get("hierarchical_levels", [])
    if levels:
        for lv in levels:
            lines.append(
                f"- grouped by {lv.get('grouping_var')} ({lv.get('n_groups')} groups); "
                f"parameters affected: {lv.get('params_affected')}"
            )
    else:
        lines.append("- (no hierarchy detected)")
    lines += ["", "Potential issues flagged by the analyst:"]
    issues = a.get("potential_issues", [])
    if issues:
        for issue in issues:
            lines.append(
                f"- [{issue.get('severity')}] {issue.get('issue')} "
                f"(suggested direction: {issue.get('suggestion')})"
            )
    else:
        lines.append("- (none flagged)")
    return "\n".join(lines)


def _render_findings(findings: list) -> str:
    """Human-readable block: title, technique, applicability, summary, snippet presence."""
    if not findings:
        return "(no literature findings available)"
    blocks = []
    for i, raw in enumerate(findings, 1):
        f = _as_dict(raw)
        has_snippet = "yes" if f.get("code_snippet") else "no"
        blocks.append(
            f'{i}. "{f.get("paper_title")}" (arXiv {f.get("arxiv_id")}, {f.get("year")})\n'
            f"   technique: {f.get('technique')}\n"
            f"   applicable to: {f.get('applicable_to')}\n"
            f"   relevance score: {f.get('relevance_score')}\n"
            f"   summary: {f.get('summary')}\n"
            f"   code snippet available: {has_snippet}"
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------
def create_strategist_agent(llm, directive: str = "", context_fn=None):
    """Build the strategist node: run(state) -> dict of changed fields.

    ``directive`` is an optional diversification instruction appended to the
    prompt — used by the refinement tournament (``tournament.run_tournament``)
    to steer distinct candidates (e.g. "favor the most aggressive structural
    change" vs "favor the most conservative prior-only change"). It does NOT
    change which candidate wins — the MCMC validator decides that.

    ``context_fn(state) -> str`` (optional) supplies the DESIGN RESOLUTION
    block (resolution.format_resolution_block, wired by graph.build_graph):
    the STRUCTURE hint channel — iteration-1 facts about the design plus, at
    hint_level "structure+ladder", the symptom-level failure-mode ladder for
    the detected latent class. Advisory text only; it never gates.
    """
    chain = STRATEGIST_PROMPT | llm.with_structured_output(StrategyOutput)

    def run(state: WorkflowState) -> dict:
        design_resolution = ""
        if context_fn is not None:
            try:
                design_resolution = context_fn(state) or ""
            except Exception:  # advisory only — never break the run
                design_resolution = ""
        if state.feedback_history:
            do_not_propose = (
                render_feedback(state.feedback_history)
                + "\n\nYour `reasoning` MUST explain how each new proposal "
                "avoids these recorded failure modes."
            )
        else:
            do_not_propose = "(none — first iteration, no failed refinement attempts yet)"

        result: StrategyOutput = invoke_structured(
            chain,
            {
                "problem_description": state.problem_description,
                "dataset_summary": state.dataset_summary,
                "analysis_block": _render_analysis(state.analysis),
                "findings_block": _render_findings(state.literature_findings),
                "do_not_propose_block": do_not_propose,
                "design_resolution_block": (
                    design_resolution or "(none — no structural audit available)"
                ),
                "diversification_directive": (
                    directive or "(none — propose the single best refinement set)"
                ),
            },
            nudge_field="diversification_directive",
            schema_hint="`proposals` as a JSON array of proposal objects and "
                        "`implementation_order` as a list of their names",
            what="StrategyOutput",
        )

        # --- post-validation pass (deterministic, no LLM) -------------------
        logged: list[str] = []
        kept = []
        banned = _banned_technique_names(state.feedback_history)
        # The citation rule ("every proposal must cite >= 1 finding") is only
        # satisfiable when the scout delivered findings. With none (real data
        # where the applicability gate passed nothing), an uncited proposal is
        # the honest outcome, not a defect: keep it as LLM-prior grounding.
        require_citation = bool(state.literature_findings)
        if not require_citation and result.proposals:
            logged.append(
                "strategist: no literature findings reached the strategist; the "
                "citation rule is waived and proposals are LLM-prior grounded"
            )
        for p in result.proposals:
            # Failed-technique ban, enforced deterministically (the prompt's
            # rule 8 alone is advisory): exact case-insensitive match on the
            # normalized name only — a materially modified variant (different
            # name) is allowed through, per the prompt contract. No fuzzy
            # matching.
            if (p.name or "").strip().lower() in banned:
                logged.append(
                    f"strategist: dropped proposal {p.name!r} — re-proposes a "
                    "technique recorded as failed in feedback_history "
                    "(DO NOT PROPOSE ban)"
                )
                continue
            if p.change_type not in CHANGE_TYPES:
                logged.append(
                    f"strategist: dropped proposal {p.name!r} — change_type "
                    f"{p.change_type!r} is not one of {CHANGE_TYPES}"
                )
                continue
            if require_citation and not p.supporting_papers:
                logged.append(
                    f"strategist: dropped proposal {p.name!r} — empty "
                    "supporting_papers (every proposal must cite >= 1 finding)"
                )
                continue
            kept.append(p)

        kept.sort(key=lambda p: p.priority)  # (a) priority ascending, stable

        if len(kept) > MAX_PROPOSALS:  # (b) enforce the 3-5 limit
            overflow = [p.name for p in kept[MAX_PROPOSALS:]]
            logged.append(
                f"strategist: trimmed proposals from {len(kept)} to "
                f"{MAX_PROPOSALS} (spec limit); dropped lowest-priority: {overflow}"
            )
            kept = kept[:MAX_PROPOSALS]

        if not kept:  # (d) nothing usable survived
            # Defect fixed here (P1): this used to `raise RuntimeError`, which
            # escaped app.invoke / run_refinement — no report, candidate_history
            # lost, and the benchmark driver booked a legitimately refused
            # candidate as a pipeline error (typically the proposer re-serving
            # the names banned after a refusal). No caller relies on the raise
            # (tournament.py and bench/run.py catch Exception as a dead
            # candidate — exactly the mis-booking), so the node now returns an
            # EMPTY bundle with the diagnostics in error_log: the feasibility
            # gate books the failed iteration (graph.create_feasibility_gate)
            # and route_after_feasibility retries the strategist within
            # max_iterations or routes to prepare_report. status is left
            # untouched — nothing was proposed.
            dropped = "; ".join(
                line.split("strategist: dropped proposal ", 1)[1]
                for line in logged if line.startswith("strategist: dropped proposal ")
            )
            logged.append(
                f"{EMPTY_BUNDLE_LOG_PREFIX}: the LLM returned "
                f"{len(result.proposals)} proposal(s), all with an invalid "
                "change_type, no supporting papers, or a banned already-failed "
                "technique name; nothing to synthesize this round"
                + (f" — {dropped}" if dropped else "")
            )
            state.error_log.extend(logged)
            state.proposals = []
            return {"proposals": [], "error_log": state.error_log}

        # --- provenance + falsifiable-prediction normalization (no LLM) -------
        # Work on COPIES: FixtureLLM replays the same StrategyOutput object, so
        # mutating proposals in place would corrupt a shared/global fixture
        # (e.g. fixtures.frozen_outputs.FROZEN_PROPOSALS) across calls.
        kept = [p.model_copy(deep=True) for p in kept]
        # evidence_basis is cross-checked against the actual findings, and every
        # surviving proposal is guaranteed a validator-scorable prediction.
        finding_titles = {
            (_as_dict(f).get("paper_title") or "")
            for f in (state.literature_findings or [])
        }
        for p in kept:
            basis = (getattr(p, "evidence_basis", "") or "llm_prior")
            if basis not in EVIDENCE_BASES:
                logged.append(
                    f"strategist: proposal {p.name!r} had unknown evidence_basis "
                    f"{basis!r}; reset to 'llm_prior'"
                )
                basis = "llm_prior"
            if basis == "literature" and not (set(p.supporting_papers) & finding_titles):
                logged.append(
                    f"strategist: proposal {p.name!r} declared evidence_basis "
                    "'literature' but cites no finding from the scout; downgraded "
                    "to 'model_structure'"
                )
                basis = "model_structure"
            p.evidence_basis = basis
            if p.prediction is None:
                metric = _DEFAULT_PREDICTION_METRIC.get(p.change_type, "loo_delta")
                p.prediction = FalsifiablePrediction(
                    metric=metric,
                    direction="improve",
                    rationale="(default prediction inferred from change_type)",
                    failure_mode=f"{metric} does not improve after applying this proposal",
                )
                logged.append(
                    f"strategist: proposal {p.name!r} carried no prediction; "
                    f"defaulted its metric to {metric!r}"
                )

        state.proposals = [p.model_dump() for p in kept]
        state.status = "proposed"
        update: dict = {"proposals": state.proposals, "status": state.status}
        if logged:
            state.error_log.extend(logged)
            update["error_log"] = state.error_log
        return update

    return run
