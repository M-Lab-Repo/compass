"""Multi-model refinement tournament (MEDEA borrow).

Diversify candidate refinements, then let the MCMC VALIDATOR — not a vote —
pick the winner. Each candidate runs the full refinement pipeline with a
distinct diversification directive (and optionally a distinct LLM / model tier);
the winner is chosen ONLY from the validator's ground-truth diagnostics:

    1. prefer candidates that strictly pass (converged AND no failure reasons),
    2. then the most negative loo_delta (deviance scale: more negative = better
       out-of-sample prediction),
    3. tie-broken by posterior predictive check, then ESS headroom.

The LLM only diversifies the *proposing* step (peer-diverse candidates catch
e.g. "horseshoe diverges on 8 groups" before MCMC pays for it). It never
adjudicates — that stays the validator's job, preserving the system invariant
that the MCMC validator is the sole arbiter (no LLM judge).

This is a thin OUTER driver over ``graph.run_refinement`` — the core state
machine is untouched, so a tournament is an optional layer a caller opts into.
"""

from __future__ import annotations

import math
from typing import Any

from compass.graph import run_refinement

# Default 2-candidate bracket: one aggressive, one conservative proposer. Each
# directive is appended to the strategist prompt via ``strategist_directive``.
DEFAULT_CANDIDATES = [
    {
        "label": "aggressive",
        "strategist_directive": (
            "Favor the most aggressive, highest-impact refinement set — prefer "
            "structural/hierarchy changes that could most improve the fit, even "
            "at higher implementation risk."
        ),
    },
    {
        "label": "conservative",
        "strategist_directive": (
            "Favor the most conservative, lowest-risk refinement set — prefer "
            "minimal prior-only adjustments that are very unlikely to harm the "
            "model."
        ),
    },
]


def _candidate_key(cand: dict) -> tuple:
    """Sort key (smaller is better) ranking a candidate purely by validator
    ground truth. Tiers: 0 = strictly passes, 1 = converged, 2 = otherwise."""
    v = cand.get("validation")
    if v is None:
        return (3, 0.0, 0, 0.0)
    passed = bool(v.converged and not v.failure_reasons)
    tier = 0 if passed else (1 if v.converged else 2)
    loo = v.loo_delta if v.loo_delta is not None else 0.0   # more negative = better
    ppc = 1 if v.posterior_predictive_check else 0           # prefer pass
    # NaN is truthy and poisons the tuple comparison (order-dependent min):
    # guard so a NaN ess_min never beats a real one.
    ess = v.ess_min if (v.ess_min and math.isfinite(v.ess_min)) else 0.0  # prefer higher
    return (tier, loo, -ppc, -ess)


def select_winner(candidates: list[dict]) -> dict | None:
    """Pick the best candidate by the MCMC validator's diagnostics alone.

    ``candidates`` is a list of ``{"label", "state", "validation"}`` dicts.
    Returns the winning candidate dict, or None if the list is empty. NEVER
    uses an LLM judgement or a vote — the validator margin decides.
    """
    scored = [c for c in candidates if c is not None]
    if not scored:
        return None
    return min(scored, key=_candidate_key)


def _confidence(v) -> str:
    return getattr(v, "confidence", "") if v is not None else ""


def compute_ties(candidates: list[dict], winner: dict | None) -> list[str]:
    """Labels of candidates statistically TIED with the winner (v4, C4).

    Ties-as-ties (the identifiability explainer's mandate): near-equivalent
    rivals must be reported as an equivalence class, because the data cannot
    arbitrate within it — selection inside the tie set is margin ordering, a
    CONVENTION. A candidate ties the winner iff BOTH strictly pass (converged,
    zero failure reasons — only certified refinements can tie) and their
    S-gate margins agree within noise: |elpd_diff_c − elpd_diff_w| ≤
    2·max(SE_c, SE_w). Pairwise dse between candidates is unavailable (each
    was compared to the ORIGINAL, not to each other), so each side's own SE
    is the documented fallback. Candidates without elpd fields never tie.

    NEVER affects selection: ``select_winner`` is byte-identical (I2).
    """
    if winner is None:
        return []
    wv = winner.get("validation")
    if wv is None or not (wv.converged and not wv.failure_reasons):
        return []
    w_diff = getattr(wv, "elpd_diff", None)
    w_se = getattr(wv, "elpd_diff_se", None)
    if w_diff is None or w_se is None or not math.isfinite(w_diff) \
            or not math.isfinite(w_se):
        return []
    ties = []
    for c in candidates:
        if c is None or c is winner or c.get("label") == winner.get("label"):
            continue
        v = c.get("validation")
        if v is None or not (v.converged and not v.failure_reasons):
            continue
        d = getattr(v, "elpd_diff", None)
        se = getattr(v, "elpd_diff_se", None)
        if d is None or se is None or not math.isfinite(d) or not math.isfinite(se):
            continue
        if abs(d - w_diff) <= 2.0 * max(se, w_se):
            ties.append(c.get("label"))
    return ties


TIE_NOTE = ("selection within the tie set follows margin ordering — a "
            "convention, not a data verdict; the tied candidates are "
            "predictively equivalent at this resolution")


def build_tournament_report(candidates: list[dict], winner: dict | None,
                            ties: list[str] | None = None) -> str:
    """Deterministic markdown scoreboard (no LLM)."""
    lines = ["# Refinement Tournament", ""]
    win_label = winner["label"] if winner else None
    ties = ties if ties is not None else compute_ties(candidates, winner)
    lines.append(f"**Winner:** {win_label or '(none)'} — selected by the MCMC "
                 "validator (most negative ΔLOO among converged candidates), "
                 "not by vote.\n")
    if ties:
        lines.append(f"**Statistically tied (within 2·SE):** {', '.join(ties)} — "
                     f"{TIE_NOTE}.\n")
    lines.append("| candidate | converged | ΔLOO | ΔWAIC | PPC | confidence | failures |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in candidates:
        v = c.get("validation")
        if v is None:
            lines.append(f"| {c.get('label')} | — | — | — | — | — | "
                         f"{c.get('error', 'no validation')} |")
            continue
        mark = ("**←win**" if c.get("label") == win_label
                else "←tied" if c.get("label") in ties else "")
        lines.append(
            f"| {c.get('label')} {mark} | {v.converged} | {v.loo_delta} | "
            f"{v.waic_delta} | {'pass' if v.posterior_predictive_check else 'FAIL'} | "
            f"{_confidence(v) or '—'} | {'; '.join(v.failure_reasons) or 'none'} |"
        )
    return "\n".join(lines) + "\n"


def run_tournament(
    problem_description: str,
    dataset_summary: str = "",
    model_code: str = "",
    framework: str = "numpyro",
    data: dict | None = None,
    max_iterations: int = 3,
    candidates: list[dict] | None = None,
    base_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run N diversified refinement candidates and let the validator pick.

    ``candidates`` is a list of per-candidate override dicts; each may carry a
    ``label``, a ``strategist_directive``, and any ``build_refinement_graph``
    kwargs (e.g. a distinct ``strategist_llm`` / model tier, or a stub
    ``validator_kwargs`` in tests). ``base_kwargs`` are shared across all
    candidates. Defaults to a 2-candidate aggressive/conservative bracket.

    Returns ``{"winner", "winner_state", "candidates", "report"}``.
    """
    base = dict(base_kwargs or {})
    cands = candidates if candidates is not None else DEFAULT_CANDIDATES

    results: list[dict] = []
    for i, cand in enumerate(cands):
        label = cand.get("label", f"candidate_{i + 1}")
        kwargs = dict(base)
        kwargs.update({k: v for k, v in cand.items() if k != "label"})
        try:
            final = run_refinement(
                problem_description, dataset_summary, model_code,
                framework=framework, data=data, max_iterations=max_iterations,
                **kwargs,
            )
            results.append({"label": label, "state": final,
                            "validation": final.get("validation")})
        except Exception as exc:  # a dead candidate must not kill the bracket
            results.append({"label": label, "state": None, "validation": None,
                            "error": str(exc)})

    winner = select_winner(results)
    ties = compute_ties(results, winner)
    return {
        "winner": winner["label"] if winner else None,
        "winner_state": winner["state"] if winner else None,
        "candidates": results,
        "ties": ties,
        "tie_note": TIE_NOTE if ties else "",
        "report": build_tournament_report(results, winner, ties),
    }


__all__ = ["run_tournament", "select_winner", "compute_ties",
           "build_tournament_report", "DEFAULT_CANDIDATES", "TIE_NOTE"]
