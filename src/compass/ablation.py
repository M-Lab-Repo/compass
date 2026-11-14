"""Per-proposal ablation (v3) — both-directions causal attribution.

The v2 validator scores each proposal's prediction against the AGGREGATE refined
model (all proposals applied together), so it cannot isolate which proposal
actually drove a win. Ablation closes that gap by fitting the proposal subsets:

  * full            — all proposals applied               (1 fit)
  * one-at-a-time   — each proposal ALONE vs the original (N fits)  [standalone]
  * leave-one-out   — the full set MINUS each proposal     (N fits)  [contribution]
  * original        — the baseline                          (1 fit, cached)

= 2N + 2 fits (the trace cache collapses any duplicate subset, e.g. N=1 where
"only-i" == "full"). Each subset is SYNTHESIZED into code then VALIDATED with
real MCMC; both are injected so the attribution math is unit-testable without an
LLM or a sampler. The MCMC validator remains the sole arbiter — ablation only
re-runs it on subsets; it never introduces an LLM judgement.

Attribution (deviance scale, ``loo_delta = subset − original``, negative = better):
  * standalone[i]    = loo_delta of {i} alone           (does i help on its own?)
  * contribution[i]  = loo_delta(full) − loo_delta(full∖{i})
                       (how much of the bundle's win is lost by removing i;
                        negative = removing i hurts, i.e. i is pulling its weight)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def _summary(v) -> dict:
    """Extract the routing-relevant numbers from a ValidationResult (or None)."""
    if v is None:
        return {"converged": False, "loo_delta": None, "waic_delta": None,
                "divergences": None, "passed": False, "failure_reasons": []}
    failures = list(getattr(v, "failure_reasons", []) or [])
    return {
        "converged": bool(getattr(v, "converged", False)),
        "loo_delta": getattr(v, "loo_delta", None),
        "waic_delta": getattr(v, "waic_delta", None),
        "divergences": getattr(v, "divergences", None),
        "passed": bool(getattr(v, "converged", False) and not failures),
        "failure_reasons": failures,
    }


def _proposal_name(p) -> str:
    if hasattr(p, "model_dump"):
        p = p.model_dump()
    return (p.get("name") if isinstance(p, dict) else str(p)) or "?"


def _diff(a, b):
    """a - b with None-safety (returns None if either is missing)."""
    return None if (a is None or b is None) else float(a) - float(b)


def ablate_proposals(
    original_code: str,
    proposals: Sequence,
    data: dict,
    *,
    synthesize: Callable[[list], str],
    validate: Callable[[str], Any],
    framework: str = "numpyro",
) -> dict:
    """Both-directions per-proposal ablation. Returns an attribution dict.

    ``synthesize(subset) -> code`` applies a list of proposals to the original
    model (the real synthesizer, or a stub in tests). ``validate(code) ->
    ValidationResult`` runs MCMC for that code against the original
    (``run_full_validation`` bound to ``original_code``/``data``, or a stub). A
    subset that fails to synthesize/validate degrades to ``None`` (recorded, not
    raised), so one bad variant never aborts the whole ablation.
    """
    proposals = list(proposals)
    n = len(proposals)

    def _run(subset: list):
        try:
            code = synthesize(subset)
            if not (code or "").strip():
                return None
            return validate(code)
        except Exception:  # a dead subset must not kill the ablation
            return None

    full_v = _run(proposals)
    full = _summary(full_v)

    per_proposal: list[dict] = []
    for i, p in enumerate(proposals):
        standalone = _summary(_run([p]))                       # one-at-a-time
        loo_subset = proposals[:i] + proposals[i + 1:]         # leave-one-out
        leave_one_out = _summary(_run(loo_subset))
        per_proposal.append({
            "name": _proposal_name(p),
            "standalone": standalone,
            "leave_one_out": leave_one_out,
            # marginal contribution to the bundle's predictive win (neg = i helps)
            "contribution_loo": _diff(full["loo_delta"], leave_one_out["loo_delta"]),
            "contribution_waic": _diff(full["waic_delta"], leave_one_out["waic_delta"]),
            # "necessary": removing i flips a converged bundle from improved to not.
            # Require the leave-one-out fit to have ACTUALLY run (loo_delta is not
            # None) so a failed/None LOO fit is not misread as "necessary".
            "necessary": bool(
                full["passed"]
                and leave_one_out["loo_delta"] is not None
                and not leave_one_out["passed"]
            ),
        })

    return {
        "framework": framework,
        "n_proposals": n,
        "n_fits": 2 * n + 2,  # conceptual; the trace cache collapses duplicates
        "full": full,
        "per_proposal": per_proposal,
    }


def format_ablation(result: dict) -> str:
    """Deterministic markdown summary of an ablation result (no LLM)."""
    lines = [
        "## Per-proposal ablation",
        f"\n{result['n_proposals']} proposals · up to {result['n_fits']} MCMC fits "
        "(both directions: standalone + leave-one-out).\n",
        f"**Full set:** ΔLOO {result['full']['loo_delta']} · "
        f"converged {result['full']['converged']} · "
        f"{'PASS' if result['full']['passed'] else 'no significant win'}\n",
        "| proposal | standalone ΔLOO | leave-one-out ΔLOO | contribution | necessary |",
        "|---|---|---|---|---|",
    ]
    for p in result["per_proposal"]:
        lines.append(
            f"| {p['name']} | {p['standalone']['loo_delta']} | "
            f"{p['leave_one_out']['loo_delta']} | {p['contribution_loo']} | "
            f"{'yes' if p['necessary'] else 'no'} |"
        )
    return "\n".join(lines) + "\n"


def make_synth_validate(original_code, data, *, synthesizer_run, validation_fn,
                        framework="numpyro", make_state):
    """Wire the REAL synthesizer + validator into ``(synthesize, validate)`` for
    a live ablation run.

    ``synthesizer_run`` is a ``create_synthesizer_agent(llm)`` node and
    ``make_state(subset) -> WorkflowState`` seeds a state carrying the original
    model + the proposal subset; ``validation_fn`` is ``run_full_validation``.
    Kept thin and dependency-injected so production wiring lives at the call site
    and the core stays import-light + testable.
    """
    def synthesize(subset: list) -> str:
        state = make_state(subset)
        synthesizer_run(state)
        return state.refined_model_code

    def validate(code: str):
        return validation_fn(code, original_code, data, framework=framework)

    return synthesize, validate


__all__ = ["ablate_proposals", "format_ablation", "make_synth_validate"]
