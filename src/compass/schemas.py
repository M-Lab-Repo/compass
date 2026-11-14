"""Pydantic schemas for every LLM structured output in the system.

Single source of truth: the analyst emits `ModelAnalysis`, the scout's
applicability filter emits `PaperApplicability`, the strategist emits
`StrategyOutput`, the synthesizer emits `SynthesizedCode`. Keeping them in one
module (rather than per-agent) prevents schema drift between the agent that
produces an object and the agents/fixtures that consume it.

Spec references: §5.1 (ModelAnalysis), §5.2 (applicability filter),
§5.3 (Proposal/StrategyOutput), §5.4 (synthesizer).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

CHANGE_TYPES = ("prior", "hierarchy", "likelihood", "structure")

# Evidence provenance for a proposal (MEDEA borrow): is the refinement grounded
# in a real literature finding, entailed by the model's structure (the citation
# is incidental — the reason is geometry/identifiability), or an unbacked LLM
# hunch? DESCRIPTIVE ONLY: surfaced in the report's grounding block. It is
# deliberately kept OUT of the validator's scoring and the tournament's winner
# selection — an LLM-declared provenance tag must never influence a ground-truth
# decision (the MCMC validator stays the sole arbiter).
EVIDENCE_BASES = ("literature", "model_structure", "llm_prior")

# Validator-measured quantities a falsifiable prediction can target (asta-theorizer
# borrow). Each maps to a field the no-LLM validator computes, so a prediction is
# scored against real MCMC ground truth — never against an LLM judge.
PREDICTION_METRICS = (
    "divergences",   # good = 0
    "rhat_max",      # good = below the R-hat threshold
    "ess_min",       # good = above the ESS threshold
    "ebfmi_min",     # good = above the E-BFMI threshold
    "loo_delta",     # good = negative (refined predicts better; deviance scale)
    "waic_delta",    # good = negative
    "ppc",           # good = posterior predictive check passes
)


class ModelAnalysis(BaseModel):
    """Analyst output: complete structural read of the user's model."""

    framework: str = Field(description="numpyro or stan")
    parameters: list[dict] = Field(
        description="List of {name, prior_dist, hyperparams, role}"
    )
    likelihood: dict = Field(
        description="{family, link_function, observed_variable}"
    )
    hierarchical_levels: list[dict] = Field(
        description="List of {grouping_var, n_groups, params_affected}"
    )
    potential_issues: list[dict] = Field(
        description="List of {issue, severity, suggestion}; severity in low|medium|high"
    )
    data_summary_needed: list[str] = Field(
        description="What dataset characteristics matter for refinement"
    )


class PaperApplicability(BaseModel):
    """Scout's per-paper LLM applicability filter verdict (spec §5.2 step 4)."""

    score: int = Field(ge=0, le=10, description="Applicability 0-10; <6 drop, >=8 fetch method section")
    addresses_weakness: bool = Field(description="Does it address a real weakness in THIS model?")
    framework_compatible: bool = Field(description="Compatible with the model's framework?")
    data_size_appropriate: bool = Field(description="Appropriate for the dataset size N?")
    scope: str = Field(description="'component' (modify specific parts) or 'restructure' (whole model)")
    effort_to_improvement: str = Field(description="Expected effort-to-improvement ratio, concretely")
    what_would_change: str = Field(description="Concretely what in the model would change")


class FalsifiablePrediction(BaseModel):
    """A single operational, validator-scorable prediction for one proposal
    (asta-theorizer borrow).

    The strategist states which measurable diagnostic the proposal should move
    and in which direction; the no-LLM validator later scores it against the
    real MCMC result (`agents.validator.score_proposal_predictions`). This makes
    each proposal individually FALSIFIABLE by ground truth, so the feedback /
    ban-list can say which proposal failed which prediction — not just that the
    bundle failed.
    """

    metric: str = Field(
        description="Validator quantity this proposal targets: one of "
        "divergences | rhat_max | ess_min | ebfmi_min | loo_delta | waic_delta | ppc"
    )
    direction: str = Field(
        default="improve",
        description="Operational direction of the predicted effect: 'improve' "
        "(the canonical good direction for the metric) or decrease|increase|pass",
    )
    rationale: str = Field(
        default="",
        description="Why the proposal should move that metric — what SUPPORT looks like",
    )
    failure_mode: str = Field(
        default="",
        description="What a CONTRADICTION looks like — the observation that falsifies it",
    )


class Proposal(BaseModel):
    """One concrete, citable, ranked refinement (strategist output item)."""

    name: str
    description: str
    change_type: str = Field(description="prior|hierarchy|likelihood|structure")
    changes: list[dict] = Field(
        description=(
            '[{"parameter": "sigma", "current": "HalfNormal(1)", '
            '"proposed": "HalfCauchy(2.5)", "reason": "..."}]'
        )
    )
    supporting_papers: list[str]
    expected_improvement: str
    risk_assessment: str
    priority: int = Field(ge=1, description="1 = highest priority")
    combine_with: list[str] = Field(default_factory=list)
    # --- asta-theorizer / MEDEA borrows (optional, default-safe) -------------
    prediction: FalsifiablePrediction | None = Field(
        default=None,
        description="The proposal's falsifiable, validator-scorable prediction",
    )
    evidence_basis: str = Field(
        default="llm_prior",
        description="Provenance: literature | model_structure | llm_prior",
    )


class StrategyOutput(BaseModel):
    """Strategist output: reasoning + 3-5 ranked proposals + ordering."""

    reasoning: str
    proposals: list[Proposal]
    implementation_order: list[str]


class SynthesizedCode(BaseModel):
    """Synthesizer LLM output: the refined model module, ready to smoke-test."""

    code: str = Field(description="Complete runnable Python (NumPyro) or Stan program text")
    applied_proposals: list[str] = Field(description="Names of the proposals actually applied")
    notes: str = Field(default="", description="Implementation notes / deviations from proposals")


class JudgeVerdict(BaseModel):
    """An LLM judge's verdict on whether a refined model improves on the original.

    This is the BASELINE arbiter for the v3 evaluation (judge.py): it judges the
    code WITHOUT running MCMC, so the eval can quantify how often an LLM judge
    accepts a model that real MCMC rejects (the false-acceptance the system's
    MCMC-as-sole-arbiter design avoids). It is NEVER wired into the pipeline as
    an arbiter — the MCMC validator remains the sole arbiter.
    """

    passed: bool = Field(
        description="Is the refined model a genuine improvement over the original?"
    )
    confidence: str = Field(default="medium", description="high | medium | low")
    reasoning: str = Field(description="Brief justification (no MCMC was run)")
