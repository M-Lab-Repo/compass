"""Shared workflow state — the contract every agent reads from and writes to.

Spec reference: SPEC_bayesian_refinement_system.md §4. Defined before any
agent; flows through the LangGraph graph as a dataclass. Semantic model
representation (`ModelSpec`) is deliberately separate from executable code
(`WorkflowState.refined_model_code`): the analyst produces the former, the
synthesizer produces the latter.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Allowed values for WorkflowState.status
STATUSES = (
    "initialized",
    "analyzing",
    "analyzed",
    "searching",
    "searched",
    "proposing",
    "proposed",
    "synthesizing",
    "synthesized",
    "validating",
    "complete",
    "failed",
)


@dataclass
class ModelSpec:
    """Semantic representation of a probabilistic model (framework + structure)."""

    framework: str = ""  # "numpyro" or "stan"
    source_code: str = ""
    parameters: list = field(default_factory=list)
    priors: dict = field(default_factory=dict)
    likelihood: str = ""
    hierarchical_structure: dict = field(default_factory=dict)
    data_shapes: dict = field(default_factory=dict)


@dataclass
class PaperFinding:
    """One literature hit that survived the scout's retrieval funnel."""

    paper_title: str = ""
    arxiv_id: str = ""
    year: int = 0
    relevance_score: float = 0.0  # 0-1
    technique: str = ""
    applicable_to: str = ""  # which part of the model
    summary: str = ""
    code_snippet: str | None = None


@dataclass
class ValidationResult:
    """Deterministic verdict from the validator (no LLM involved)."""

    converged: bool = False
    rhat_max: float = 0.0
    ess_min: float = 0.0
    divergences: int = 0
    waic: float = 0.0
    loo: float = 0.0
    waic_delta: float | None = None  # vs original; negative = better
    loo_delta: float | None = None
    # the S-gate margin itself (az.compare convention: negative elpd_diff
    # favors refined; SE on the sum scale) — needed by the tournament's
    # ties-as-ties reporting and by the repair clause:
    elpd_diff: float | None = None
    elpd_diff_se: float | None = None
    khat_max: float | None = None  # worst Pareto-k from PSIS-LOO (descriptive only)
    # which instrument adjudicated the significance gate; None = default
    # PSIS-LOO, else e.g. "kfold_unit(K=5) over plate 'unit' [auto-escalated:
    # PSIS k-hat 3.20 > 0.7]" (latent-variable models, crossval.py):
    s_gate: str | None = None
    posterior_predictive_check: bool = False
    # mixed-replication PPC + unit-aware statistics (v4, DESCRIPTIVE — never
    # gates): {passed: bool|None, details: [...], local_sites: [...]} | None.
    # Redraws local latents from p(t|theta_s) before replicating — the check
    # with power against latent-law misspecification and the sponge:
    ppc_mixed: dict | None = None
    # repair-equivalence clause (v4): set (with the reason text) when a
    # refinement was ACCEPTED as a computational repair — baseline failed the
    # convergence gates, refinement passes them, and the predictive
    # comparison is TOST-equivalent (|delta| <= 2*SE and CI within ±4 nats).
    # None on every ordinary accept/refuse:
    repair: str | None = None
    failure_reasons: list = field(default_factory=list)
    # --- MEDEA / asta-theorizer borrows (default-safe, computed by the validator) ---
    confidence: str = ""  # "" until scored; then HIGH | MEDIUM | LOW (margin-calibrated)
    proposal_scores: list = field(default_factory=list)
    # per-proposal verdicts vs each proposal's falsifiable prediction:
    # [{name, metric, direction, predicted, observed, passed, note}]
    # --- Calibration-Not-Compilation [13] borrows (descriptive; never routing) ---
    # repair hints localizing a failed diagnostic to a bug type + fix class:
    misspecification_hints: list = field(default_factory=list)
    # out-of-sample confirmation for the winner (heldout_predictive_check); dict
    # {available, refined_lppd, original_lppd, lppd_delta, refined_better, ...}:
    heldout: dict | None = None


@dataclass
class WorkflowState:
    """Everything the orchestrator routes on. One instance flows through the graph."""

    # Inputs
    problem_description: str = ""
    dataset_summary: str = ""
    original_model: ModelSpec | None = None
    # Agent outputs
    analysis: dict = field(default_factory=dict)
    literature_findings: list = field(default_factory=list)
    proposals: list = field(default_factory=list)
    refined_model_code: str = ""
    validation: ValidationResult | None = None
    report: str = ""  # final markdown report (prepare_report node, spec §5.6)
    # Control flow
    iteration: int = 0
    max_iterations: int = 3
    status: str = "initialized"
    error_log: list = field(default_factory=list)
    feedback_history: list = field(default_factory=list)
    # every validated candidate (pass OR fail), in order: refined_model_code /
    # validation are overwritten each iteration, so this is the only record of
    # the refused programs the benchmark scores on the hidden fold.
    candidate_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Plain-dict view (for checkpointing, logging, fixture promotion)."""
        return asdict(self)
