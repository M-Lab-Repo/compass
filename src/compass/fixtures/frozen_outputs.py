"""Hand-written expected agent outputs for the GOLDEN model (spec §9, §10).

The fixture cascade: each agent's output is the test fixture for the next
agent downstream. These frozen objects are the hand-written starting point;
once an agent passes its tests, `scripts/promote_fixture.py <agent>` runs it
on the golden model and writes `fixtures/real_<name>.json`, which
`load_real_or_frozen(<name>)` then prefers over the frozen object.

Golden model (fixtures/golden_model.py, spec §10): a deliberately flawed
NumPyro hierarchical regression over 8 schools —

    mu_alpha    ~ Normal(0, 10)                  # vague hyperprior
    sigma_alpha ~ HalfNormal(10)                 # vague hyperprior
    alpha       ~ Normal(mu_alpha, sigma_alpha)  # CENTERED, plate of 8 groups
    beta        ~ Normal(0, 100)                 # absurdly vague slope prior
    sigma       ~ HalfNormal(10)
    y           ~ Normal(alpha[group_idx] + X * beta, sigma)

Known correct refinement: non-centered parameterization + scale-appropriate
hyperpriors + a regularized slope prior.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from compass.schemas import FalsifiablePrediction, ModelAnalysis, Proposal, StrategyOutput
from compass.state import PaperFinding, ValidationResult

FIXTURES_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Canonical paper titles (spec §12) — single source of truth within this module
# ---------------------------------------------------------------------------
PAPER_NONCENTERED = "A General Framework for the Parametrization of Hierarchical Models"
PAPER_PRIORS = "The Prior Can Often Only Be Understood in the Context of the Likelihood"
PAPER_LOO = (
    "Practical Bayesian Model Evaluation Using Leave-One-Out "
    "Cross-Validation and WAIC"
)
PAPER_HORSESHOE = (
    "Sparsity Information and Regularization in the Horseshoe "
    "and Other Shrinkage Priors"
)

# ---------------------------------------------------------------------------
# FROZEN_ANALYSIS — what the Model Analyst must say about the golden model
# ---------------------------------------------------------------------------
FROZEN_ANALYSIS = ModelAnalysis(
    framework="numpyro",
    parameters=[
        {
            "name": "mu_alpha",
            "prior_dist": "Normal(0, 10)",
            "hyperparams": {"loc": 0.0, "scale": 10.0},
            "role": "group_mean_hyperprior",
        },
        {
            "name": "sigma_alpha",
            "prior_dist": "HalfNormal(10)",
            "hyperparams": {"scale": 10.0},
            "role": "group_scale_hyperprior",
        },
        {
            "name": "alpha",
            "prior_dist": "Normal(mu_alpha, sigma_alpha)",
            "hyperparams": {"loc": "mu_alpha", "scale": "sigma_alpha"},
            "role": "group_intercept",
        },
        {
            "name": "beta",
            "prior_dist": "Normal(0, 100)",
            "hyperparams": {"loc": 0.0, "scale": 100.0},
            "role": "slope",
        },
        {
            "name": "sigma",
            "prior_dist": "HalfNormal(10)",
            "hyperparams": {"scale": 10.0},
            "role": "observation_noise",
        },
    ],
    likelihood={
        "family": "normal",
        "link_function": "identity",
        "observed_variable": "y",
    },
    hierarchical_levels=[
        {"grouping_var": "group_idx", "n_groups": 8, "params_affected": ["alpha"]}
    ],
    potential_issues=[
        {
            "issue": (
                "alpha uses a centered parameterization "
                "(alpha ~ Normal(mu_alpha, sigma_alpha) inside the groups plate) "
                "with only 8 groups; this funnel geometry causes divergent "
                "transitions and poor HMC mixing"
            ),
            "severity": "high",
            "suggestion": "non-centered parameterization hierarchical model",
        },
        {
            "issue": (
                "beta has an absurdly vague prior Normal(0, 100) for a regression "
                "slope on study hours; the prior is ~50x wider than any plausible "
                "effect size given an outcome sd of 15"
            ),
            "severity": "high",
            "suggestion": "weakly informative prior regression coefficient",
        },
        {
            "issue": (
                "hyperpriors mu_alpha ~ Normal(0, 10) and sigma_alpha ~ "
                "HalfNormal(10) are vague relative to the outcome scale (test "
                "scores 40-100); Normal(0, 10) actually places most hyperprior "
                "mass far below the observed mean of ~72"
            ),
            "severity": "medium",
            "suggestion": "prior choice hierarchical scale parameter",
        },
    ],
    data_summary_needed=[
        "outcome scale: mean and standard deviation of test scores",
        "predictor scale: mean and standard deviation of study_hours",
        "group sizes: number of students per school",
        "presence of outliers in the outcome",
    ],
)

# ---------------------------------------------------------------------------
# FROZEN_LITERATURE — what the Literature Scout must find (canonical papers
# 1, 2, 4, 5 of spec §12). The Piironen & Vehtari horseshoe paper is a REAL
# finding but a judgment trap: the strategist must NOT propose it for this
# simple single-predictor hierarchical regression.
# ---------------------------------------------------------------------------
NONCENTERED_SNIPPET = '''with numpyro.plate("groups", N_groups):
    z = numpyro.sample("z", dist.Normal(0.0, 1.0))
alpha = numpyro.deterministic("alpha", mu_alpha + sigma_alpha * z)'''

FROZEN_LITERATURE = [
    PaperFinding(
        paper_title=PAPER_NONCENTERED,
        arxiv_id="0708.3797",
        year=2007,
        relevance_score=0.95,
        technique="non-centered parameterization",
        applicable_to="alpha (group intercepts drawn in the 8-group plate)",
        summary=(
            "Shows that centered parameterizations of hierarchical models mix "
            "poorly when groups are few and group-level variance is weakly "
            "identified; reparameterizing alpha = mu_alpha + sigma_alpha * z "
            "with z ~ Normal(0, 1) removes the funnel geometry that causes "
            "divergent transitions in HMC."
        ),
        code_snippet=NONCENTERED_SNIPPET,
    ),
    PaperFinding(
        paper_title=PAPER_PRIORS,
        arxiv_id="1708.07487",
        year=2017,
        relevance_score=0.85,
        technique="weakly informative priors",
        applicable_to="beta, mu_alpha, sigma_alpha (vague priors)",
        summary=(
            "Argues priors must be chosen relative to the likelihood and the "
            "scale of the data: with test scores of mean 72 and sd 15, "
            "Normal(0, 100) on the slope and Normal(0, 10)/HalfNormal(10) on "
            "the hyperparameters are uninformative in the wrong places; "
            "scale-appropriate weakly informative priors regularize inference "
            "without dominating the data."
        ),
        code_snippet=None,
    ),
    PaperFinding(
        paper_title=PAPER_LOO,
        arxiv_id="1510.07151",
        year=2017,
        relevance_score=0.7,
        technique="LOO-CV and WAIC model comparison (PSIS-LOO)",
        applicable_to="model comparison",
        summary=(
            "Practical estimators (PSIS-LOO, WAIC) for out-of-sample predictive "
            "accuracy; the validator uses these to compare the refined model "
            "against the original (negative delta = improvement)."
        ),
        code_snippet=None,
    ),
    PaperFinding(
        paper_title=PAPER_HORSESHOE,
        arxiv_id="1707.01694",
        year=2017,
        relevance_score=0.55,
        technique="regularized horseshoe shrinkage prior",
        applicable_to="beta (regression coefficients, if many and sparse)",
        summary=(
            "Develops the regularized horseshoe for sparse regression with many "
            "predictors, where most coefficients are near zero. Topically "
            "related to the vague slope prior, but the golden model has a "
            "single predictor with a real effect, so sparsity-inducing "
            "shrinkage is not appropriate here."
        ),
        code_snippet=None,
    ),
]

# ---------------------------------------------------------------------------
# FROZEN_PROPOSALS — what the Refinement Strategist must propose (spec §10's
# three expected refinements, ranked).
# ---------------------------------------------------------------------------
FROZEN_PROPOSALS = StrategyOutput(
    reasoning=(
        "The analysis flags two high-severity issues — a centered "
        "parameterization of alpha over only 8 groups, and an absurdly vague "
        "Normal(0, 100) prior on beta — plus vague hyperpriors relative to the "
        "outcome scale (test scores, mean 72, sd 15). The literature findings "
        "map directly onto these: Papaspiliopoulos et al. (2007) for the "
        "non-centered reparameterization that removes the funnel, and Gelman "
        "et al. (2017) for scale-appropriate weakly informative priors on the "
        "hyperparameters and the slope. The Piironen & Vehtari (2017) "
        "regularized horseshoe finding was considered and REJECTED: it targets "
        "sparse regression with many predictors, while this model has a single "
        "predictor with a clearly nonzero effect, so a simple Normal(0, 10) "
        "regularizing prior is the right tool. Vehtari et al. (2017) LOO/WAIC "
        "is used for validation-stage comparison, not as a model change."
    ),
    proposals=[
        Proposal(
            name="non_centered_parameterization",
            description=(
                "Reparameterize the group intercepts non-centrally: draw "
                "standard-normal z in the groups plate and set "
                "alpha = mu_alpha + sigma_alpha * z, eliminating the funnel "
                "between alpha and sigma_alpha with only 8 groups."
            ),
            change_type="hierarchy",
            changes=[
                {
                    "parameter": "alpha",
                    "current": "alpha ~ Normal(mu_alpha, sigma_alpha)",
                    "proposed": "z ~ Normal(0,1); alpha = mu_alpha + sigma_alpha * z",
                    "reason": (
                        "Centered hierarchical parameterizations with few groups "
                        "(8 schools) create funnel geometry that HMC cannot "
                        "traverse, producing divergent transitions and biased "
                        "sigma_alpha estimates."
                    ),
                }
            ],
            supporting_papers=[PAPER_NONCENTERED],
            expected_improvement=(
                "Eliminates divergent transitions, brings R-hat to ~1.0, and "
                "substantially raises effective sample size for sigma_alpha "
                "and alpha."
            ),
            risk_assessment=(
                "Low: purely computational reparameterization; the implied "
                "posterior is identical, only the sampling geometry changes."
            ),
            priority=1,
            combine_with=["weakly_informative_hyperpriors"],
            evidence_basis="model_structure",
            prediction=FalsifiablePrediction(
                metric="divergences",
                direction="improve",
                rationale=(
                    "Removing the centered funnel lets HMC explore sigma_alpha, "
                    "so divergent transitions should fall to zero."
                ),
                failure_mode=(
                    "Divergent transitions persist (or R-hat stays above 1.01)."
                ),
            ),
        ),
        Proposal(
            name="weakly_informative_hyperpriors",
            description=(
                "Replace the vague hyperpriors with weakly informative priors "
                "matched to the outcome scale (test scores: mean 72, sd 15)."
            ),
            change_type="prior",
            changes=[
                {
                    "parameter": "mu_alpha",
                    "current": "mu_alpha ~ Normal(0, 10)",
                    "proposed": "mu_alpha ~ Normal(70, 20)",
                    "reason": (
                        "Test scores have mean 72 (sd 15); Normal(0, 10) puts "
                        "almost no mass near the observed outcome scale, while "
                        "Normal(70, 20) is centered on it and still weakly "
                        "informative."
                    ),
                },
                {
                    "parameter": "sigma_alpha",
                    "current": "sigma_alpha ~ HalfNormal(10)",
                    "proposed": "sigma_alpha ~ HalfNormal(5)",
                    "reason": (
                        "Between-school spread plausibly spans a fraction of the "
                        "outcome sd of 15; HalfNormal(5) keeps mass on realistic "
                        "group-level scales instead of allowing implausibly "
                        "large ones."
                    ),
                },
            ],
            supporting_papers=[PAPER_PRIORS],
            expected_improvement=(
                "Better-identified hyperparameters, faster mixing, and prior "
                "predictive draws on the actual scale of test scores."
            ),
            risk_assessment=(
                "Low: priors remain weakly informative; with N=280 the "
                "likelihood dominates, so inferences are insensitive to the "
                "exact constants."
            ),
            priority=2,
            combine_with=[],
            evidence_basis="literature",
            prediction=FalsifiablePrediction(
                metric="ess_min",
                direction="improve",
                rationale=(
                    "Better-identified hyperparameters mix faster, raising the "
                    "minimum bulk ESS above the 400 threshold."
                ),
                failure_mode="ESS_min stays below the 400 threshold.",
            ),
        ),
        Proposal(
            name="regularize_slope_prior",
            description=(
                "Replace the absurdly vague slope prior with a weakly "
                "informative regularizing prior."
            ),
            change_type="prior",
            changes=[
                {
                    "parameter": "beta",
                    "current": "beta ~ Normal(0, 100)",
                    "proposed": "beta ~ Normal(0, 10)",
                    "reason": (
                        "A study-hours effect anywhere near 100 points per hour "
                        "is impossible when scores range 40-100; Normal(0, 10) "
                        "still comfortably covers any plausible slope."
                    ),
                }
            ],
            supporting_papers=[PAPER_PRIORS],
            expected_improvement=(
                "Mild regularization of the slope and a sane prior predictive "
                "distribution; protects against pathological tail behavior."
            ),
            risk_assessment=(
                "Low: the data strongly identify beta, so the posterior barely "
                "moves; only the prior tails are tamed."
            ),
            priority=3,
            combine_with=[],
            evidence_basis="literature",
            prediction=FalsifiablePrediction(
                metric="loo_delta",
                direction="improve",
                rationale=(
                    "Mild regularization of the slope improves out-of-sample "
                    "predictive accuracy (negative LOO delta on the deviance scale)."
                ),
                failure_mode="LOO does not improve (loo_delta >= 0).",
            ),
        ),
    ],
    implementation_order=[
        "non_centered_parameterization",
        "weakly_informative_hyperpriors",
        "regularize_slope_prior",
    ],
)

# ---------------------------------------------------------------------------
# Frozen validator verdicts — used by routing tests (success vs retry paths)
# ---------------------------------------------------------------------------
FROZEN_VALIDATION_PASS = ValidationResult(
    converged=True,
    rhat_max=1.003,
    ess_min=1200.0,
    divergences=0,
    waic=1694.2,
    loo=1694.8,
    waic_delta=-18.5,
    loo_delta=-17.9,
    posterior_predictive_check=True,
    failure_reasons=[],
)

FROZEN_VALIDATION_FAIL = ValidationResult(
    converged=False,
    rhat_max=1.08,
    ess_min=42.0,
    divergences=47,
    waic=1712.7,
    loo=1713.3,
    waic_delta=None,
    loo_delta=None,
    posterior_predictive_check=False,
    failure_reasons=["47 divergent transitions", "R-hat too high: 1.08 (tau)"],
)


# ---------------------------------------------------------------------------
# Promote-fixture cascade: prefer the real (promoted) JSON when it exists
# ---------------------------------------------------------------------------
def load_real_or_frozen(name: str):
    """Return fixtures/real_<name>.json if promoted, else the frozen object as
    plain dicts/lists. ``name`` is one of "analysis", "literature", "proposals".
    """
    frozen_dumpers = {
        "analysis": lambda: FROZEN_ANALYSIS.model_dump(),
        "literature": lambda: [asdict(f) for f in FROZEN_LITERATURE],
        "proposals": lambda: FROZEN_PROPOSALS.model_dump(),
    }
    if name not in frozen_dumpers:
        raise ValueError(
            f"unknown fixture name {name!r}; expected one of {sorted(frozen_dumpers)}"
        )
    real_path = FIXTURES_DIR / f"real_{name}.json"
    if real_path.exists():
        with real_path.open() as f:
            return json.load(f)
    return frozen_dumpers[name]()
